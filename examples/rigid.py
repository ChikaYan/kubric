# Copyright 2021 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 20   # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(5, 5, 0.1), position=(0, 0, -0.1), static=True, background=True, segmentation_id=1)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -2, 4), look_at=(0, 0, 0))


velocity = np.array([0.5, 0, -1])
color = kb.random_hue_color()
# quaternion = [0.871342, 0.401984, -0.177436, 0.218378]
material = kb.PrincipledBSDFMaterial(color=color)
cube = kb.Cube(scale=0.3, velocity=velocity, angular_velocity=[0,0,0], position=(0, 0, 1), mass=0.2, restitution=1, material=material, friction=1, segmentation_id=2)
# segmentation id doesn't seem to be working -- the segmentation mask still uses object id
scene += cube

# Render cameras at the same general distance from the origin, but at
# different positions.
#
# We will use spherical coordinates (r, theta, phi) to do this.
#   x = r * cos(theta) * sin(phi)
#   y = r * sin(theta) * sin(phi)
#   z = r * cos(phi)
original_camera_position = (2, -2, 4)
r = np.sqrt(sum(a * a for a in original_camera_position))
phi = np.arccos(original_camera_position[2] / r) # (180 - elevation)
theta = np.arccos(original_camera_position[0] / (r * np.sin(phi))) # azimuth
num_phi_values_per_theta = 1
theta_change = (2 * np.pi) / ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

for frame in range(scene.frame_start, scene.frame_end + 1):
  i = (frame - scene.frame_start)
  theta_new = (i // num_phi_values_per_theta) * theta_change + theta

  # These values of (x, y, z) will lie on the same sphere as the original camera.
  x = r * np.cos(theta_new) * np.sin(phi)
  y = r * np.sin(theta_new) * np.sin(phi)
  z = r * np.cos(phi)

  scene.camera.position = (x, y, z)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)



# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
kb.as_path("output").mkdir(exist_ok=True)
renderer.save_state("output/simulator.blend")
frames_dict = renderer.render()

del frames_dict["uv"]
del frames_dict["forward_flow"]
del frames_dict["backward_flow"]
del frames_dict["depth"]
del frames_dict["normal"]

# import pickle
# with open('frames.dict', 'wb') as file:
#     pickle.dump(frames_dict, file)

# kb.write_image_dict(frames_dict, "output/rigid")


# convert segmentation mask to LASR style
palette = [[0,0,0],[0,0,0],[128,128,128],[128,128,128],[128,128,128],[128,128,128]]
kb.file_io.multi_write_image(frames_dict['segmentation'], str(kb.as_path("output/rigid/segmentation") / "{:05d}.png"), write_fn=kb.write_palette_png,
                    max_write_threads=16, palette=palette)
# kb.file_io.write_rgba_batch(frames_dict['rgba'], str(kb.as_path("output/rigid/rgba") / "{:05d}.png"))
kb.file_io.multi_write_image(frames_dict['rgba'], str(kb.as_path("output/rigid/rgba") / "{:05d}.png"), write_fn=kb.write_png,
                    max_write_threads=16)