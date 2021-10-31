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
cube = kb.Cube(scale=0.3, velocity=velocity, position=(0, 0, 1), mass=0.2, restitution=1, material=material, friction=1, segmentation_id=2)
# segmentation id doesn't seem to be working -- the segmentation mask still uses object id
scene += cube

# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
kb.as_path("output").mkdir(exist_ok=True)
renderer.save_state("output/simulator.blend")
frames_dict = renderer.render()

# import pickle
# with open('frames.dict', 'wb') as file:
#     pickle.dump(frames_dict, file)

kb.write_image_dict(frames_dict, "output/rigid")

from kubric.file_io import multi_write_image

# convert segmentation mask to LASR style
palette = [[0,0,0],[0,0,0],[128,128,128],[128,128,128],[128,128,128],[128,128,128]]
multi_write_image(frames_dict['segmentation'], str(kb.as_path("output/rigid") / "segmentation_{:05d}.png"), write_fn=kb.write_palette_png,
                    max_write_threads=16, palette=palette)
