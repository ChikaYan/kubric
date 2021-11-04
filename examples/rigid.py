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
from kubric.assets import asset_source
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import sys
import imageio
import bpy
import random

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

ROT_CAM = False
OBJNAME = 'textured_cube'

random.seed(0)

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


velocity = np.array([2, 0, -1])
# color = kb.random_hue_color()
# input(color)
color = kb.Color(r=1, g=0.1, b=0.1, a=1.0)
# quaternion = [0.871342, 0.401984, -0.177436, 0.218378]
material = kb.PrincipledBSDFMaterial(color=color)
cube = kb.Cube(name='cube', scale=0.3, velocity=velocity, angular_velocity=[0,0,0], position=(0, 0, 0), mass=0.2, restitution=1, material=material, friction=1, segmentation_id=2)
# segmentation id doesn't seem to be working -- the segmentation mask still uses object id



bpy_scene = bpy.context.scene
cube.material = kb.PrincipledBSDFMaterial(name="material")
cube.material.metallic = 0
cube.material.specular = 1
cube.material.roughness = random.random()**0.2
scene += cube

mat = bpy_scene.objects[f"cube"].active_material
tree = mat.node_tree

mat_node = tree.nodes["Principled BSDF"]
ramp_node = tree.nodes.new(type="ShaderNodeValToRGB")
tex_node = tree.nodes.new(type="ShaderNodeTexNoise")
scaling_node = tree.nodes.new(type="ShaderNodeMapping")
rotation_node = tree.nodes.new(type="ShaderNodeMapping")
vector_node = tree.nodes.new(type="ShaderNodeNewGeometry")

tree.links.new(vector_node.outputs["Position"], rotation_node.inputs["Vector"])
tree.links.new(rotation_node.outputs["Vector"], scaling_node.inputs["Vector"])
tree.links.new(scaling_node.outputs["Vector"], tex_node.inputs["Vector"])
tree.links.new(tex_node.outputs["Fac"], ramp_node.inputs["Fac"])
tree.links.new(ramp_node.outputs["Color"], mat_node.inputs["Base Color"])

rotation_node.inputs["Rotation"].default_value = (
    random.random() * 3.141,
    random.random() * 3.141,
    random.random() * 3.141,
)

scaling_node.inputs["Scale"].default_value = (
    random.random()**2 * 2.0,
    random.random()**2 * 2.0,
    random.random()**2 * 2.0,
)

tex_node.inputs["Roughness"].default_value = random.random()
tex_node.inputs["Detail"].default_value = 10.0 * random.random()

for i in range(random.randint(3, 6)):
    ramp_node.color_ramp.elements.new(random.random())

base_color = color
for element in ramp_node.color_ramp.elements:
    mult = random.random()**2
    element.color = (
        0.3 * random.random() + base_color.r * mult,
        0.3 * random.random() + base_color.g * mult,
        0.3 * random.random() + base_color.b * mult,
        1
    )


scene += cube

# # set up assets
# asset_source = kb.AssetSource("examples/KuBasic")

# torus = asset_source.create(name="torus",
#                           asset_id='Torus', scale=0.5)
# torus.material = kb.PrincipledBSDFMaterial(color=kb.Color(r=1, g=0.030765511645494348, b=0.0, a=1.0), metallic=0., ior=1.25,
#                 roughness=0.7, specular=0.33)
# torus.position = (0, 0, 1)
# torus.velocity = (0.5, 0, -1)
# scene += torus

# gso = kb.AssetSource("gs://kubric-public/GSO")
# train_split, test_split = asset_source.get_test_split(fraction=0.1)
# # textured_cube = gso.create("gs://kubric-public/GSO")

# import pdb; pdb.set_trace()




# Render cameras at the same general distance from the origin, but at
# different positions.
#
# We will use spherical coordinates (r, theta, phi) to do this.
#   x = r * cos(theta) * sin(phi)
#   y = r * sin(theta) * sin(phi)
#   z = r * cos(phi)
if ROT_CAM:
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
renderer.save_state(f"output/{OBJNAME}/{OBJNAME}.blend")
frames_dict = renderer.render()

# del frames_dict["uv"]
# del frames_dict["forward_flow"]
# del frames_dict["backward_flow"]
# del frames_dict["depth"]
# del frames_dict["normal"]

import pickle
with open(f'output/{OBJNAME}/frames.dict', 'wb') as file:
    pickle.dump(frames_dict, file)

# kb.write_image_dict(frames_dict, f"output/{OBJNAME}")


# convert segmentation mask to LASR style
palette = [[0,0,0],[0,0,0],[128,128,128],[128,128,128],[128,128,128],[128,128,128]]
kb.file_io.multi_write_image(frames_dict['segmentation'], str(kb.as_path(f"output/{OBJNAME}/LASR/Annotations/Full-Resolution/{OBJNAME}") / "{:05d}.png"), write_fn=kb.write_palette_png,
                    max_write_threads=16, palette=palette)
# kb.file_io.write_rgba_batch(frames_dict['rgba'], str(kb.as_path("output/rigid/rgba") / "{:05d}.png"))
kb.file_io.multi_write_image(frames_dict['rgba'], str(kb.as_path(f"output/{OBJNAME}/LASR/JPEGImages/Full-Resolution/{OBJNAME}") / "{:05d}.png"), write_fn=kb.write_png,
                    max_write_threads=16)


# write optical flow and occlusion map in LASR format
def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n".encode() if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)



fw = frames_dict['forward_flow'][:-1,...] * 256
bw = frames_dict['backward_flow'][1:,...] * 256
imgs = frames_dict['rgba']
M, N = imgs.shape[1:3]

occs = np.ones(fw.shape[:-1]).astype('float32')

# for img_i in range(len(imgs)-1):
#     img1, img2 = imgs[img_i,...], imgs[img_i+1,...]
#     f = fw[img_i,...] # corresponding forward floAw image
#     b = bw[img_i,...] # corresponding backward flow image
    
#     # loop through all pixel to check occlusion
#     for i in range(M):
#         for j in range(N):
#             # flow forward
#             fi, fj = np.round(np.array([i,j]) + f[i,j]).astype(int)

#             # ignore the pixel if it goes out of range
#             if not (0<=fi<M  and 0<=fj<N):
#                 continue
            
#             # flow backward
#             bi, bj = np.round(np.array([fi,fj]) - b[fi,fj]).astype(int)

#             THRESHOLD = 1
#             # occlusion detected
#             if np.abs(i - bi) + np.abs(j - bj) > THRESHOLD:
#                 occs[img_i,i,j] = 1
#                 # pdb.set_trace()

import os
os.makedirs(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}',exist_ok=True)
os.makedirs(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}',exist_ok=True)

# write flows into pfm
for i in range(len(fw)):
    f = fw[i,...]
    ones = np.ones_like(f[...,:1])
    f = np.concatenate([ones,f],-1)
    b = np.concatenate([ones,bw[i,...]],-1)
    
    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/flo-{i:05d}.pfm',f)
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/flo-{i+1:05d}.pfm',b)
    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/occ-{i:05d}.pfm',np.ones_like(occs[i,...]))
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/occ-{i+1:05d}.pfm',np.ones_like(occs[i,...]))
    

# write gif
imageio.mimsave(str(kb.as_path(f"output/{OBJNAME}/") / f"{OBJNAME}.gif"),frames_dict['rgba'])
kb.file_io.write_flow_batch(frames_dict['forward_flow'], directory= f"output/{OBJNAME}/FlowFW", file_template="{:05d}.png", name="forward_flow",
                    max_write_threads=16)
kb.file_io.write_flow_batch(frames_dict['backward_flow'], directory= f"output/{OBJNAME}/FlowBW", file_template="{:05d}.png", name="backward_flow",
                    max_write_threads=16)
