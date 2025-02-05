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
from numpy.lib.function_base import append
import kubric as kb
from kubric.assets import asset_source
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import sys
import imageio
import bpy
import pdb
import random
from scipy.spatial import transform

logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

ROT_CAM = True
ROT_RANGE = np.pi / 4 # 2 * np.pi # 
OBJNAME = 'shapenet-less-rot'
POSITION = (0,0,1) #(0,0,0.2)
VELOCITY = (0.5,0,-1) # (4,-4,0) 
OBJ_TYPE = 'shapenet'
TEXTURE = False

random.seed(0)

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 30   # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(5, 5, 0.1), position=(0, 0, -0.1), static=True, background=True, segmentation_id=1)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3), look_at=(0, 0, 0), intensity=1.5)
scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -2, 4), look_at=(0, 0, 0))


# color = kb.random_hue_color()
color = kb.Color(r=1, g=0.1, b=0.1, a=1.0)
# quaternion = [0.871342, 0.401984, -0.177436, 0.218378]
material = kb.PrincipledBSDFMaterial(color=color)

if OBJ_TYPE == 'cube':
  obj = kb.Cube(name='cube', scale=0.3, velocity=VELOCITY, angular_velocity=[0,0,0], position=POSITION, mass=0.2, restitution=1, material=material, friction=1, segmentation_id=2)
  objname = 'cube'
  # segmentation id doesn't seem to be working -- the segmentation mask still uses object id

elif OBJ_TYPE == 'torus':
  # set up assets
  asset_source = kb.AssetSource("examples/KuBasic")

  obj = asset_source.create(name="torus",
                            asset_id='Torus', scale=0.5)
  objname = 'torus'
  obj.material = material # kb.PrincipledBSDFMaterial(color=kb.Color(r=1, g=0.030765511645494348, b=0.0, a=1.0), metallic=0., ior=1.25, roughness=0.7, specular=0.33)
  obj.position = POSITION
  obj.velocity = VELOCITY

elif OBJ_TYPE == 'shapenet':
  asset_source = kb.AssetSource('gs://tensorflow-graphics/public/60c9de9c410be30098c297ac/ShapeNetCore.v2')
  ids = list(asset_source.db.loc[asset_source.db['id'].str.startswith('02691156')]['id'])
  rng = np.random.RandomState(0)
  asset_id = rng.choice(ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87
  obj = asset_source.create(asset_id=asset_id)
  obj.position = POSITION
  obj.velocity = VELOCITY
  obj.metadata = {
    "asset_id": obj.asset_id,
    "category": asset_source.db[
      asset_source.db["id"] == obj.asset_id].iloc[0]["category_name"],
  }
  obj.scale = 2
  objname = obj.name
else:
  raise NotImplementedError

if TEXTURE:
  bpy_scene = bpy.context.scene
  obj.material = kb.PrincipledBSDFMaterial(name="material")
  obj.material.metallic = random.random()
  obj.material.roughness = random.random()**0.2

  scene += obj

  mat = bpy_scene.objects[objname].active_material
  tree = mat.node_tree

  mat_node = tree.nodes["Principled BSDF"]
  texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
  texImage.image = bpy.data.images.load('examples/tex/tex.jpg')
  tree.links.new(mat_node.inputs['Base Color'], texImage.outputs['Color'])
else:
  scene += obj


cam_params = []

if ROT_CAM:
  # Render cameras at the same general distance from the origin, but at
  # different positions.
  #
  # We will use spherical coordinates (r, theta, phi) to do this.
  #   x = r * cos(theta) * sin(phi)
  #   y = r * sin(theta) * sin(phi)
  #   z = r * cos(phi)
  original_camera_position = scene.camera.position
  r = np.sqrt(sum(a * a for a in original_camera_position))
  phi = np.arccos(original_camera_position[2] / r) # (180 - elevation)
  theta = np.arccos(original_camera_position[0] / (r * np.sin(phi))) # azimuth
  num_phi_values_per_theta = 1
  theta_change = ROT_RANGE / ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

  # pdb.set_trace()

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

    cam_param = np.zeros([1,8])
    quat = scene.camera.quaternion
    rot = transform.Rotation.from_quat(quat)
    inv_quat = rot.inv().as_quat()


    cam_param[0,0] = scene.camera.focal_length
    cam_param[0,1] = x
    cam_param[0,2] = y
    cam_param[0,3] = quat[3]
    cam_param[0,4:7] = quat[:3]
    cam_param[0,7] = z

    # cam_param[0,0] = scene.camera.focal_length
    # cam_param[0,1] = -x
    # cam_param[0,2] = -y
    # cam_param[0,3] = inv_quat[3]
    # cam_param[0,4:7] = inv_quat[:3]
    # cam_param[0,7] = -z

    cam_params.append(cam_param)

    # pdb.set_trace()
else:
    x,y,z = scene.camera.position
    cam_param = np.zeros([1,8])
    quat = scene.camera.quaternion
    rot = transform.Rotation.from_quat(quat)
    inv_quat = rot.inv().as_quat()

    cam_param[0,0] = scene.camera.focal_length
    cam_param[0,1] = x
    cam_param[0,2] = y
    cam_param[0,3] = quat[3]
    cam_param[0,4:7] = quat[:3]
    cam_param[0,7] = z

    # cam_param[0,0] = scene.camera.focal_length
    # cam_param[0,1] = -x
    # cam_param[0,2] = -y
    # cam_param[0,3] = inv_quat[3]
    # cam_param[0,4:7] = inv_quat[:3]
    # cam_param[0,7] = -z

    for _ in range(scene.frame_end):
        cam_params.append(cam_param)


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
os.makedirs(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}',exist_ok=True)
os.makedirs(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}',exist_ok=True)
os.makedirs(f'output/{OBJNAME}/LASR/Camera/Full-Resolution/{OBJNAME}',exist_ok=True)
os.makedirs(f'output/{OBJNAME}/LASR/Camera/Full-Resolution/r{OBJNAME}',exist_ok=True)

# write flows into pfm

# Kubric optical forward flow format: 
# fw[i,j,k] = [dy, dx] for pixel [j,k] from img[i] to img[i+1]
# Kubric optical backward flow format: 
# bw[i,j,k] = [-dy, -dx] for pixel [j,k] from img[i] to img[i-1]

# VCN optical forward flow format:
# fw[i,j,k] = [dx, dy] for pixel [256-j,k] from img[i] to img[i+1]
# VCN optical backward flow format:
# bw[i,j,k] = [dx, dy] for pixel [256-j,k] from img[i] to img[i-1]
for i in range(len(fw)):
    f = fw[i,...]
    ones = np.ones_like(f[...,:1])  
    f = np.concatenate([f[...,1:], f[...,:1], ones],-1)
    b = np.concatenate([-bw[i,...,1:],-bw[i,...,:1], ones],-1)

    f = np.flip(f,0)
    b = np.flip(b,0)
    
    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/flo-{i:05d}.pfm',f)
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/flo-{i+1:05d}.pfm',b)
    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/{OBJNAME}/occ-{i:05d}.pfm',np.ones_like(occs[i,...]))
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/{OBJNAME}/occ-{i+1:05d}.pfm',np.ones_like(occs[i,...]))

    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}/flo-{i:05d}.pfm',f)
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}/flo-{i+1:05d}.pfm',b)
    write_pfm(f'output/{OBJNAME}/LASR/FlowFW/Full-Resolution/r{OBJNAME}/occ-{i:05d}.pfm',np.ones_like(occs[i,...]))
    write_pfm(f'output/{OBJNAME}/LASR/FlowBW/Full-Resolution/r{OBJNAME}/occ-{i+1:05d}.pfm',np.ones_like(occs[i,...]))

for i in range(len(cam_params)):
    # save camera parameters
    np.savetxt(f'output/{OBJNAME}/LASR/Camera/Full-Resolution/{OBJNAME}/{i:05d}.txt',cam_params[i].T)
    np.savetxt(f'output/{OBJNAME}/LASR/Camera/Full-Resolution/r{OBJNAME}/{i:05d}.txt',cam_params[i].T)

# write gif
imageio.mimsave(str(kb.as_path(f"output/{OBJNAME}/") / f"{OBJNAME}.gif"),frames_dict['rgba'])
kb.file_io.write_flow_batch(frames_dict['forward_flow'], directory= f"output/{OBJNAME}/FlowFW", file_template="{:05d}.png", name="forward_flow",
                    max_write_threads=16)
kb.file_io.write_flow_batch(frames_dict['backward_flow'], directory= f"output/{OBJNAME}/FlowBW", file_template="{:05d}.png", name="backward_flow",
                    max_write_threads=16)


# cp -r output/moving_cube/LASR/*s/ ../lasr/database/DAVIS/