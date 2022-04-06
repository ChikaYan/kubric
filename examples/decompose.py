# Copyright 2020 The Kubric Authors
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

from enum import Flag
import logging

import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import pandas as pd
from kubric import randomness
import imageio
import bpy
import json
import pdb
from kubric import core
from kubric import randomness
from kubric.core import color
import shutil


# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-.1, -.1, 1), (.1, .1, 0)]
DYNAMIC_SPAWN_REGION = [(0, 0, 3), (0, 0, 3)]
VELOCITY_RANGE = [(.5, .5, 0.), (.5, .5, 0.)]
# CAMERA_RANGE = [[-10, -10, 1], [10, 10, 3]]

# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument(
    "--objects_set", choices=["clevr", "kubasic", "gso", "shapenet"], default="clevr")
# only used for gso dataset
parser.add_argument(
    "--objects_split", choices=["train", "test"], default="train")


parser.add_argument("--object_friction", type=float, default=None)
parser.add_argument("--object_restitution", type=float, default=None)
parser.add_argument("--object_size", type=float, default=None)
parser.add_argument("--shape_name", type=str, default=None)
parser.add_argument("--objects_all_same", action="store_true", default=False)
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument(
    "--background", choices=["clevr", "colored", "hdri"], default="clevr")
# only used for hdri backgrounds
parser.add_argument("--backgrounds_split",
                    choices=["train", "test"], default="train")

# Configuration for the camera
parser.add_argument("--camera", choices=["clevr", "katr", "random", "linear_movement",
                    "multiview", "rotate", "static", "rotate_repeat", "multiview_rot"], default="multiview")
parser.add_argument("--max_camera_movement", type=float, default=4.0)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets_dir", type=str,
                    default="./examples/KuBasic")
parser.add_argument("--gso_assets_dir", type=str,
                    default="gs://kubric-public/GSO")
parser.add_argument("--hdri_dir", type=str,
                    default="gs://kubric-public/hdri_haven/4k")

parser.add_argument("--no_save_state", dest="save_state", action="store_false")
parser.add_argument("--save_state", dest="save_state", action="store_true")

parser.add_argument("--ncam", type=int, default=10,
                    help="number of possible camera positions")
parser.add_argument("--icam", type=int, default=0,
                    help="camera index")
parser.add_argument("--reallocate", type=bool, default=True,
                    help="whether to use mov_till_no_overlap to reallocate objects")

parser.add_argument("--waypoint_seed", type=int, default=0,
                    help="random seed used for randomly sampled waypoints")
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--job_dir", type=str, default="output/movid_multiview")

RUN_RENDERING = True
STATIC = False

FRAME_END = 200

# render test mode: freeze scene at specified frame and render with extra camera
RENDER_TEST = True
TEST_FREEZE_FRAME = 99 if RENDER_TEST else 0
TEST_NUM_FRAME = 100
TEST_CAMERA = 'circle'
TEST_CAM_R = 2.5
# TEXTURE_PATH = 'examples/tex/plywood_diff_4k.jpg'
TEXTURE_PATH = 'examples/tex/t_brick_floor_002_diffuse_4k.jpg'

parser.set_defaults(
    save_state=True,
    frame_start=TEST_FREEZE_FRAME + 1,
    frame_end=FRAME_END if not RENDER_TEST else TEST_NUM_FRAME + TEST_FREEZE_FRAME,
    frame_rate=30,
    width=512,
    height=512
)
FLAGS = parser.parse_args()

FLAGS.objects_set = 'shapenet'
FLAGS.seed = 1
np.random.seed(FLAGS.seed)
FLAGS.job_dir = 'output/decompose/chair_v2/'
if STATIC:
  FLAGS.job_dir += 'static'
else:
  FLAGS.job_dir += 'dynamic'
if RENDER_TEST:
  FLAGS.job_dir += '_test'
FLAGS.background = 'clevr'
FLAGS.camera = 'rotate_random'
FLAGS.waypoint_seed = 321
FLAGS.object_size = 5
FLAGS.object_friction = 0
FLAGS.object_restitution = 1
FLAGS.shape_name = 'car'
FLAGS.reallocate = True

STATIC_OBJS = [
    {
        "asset_id": "02871439_586356e8809b6a678d44b95ca8abc7b2",  # bookshelf
        "scale": 4, "object_restitution": 0, "position": (-2.5, -4, 0.82), "quaternion": (0.706474, 0.706474, -0.029844, -0.029844)
    },
    {
        "asset_id": "04256520_e9d6a366a92a61d979c851829c339535",  # sofa
        "scale": 6, "object_restitution": 0, "position": (0.662423, 3.70227, 0.417688), "quaternion": (0.270095, 0.270095, 0.653493, 0.653493)
    },
    {
        "asset_id": "03636649_de5799693fbddbe3c4560b8f156e749",  # lamp
        "scale": 6, "object_restitution": 0, "position": (2.43664, -3.8925, 1.73538), "quaternion": (0.699312, 0.699312, 0.104714, 0.104714)
    },
]

DYNAMIC_OBJS = [
    {
        "asset_id": "03001627_653c0f8b819872b41a6af642cfc1a2bc", # chair
        "scale": 2, "object_restitution": 0.5, "position": (-7.49685, 7.00034, 0.8),
        "init_velocity": (1.5, -3, 0), "angular_velocity": (0, 0, 2),
        "quaternion": (1, 1, -0.5, -0.5), "object_friction": 0
    },
    {
        "asset_id": "02773838_c4cb5bcb1a9399ae504721639e19f609", # bag
        "scale": 2, "object_restitution": 0.9, "position": (-7.49685, -7.00034, 4),
        "init_velocity": (2, 4, 0), "angular_velocity": (0, 0, 1),
        "quaternion": (1, 1, -0.5, -0.5), "object_friction": 0
    },
]
# STATIC_OBJS = [
#     {
#         "asset_id": "02958343_f6906412622f5853413113c385698bb2",
#         "scale": 4, "object_restitution": 0, "position": (-6.2, -5, 1), "quaternion": (1, 1, 0.15, 0.15)
#     },
#     {
#         "asset_id": "02958343_2928f77d711bf46eaa69dfdc5532bb13",
#         "scale": 4, "object_restitution": 0, "position": (7, 3/2, 1), "quaternion": (1, 1, -0.5, -0.5)
#     },
#     {
#         "asset_id": "02958343_9752827fb7788c2d5c893a899536502e",
#         "scale": 4, "object_restitution": 0, "position": (10, -3.5, 1), "quaternion": (1, 1, 0.5, 0.5)
#     },
# ]

# DYNAMIC_OBJS = [
#     {
#         "asset_id": "02958343_d4d7d596cf08754e2dfac2620a0cf07b",
#         "scale": 4, "object_restitution": 0.9, "position": (-1, -5, 4),
#         "init_velocity": (-1, 3, -1), "object_friction": 0
#     },
#     {
#         "asset_id": "02958343_1a56d596c77ad5936fa87a658faf1d26",
#         "scale": 4, "object_restitution": 0.9, "position": (4, 11, 4),
#         "init_velocity": (-2, -6, 0), "quaternion": (1, 1, -0.5, -0.5), "object_friction": 0
#     },
#     {
#         "asset_id": "02958343_5876e90c8f0b15e112ed57dd1bc82aa3",
#         "scale": 4, "object_restitution": 0.9, "position": (-10, 10, 1),
#         "init_velocity": (6, -6, -4), "quaternion": (0.2, 0.2, -0.67, -0.67), "object_friction": 0
#     },
# ]


def euler_to_xyz(r, theta, phi):
  x = r * np.cos(theta) * np.sin(phi)
  y = r * np.sin(theta) * np.sin(phi)
  z = r * np.cos(phi)
  return [x, y, z]


if FLAGS.camera == 'rotate':
  # rotate camera around center
  ROT_RANGE = np.pi / 4
  r = 15
  THETA = 2.5
  PHI = 1
  # camera_waypoints = [
  #   {'position': euler_to_xyz(r, THETA, PHI), 'rot': [THETA, PHI], 'frame': 0},
  #   {'position': euler_to_xyz(r, THETA+np.pi/4, PHI), 'rot': [THETA + np.pi/4, PHI], 'frame': int(0.3*FRAME_END)},
  #   {'position': euler_to_xyz(r, THETA+np.pi/4, PHI+0.1), 'rot': [THETA + np.pi/4, PHI+0.1], 'frame': int(0.36*FRAME_END)},
  #   {'position': euler_to_xyz(r, THETA, PHI+0.1), 'rot': [THETA, PHI+0.1], 'frame': int(0.63*FRAME_END)},
  #   {'position': euler_to_xyz(r, THETA, PHI+0.2), 'rot': [THETA, PHI+0.1], 'frame': int(0.66*FRAME_END)},
  #   {'position': euler_to_xyz(r, THETA+np.pi/4, PHI+0.2), 'rot': [THETA, PHI+0.1], 'frame': int(0.63*FRAME_END)},
  #   ]

  # camera_waypoints = [
  #   {'rot': [THETA, PHI], 'frame': 0},
  #   {'rot': [THETA + np.pi/4, PHI], 'frame': int(0.3*FRAME_END)},
  #   {'rot': [THETA + np.pi/4, PHI+0.1], 'frame': int(0.36*FRAME_END)},
  #   {'rot': [THETA, PHI+0.1], 'frame': int(0.63*FRAME_END)},
  #   {'rot': [THETA, PHI+0.2], 'frame': int(0.69*FRAME_END)},
  #   {'rot': [THETA+ np.pi/4, PHI+0.2], 'frame': int(FRAME_END)},
  #   ]

  camera_waypoints = [
      {'rot': [THETA, PHI]},
      {'rot': [THETA + np.pi/4, PHI]},
      {'rot': [THETA, PHI+0.1]},
      {'rot': [THETA + np.pi/4, PHI+0.1]},
      {'rot': [THETA, PHI]},
      {'rot': [THETA + np.pi/4, PHI]},
      {'rot': [THETA, PHI+0.1]},
      {'rot': [THETA + np.pi/4, PHI+0.1]},
  ]
  for i in range(len(camera_waypoints)):
    camera_waypoints[i]['frame'] = int(
        i / (len(camera_waypoints)-1) * FRAME_END)

elif FLAGS.camera == 'static':
  r = 15
  # THETA = np.random.uniform(low=0, high=2 * np.pi)
  # PHI = np.random.uniform(low=0, high=np.pi/2)
  THETA = 2.5
  PHI = 1
elif FLAGS.camera == 'rotate_random':
  # rotate camera with random waypoints
  r = 15
  THETA = 2.5
  PHI = 1
  THETA_RANGE = [2, 2 + np.pi/4]
  PHI_RANGE = [1, 1.2]
  # THETA_RANGE = [2.5, 2.5 + np.pi/4]
  # PHI_RANGE = [1, 1.2]
  #   THETA_RANGE = [2, 2 + np.pi/2]
  # PHI_RANGE = [0.75, 1.25]
  N_WAYPOINTS = 10

  rng = np.random.default_rng(FLAGS.waypoint_seed)
  thetas = rng.uniform(THETA_RANGE[0], THETA_RANGE[1], N_WAYPOINTS)
  phis = rng.uniform(PHI_RANGE[0], PHI_RANGE[1], N_WAYPOINTS)
  camera_waypoints = []
  for i in range(N_WAYPOINTS):
    camera_waypoints.append({'rot': [thetas[i], phis[i]]})

  for i in range(len(camera_waypoints)):
    camera_waypoints[i]['frame'] = int(
        i / (len(camera_waypoints)-1) * FRAME_END)
else:
  raise NotImplementedError

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True,
                   adaptive_sampling=False)

# Log arguments:
with open(FLAGS.job_dir + '/args.config', 'w') as file:
  json.dump(FLAGS.__dict__, file, indent=2)

# Log script
shutil.copy(__file__, FLAGS.job_dir + '/script.py')


# To load arguments:
# with open(FLAGS.job_dir + '/args.config', 'r') as file:
#     FLAGS.__dict__ = json.load(file)


# --- Populate the scene
logging.info("Creating a large gray floor...")
floor_material = kb.PrincipledBSDFMaterial(color=kb.Color.from_name("gray"),
                                           roughness=1., specular=0.)

FLOOR_SIZE = 50
WALL_HEIGHT = 20
floor = kb.Cube(name="floor0", scale=(FLOOR_SIZE, FLOOR_SIZE, 1), position=(0, 0, -1),
                material=floor_material, friction=1.0,
                restitution=0,  # friction and restitution are set later
                static=True, background=True)
floor.friction = 0.
floor.restitution = 1.

east_wall = kb.Cube(name="floor1", scale=(1, FLOOR_SIZE, WALL_HEIGHT), position=(FLOOR_SIZE, 0, WALL_HEIGHT),
                    material=floor_material, static=True, background=True)
west_wall = kb.Cube(name="floor2", scale=(1, FLOOR_SIZE, WALL_HEIGHT), position=(-FLOOR_SIZE, 0, WALL_HEIGHT),
                    material=floor_material, static=True, background=True)
north_wall = kb.Cube(name="floor3", scale=(FLOOR_SIZE, 1, WALL_HEIGHT), position=(0, FLOOR_SIZE, WALL_HEIGHT),
                     material=floor_material, static=True, background=True)
south_wall = kb.Cube(name="floor4", scale=(FLOOR_SIZE, 1, WALL_HEIGHT), position=(0, -FLOOR_SIZE, WALL_HEIGHT),
                     material=floor_material, static=True, background=True)
scene.add([floor, north_wall, south_wall, east_wall, west_wall])

bpy_scene = bpy.context.scene
# for i in range(5):
mat = bpy_scene.objects[f'floor{0}'].active_material
tree = mat.node_tree
mat_node = tree.nodes["Principled BSDF"]
tex_node = tree.nodes.new('ShaderNodeTexImage')
tex_node.image = bpy.data.images.load(TEXTURE_PATH)
# tex_node.image = bpy.data.images.load('examples/tex/t_brick_floor_002_diffuse_4k.jpg')
tree.links.new(tex_node.outputs['Color'], mat_node.inputs['Base Color'])
map_node = tree.nodes.new('ShaderNodeMapping')
map_node.vector_type = "TEXTURE"
# pdb.set_trace()
map_node.inputs['Scale'].default_value = (.1, .1, .1)
tree.links.new(map_node.outputs['Vector'], tex_node.inputs['Vector'])
tex_coord = tree.nodes.new(type='ShaderNodeTexCoord')
tree.links.new(tex_coord.outputs["UV"], map_node.inputs["Vector"])


scene_metadata = {}
sun = core.DirectionalLight(name="sun",
                            color=color.Color.from_name("white"), shadow_softness=0.2,
                            intensity=1.5, position=(-11.6608, 6.62799, 25.8232))
# sun = core.PointLight(name="lamp_key",
#                               color=color.Color.from_name("white"), intensity=6000,
#                               position=(13, -5, 13))
# lamp_key = core.PointLight(name="lamp_key",
#                            color=color.Color.from_name("white"), intensity=300,
#                            position=(-6.44671, 2.90517, 4.2584))
lights = [sun]

# jitter lights
for light in lights:
  light.look_at((0, 0, 0))
scene.add(lights)
scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)
scene_metadata["background"] = "clevr"


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
scene.camera.position = euler_to_xyz(r, THETA, PHI)
scene.camera.look_at((0, 0, 0))

if FLAGS.camera.startswith('rotate') and not RENDER_TEST:
  for i, waypoint in enumerate(camera_waypoints):
    if i == 0:
      continue
    change_rate = 1
    last_waypoint = camera_waypoints[i-1]
    frame_start = last_waypoint['frame']
    frame_end = waypoint['frame']
    theta_change = (waypoint['rot'][0] - last_waypoint['rot']
                    [0]) / ((frame_end - frame_start) / change_rate)
    phi_change = (waypoint['rot'][1] - last_waypoint['rot']
                  [1]) / ((frame_end - frame_start) / change_rate)

    for frame in range(frame_start, frame_end + 1):
      i = (frame - frame_start)
      theta_new = (i // change_rate) * theta_change + last_waypoint['rot'][0]
      phi_new = (i // change_rate) * phi_change + last_waypoint['rot'][1]

      # These values of (x, y, z) will lie on the same sphere as the original camera.
      x = r * np.cos(theta_new) * np.sin(phi_new)
      y = r * np.sin(theta_new) * np.sin(phi_new)
      z = r * np.cos(phi_new)

      scene.camera.position = (x, y, z)
      scene.camera.look_at((0, 0, 0))
      scene.camera.keyframe_insert("position", frame)
      scene.camera.keyframe_insert("quaternion", frame)
elif RENDER_TEST:
  # change camera path to be a circle around center of original camera waypoints
  center_rot = np.average([waypoint['rot']
                          for waypoint in camera_waypoints], axis=0)
  center_coord = [
      r * np.cos(center_rot[0]) * np.sin(center_rot[1]),
      r * np.sin(center_rot[0]) * np.sin(center_rot[1]),
      r * np.cos(center_rot[1])
  ]

  angles = np.linspace(0, 2 * np.pi, scene.frame_end + 1 - scene.frame_start)

  # pdb.set_trace()

  for i, frame in enumerate(range(scene.frame_start, scene.frame_end + 1)):
    x = center_coord[0]  # + TEST_CAM_R * np.cos(angles[i])
    y = center_coord[1] + TEST_CAM_R * np.sin(angles[i])
    z = center_coord[2] + TEST_CAM_R * np.cos(angles[i])

    scene.camera.position = (x, y, z)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)

else:
  raise NotImplementedError


asset_source = kb.AssetSource(
    'gs://tensorflow-graphics/public/60c9de9c410be30098c297ac/ShapeNetCore.v2')
active_split = None

OBJ_ID = 0
# --- Place random objects


def add_random_shapnet_object(
        rng,
        spawn_region=((-1, -1, -1), (1, 1, 1)),
        asset_id='02958343_d4d7d596cf08754e2dfac2620a0cf07b',
        position=(0, 0, 0),
        scale=1,
        quaternion=(1, 1, 0, 0),
        object_friction=FLAGS.object_friction,
        object_restitution=FLAGS.object_restitution,
        init_velocity=(0, 0, 0),
        angular_velocity=(0, 0, 0),
        reallocate=False,
        static=False,):
  global OBJ_ID
  obj = asset_source.create(asset_id=asset_id, name=f'obj_{OBJ_ID}')
  OBJ_ID += 1
  obj.scale = scale
  obj.position = position

  if object_friction is not None:
    obj.friction = object_friction
  if object_restitution is not None:
    obj.restitution = object_restitution
  scene.add(obj)
  if reallocate:
    kb.move_until_no_overlap(
        obj, simulator, spawn_region=spawn_region, rng=rng)

  # obj.quaternion = kb.Quaternion(axis=quaternion, degrees=90)
  obj.quaternion = quaternion
  obj.velocity = init_velocity
  obj.angular_velocity = angular_velocity
  logging.info("    Added %s at %s", obj.asset_id, obj.position)
  if static:
    obj.friction = 1.0
    obj.metadata["is_dynamic"] = False
  else:
    obj.metadata["is_dynamic"] = True

  return obj


logging.info("Placing static objects:")
for params in STATIC_OBJS:
  obj = add_random_shapnet_object(rng=rng, **params)

# obj = add_random_shapnet_object(rng=rng, asset_id="02958343_f6906412622f5853413113c385698bb2",
#                                 scale=4, object_restitution=0, position=(-6.2, -5, 1), quaternion=(1, 1, 0.15, 0.15))
# obj = add_random_shapnet_object(rng=rng, asset_id="02958343_2928f77d711bf46eaa69dfdc5532bb13",
#                                 scale=4, object_restitution=0, position=(7, 3/2, 1), quaternion=(1, 1, -0.5, -0.5))
# obj = add_random_shapnet_object(rng=rng, asset_id="02958343_9752827fb7788c2d5c893a899536502e",
#                                 scale=4, object_restitution=0, position=(10, -3.5, 1), quaternion=(1, 1, 0.5, 0.5))

# # chairs
# obj = add_random_shapnet_object(rng=rng, asset_id="03001627_653c0f8b819872b41a6af642cfc1a2bc",
#                                 scale=4, object_restitution=0, position=(5, 5, 1), quaternion=(1, 1, 0.5, 0.5))
# obj = add_random_shapnet_object(rng=rng, asset_id="04379243_229af4d0700b3fab29f2e5c9212b176c",
#                                 scale=4, object_restitution=0, position=(8, 8, 1), quaternion=(1, 1, 0.5, 0.5))


# --- Simulation
logging.info(
    "Running 500 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-500, frame_end=0)
for obj in scene.foreground_assets:
  # stop any objects that are still moving/rolling
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)
    obj.static = True

key_frame = 0

# Add dynamic objects
if not STATIC:
  logging.info("Placing dynamic objects:")
  dynamic_objs = []
  for params in DYNAMIC_OBJS:
    obj = add_random_shapnet_object(rng=rng, **params)
    dynamic_objs.append(obj)
  # obj = add_random_shapnet_object(rng=rng, asset_id='02958343_d4d7d596cf08754e2dfac2620a0cf07b', scale=4,
  #                                 object_restitution=0.9, position=(-1, -5, 4), init_velocity=(-1, 3, -1), object_friction=0)
  # obj = add_random_shapnet_object(rng=rng, asset_id="02958343_1a56d596c77ad5936fa87a658faf1d26", scale=4, object_restitution=0.9, position=(
  #     4, 11, 4), init_velocity=(-2, -6, 0), quaternion=(1, 1, 3, 3), object_friction=0)
  # obj = add_random_shapnet_object(rng=rng, asset_id="02958343_5876e90c8f0b15e112ed57dd1bc82aa3", scale=4, object_restitution=0.9,
  #                                 position=(-10, 10, 1), init_velocity=(6, -6, -4), quaternion=(0.2, 0.2, -0.67, -0.67), object_friction=0)

  if RENDER_TEST:
    if TEST_FREEZE_FRAME < key_frame:
      simulator.run(frame_start=0, frame_end=TEST_FREEZE_FRAME)
      for obj in dynamic_objs:
        obj.static = True
    else:
      simulator.run(frame_start=0, frame_end=key_frame)
      # obj.velocity = (1, -2, 0)
      simulator.run(frame_start=key_frame, frame_end=TEST_FREEZE_FRAME)
      for obj in dynamic_objs:
        obj.static = True
    # replace key_frame for rendering of rest of frames
    key_frame = TEST_FREEZE_FRAME

  else:
    if key_frame > 0:
      simulator.run(frame_start=0, frame_end=key_frame)
      obj.velocity = (1, -2, 0)
    pass


if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' before starting the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(
    frame_start=key_frame, frame_end=scene.frame_end+1)

all_bboxes = np.array([obj.aabbox for obj in scene.foreground_assets])
scene_aabbox = all_bboxes[:, 0, :].min(), all_bboxes[:, 1, :].max()

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' before starting the rendering.",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


if RUN_RENDERING:
  logging.info("Rendering the scene ...")
  data_stack = renderer.render()

  # --- Postprocessing
  kb.compute_visibility(data_stack["segmentation"], scene.assets)
  visible_foreground_assets = [asset for asset in scene.foreground_assets
                               if np.max(asset.metadata["visibility"]) > 0]
  visible_foreground_assets = sorted(visible_foreground_assets,
                                     key=lambda asset: np.sum(
                                         asset.metadata["visibility"]),
                                     reverse=True)
  data_stack["segmentation"] = kb.adjust_segmentation_idxs(
      data_stack["segmentation"],
      scene.assets,
      visible_foreground_assets)
  scene_metadata["num_instances"] = len(visible_foreground_assets)

  del data_stack["uv"]
  # del data_stack["forward_flow"]
  # del data_stack["backward_flow"]
  # del data_stack["depth"]
  del data_stack["normal"]
  del data_stack["object_coordinates"]

  # Save to image files
  kb.write_image_dict(data_stack, output_dir)

  kb.post_processing.compute_bboxes(
      data_stack["segmentation"], visible_foreground_assets)

  # --- Metadata
  logging.info("Collecting and storing metadata for each object.")
  kb.write_json(filename=output_dir / "metadata.json", data={
      "metadata_kubric": kb.get_scene_metadata(scene, **scene_metadata),
      "camera": kb.get_camera_info(scene.camera),
      "instances": kb.get_instance_info(scene, assets_subset=visible_foreground_assets),
      "test_freeze_frame": TEST_FREEZE_FRAME
  })
  # kb.write_json(filename=output_dir / "events.json", data={
  #     "collisions":  kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
  # })

  imageio.mimsave(str(kb.as_path(output_dir) / "movid.gif"),
                  data_stack['rgba'])
  # to convert to video, run:
  # ffmpeg -framerate 60 -i rgba_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

  kb.done()
