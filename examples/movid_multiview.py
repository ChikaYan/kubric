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
"""
Worker file for the Multi-Object Video (MOVid) dataset.

It creates a scene with a number of static objects lying on the ground,
and a few objects being tossed onto them.
Many aspects of this scene are configurable.

Objects
  * The number of static objects is randomly chosen between
    --min_num_static_objects and --max_num_static_objects
  * The number of dynamic objects is randomly chosen between
    --min_num_dynamic_objects and --max_num_dynamic_objects
  * The objects are randomly chosen from one of three sets (--objects_set):
    1. "clevr" refers to the objects from the CLEVR dataset i.e. plastic or metallic
       cubes, cylinders and spheres in one of eight different colors and two sizes.
    2. "kubasic" is a superset of "clevr" that contains eight additional shapes:
       Cone, Torus, Gear, TorusKnot, Sponge, Spot, Teapot, and Suzanne
       The "kubasic" objects also use uniformly sampled hues as color and vary
       continuously in size
    3. "gso" refers to the set of Google Scanned Objects and consists of roughly
       1000 scanned household items (shoes, toys, appliances, and other products)
       They come with a fixed high-quality texture and are not varied in scale.
  * --object_friction and --object_restitution control the friction and bounciness of
    dynamic objects during the physics simulation. They default to None in which case
    "clevr" and "kubasic" objects have friction and restitution according to their material,
    and "gso" objects have a friction and restitution of 0.5 each.

Background

  * background
    1. "clevr"
    2. "hdri"
  * backgrounds_split
  * --floor_friction and --floor_restitution control the friction and bounciness of
    the floor during the physics simulation.

Camera
  1. clevr
  2. random
  3. linear_movement
    - max_camera_movement


MOVid-A
  --camera=clevr --background=clevr --objects_set=clevr
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0

MOVid-B
  --camera=random --background=colored --objects_set=kubasic
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0

MOVid-C
  --camera=random --background=hdri --objects_set=gso
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0
  --save_state=False

MOVid-CC
  --camera=linear_movement --background=hdri --objects_set=gso
  --min_num_dynamic_objects=3 --max_num_dynamic_objects=10
  --min_num_static_objects=0 --max_num_static_objects=0
  --save_state=False

MOVid-D
  --camera=random --background=hdri --objects_set=gso
  --min_num_dynamic_objects=1 --max_num_dynamic_objects=3
  --min_num_static_objects=10 --max_num_static_objects=20
  --save_state=False

MOVid-E
  --camera=linear_movement --background=hdri --objects_set=gso
  --min_num_dynamic_objects=1 --max_num_dynamic_objects=3
  --min_num_static_objects=10 --max_num_static_objects=20
  --save_state=False
"""

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


# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-.1, -.1, 1), (.1, .1, 0)]
DYNAMIC_SPAWN_REGION = [(0, 0, 3), (0, 0, 3)]
VELOCITY_RANGE = [(.5, .5, 0.), (.5, .5, 0.)]
# CAMERA_RANGE = [[-10, -10, 1], [10, 10, 3]]

# --- CLI arguments
parser = kb.ArgumentParser()
# Configuration for the objects of the scene
parser.add_argument("--objects_set", choices=["clevr", "kubasic", "gso", "shapenet"], default="clevr")
parser.add_argument("--objects_split", choices=["train", "test"], default="train") # only used for gso dataset
# parser.add_argument("--min_num_static_objects", type=int, default=10,
#                     help="minimum number of static (distractor) objects")
# parser.add_argument("--max_num_static_objects", type=int, default=20,
#                     help="maximum number of static (distractor) objects")
# parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
#                     help="minimum number of dynamic (tossed) objects")
# parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
#                     help="maximum number of static (distractor) objects")
parser.add_argument("--num_static_objects", type=int, default=0,
                    help="number of static (distractor) objects")
parser.add_argument("--num_dynamic_objects", type=int, default=1,
                    help="number of dynamic (tossed) objects")


parser.add_argument("--object_friction", type=float, default=None)
parser.add_argument("--object_restitution", type=float, default=None)
parser.add_argument("--object_size", type=float, default=None)
parser.add_argument("--shape_name", type=str, default=None)
parser.add_argument("--objects_all_same", action="store_true", default=False)
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--background", choices=["clevr", "colored", "hdri"], default="clevr")
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train") # only used for hdri backgrounds

# Configuration for the camera
parser.add_argument("--camera", choices=["clevr", "katr", "random", "linear_movement", "multiview", "rotate", "static", "rotate_repeat", "multiview_rot"], default="multiview")
parser.add_argument("--max_camera_movement", type=float, default=4.0)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets_dir", type=str, default="./examples/KuBasic")
parser.add_argument("--gso_assets_dir", type=str, default="gs://kubric-public/GSO")
parser.add_argument("--hdri_dir", type=str, default="gs://kubric-public/hdri_haven/4k")

parser.add_argument("--no_save_state", dest="save_state", action="store_false")
parser.add_argument("--save_state", dest="save_state", action="store_true")

parser.add_argument("--ncam", type=int, default=10,
                    help="number of possible camera positions")
parser.add_argument("--icam", type=int, default=0,
                    help="camera index")
parser.add_argument("--reallocate", type=bool, default=True,
help="whether to use mov_till_no_overlap to reallocate objects")

# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--job_dir", type=str, default="output/movid_multiview")

parser.set_defaults(save_state=True, frame_end=100, frame_rate=60, width=512, height=512)
FLAGS = parser.parse_args()

FLAGS.objects_set = 'shapenet'
FLAGS.seed = 1
np.random.seed(FLAGS.seed)
FLAGS.job_dir = 'output/car'
FLAGS.background = 'clevr'
FLAGS.camera = 'rotate_repeat'
FLAGS.object_size = 5
FLAGS.object_restitution = 1
FLAGS.shape_name = 'car'
FLAGS.num_static_objects = 0
FLAGS.num_dynamic_objects = 1
FLAGS.ncam = 4
FLAGS.reallocate = True

if FLAGS.camera == 'multiview':
  # generate a set of camera positionsfrom upper hemisphere

  # We will use spherical coordinates (r, theta, phi) to do this.
  #   x = r * cos(theta) * sin(phi)
  #   y = r * sin(theta) * sin(phi)
  #   z = r * cos(phi)
  r = 10
  ncam = FLAGS.ncam
  cam_poses = []

  for _ in range(ncam):
    theta = np.random.uniform(low=0, high=2 * np.pi)
    phi = np.random.uniform(low=0, high=np.pi/2)
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    cam_poses.append((x,y,z))

  # re-run rendering for each camera position
  # TODO: find a better approach
  icam = FLAGS.icam
  if icam >= ncam:
    raise IndexError
  FLAGS.job_dir = f'{FLAGS.job_dir}/cam_{icam}'

elif FLAGS.camera == 'rotate' or FLAGS.camera == 'rotate_repeat':
  # rotate camera around center
  ROT_RANGE = np.pi / 4
  r = 10
  theta = np.random.uniform(low=0, high=2 * np.pi)
  phi = np.random.uniform(low=0, high=np.pi/2)

  pdb.set_trace()

  if FLAGS.camera == 'rotate':
    FLAGS.job_dir = f'{FLAGS.job_dir}_rot_cam'
  else:
    FLAGS.job_dir = f'{FLAGS.job_dir}_rot_rep'
elif FLAGS.camera == 'static':
  r = 10
  theta = np.random.uniform(low=0, high=2 * np.pi)
  phi = np.random.uniform(low=0, high=np.pi/2)
else:
  raise NotImplementedError

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
# scene.background = (75/255,75/255,75/255,1)

simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, adaptive_sampling=False)

# Log arguments:
with open(FLAGS.job_dir + '/args.config','w') as file:
  json.dump(FLAGS.__dict__, file, indent=2)

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

east_wall = kb.Cube(name="floor1",scale=(1, FLOOR_SIZE, WALL_HEIGHT), position=(FLOOR_SIZE, 0, WALL_HEIGHT),
                    material=floor_material,static=True, background=True)
west_wall = kb.Cube(name="floor2",scale=(1, FLOOR_SIZE, WALL_HEIGHT), position=(-FLOOR_SIZE, 0, WALL_HEIGHT),
                    material=floor_material,static=True, background=True)
north_wall = kb.Cube(name="floor3",scale=(FLOOR_SIZE, 1, WALL_HEIGHT), position=(0, FLOOR_SIZE, WALL_HEIGHT),
                    material=floor_material,static=True, background=True)
south_wall = kb.Cube(name="floor4",scale=(FLOOR_SIZE, 1, WALL_HEIGHT), position=(0, -FLOOR_SIZE, WALL_HEIGHT),
                    material=floor_material,static=True, background=True)
scene.add([floor, north_wall, south_wall, east_wall, west_wall])

bpy_scene = bpy.context.scene
# for i in range(5):
mat = bpy_scene.objects[f'floor{0}'].active_material
tree = mat.node_tree
mat_node = tree.nodes["Principled BSDF"]
tex_node = tree.nodes.new('ShaderNodeTexImage')
tex_node.image = bpy.data.images.load('examples/tex/cobblestone_floor_001_diff_4k.jpg')
tree.links.new(tex_node.outputs['Color'], mat_node.inputs['Base Color'])
map_node = tree.nodes.new('ShaderNodeMapping')
map_node.vector_type = "TEXTURE"
# pdb.set_trace()
map_node.inputs['Scale'].default_value = (.1,.1,.1)
tree.links.new(map_node.outputs['Vector'], tex_node.inputs['Vector'])
tex_coord = tree.nodes.new(type = 'ShaderNodeTexCoord')
tree.links.new(tex_coord.outputs["UV"], map_node.inputs["Vector"])


scene_metadata = {}
if FLAGS.background == "clevr":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
  scene_metadata["background"] = "clevr"
if FLAGS.background == "colored":
  logging.info("Adding four (studio) lights to the scene similar to the CLEVR setup...")
  scene.add(kb.assets.utils.get_clevr_lights(rng=rng))
  scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)
  hdri_source = kb.TextureSource(FLAGS.hdri_dir)
  dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, None)
  bg_color = kb.random_hue_color()
  dome.material = kb.PrincipledBSDFMaterial(color=bg_color, roughness=1., specular=0.)
  scene_metadata["background"] = bg_color.hexstr
elif FLAGS.background == "hdri":
  logging.info("Loading background HDRIs from %s", FLAGS.hdri_dir)
  hdri_source = kb.TextureSource(FLAGS.hdri_dir)
  train_backgrounds, held_out_backgrounds = hdri_source.get_test_split(fraction=0.1)
  if FLAGS.backgrounds_split == "train":
    logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(train_backgrounds))
  else:
    logging.info("Choosing one of the %d held-out backgrounds...", len(held_out_backgrounds))
    background_hdri = hdri_source.create(texture_name=rng.choice(held_out_backgrounds))

  dome = kb.assets.utils.add_hdri_dome(hdri_source, scene, background_hdri)
  renderer._set_ambient_light_hdri(background_hdri.filename)
  scene_metadata["background"] = kb.as_path(background_hdri.filename).stem


logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == 'multiview':
  scene.camera.position = cam_poses[icam]
  scene.camera.look_at((0, 0, 0))

elif FLAGS.camera == 'rotate' or FLAGS.camera == 'rotate_repeat':
  num_phi_values_per_theta = 1
  theta_change = ROT_RANGE / ((scene.frame_end - scene.frame_start) / num_phi_values_per_theta)

  for frame in range(scene.frame_start, scene.frame_end + 1):
    i = (frame - scene.frame_start)
    if FLAGS.camera == 'rotate_repeat' and i >= (scene.frame_end + 1) // 2:
      # go back in the same trajectory
      i = scene.frame_end + 1 - i
    theta_new = (i // num_phi_values_per_theta) * theta_change + theta

    # These values of (x, y, z) will lie on the same sphere as the original camera.
    x = r * np.cos(theta_new) * np.sin(phi)
    y = r * np.sin(theta_new) * np.sin(phi)
    z = r * np.cos(phi)

    scene.camera.position = (x, y, z)
    scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)
else:
  raise NotImplementedError

# --- Set up asset sources and held-out splits for random objects
assert FLAGS.objects_set in {"clevr", "kubasic", "gso", "shapenet"}, FLAGS.objects_set
if FLAGS.objects_set in {"clevr", "kubasic"}:
  logging.info("Loading assets from %s", FLAGS.kubasic_assets_dir)
  asset_source = kb.AssetSource(FLAGS.kubasic_assets_dir)
  active_split = None
elif FLAGS.objects_set == 'gso':  # FLAGS.objects_set == "gso":
  logging.info("Loading assets from %s", FLAGS.gso_assets_dir)
  asset_source = kb.AssetSource(FLAGS.gso_assets_dir)
  train_split, test_split = asset_source.get_test_split(fraction=0.1)
  active_split = train_split if FLAGS.objects_split == "train" else test_split
  if FLAGS.objects_all_same:
    active_split = [rng.choice(active_split)]
else:
  # FLAGS.objects_set == "shapenet":
  asset_source = kb.AssetSource('gs://tensorflow-graphics/public/60c9de9c410be30098c297ac/ShapeNetCore.v2')
  active_split = None


def get_kubasic_object(
    asset_source,
    objects_set="kubasic",
    shape_name=None,
    color_strategy="uniform_hue",
    size_strategy="uniform",
    rng=randomness.default_rng()):
  if shape_name is None:
    if objects_set == "clevr":
      shape_name = rng.choice(kb.assets.utils.CLEVR_OBJECTS)
    elif objects_set == "kubasic":
      shape_name = rng.choice(kb.assets.utils.KUBASIC_OBJECTS)
    else:
      raise ValueError(f"Unknown object set {objects_set}")

  size_label, size = randomness.sample_sizes(size_strategy, rng)
  if FLAGS.object_size is not None:
    size = FLAGS.object_size

  color_label, random_color = randomness.sample_color(color_strategy, rng)
  material_name = rng.choice(["Metal", "Rubber"])
  obj = asset_source.create(name=f"{size_label} {color_label} {material_name} {shape_name}",
                            asset_id=shape_name, scale=size)

  if material_name == "Metal":
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=1.0, roughness=0.2,
                                             ior=2.5)
    obj.friction = 0.4
    obj.restitution = 0.3
    obj.mass *= 2.7 * size**3
  else:  # material_name == "Rubber"
    obj.material = kb.PrincipledBSDFMaterial(color=random_color, metallic=0., ior=1.25,
                                             roughness=0.7, specular=0.33)
    obj.friction = 0.8
    obj.restitution = 0.7
    obj.mass *= 1.1 * size**3

  obj.metadata = {
      "shape": shape_name.lower(),
      "size": size,
      "size_label": size_label,
      "material": material_name.lower(),
      "color": random_color.rgb,
      "color_label": color_label,
  }
  return obj


# --- Place random objects
def add_random_object(spawn_region, rng, use_init_velocity=True):
  if FLAGS.objects_set == "clevr":
    obj = kb.assets.utils.get_random_kubasic_object(
        asset_source, objects_set="clevr", color_strategy="clevr", size_strategy="clevr", rng=rng)
  elif FLAGS.objects_set == "kubasic":
    obj = get_kubasic_object(
        asset_source, objects_set="kubasic", shape_name=FLAGS.shape_name, color_strategy="uniform_hue",
        size_strategy="uniform", rng=rng)
  elif FLAGS.objects_set == "gso":

    scale = rng.uniform(0.5, 2.0)
    asset_id = rng.choice(active_split)
    obj = asset_source.create(asset_id=asset_id, scale=scale,
                              friction=0.5, restitution=0.5)
    min_bounds, max_bounds = obj.bounds
    max_dim = np.max(np.abs(min_bounds - max_bounds))
    obj.scale = scale / max_dim

    cat_name = "jft_category" if "jft_category" in asset_source.db.columns else "category_id"
    category_id = int(asset_source.db[
                      asset_source.db["id"] == asset_id].iloc[0][cat_name])
    categories = sorted(pd.unique(asset_source.db[cat_name]))
    obj.metadata = {
        "scale": scale,
        "asset_id": obj.asset_id,
        "category": categories.index(category_id),
    }
  elif FLAGS.objects_set == "shapenet":
    if FLAGS.shape_name == 'car':
      obj = asset_source.create(asset_id='02958343_d4d7d596cf08754e2dfac2620a0cf07b')
      obj.scale = FLAGS.object_size
      obj.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
      # pdb.set_trace()
    else:
      raise NotImplementedError(f'ShapeNet object name {FLAGS.shape_name} not implemented')

  if FLAGS.object_friction is not None:
    obj.friction = FLAGS.object_friction
  if FLAGS.object_restitution is not None:
    obj.restitution = FLAGS.object_restitution
  scene.add(obj)
  if FLAGS.reallocate:
    kb.move_until_no_overlap(obj, simulator, spawn_region=spawn_region, rng=rng)
    if FLAGS.objects_set == "shapenet":
      obj.quaternion = kb.Quaternion(axis=[1,0,0], degrees=90)
  # bias velocity towards center
  if use_init_velocity:
    obj.velocity = rng.uniform(*VELOCITY_RANGE) - [obj.position[0], obj.position[1], 0]
  else:
    obj.velocity = (0., 0., 0.)
  logging.info("    Added %s at %s", obj.asset_id, obj.position)
  return obj


num_static_objects = FLAGS.num_static_objects
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
  obj = add_random_object(spawn_region=STATIC_SPAWN_REGION, rng=rng, use_init_velocity=False)
  obj.friction = 1.0
  obj.metadata["is_dynamic"] = False


# --- Simulation
logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)
for obj in scene.foreground_assets:
  # stop any objects that are still moving/rolling
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)

floor.friction = FLAGS.floor_friction
floor.restitution = FLAGS.floor_restitution

# Add dynamic objects
num_dynamic_objects = FLAGS.num_dynamic_objects
logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
for i in range(num_dynamic_objects):
  obj = add_random_object(spawn_region=DYNAMIC_SPAWN_REGION, rng=rng, use_init_velocity=True)
  obj.metadata["is_dynamic"] = True


if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' before starting the simulation.",
              output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end+1)

all_bboxes = np.array([obj.aabbox for obj in scene.foreground_assets])
scene_aabbox = all_bboxes[:, 0, :].min(), all_bboxes[:, 1, :].max()

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' before starting the rendering.",
              output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")

logging.info("Rendering the scene ...")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                            if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(visible_foreground_assets,
                                  key=lambda asset: np.sum(asset.metadata["visibility"]),
                                  reverse=True)
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene_metadata["num_instances"] = len(visible_foreground_assets)


del data_stack["uv"]
del data_stack["forward_flow"]
del data_stack["backward_flow"]
del data_stack["depth"]
del data_stack["normal"]
del data_stack["object_coordinates"]

# Save to image files
kb.write_image_dict(data_stack, output_dir)

kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)


# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "metadata": kb.get_scene_metadata(scene, **scene_metadata),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, assets_subset=visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
})


imageio.mimsave(str(kb.as_path(output_dir) / "movid.gif"), data_stack['rgba'])
# to convert to video, run:
# ffmpeg -framerate 60 -i rgba_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

kb.done()