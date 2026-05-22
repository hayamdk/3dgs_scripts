import os
import sys
import cv2
from scipy.spatial.transform import Rotation
import numpy as np
import math
import json
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "utils"))

import fisheye_util

parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--camera_settings', required=True)
parser.add_argument('--use_camera_translation', action='store_true')

parser.add_argument('--generate_gps', action='store_true')
parser.add_argument('--gps_lat_origin', type=float, default=0)
parser.add_argument('--gps_lng_origin', type=float, default=0)
parser.add_argument('--gps_alt_origin', type=float, default=0)
parser.add_argument('--gps_scale', type=float, default=1000)

parser.add_argument('--images_text', default=None)

parser.add_argument('--output_dir', required=True)
parser.add_argument('--camera_model', required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--fov_x', type=float, required=True)
parser.add_argument('--fov_y', type=float, required=True)
parser.add_argument('--rotations', nargs='*', required=True)

args = parser.parse_args()

img_exts = [".jpg", ".png"]

f = open(args.camera_settings, "rt")
cam_settings = json.load(f)
f.close()

pers_camera_model = args.camera_model
width = args.width
height = args.height
fov_x = args.fov_x
fov_y = args.fov_y

def parse_rotations_opt(rotations_opt):
	rotations = []
	for rotation in rotations_opt:
		if rotation[0] != "(" or rotation[-1] != ")":
			print("Invalid rotation tuple!")
			raise
		spl = rotation[1:len(rotation)-1].split(",")
		yaw = int(spl[0])
		pitch = int(spl[1])
		rotations.append((yaw, pitch))
	
	return rotations

rotations = parse_rotations_opt(args.rotations)

def np_to_list(nplist):
    pylist = []
    for val in list(nplist):
        pylist.append(float(val))
    return pylist

def render_pers_images(camera_group_name, rotations, camera_params, fisheye_images_dir, mask_img_path, output_dir, fisheye_image_dirname=None, fisheye_sparse_images=None):
	f2p_map_cache = {}
	
	fisheye_mask_img = None
	#if os.path.isfile(mask_img_path):
	#	fisheye_mask_img = cv2.imread(mask_img_path, cv2.IMREAD_COLOR)
	
	for fisheye_img_name in os.listdir(fisheye_images_dir):
		basename, ext = os.path.splitext(fisheye_img_name)
		if ext.lower() not in img_exts:
			continue
		
		if args.generate_gps:
			fimg_name = fisheye_image_dirname + "/" + fisheye_img_name
			fimg = fisheye_sparse_images[fimg_name.lower().replace("\\", "/")]
			
			img_rotation = Rotation.from_quat(fimg.quat).as_matrix()
			image_translation = np.array(fimg.translation, dtype=np.float64)
			
			x, y, z = np_to_list(-img_rotation.T @ image_translation)
			lat_c = args.gps_lat_origin
			lng_c = args.gps_lng_origin
			alt_c = args.gps_alt_origin
			lat, lng, alt = gps_util.xyz_to_wgs(x, y, z, (lat_c, lng_c, alt_c), scale=args.gps_scale)
			
			gps_ifd = gps_util.gen_gps_ifd(lat, lng, alt)
			#print(fimg_name, (x, y, z), lat, lng, alt)
		
		fisheye_img_path = os.path.join(fisheye_images_dir, fisheye_img_name)
		fisheye_img = cv2.imread(fisheye_img_path, cv2.IMREAD_COLOR)
		
		if fisheye_mask_img is None:
			if os.path.isdir(mask_img_path):
				fisheye_mask_img_path = os.path.join(mask_img_path, fisheye_img_name + ".png")
				fisheye_mask_img = cv2.imread(fisheye_mask_img_path, cv2.IMREAD_COLOR)
			else:
				raise
		
		print(fisheye_img_path)
		
		for rotation in rotations:
			yaw = rotation[0]
			pitch = rotation[1]
			image_dir = os.path.join(output_dir, "images", camera_group_name + "_" + str(yaw) + "_" + str(pitch))
			mask_dir = os.path.join(output_dir, "masks", camera_group_name + "_" + str(yaw) + "_" + str(pitch))
			os.makedirs(image_dir, exist_ok=True)
			os.makedirs(mask_dir, exist_ok=True)
			
			image_path = os.path.join(image_dir, fisheye_img_name)
			mask_path = os.path.join(mask_dir, fisheye_img_name + ".png")
			
			if rotation in f2p_map_cache:
				f2p_map = f2p_map_cache[rotation]
			else:
				f2p_map = fisheye_util.make_f2p_map(width, height, fov_x, fov_y, yaw, pitch, camera_params)
				print("Made distortion map: rotation=", rotation, "   ")
				f2p_map_cache[rotation] = f2p_map
			
			persp_img = cv2.remap(fisheye_img, f2p_map, None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			cv2.imwrite(image_path, persp_img)
			
			if args.generate_gps:
				gps_util.insert_gps(image_path, gps_ifd)
			
			#mask_persp_img = cv2.remap(fisheye_mask_img, f2p_map, None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			mask_persp_img = cv2.remap(fisheye_mask_img, f2p_map, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			cv2.imwrite(mask_path, mask_persp_img)

def perspective_rig_cameras(rig_setting):
	rig_cameras = []
	
	fx = width / 2.0 / math.tan(math.radians(fov_x / 2.0))
	fy = height / 2.0 / math.tan(math.radians(fov_y / 2.0))
	
	if pers_camera_model == "SIMPLE_PINHOLE":
		if fov_x != fov_y:
			raise
		pers_camera_params = [fx, width / 2.0, height / 2.0]
	elif pers_camera_model == "PINHOLE":
		pers_camera_params = [fx, fy, width / 2.0, height / 2.0]
	elif pers_camera_model == "SIMPLE_RADIAL":
		if fov_x != fov_y:
			raise
		pers_camera_params = [fx, width / 2.0, height / 2.0, 0]
	elif pers_camera_model == "RADIAL":
		if fov_x != fov_y:
			raise
		pers_camera_params = [fx, width / 2.0, height / 2.0, 0, 0]
	elif pers_camera_model == "OPENCV":
		pers_camera_params = [fx, fy, width / 2.0, height / 2.0, 0, 0, 0, 0]
	elif pers_camera_model == "FULL_OPENCV":
		pers_camera_params = [fx, fy, width / 2.0, height / 2.0, 0, 0, 0, 0, 0, 0, 0, 0]
	else:
		# unknown camera model
		raise
	
	for rig_cam in rig_setting:
		cam_name = rig_cam["name"]
		sensor_rotation = rig_cam["rotation"]
		
		if args.use_camera_translation:
			print("######################## use_camera_translation")
			sensor_offset = rig_cam["translation"]
		else:
			print("######################## NOT use_camera_translation")
			sensor_offset = None
		
		if sensor_rotation is not None:
			R_sensor_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(-sensor_rotation[0]), 0]))
			R_sensor_pitch, _ = cv2.Rodrigues(np.array([math.radians(-sensor_rotation[1]), 0, 0]))
			R_sensor_roll, _ = cv2.Rodrigues(np.array([0, 0, math.radians(-sensor_rotation[2])]))
			R_sensor = R_sensor_yaw @ R_sensor_pitch @ R_sensor_roll
			
			print("!!!!!!!!!!! sensor_rotation !!!!!!!!!!!!!", "\n", Rotation.from_matrix(R_sensor).as_quat())
		
		for rotation in rotations:
			yaw = rotation[0]
			pitch = rotation[1]
			camera_name = cam_name + "_" + str(yaw) + "_" + str(pitch)
			
			R_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(-yaw), 0]))
			R_pitch, _ = cv2.Rodrigues(np.array([math.radians(-pitch), 0, 0]))
			
			if sensor_rotation is None:
				rotation_matrix = R_pitch @ R_yaw
			else:
			
				R_yaw2, _ = cv2.Rodrigues(np.array([0, math.radians(-yaw-180), 0]))
				R_pitch2, _ = cv2.Rodrigues(np.array([math.radians(-pitch), 0, 0]))
				
				
				rotation_matrix = R_pitch @ R_yaw @ R_sensor
				rotation_matrix2 = R_pitch2 @ R_yaw2
				
				print("rotation_matrix", "\n", rotation_matrix)
				print("rotation_matrix2", "\n", rotation_matrix2)
			
			if len(rig_cameras) == 0:
				rig_camera = {
						"image_prefix": camera_name + "/",
						"ref_sensor": True,
						"camera_model_name": pers_camera_model,
						"camera_params": pers_camera_params,
					}
				
				ref_rotation_matrix = rotation_matrix
				
				if sensor_offset is not None:
					ref_sensor_offset = sensor_offset
				else:
					ref_sensor_offset = np.array([0, 0, 0])
			else:
				rotation_matrix_from_ref = rotation_matrix @ ref_rotation_matrix.T
				rotation_from_ref = Rotation.from_matrix(rotation_matrix_from_ref).as_quat()
				rotation_from_ref = np_to_list(rotation_from_ref)
				
				if sensor_offset is not None:
					print(sensor_offset)
					offset_from_ref = rotation_matrix @ (np.array(sensor_offset) - ref_sensor_offset).T
				else:
					offset_from_ref = rotation_matrix @ (-ref_sensor_offset).T
				
				rig_camera = {
						"image_prefix": camera_name + "/",
						"ref_sensor": False,
						"camera_model_name": pers_camera_model,
						"camera_params": pers_camera_params,
						"cam_from_rig_rotation": [rotation_from_ref[3]] + rotation_from_ref[0:3],
						"cam_from_rig_translation": np_to_list(offset_from_ref)
					}
			
			rig_cameras.append(rig_camera)
	
	return rig_cameras

rig_config = []

if args.generate_gps:
	import gps_util
	import colmap_sparse_util as csu
	
	print("Reading images.txt")
	images = {}
	for image in csu.read_images(args.images_text):
		images[image.name.lower().replace("\\", "/")] = image

for rig in cam_settings:
	rig_cameras = perspective_rig_cameras(rig)
	rig_config.append({"cameras": rig_cameras})
	print("rig_cameras", "\n", rig_cameras)
	
	for camera in rig:
		print(camera)
		camera_model = camera["camera_model"]
		
		if camera_model == "OPENCV_FISHEYE":
			camera_params = camera["camera_params"]
		else:
			raise
		
		camera_image_dir = os.path.join(args.image_dir, camera["name"])
		camera_mask_dir = os.path.join(args.mask_dir, camera["name"])
		
		if args.generate_gps:
			render_pers_images(
					camera["name"],
					rotations,
					camera_params,
					camera_image_dir,
					camera_mask_dir,
					args.output_dir,
					fisheye_image_dirname = camera["name"],
					fisheye_sparse_images = images
				)
		else:
			render_pers_images(
					camera["name"],
					rotations,
					camera_params,
					camera_image_dir,
					camera_mask_dir,
					args.output_dir
				)

rigconfig_path = os.path.join(args.output_dir, "rig_config.json")
f = open(rigconfig_path, "wt")
json.dump(rig_config, f)
f.close()

print(rig_cameras)
