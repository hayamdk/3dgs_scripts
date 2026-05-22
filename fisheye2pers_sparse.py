import os
import sys
import math
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "utils"))

import fisheye_util
import colmap_sparse_util as csu

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--sparse_dir', required=True)

#parser.add_argument('--threads', required=False)

parser.add_argument('--generate_gps', action='store_true')
parser.add_argument('--gps_lat_origin', type=float, default=0)
parser.add_argument('--gps_lng_origin', type=float, default=0)
parser.add_argument('--gps_alt_origin', type=float, default=0)
parser.add_argument('--gps_scale', type=float, default=1000)

parser.add_argument('--output_dir', required=True)
parser.add_argument('--camera_model', required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--fov_x', type=float, required=True)
parser.add_argument('--fov_y', type=float, required=True)
parser.add_argument('--rotations', nargs='*', required=True)

args = parser.parse_args()

if args.generate_gps:
	import gps_util

fisheye_image_dir = args.image_dir
fisheye_mask_dir = args.mask_dir
fisheye_sparse_dir = args.sparse_dir

output_dir = args.output_dir

#threads = args.threads
#if threads == -1:
#	threads = os.process_cpu_count()

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

print("Reading images.txt")
images_text = os.path.join(fisheye_sparse_dir, "images.txt")
images = csu.read_images(images_text)

print("Reading cameras.txt")
cameras_text = os.path.join(fisheye_sparse_dir, "cameras.txt")
cameras = {}
for cam in csu.read_cameras(cameras_text):
	cameras[cam.camera_id] = cam

print("Reading points3D.txt")
points3d_text = os.path.join(fisheye_sparse_dir, "points3D.txt")
points3d_list = csu.read_points3d(points3d_text)

print("Reading rigs.txt")
rigs_text = os.path.join(fisheye_sparse_dir, "rigs.txt")
rigs = csu.read_rigs(rigs_text)

print("Reading frames.txt")
frames_text = os.path.join(fisheye_sparse_dir, "frames.txt")
frames = csu.read_frames(frames_text)

def np_to_list(nplist):
    pylist = []
    for val in list(nplist):
        pylist.append(float(val))
    return pylist

def split_path_r(filepath):
	path_spl = []
	spl = os.path.split(filepath)
	while True:
		path_spl.append(spl[1])
		if spl[0] == "":
			break
		spl = os.path.split(spl[0])
	return list(reversed(path_spl))


f2p_camera_table = {}
camera_id = 1
new_cameras = []

for cam in cameras.values():
	if cam.model == "OPENCV_FISHEYE":
		f2p_camera_table[cam.camera_id] = {}
		
		for rotation in rotations:
			yaw = rotation[0]
			pitch = rotation[1]
			fx = width / 2.0 / math.tan(math.radians(fov_x / 2.0))
			fy = height / 2.0 / math.tan(math.radians(fov_y / 2.0))
			
			if pers_camera_model == "SIMPLE_PINHOLE":
				if fov_x != fov_y:
					raise
				params = [fx, width / 2.0, height / 2.0]
			elif pers_camera_model == "PINHOLE":
				params = [fx, fy, width / 2.0, height / 2.0]
			elif pers_camera_model == "SIMPLE_RADIAL":
				if fov_x != fov_y:
					raise
				params = [fx, width / 2.0, height / 2.0, 0]
			elif pers_camera_model == "RADIAL":
				if fov_x != fov_y:
					raise
				params = [fx, width / 2.0, height / 2.0, 0, 0]
			elif pers_camera_model == "OPENCV":
				params = [fx, fy, width / 2.0, height / 2.0, 0, 0, 0, 0]
			elif pers_camera_model == "FULL_OPENCV":
				params = [fx, fy, width / 2.0, height / 2.0, 0, 0, 0, 0, 0, 0, 0, 0]
			else:
				#unknown model
				raise
			
			new_cam = csu.ColmapCamera(
					camera_id = camera_id,
					model = pers_camera_model,
					width = width,
					height = height,
					params = params,
				)
			
			camera_id += 1
			new_cameras.append(new_cam)
			
			f2p_camera_table[cam.camera_id][rotation] = new_cam
	else:
		new_cam = dataclasses.replace(cam, camera_id=camera_id)
		new_cameras.append(new_cam)
		camera_id += 1

new_rigs = []
f2p_rig_table = {}
rig_id = camera_id

for rig in rigs:
	new_cameras_d = f2p_camera_table[rig.ref_sensor_id]
	rotations = list(new_cameras_d.keys())
	new_ref_sensor_id = new_cameras_d[rotations[0]].camera_id
	
	ref_yaw = rotations[0][0]
	ref_pitch = rotations[0][1]
	new_ref_rotation_matrix = fisheye_util.make_rotation_matrix(ref_yaw, ref_pitch, inverse=False)
	
	new_sub_sensors = []
	
	for rotation in rotations[1:]:
		new_camera = new_cameras_d[rotation]
		yaw = rotation[0]
		pitch = rotation[1]
		rotation_matrix = fisheye_util.make_rotation_matrix(yaw, pitch, inverse=True)
		new_rotation_matrix_from_ref = rotation_matrix @ new_ref_rotation_matrix
		
		sub_sensor = csu.ColmapRigSensor(
				sensor_type = "CAMERA",
				sensor_id = new_camera.camera_id,
				has_pose = 1,
				quat = np_to_list(Rotation.from_matrix(new_rotation_matrix_from_ref).as_quat()),
				translation = [0.0, 0.0, 0.0]
			)
		new_sub_sensors.append(sub_sensor)
	
	for sub_sensor in rig.sub_sensors:
		sub_rotation_matrix = Rotation.from_quat(sub_sensor.quat).as_matrix()
		sub_translation = np.array(sub_sensor.translation, dtype=np.float64)
		
		sub_rotation_matrix_from_ref = sub_rotation_matrix @ new_ref_rotation_matrix
		
		for rotation, new_camera in f2p_camera_table[sub_sensor.sensor_id].items():
			yaw = rotation[0]
			pitch = rotation[1]
			rotation_matrix = fisheye_util.make_rotation_matrix(yaw, pitch, inverse=True)
			
			new_rotation_matrix_from_ref = rotation_matrix @ sub_rotation_matrix_from_ref
			new_translation_from_ref = rotation_matrix @ sub_rotation_matrix @ sub_translation
			
			new_sub_sensor = csu.ColmapRigSensor(
				sensor_type = "CAMERA",
				sensor_id = new_camera.camera_id,
				has_pose = 1,
				quat = np_to_list(Rotation.from_matrix(new_rotation_matrix_from_ref).as_quat()),
				translation = np_to_list(new_translation_from_ref)
			)
			
			new_sub_sensors.append(new_sub_sensor)
	
	new_rig = csu.ColmapRig(
			rig_id = rig_id,
			ref_sensor_type = rig.ref_sensor_type,
			ref_sensor_id = new_ref_sensor_id,
			sub_sensors = new_sub_sensors
		)
	new_rigs.append(new_rig)
	f2p_rig_table[rig.rig_id] = new_rig
	
	rig_id += 1

f2p_maps = {}
rotation_cache = {}
dedistortion_funcs = {}
image_id = 1

f2p_image_table = {}
new_imgs = []
remap_job_list = []

for count, img in enumerate(images, start=1):
	print("Processing colmap sparse: [" + str(count) + "/" + str(len(images)) + "]", img.name, end="    \r")
	f2p_image_table[img.image_id] = []
	
	img_rotation = Rotation.from_quat(img.quat).as_matrix()
	if args.generate_gps:
		img_translation = np.array(img.translation, dtype=np.float64)
		x, y, z = np_to_list(-img_rotation.T @ img_translation)
		lat_c = args.gps_lat_origin
		lng_c = args.gps_lng_origin
		alt_c = args.gps_alt_origin
		lat, lng, alt = gps_util.xyz_to_wgs(x, y, z, (lat_c, lng_c, alt_c), scale=args.gps_scale)
		
		gps_ifd = gps_util.gen_gps_ifd(lat, lng, alt)
	else:
		gps_ifd = None
	
	if img.camera_id in f2p_camera_table:
		fimg_path = os.path.join(fisheye_image_dir, img.name)
		fimg = cv2.imread(fimg_path, cv2.IMREAD_COLOR)
		
		fmask_path = None
		if fisheye_mask_dir is not None:
			fmask_path = os.path.join(fisheye_mask_dir, img.name + ".png")
			#if os.path.isfile(fmask_path):
			#	fmask = cv2.imread(fmask_path, cv2.IMREAD_COLOR)
		
		f2p_list = f2p_camera_table[img.camera_id]
		for rotation, new_cam in f2p_list.items():
			yaw = rotation[0]
			pitch = rotation[1]
			
			rotation_key = (yaw, pitch)
			if rotation_key in rotation_cache:
				rotation_matrix = rotation_cache[rotation_key]
			else:
				rotation_matrix = fisheye_util.make_rotation_matrix(yaw, pitch, inverse=True)
				rotation_cache[rotation_key] = rotation_matrix
			
			f2p_map_key = (img.camera_id, yaw, pitch)
			if f2p_map_key in f2p_maps:
				f2p_map = f2p_maps[f2p_map_key]
			else:
				cam = cameras[img.camera_id]
				f2p_map = fisheye_util.make_f2p_map(width, height, fov_x, fov_y, yaw, pitch, cam.params)
				print("Made fisheye to pers map: camera_id=", img.camera_id, ", rotation= (" + str(yaw) + "," + str(pitch) + ")   ")
				f2p_maps[f2p_map_key] = f2p_map
			
			if f2p_map_key in dedistortion_funcs:
				dedistortion_func = dedistortion_funcs[f2p_map_key]
			else:
				dedistortion_func = fisheye_util.make_dedistortion_func_interp(cam.params)
				dedistortion_funcs[f2p_map_key] = dedistortion_func
			
			spl_path = split_path_r(img.name)
			if len(spl_path) > 1:
				imgdir = spl_path[0]
				imgname = os.sep.join(spl_path[1:])
				imgdir += "_" + str(yaw) + "_" + str(pitch)
				
				imgname_p = os.path.join(imgdir, imgname)
			else:
				imgname_p = img.name
			
			pimg_path = os.path.join(output_dir, "images", imgname_p)
			#pimg = cv2.remap(fimg, f2p_map, None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			pimg_dir = os.path.dirname(pimg_path)
			os.makedirs(pimg_dir, exist_ok=True)
			#cv2.imwrite(pimg_path, pimg)
			
			remap_job = (fimg_path, f2p_map, cv2.INTER_LANCZOS4, pimg_path, gps_ifd)
			remap_job_list.append(remap_job)
			
			if fmask_path is not None:
				pmask_path = os.path.join(output_dir, "masks", imgname_p + ".png")
				#pmask = cv2.remap(fmask, f2p_map, None, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
				#pmask = cv2.remap(fmask, f2p_map, None, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
				pmask_dir = os.path.dirname(pmask_path)
				os.makedirs(pmask_dir, exist_ok=True)
				#cv2.imwrite(pmask_path, pmask)
				
				remap_job = (fmask_path, f2p_map, cv2.INTER_NEAREST, pmask_path, None)
				remap_job_list.append(remap_job)
			
			img_rotation_p = rotation_matrix @ img_rotation
			quat_p = np_to_list(Rotation.from_matrix(img_rotation_p).as_quat())
			
			img_translation = np.array(img.translation, dtype=np.float64)
			img_translation_p = np_to_list(rotation_matrix @ img_translation)
			
			img_p = csu.ColmapImage(
					image_id,
					quat_p,
					img_translation_p,
					new_cam.camera_id,
					imgname_p,
					[]
				)
			
			image_id += 1
			
			for point2d in img.points2d:
				xy_pers = fisheye_util.f2p_coord(
						point2d.x, point2d.y,
						width, height,
						fov_x, fov_y,
						yaw, pitch,
						cam.params, #not used
						dedistortion_func=dedistortion_func,
						rotation_matrix=rotation_matrix
					)
				
				if xy_pers is None:
					continue
				
				x_pers = xy_pers[0]
				y_pers = xy_pers[1]
				if x_pers < 0.0 or width <= x_pers or y_pers < 0.0 or height <= y_pers:
					continue
				
				point2d_p = csu.ColmapImagePoint2D(x_pers, y_pers, point2d.point3d_id)
				img_p.points2d.append(point2d_p)
				
			
			new_imgs.append(img_p)
			f2p_image_table[img.image_id].append(img_p)
	
for count, remap_job in enumerate(remap_job_list, start=1):
	fimg_path = remap_job[0]
	f2p_map = remap_job[1]
	interpolation = remap_job[2]
	pimg_path = remap_job[3]
	gps_ifd = remap_job[4]
	
	print("Processing image: [" + str(count) + "/" + str(len(remap_job_list)) + "]", pimg_path, end="    \r")
	
	fimg = cv2.imread(fimg_path, cv2.IMREAD_COLOR)
	pimg = cv2.remap(fimg, f2p_map, None, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
	cv2.imwrite(pimg_path, pimg)
	
	if gps_ifd is not None:
		gps_util.insert_gps(pimg_path, gps_ifd)

new_frame_id = image_id

new_frames = []
for frame in frames:
	ref_img = None
	new_rig = f2p_rig_table[frame.rig_id]
	data_ids = []
	
	sub_sensor_idx = 0
	for imgdata in frame.data_ids:
		for new_img in f2p_image_table[imgdata.data_id]:
			if ref_img is None:
				ref_img = new_img
				data_id = csu.ColmapFrameDataID(
						sensor_type = new_rig.ref_sensor_type,
						sensor_id = new_rig.ref_sensor_id,
						data_id = new_img.image_id
					)
			else:
				sub_sensor = new_rig.sub_sensors[sub_sensor_idx]
				sub_sensor_idx += 1
				
				data_id = csu.ColmapFrameDataID(
						sensor_type = sub_sensor.sensor_type,
						sensor_id = sub_sensor.sensor_id,
						data_id = new_img.image_id
					)
			
			data_ids.append(data_id)
	
	if ref_img is None:
		raise
	
	new_frame = csu.ColmapFrame(
			frame_id = new_frame_id,
			rig_id = new_rig.rig_id,
			rig_from_world_quat = ref_img.quat,
			rig_from_world_translation = ref_img.translation,
			data_ids = data_ids
		)
	new_frame_id += 1
	
	new_frames.append(new_frame)

print("Clear tracks    ")
point3d_idx = {}
for point3d in points3d_list:
	point3d_idx[point3d.point3d_id] = point3d
	
	#clear track
	point3d.track = []

new_img_idx = {}
for count, new_img in enumerate(new_imgs, start=1):
	print("Setting new tracks: [" + str(count) + "/" + str(len(new_imgs)) + "]", new_img.name, end="    \r")
	
	new_img_idx[new_img.image_id] = new_img
	
	#set new track
	for point2d_id, point2d in enumerate(new_img.points2d):
		if point2d.point3d_id < 0:
			continue
		
		point3d = point3d_idx[point2d.point3d_id]
		point3dtrack = csu.ColmapPoint3DTrack(new_img.image_id, point2d_id)
		point3d.track.append(point3dtrack)

# remove point3d of tracklen<=1
del point3d_idx
new_points3d_list = []
for point3d in points3d_list:
	if len(point3d.track) == 0:
		pass
	elif len(point3d.track) == 1:
		point3dtrack = point3d.track[0]
		img = new_img_idx[point3dtrack.image_id]
		for point2d_id, point2d in enumerate(img.points2d):
			if point2d_id == point3dtrack.point2d_id:
				point2d.point3d_id = -1
	else:
		new_points3d_list.append(point3d)

points3d_list = new_points3d_list

print("Set new track    ")

images_dir = os.path.join(output_dir, "sparse", "0")
os.makedirs(images_dir, exist_ok=True)

print("Writing images.txt")
csu.save_images(new_imgs, os.path.join(images_dir, "images.txt"))

print("Writing cameras.txt")
csu.save_cameras(new_cameras, os.path.join(images_dir, "cameras.txt"))

print("Writing points3D.txt")
csu.save_points3d(points3d_list, os.path.join(images_dir, "points3D.txt"))

print("Writing rigs.txt")
csu.save_rigs(new_rigs, os.path.join(images_dir, "rigs.txt"))

print("Writing frames.txt")
csu.save_frames(new_frames, os.path.join(images_dir, "frames.txt"))
