import os
import sys
import argparse
import json
from scipy.spatial.transform import Rotation

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "utils"))

import fisheye_util
import colmap_sparse_util as csu

parser = argparse.ArgumentParser()
parser.add_argument('--sparse_dir', required=True)
parser.add_argument('--output', required=True)


args = parser.parse_args()

print("Reading images.txt")
images_text = os.path.join(args.sparse_dir, "images.txt")
images = csu.read_images(images_text, max_images=100)

print("Reading cameras.txt")
cameras_text = os.path.join(args.sparse_dir, "cameras.txt")
cameras = csu.read_cameras(cameras_text)

print("Reading rigs.txt")
rigs_text = os.path.join(args.sparse_dir, "rigs.txt")
rigs = csu.read_rigs(rigs_text)

print("Reading frames.txt")
frames_text = os.path.join(args.sparse_dir, "frames.txt")
frames = csu.read_frames(frames_text)

def find_frame_by_rig_id(frames, rig_id):
	for frame in frames:
		if frame.rig_id == rig_id:
			return frame

def get_img(frame, images, sensor_id):
	found_data = None
	for data in frame.data_ids:
		if data.sensor_id == sensor_id:
			found_data = data
			break
	
	if found_data is None:
		raise
	
	
	found_image = None
	for image in images:
		if image.image_id == found_data.data_id:
			found_image = image
			break
	
	if found_image is None:
		raise
	
	return found_image

def get_imgname(image):
	return image.name.split("/")[0]

def get_camera_by_id(cameras, sensor_id):
	for camera in cameras:
		if camera.camera_id == sensor_id:
			return camera
	raise

def np_to_list(nplist):
    listval = []
    for val in list(nplist):
        listval.append(float(val))
    return listval

rig_settings = []
for rig in rigs:
	rig_setting = []
	
	frame = find_frame_by_rig_id(frames, rig.rig_id)
	ref_img = get_img(frame, images, rig.ref_sensor_id)
	ref_camera = get_camera_by_id(cameras, rig.ref_sensor_id)
	
	rig_camera = {"name": get_imgname(ref_img), "camera_model": ref_camera.model, "camera_params": ref_camera.params, "rotation": None, "translation": [0, 0, 0]}
	rig_setting.append(rig_camera)
	
	for sub_sensor in rig.sub_sensors:
		sub_img = get_img(frame, images, sub_sensor.sensor_id)
		sub_camera = get_camera_by_id(cameras, sub_sensor.sensor_id)
		
		rotation = Rotation.from_quat(sub_sensor.quat)
		print("rotation!!!!!!!!", rotation)
		eular = np_to_list(rotation.as_euler('YXZ', degrees=True))
		#eular = np_to_list(rotation.as_euler('ZXY', degrees=True))
		
		yaw = -eular[0]
		pitch = -eular[1]
		roll = -eular[2]
		
		rotation = [yaw, pitch, roll]
		
		rig_camera = {"name": get_imgname(sub_img), "camera_model": sub_camera.model, "camera_params": sub_camera.params, "rotation": rotation, "translation": sub_sensor.translation}
		rig_setting.append(rig_camera)
	
	rig_settings.append(rig_setting)


f = open(args.output, "wt")
json.dump(rig_settings, f, indent=4)
f.close()

print(rig_settings)
