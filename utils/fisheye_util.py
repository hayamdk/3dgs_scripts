import cv2
from scipy.spatial.transform import Rotation
from scipy import interpolate
import numpy as np
import math
import functools

def np_to_list(nplist):
    pylist = []
    for val in list(nplist):
        pylist.append(float(val))
    return pylist

def get_angle_limit(camera_params):
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]
	
	theta_max = None
	
	for distortion_root in np.polynomial.Polynomial([1.0, 3.0*k1, 5.0*k2, 7.0*k3, 9.0*k4]).roots():
		if np.isreal(distortion_root) and 0.0 < distortion_root.real:
			theta = math.sqrt(float(distortion_root.real))
			if theta_max is None or theta < theta_max:
				theta_max = theta
	return theta_max

def dedistortion_func(theta_d, camera_params):
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]

	theta = None
	roots = np.polynomial.Polynomial([-theta_d, 1, 0, k1, 0, k2, 0, k3, 0, k4]).roots()
	for distortion_root in roots:
		if np.isreal(distortion_root) and 0.0 <= distortion_root.real:
			theta_ = float(distortion_root.real)
			if theta is None or theta_ < theta:
				theta = theta_
	return theta

def dedistortion_func_interp(theta_d, theta_d_max, interp):
	if theta_d < 0 or theta_d_max < theta_d:
		return None
	return interp(theta_d)

def make_dedistortion_func_interp(camera_params, divs=1000):
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]
	
	theta_max = get_angle_limit(camera_params)
	if theta_max is None:
		return None
	
	x = [0.0]
	y = [0.0]
	for i in range(0, divs):
		theta = theta_max * (i+1) / divs
		theta_d = theta * (1 + k1 * (theta**2) + k2 * (theta**4) + k3 * (theta**6) + k4 * (theta**8))
		x.append(theta_d)
		y.append(theta)
	
	interp = interpolate.interp1d(x, y, kind="cubic")
	
	return functools.partial(dedistortion_func_interp, theta_d_max=theta_d, interp=interp)

def dedistortion_func_linear(theta_d, theta_d_max, table):
	if theta_d < 0.0 or theta_d_max < theta_d:
		return None
	
	divs = len(table) - 1
	i = int(theta_d * divs / theta_d_max)
	
	if i == divs:
		upper = table[i]
	else:
		upper = table[i+1]
	lower = table[i]
	
	factor = theta_d * divs / theta_d_max - i
	return lower + (upper - lower) * factor

def make_dedistortion_func_linear(camera_params, divs=1000):
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]
	
	theta_max = get_angle_limit(camera_params)
	if theta_max is None:
		return None
	
	theta_d_max = theta_max * (1 + k1 * (theta_max**2) + k2 * (theta_max**4) + k3 * (theta_max**6) + k4 * (theta_max**8))
	
	table = []
	for i in range(0, divs):
		theta_d = theta_d_max * i / divs
		theta = dedistortion_func(theta_d, camera_params)
		table.append(theta)
	
	table.append(theta_max)
	
	return functools.partial(dedistortion_func_linear, theta_d_max=theta_d_max, table=table)

def make_rotation_matrix(yaw, pitch, inverse=False):
	if not inverse:
		R_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(yaw), 0]))
		R_pitch, _ = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0]))
		return R_yaw @ R_pitch
	else:
		R_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(-yaw), 0]))
		R_pitch, _ = cv2.Rodrigues(np.array([math.radians(-pitch), 0, 0]))
		return R_pitch @ R_yaw

def f2p_coord(x_fisheye, y_fisheye, width, height, fov_x, fov_y, yaw, pitch, camera_params, dedistortion_func=None, rotation_matrix=None):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]
	
	ud = (x_fisheye - cx) / fx
	vd = (y_fisheye - cy) / fy
	theta_d = math.sqrt(ud**2 + vd**2)
	
	if theta_d == 0.0:
		theta = 0.0
		u = 0.0
		v = 0.0
	else:
		if dedistortion_func is None:
			theta = None
			roots = np.polynomial.Polynomial([-theta_d, 1, 0, k1, 0, k2, 0, k3, 0, k4]).roots()
			for distortion_root in roots:
				if np.isreal(distortion_root) and 0.0 <= distortion_root.real:
					theta_ = float(distortion_root.real)
					if theta is None or theta_ < theta:
						theta = theta_
		else:
			theta = dedistortion_func(theta_d)
		
		if theta is None:
			return None
		
		u = ud / theta_d
		v = vd / theta_d
	
	x_rot = u * math.sin(theta)
	y_rot = v * math.sin(theta)
	z_rot = math.cos(theta)
	xyz_rot = np.array([x_rot, y_rot, z_rot])
	
	if rotation_matrix is None:
		rotation_matrix = make_rotation_mateix(yaw, pitch)
	
	xyz = rotation_matrix @ xyz_rot
	x_world = float(xyz[0])
	y_world = float(xyz[1])
	z_world = float(xyz[2])
	
	if z_world <= 0.0:
		return None
	
	x_norm = (x_world / z_world) / math.tan(math.radians(float(fov_x) / 2.0))
	y_norm = (y_world / z_world) / math.tan(math.radians(float(fov_y) / 2.0))
	
	x = (x_norm / 2.0 + 0.5) * width - 0.5
	y = (y_norm / 2.0 + 0.5) * height - 0.5
	
	return [x, y]

def make_f2p_map(width, height, fov_x, fov_y, yaw, pitch, camera_params):
#	fx = camera_params[0]
#	fy = camera_params[0]
#	cx = camera_params[1]
#	cy = camera_params[2]
#	k1 = camera_params[3]
#	k2 = camera_params[4]
#	k3 = 0.0
#	k4 = 0.0
	
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	k3 = camera_params[6]
	k4 = camera_params[7]
	
	theta_max = get_angle_limit(camera_params)
	
	R_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(yaw), 0]))
	R_pitch, _ = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0]))
	
	map_py = []
	for y in range(0, height):
		map_py.append([])
		print("Making fisheye to pers map:", y+1, "of", height, end="   \r")
		for x in range(0, width):
			x_norm = (x + 0.5) / width * 2 - 1.0
			y_norm = (y + 0.5) / height * 2 - 1.0
			
			x_world = math.tan(math.radians(float(fov_x) / 2.0)) * x_norm
			y_world = math.tan(math.radians(float(fov_y) / 2.0)) * y_norm
			z_world = 1.0
			
			xyz = np.array([x_world, y_world, z_world])
			
			xyz_rot = R_yaw @ R_pitch @ xyz
			x_rot = float(xyz_rot[0])
			y_rot = float(xyz_rot[1])
			z_rot = float(xyz_rot[2])
			
			theta = math.acos(z_rot / math.sqrt(x_rot**2 + y_rot**2 + z_rot**2))
			if x_rot == 0.0 and y_rot == 0.0:
				u = 0.0
				v = 0.0
			else:
				u = x_rot / math.sqrt(x_rot**2 + y_rot**2)
				v = y_rot / math.sqrt(x_rot**2 + y_rot**2)
			
			if theta > theta_max:
				theta = theta_max
			theta_d = theta * (1 + k1 * (theta**2) + k2 * (theta**4) + k3 * (theta**6) + k4 * (theta**8))
			x_fisheye = cx + u * theta_d * fx
			y_fisheye = cy + v * theta_d * fy
			
			map_py[y].append([x_fisheye, y_fisheye])

	map_np = np.array(map_py, np.float32)
	return map_np

def render_pers_images(camera_group_name, rotations, camera_params, fisheye_images_dir, mask_img_path, output_dir):
	f2p_map_cache = {}
	fisheye_mask_img = cv2.imread(mask_img_path, cv2.IMREAD_COLOR)
	
	for fisheye_img_name in os.listdir(fisheye_images_dir):
		basename, ext = os.path.splitext(fisheye_img_name)
		if ext.lower() not in img_exts:
			continue
		
		fisheye_img_path = os.path.join(fisheye_images_dir, fisheye_img_name)
		fisheye_img = cv2.imread(fisheye_img_path, cv2.IMREAD_COLOR)
		
		print(fisheye_img_path)
		
		for rotation in rotations:
			yaw = rotation[0]
			pitch = rotation[1]
			image_dir = os.path.join(output_dir, "images", camera_group_name + "_" + str(yaw).zfill(3) + "_" + str(pitch).zfill(3))
			mask_dir = os.path.join(output_dir, "masks", camera_group_name + "_" + str(yaw).zfill(3) + "_" + str(pitch).zfill(3))
			os.makedirs(image_dir, exist_ok=True)
			os.makedirs(mask_dir, exist_ok=True)
			
			image_path = os.path.join(image_dir, fisheye_img_name)
			mask_path = os.path.join(mask_dir, fisheye_img_name + ".png")
			
			if rotation in f2p_map_cache:
				f2p_map = f2p_map_cache[rotation]
			else:
				f2p_map = get_f2p_mapping(width, height, fov_x, fov_y, yaw, pitch, camera_params)
				print("Made distortion map: rotation=", rotation)
				f2p_map_cache[rotation] = f2p_map
			
			persp_img = cv2.remap(fisheye_img, f2p_map, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			cv2.imwrite(image_path, persp_img)
			
			mask_persp_img = cv2.remap(fisheye_mask_img, f2p_map, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
			cv2.imwrite(mask_path, mask_persp_img)

def rig_config(cam_groups):
	if fov_x != fov_y:
		#TODO PINHOLE model
		raise
	
	rig_cameras = []

	fx = width / 2.0 / math.tan(math.radians(fov_x / 2.0))
	pinhole_camera_params = [fx, width/2, height/2]
	
	for cam_group in cam_groups:
		cam_group_name = cam_group[0]
		base_yaw = cam_group[1]
		sensor_offset = cam_group[2]
		rotations = cam_group[3]
		
		for rotation in rotations:
			yaw = rotation[0]
			pitch = rotation[1]
			camera_name = cam_group_name + "_" + str(yaw).zfill(3) + "_" + str(pitch).zfill(3)
			
			if base_yaw is not None:
				yaw += base_yaw
			
			# Equivalent to:
			#rotation_matrix = Rotation.from_euler(
			#		"XY", [-pitch, -yaw], degrees=True
			#	).as_matrix()
			R_yaw, _ = cv2.Rodrigues(np.array([0, math.radians(-yaw), 0]))
			R_pitch, _ = cv2.Rodrigues(np.array([math.radians(-pitch), 0, 0]))
			rotation_matrix = R_pitch @ R_yaw
			
			if len(rig_cameras) == 0:
				rig_camera = {
						"image_prefix": camera_name + "/",
						"ref_sensor": True,
						"camera_model_name": "SIMPLE_PINHOLE",
						"camera_params": pinhole_camera_params,
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
						"camera_model_name": "SIMPLE_PINHOLE",
						"camera_params": pinhole_camera_params,
						"cam_from_rig_rotation": [rotation_from_ref[3]] + rotation_from_ref[0:3],
						"cam_from_rig_translation": np_to_list(offset_from_ref)
					}
			
			rig_cameras.append(rig_camera)
	
	return rig_cameras
