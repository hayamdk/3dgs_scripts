from dataclasses import dataclass

@dataclass
class ColmapCamera:
	camera_id: int
	model: str
	width: int
	height: int
	params: list

@dataclass
class ColmapImagePoint2D:
	x: float
	y: float
	point3d_id: int

def image_from_line_text(imageline, pointsline):
	params = imageline.split(" ")
	
	image_id = int(params[0])
	qw = float(params[1])
	qx = float(params[2])
	qy = float(params[3])
	qz = float(params[4])
	tx = float(params[5])
	ty = float(params[6])
	tz = float(params[7])
	camera_id = int(params[8])
	name = params[9]
	
	if pointsline == "":
		spl = []
	else:
		spl = pointsline.split(" ")
	
	if len(spl) == 0:
		print("No point2D: ", self.name)
		
	if len(spl) % 3 != 0:
		raise
	
	n_points2d = len(spl) // 3
	
	points2d = []
	for point2d_idx in range(0, n_points2d):
		x = float(spl[point2d_idx * 3])
		y = float(spl[point2d_idx * 3 + 1])
		point3d_id = int(spl[point2d_idx * 3 + 2])
		
		point2d = ColmapImagePoint2D(x, y, point3d_id)
		points2d.append(point2d)
	
	return ColmapImage(image_id, [qx, qy, qz, qw], [tx, ty, tz], camera_id, name, points2d)

class ColmapImage:
	def __init__(self, image_id, quat, translation, camera_id, name, points2d):
		self.image_id = image_id
		self.quat = quat
		self.translation = translation
		self.camera_id = camera_id
		self.name = name
		self.points2d = points2d
	
	def get_line_text(self):
		imageline = " ".join([
			str(self.image_id),
			str(self.quat[3]),
			str(self.quat[0]),
			str(self.quat[1]),
			str(self.quat[2]),
			str(self.translation[0]),
			str(self.translation[1]),
			str(self.translation[2]),
			str(self.camera_id),
			str(self.name)
			])
		
		points = []
		for point2d in self.points2d:
			points.append(str(point2d.x))
			points.append(str(point2d.y))
			points.append(str(point2d.point3d_id))
		
		pointsline = " ".join(points)
		
		return (imageline, pointsline)

@dataclass
class ColmapPoint3DTrack:
	image_id: int
	point2d_id: int

class ColmapPoint3D():
	def __init__(self, line_text):
		spl = line_text.split(" ")
		if len(spl) < 8 or (len(spl) - 8) % 2 != 0:
			raise
		
		track_len = (len(spl) - 8) // 2
		if track_len < 1:
			raise
		
		self.point3d_id = int(spl[0])
		self.x = float(spl[1])
		self.y = float(spl[2])
		self.z = float(spl[3])
		self.r = int(spl[4])
		self.g = int(spl[5])
		self.b = int(spl[6])
		self.error = float(spl[7])
		
		self.track = []
		#self._image_id_table = {}
		for track_idx in range(0, track_len):
			image_id = int(spl[8 + track_idx*2])
			point2d_id = int(spl[8 + track_idx*2 + 1])
			point3dtrack = ColmapPoint3DTrack(image_id, point2d_id)
			#self._image_id_table[image_id] = point3dtrack
			self.track.append(point3dtrack)
	
	#def get_point2d_by_image(image_id):
	#	if image_id not in self._image_id_table:
	#		return None
	#	
	#	return self._image_id_table[image_id].point2d_id

@dataclass
class ColmapRigSensor:
	sensor_type: str
	sensor_id: int
	has_pose: int
	quat: list
	translation: list

@dataclass
class ColmapRig():
	rig_id: int
	ref_sensor_type: str
	ref_sensor_id: int
	sub_sensors: list

@dataclass
class ColmapFrameDataID:
	sensor_type: str
	sensor_id: int
	data_id: int

@dataclass
class ColmapFrame:
	frame_id: int
	rig_id: int
	rig_from_world_quat: list
	rig_from_world_translation: list
	data_ids: list

def read_cameras(cameras_file_path):
	camera_list = []
	
	f = open(cameras_file_path, "rt")
	for line in f.readlines():
		line = line.rstrip()
		
		if line[0] == "#":
			continue
		
		spl = line.split(" ")
		camera = ColmapCamera(
				camera_id = int(spl[0]),
				model = spl[1],
				width = int(spl[2]),
				height = int(spl[3]),
				params = list(map(lambda x: float(x), spl[4:]))
			 )
		camera_list.append(camera)
	
	return camera_list

def save_cameras(cameras, cameras_file_path):
	f = open(cameras_file_path, "wt")
	f.write("#\n#\n#\n#written by colmap_sparse_util\n")
	
	for camera in cameras:
		cols = []
		cols.append(str(camera.camera_id))
		cols.append(str(camera.model))
		cols.append(str(camera.width))
		cols.append(str(camera.height))
		for param in camera.params:
			cols.append(str(param))
		
		line = " ".join(cols)
		f.write(line + "\n")

def read_images(images_file_path, max_images=None):
	images = []
	
	f = open(images_file_path, "rt")
	lines = f.readlines()
	lineidx = 0
	
	while lineidx < len(lines):
		line = lines[lineidx].rstrip()
		lineidx += 1
		if line == "":
			continue
		if line[0] == "#":
			continue
		
		imageline = line
		pointsline = lines[lineidx].rstrip()
		lineidx += 1
		
		image = image_from_line_text(imageline, pointsline)
		images.append(image)
		
		if max_images is not None and len(images) >= max_images:
			break
	
	return images

def save_images(images, images_file_path):
	f = open(images_file_path, "wt")
	f.write("#\n#\n#\n#written by colmap_sparse_util\n")
	
	for image in sorted(images, key=lambda x: x.image_id):
		imageline, pointsline = image.get_line_text()
		f.write(imageline + "\n")
		f.write(pointsline + "\n")

def get_image_point3d_ids(images):
	point3d_ids = set()
	
	for image in images:
		for point2d in image.points2d:
			point3d_ids.add(point2d.point3d_id)
	
	return point3d_ids

def read_points3d(point3d_file_path):
	point3d_list = []
	
	f = open(point3d_file_path, "rt")
	for line in f.readlines():
		line = line.rstrip()
		
		if line == "":
			continue
		if line[0] == "#":
			continue
		
		point3d = ColmapPoint3D(line)
		point3d_list.append(point3d)
	
	return point3d_list

def save_points3d(point3ds, point3d_file_path, filter=None):
	f = open(point3d_file_path, "wt")
	f.write("#\n#\n#written by colmap_sparse_util\n")
	
	for point3d in point3ds:
		if filter is not None and point3d.point3d_id not in filter:
			continue
		
		cols = []
		cols.append(str(point3d.point3d_id))
		cols.append(str(point3d.x))
		cols.append(str(point3d.y))
		cols.append(str(point3d.z))
		cols.append(str(point3d.r))
		cols.append(str(point3d.g))
		cols.append(str(point3d.b))
		cols.append(str(point3d.error))
		
		for point3dtrack in point3d.track:
			cols.append(str(point3dtrack.image_id))
			cols.append(str(point3dtrack.point2d_id))
		
		line = " ".join(cols)
		f.write(line + "\n")

def read_rigs(rigs_file):
	rig_list = []
	
	f = open(rigs_file, "rt")
	for line in f.readlines():
		line = line.rstrip()
		
		if line[0] == "#":
			continue
		
		spl = line.split(" ")
		rig_id = int(spl[0])
		num_sensors = int(spl[1])
		ref_sensor_type = spl[2]
		ref_sensor_id = int(spl[3])
		
		if len(spl) != 4 + (num_sensors-1) * 10:
			raise
		
		sub_sensors = []
		for idx in range(0, num_sensors-1):
			sensor_type = spl[4 + idx*10]
			sensor_id = int(spl[4 + idx*10 + 1])
			has_pose = int(spl[4 + idx*10 + 2])
			qw = float(spl[4 + idx*10 + 3])
			qx = float(spl[4 + idx*10 + 4])
			qy = float(spl[4 + idx*10 + 5])
			qz = float(spl[4 + idx*10 + 6])
			tx = float(spl[4 + idx*10 + 7])
			ty = float(spl[4 + idx*10 + 8])
			tz = float(spl[4 + idx*10 + 9])
			
			sensor = ColmapRigSensor(
					sensor_type = sensor_type,
					sensor_id = sensor_id,
					has_pose = has_pose,
					quat = [qx, qy, qz, qw],
					translation = [tx, ty, tz]
				)
			
			sub_sensors.append(sensor)
		
		rig = ColmapRig(
				rig_id = rig_id,
				ref_sensor_type = ref_sensor_type,
				ref_sensor_id = ref_sensor_id,
				sub_sensors = sub_sensors
			)
		rig_list.append(rig)
	
	return rig_list

def save_rigs(rigs, rigs_file):
	f = open(rigs_file, "wt")
	f.write("#\n#\n#written by colmap_sparse_util\n")
	
	for rig in rigs:
		cols = []
		cols.append(str(rig.rig_id))
		cols.append(str(len(rig.sub_sensors)+1))
		cols.append(rig.ref_sensor_type)
		cols.append(str(rig.ref_sensor_id))
		
		for sub_sensor in rig.sub_sensors:
			cols.append(sub_sensor.sensor_type)
			cols.append(str(sub_sensor.sensor_id))
			cols.append(str(sub_sensor.has_pose))
			cols.append(str(sub_sensor.quat[3]))
			cols.append(str(sub_sensor.quat[0]))
			cols.append(str(sub_sensor.quat[1]))
			cols.append(str(sub_sensor.quat[2]))
			cols.append(str(sub_sensor.translation[0]))
			cols.append(str(sub_sensor.translation[1]))
			cols.append(str(sub_sensor.translation[2]))
		
		line = " ".join(cols)
		f.write(line + "\n")

def read_frames(frames_file):
	frame_list = []
	
	f = open(frames_file, "rt")
	for line in f.readlines():
		line = line.rstrip()
		
		if line[0] == "#":
			continue
		
		spl = line.split(" ")
		frame_id = int(spl[0])
		rig_id = int(spl[1])
		qw = float(spl[2])
		qx = float(spl[3])
		qy = float(spl[4])
		qz = float(spl[5])
		tx = float(spl[6])
		ty = float(spl[7])
		tz = float(spl[8])
		num_data_ids = int(spl[9])
		
		if len(spl) != 10 + num_data_ids * 3:
			raise
		
		data_ids = []
		for idx in range(0, num_data_ids):
			sensor_type = spl[10 + idx*3]
			sensor_id = int(spl[10 + idx*3 + 1])
			data_id = int(spl[10 + idx*3 + 2])
			
			data_id = ColmapFrameDataID(
					sensor_type = sensor_type,
					sensor_id = sensor_id,
					data_id = data_id
				)
			
			data_ids.append(data_id)
		
		frame = ColmapFrame(
				frame_id = frame_id,
				rig_id = rig_id,
				rig_from_world_quat = [qx, qy, qz, qw],
				rig_from_world_translation = [tx, ty, tz],
				data_ids = data_ids
			)
		frame_list.append(frame)
	
	return frame_list

def save_frames(frames, frames_file):
	f = open(frames_file, "wt")
	f.write("#\n#\n#written by colmap_sparse_util\n")
	
	for frame in frames:
		cols = []
		cols.append(str(frame.frame_id))
		cols.append(str(frame.rig_id))
		cols.append(str(frame.rig_from_world_quat[3]))
		cols.append(str(frame.rig_from_world_quat[0]))
		cols.append(str(frame.rig_from_world_quat[1]))
		cols.append(str(frame.rig_from_world_quat[2]))
		cols.append(str(frame.rig_from_world_translation[0]))
		cols.append(str(frame.rig_from_world_translation[1]))
		cols.append(str(frame.rig_from_world_translation[2]))
		cols.append(str(len(frame.data_ids)))
		
		for data_id in frame.data_ids:
			cols.append(data_id.sensor_type)
			cols.append(str(data_id.sensor_id))
			cols.append(str(data_id.data_id))
		
		line = " ".join(cols)
		f.write(line + "\n")

def apply_scale_to_image(images, scale):
	for image in images:
		#print(image.tx, image.ty, image.tz)
		image.tx *= scale
		image.ty *= scale
		image.tz *= scale

def apply_scale_to_point3d(point3ds, scale):
	for point3d in point3ds:
		point3d.x *= scale
		point3d.y *= scale
		point3d.z *= scale
