import cv2
from scipy.spatial.transform import Rotation
from scipy import interpolate
from scipy import optimize
import sympy as sp
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
		rotation_matrix = make_rotation_matrix(yaw, pitch, inverse=True)
	
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

def __f2p_coord_thin_prism(x_fisheye, y_fisheye, width, height, fov_x, fov_y, yaw, pitch, camera_params, r_max_fisheye, f2p_table, divs, rotation_matrix=None):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	
	un = (x_fisheye - cx) / fx
	vn = (y_fisheye - cy) / fy
	
#	if un*un + vn*vn > r_max_fisheye*r_max_fisheye:
#		return None
	
	u_idx_f = un / r_max_fisheye * (divs/2) + divs/2
	v_idx_f = vn / r_max_fisheye * (divs/2) + divs/2
	u_idx = math.floor(u_idx_f)
	v_idx = math.floor(v_idx_f)
	ur = u_idx_f - u_idx
	vr = v_idx_f - v_idx
	
	theta00, phi00 = f2p_table[v_idx][u_idx]
	theta01, phi01 = f2p_table[v_idx+1][u_idx]
	theta10, phi10 = f2p_table[v_idx][u_idx+1]
	theta11, phi11 = f2p_table[v_idx+1][u_idx+1]
	
	def all_valid(*args):
		return all(x is not None for x in args)
	
	if all_valid(theta00, phi00, theta01, phi01, theta10, phi10) and not all_valid(theta11, phi11):
		theta11 = theta01 + theta10 - theta00
		phi11   =   phi01 +   phi10 -   phi00
	elif all_valid(theta00, phi00, theta01, phi01, theta11, phi11) and not all_valid(theta10, phi10):
		theta10 = theta00 + theta11 - theta01
		phi10   =   phi00 +   phi11 -   phi01
	elif all_valid(theta00, phi00, theta10, phi10, theta11, phi11) and not all_valid(theta01, phi01):
		theta01 = theta00 + theta11 - theta10
		phi01   =   phi00 +   phi11 -   phi10
	elif all_valid(theta01, phi01, theta10, phi10, theta11, phi11) and not all_valid(theta00, phi00):
		theta00 = theta01 + theta10 - theta11
		phi00   =   phi01 +   phi10 -   phi11
	elif not all_valid(theta00, phi00, theta01, phi01, theta10, phi10, theta11, phi11):
		return None
	
	#theta = (1.0-ur)*(1.0-vr) * theta00 + ur*(1.0-vr) * theta10 + (1.0-ur)*vr * theta01 + ur*vr * theta11
	#phi =   (1.0-ur)*(1.0-vr) *   phi00 + ur*(1.0-vr) *   phi10 + (1.0-ur)*vr *   phi01 + ur*vr *   phi11
	
	x00 = math.sin(theta00) * math.cos(phi00)
	y00 = math.sin(theta00) * math.sin(phi00)
	z00 = math.cos(theta00)
	x01 = math.sin(theta01) * math.cos(phi01)
	y01 = math.sin(theta01) * math.sin(phi01)
	z01 = math.cos(theta01)
	x10 = math.sin(theta10) * math.cos(phi10)
	y10 = math.sin(theta10) * math.sin(phi10)
	z10 = math.cos(theta10)
	x11 = math.sin(theta11) * math.cos(phi11)
	y11 = math.sin(theta11) * math.sin(phi11)
	z11 = math.cos(theta11)
	
	x_rot = (1.0-ur)*(1.0-vr) * x00 + ur*(1.0-vr) * x10 + (1.0-ur)*vr * x01 + ur*vr * x11
	y_rot = (1.0-ur)*(1.0-vr) * y00 + ur*(1.0-vr) * y10 + (1.0-ur)*vr * y01 + ur*vr * y11
	z_rot = (1.0-ur)*(1.0-vr) * z00 + ur*(1.0-vr) * z10 + (1.0-ur)*vr * z01 + ur*vr * z11
	xyz_rot = np.array([x_rot, y_rot, z_rot])
	
	if rotation_matrix is None:
		rotation_matrix = make_rotation_matrix(yaw, pitch, inverse=True)
	
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
	
	print(x_fisheye, y_fisheye, x, y)

def make_f2p_coord_func_thin_prism(camera_params, divs=1024):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	p1 = camera_params[6]
	p2 = camera_params[7]
	k3 = camera_params[8]
	k4 = camera_params[9]
	sx1 = camera_params[10]
	sy1 = camera_params[11]
	
	fisheye_params = []
	
	r_max_fisheye = get_max_radius_thin_prism(camera_params, 64) * 1.01
	
	def p2f(tp):
		theta, phi = tp
		
		tr = 1 + k1 * (theta**2) + k2 * (theta**4) + k3 * (theta**6) + k4 * (theta**8)
		
		cos_phi = math.cos(phi)
		sin_phi = math.sin(phi)
		
		un = theta*cos_phi*tr + theta*theta*(2*p1*cos_phi*sin_phi + p2*(1+2*cos_phi*cos_phi) + sx1)
		vn = theta*sin_phi*tr + theta*theta*(2*p2*cos_phi*sin_phi + p1*(1+2*sin_phi*sin_phi) + sy1)
		
		return np.array([un, vn])
	
	def equations(tp, u_target, v_target):
		un, vn = p2f(tp)
		return np.array([un - u_target, vn - v_target])
	
	table = []
	
	for v_idx in range(0, divs+1):
		v_target = (v_idx - divs/2) / (divs/2) * r_max_fisheye
		print("Solving dedistortion function:", v_idx, "of", divs, end="   \r")
		table.append([])
		for u_idx in range(0, divs+1):
			u_target = (u_idx - divs/2) / (divs/2) * r_max_fisheye
			
			theta_approx = math.sqrt(u_target**2 + v_target**2)
			phi_approx = math.atan2(v_target, u_target)
			x0 = [theta_approx, phi_approx]
			sol = optimize.root(functools.partial(equations, u_target=u_target, v_target=v_target), x0)
			
			sol_theta = sol.x[0]
			sol_phi = sol.x[1]
			
			chk_x = [sol_theta, sol_phi]
			chk_diff = functools.partial(equations, u_target=u_target, v_target=v_target)(chk_x)
			chk_diff = math.sqrt(chk_diff[0]*chk_diff[0] + chk_diff[1]*chk_diff[1])
			
			if chk_diff > 0.01:
				sol_theta = None
				sol_phi = None
			
			table[v_idx].append([sol_theta, sol_phi])
	
	return functools.partial(__f2p_coord_thin_prism, r_max_fisheye=r_max_fisheye, f2p_table=table, divs=divs)

def __get_angle_limit_thin_prism(u, v, table, divs):
	rad = math.atan2(v, u)
	if rad < 0:
		rad += 2*math.pi
	
	idx_f = rad / (2*math.pi) * divs
	idx = math.floor(idx_f)
	idx_rem = idx_f - idx
	
	theta_interp = table[idx] + (table[idx+1] - table[idx]) * idx_rem
	return theta_interp

def calc_angle_limit_thin_prism(camera_params, divs):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	p1 = camera_params[6]
	p2 = camera_params[7]
	k3 = camera_params[8]
	k4 = camera_params[9]
	sx1 = camera_params[10]
	sy1 = camera_params[11]
	
	theta_max_approx = get_angle_limit([0, 0, 0, 0, k1, k2, k3, k4])
	
	theta_max_table = []
	r_fisheye_max = 0.0
	
	for i in range(0, divs+1):
		print("Calculating angle limit:", i, "of", divs, end="   \r")
		
		rad = 2*math.pi * i / divs
		
		u = math.cos(rad)
		v = math.sin(rad)
		
		theta = sp.Symbol('theta', real=True)
		theta_d = theta * (1 + k1 * (theta**2) + k2 * (theta**4) + k3 * (theta**6) + k4 * (theta**8))
		theta2 = (theta**2)
		
		f = u * theta_d + 2*p1 * theta2 * u*v + p2 * (theta2 + 2*theta2 * u*u) + sx1 * theta2
		g = v * theta_d + 2*p2 * theta2 * u*v + p1 * (theta2 + 2*theta2 * v*v) + sy1 * theta2
		h = f**2 + g**2
		dh = sp.diff(h, theta)
		
		poly = sp.Poly(dh, theta)
		coeffs = [float(c) for c in poly.all_coeffs()]
		
		theta_max = None
		for root in np.roots(coeffs):
			if np.isreal(root) and 0.0 < root.real:
				theta_max_cand = root.real
				if theta_max is None or abs(theta_max_approx - theta_max) > abs(theta_max_approx - theta_max_cand):
					theta_max = theta_max_cand
		
		if theta_max is None:
			theta_max = theta_max_approx
		else:
			r = math.sqrt(h.subs(theta, theta_max).evalf())
			if r > r_fisheye_max:
				r_fisheye_max = r
		
		theta_max_table.append(theta_max)
	
	return [functools.partial(__get_angle_limit_thin_prism, table=theta_max_table, divs=divs), r_fisheye_max]

def get_angle_limit_func_thin_prism(camera_params, divs=256):
	return calc_angle_limit_thin_prism(camera_params, divs)[0]

def get_max_radius_thin_prism(camera_params, divs=256):
	return float(calc_angle_limit_thin_prism(camera_params, divs)[1])

def make_f2p_map_thin_prism(width, height, fov_x, fov_y, yaw, pitch, camera_params):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	p1 = camera_params[6]
	p2 = camera_params[7]
	k3 = camera_params[8]
	k4 = camera_params[9]
	sx1 = camera_params[10]
	sy1 = camera_params[11]
	
	get_theta_max = get_angle_limit_func_thin_prism(camera_params)
	
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
			
			theta_max = get_theta_max(u, v)
			if theta > theta_max:
				theta = theta_max
			
			theta2 = (theta**2)
			theta_d = theta * (1 + k1 * (theta**2) + k2 * (theta**4) + k3 * (theta**6) + k4 * (theta**8))
			
			un = u * theta_d + 2*p1 * theta2 * u*v + p2 * (theta2 + 2*theta2 * u*u) + sx1 * theta2
			vn = v * theta_d + 2*p2 * theta2 * u*v + p1 * (theta2 + 2*theta2 * v*v) + sy1 * theta2
			
			x_fisheye = cx + un * fx
			y_fisheye = cy + vn * fy
			
			map_py[y].append([x_fisheye, y_fisheye])

	map_np = np.array(map_py, np.float32)
	return map_np
