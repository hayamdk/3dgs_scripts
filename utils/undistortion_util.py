import numpy as np
from scipy import optimize
import functools

def p2d_opencv(x_norm, y_norm, camera_params, full=True):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	k1 = camera_params[4]
	k2 = camera_params[5]
	p1 = camera_params[6]
	p2 = camera_params[7]
	
	if full:
		k3 = camera_params[8]
		k4 = camera_params[9]
		k5 = camera_params[10]
		k6 = camera_params[11]
	
	r2 = x_norm * x_norm + y_norm * y_norm
	r4 = r2 * r2
	
	if full:
		r6 = r4 * r2
		radial = (1.0 + k1*r2 + k2*r4 + k3*r6) / (1.0 + k4*r2 + k5*r4 + k6*r6)
	else:
		radial = 1.0 + k1*r2 + k2*r4
	
	dx = 2.0*p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm * x_norm)
	dy = p1 * (r2 + 2.0 * y_norm * y_norm) + 2.0*p2 * x_norm * y_norm
	
	x_dist = cx + (x_norm * radial + dx) * fx
	y_dist = cy + (y_norm * radial + dy) * fy
	
	return (x_dist, y_dist)

def calc_undistortion_ratio_opencv(width, height, fx_p, fy_p, camera_params, full=True):

	def equations(xy, x_target, y_target):
		x_dist, y_dist = p2d_opencv(xy[0], xy[1], camera_params, full)
		return np.array([x_dist - x_target, y_dist - y_target])
	
	xy_list = []
	for y in range(0, height):
		xy_list.append((0,     y))
		xy_list.append((width, y))
	
	for x in range(1, width):
		xy_list.append((x,      0))
		xy_list.append((x, height))
	
	max_ratio = 0.0
	
	for x_dist, y_dist in xy_list:
		x0 = [0, 0]
		sol = optimize.root(functools.partial(equations, x_target=x_dist, y_target=y_dist), x0)
		
		x_norm = sol.x[0]
		y_norm = sol.x[1]
		x = x_norm * fx_p + width / 2 + 0.5
		y = y_norm * fy_p + height / 2 + 0.5
		
		x_ratio = abs((x - width/2) / (width/2))
		y_ratio = abs((y - height/2) / (height/2))
		
		if x_ratio > max_ratio:
			max_ratio = x_ratio
		if y_ratio > max_ratio:
			max_ratio = y_ratio
	
	return max_ratio

def __undistort_coord_func(x_dist, y_dist, width_p, height_p, fx_p, fy_p, width, height, divs, d2p_table):
	if x_dist < 0 or width < x_dist or y_dist < 0 or height < y_dist:
		return None
	
	idx_x_f = x_dist * divs / width
	idx_y_f = y_dist * divs / height
	idx_x = int(idx_x_f)
	idx_y = int(idx_y_f)
	xr = idx_x_f - idx_x
	yr = idx_y_f - idx_y
	
	val00 = d2p_table[idx_y][idx_x]
	val10 = d2p_table[idx_y][idx_x+1]
	val01 = d2p_table[idx_y+1][idx_x]
	val11 = d2p_table[idx_y+1][idx_x+1]
	
	if val00 is None or val10 is None or val01 is None or val11 is None:
		return None
	
	x_norm = (1.0-xr)*(1.0-yr) * val00[0] + xr*(1.0-yr) * val10[0] + (1.0-xr)*yr * val01[0] + xr*yr * val11[0]
	y_norm = (1.0-xr)*(1.0-yr) * val00[1] + xr*(1.0-yr) * val10[1] + (1.0-xr)*yr * val01[1] + xr*yr * val11[1]
	
	x = x_norm * fx_p + width_p/2 - 0.5
	y = y_norm * fy_p + height_p/2 - 0.5
	
	return (x, y)

def make_undistort_coord_func(width, height, fx_p, fy_p, camera_params, divs=1024, full=True):
	fx = camera_params[0]
	fy = camera_params[1]
	cx = camera_params[2]
	cy = camera_params[3]
	
	def equations(xy, x_target, y_target):
		x_dist, y_dist = p2d_opencv(xy[0], xy[1], camera_params, full)
		return np.array([x_dist - x_target, y_dist - y_target])
	
	d2p_table = []
	
	for j in range(0, divs+1):
		print("Solving undistortion function:", j, "of", divs, end="   \r")
		d2p_table.append([])
		for i in range(0, divs+1):
			x_dist = i * width / divs
			y_dist = j * height / divs
			x0 = [(x_dist - cx) / fx, (y_dist - cy) / fy]
			#x0 = [0, 0]
			sol = optimize.root(functools.partial(equations, x_target=x_dist, y_target=y_dist), x0)
			x_norm = float(sol.x[0])
			y_norm = float(sol.x[1])
			
			x_dist2, y_dist2 = p2d_opencv(x_norm, y_norm, camera_params, full)
			
			diff = (x_dist - x_dist2) ** 2 + (y_dist - y_dist2) ** 2
			if diff > 0.0001:
				d2p_table[-1].append(None)
			else:
				d2p_table[-1].append((x_norm, y_norm))
	
	return functools.partial(__undistort_coord_func, width=width, height=height, divs=divs, d2p_table=d2p_table)

def make_undistortion_map_opencv(width, height, fx_p, fy_p, camera_params, full=True):
	map_py = []
	for y in range(0, height):
		map_py.append([])
		print("Making undistortion map:", y+1, "of", height, end="   \r")
		for x in range(0, width):
			x_norm = (x + 0.5 - width/2) / fx_p
			y_norm = (y + 0.5 - height/2) / fy_p
			
			x_dist, y_dist = p2d_opencv(x_norm, y_norm, camera_params, full)
			
			map_py[y].append([x_dist, y_dist])

	map_np = np.array(map_py, np.float32)
	return map_np
