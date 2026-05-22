set DATA_DIR=D:\path_to_data

cd /D %~dp0

python fisheye2pers_sparse.py ^
	--image_dir "%DATA_DIR%\images" ^
	--mask_dir "%DATA_DIR%\masks" ^
	--sparse_dir "%DATA_DIR%\sparse\0_text" ^
	--output_dir "%DATA_DIR%\f2p_output" ^
	--generate_gps ^
	--gps_lat_origin 35.0 ^
	--gps_lng_origin -135.0 ^
	--gps_alt_origin 100.0 ^
	--gps_scale 10 ^
	--camera_model OPENCV ^
	--width 2048 ^
	--height 2048 ^
	--fov_x 96 ^
	--fov_y 96 ^
	--rotations (-45,0) (45,0) (-45,90) (45,-90)

pause
