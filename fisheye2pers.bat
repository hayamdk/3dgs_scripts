set DATA_DIR=D:\path_to_data

cd /D %~dp0

python fisheye2pers.py ^
	--image_dir "%DATA_DIR%\images" ^
	--mask_dir "%DATA_DIR%\masks" ^
	--camera_settings camera_settings.json ^
	--output_dir "%DATA_DIR%\f2p" ^
	--camera_model OPENCV ^
	--width 2048 ^
	--height 2048 ^
	--fov_x 96 ^
	--fov_y 96 ^
	--rotations (-45,0) (45,0) (-45,90) (45,-90)

pause
