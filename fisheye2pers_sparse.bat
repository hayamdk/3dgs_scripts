set DATA_DIR=D:\path\to\data

cd /D %~dp0

python fisheye2pers_sparse.py ^
	--image_dir "%DATA_DIR%\images" ^
	--mask_dir "%DATA_DIR%\masks" ^
	--sparse_dir "%DATA_DIR%\sparse\0_text" ^
	--output_dir "%DATA_DIR%\f2p_output" ^
	--camera_model PINHOLE ^
	--width 2048 ^
	--height 2048 ^
	--fov_x 96 ^
	--fov_y 96 ^
	--rotations (-45,0) (45,0) (-45,90) (45,-90)

pause
