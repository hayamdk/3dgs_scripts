set SPARSE_DIR=D:\path_to_data\0_text

cd /D %~dp0

python export_camera_settings.py ^
	--sparse_dir %SPARSE_DIR% ^
	--output camera_settings.json

pause
