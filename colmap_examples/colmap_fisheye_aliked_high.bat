cd /D D:\path_to_data

set IMAGE_PATH=images
set MASK_PATH=masks
set SPARSE_DIR=sparse
set DBFILE=database.db
set RIG_CONFIG=rig_config.json

mkdir %SPARSE_DIR%

colmap feature_extractor ^
	--database_path %DBFILE% ^
	--image_path %IMAGE_PATH% ^
	--FeatureExtraction.type ALIKED_N16ROT ^
	--FeatureExtraction.max_image_size 1920 ^
	--AlikedExtraction.max_num_features 16384 ^
	--ImageReader.mask_path %MASK_PATH% ^
	--ImageReader.camera_model THIN_PRISM_FISHEYE ^
	--ImageReader.single_camera_per_folder 1

colmap rig_configurator ^
	--database_path %DBFILE% ^
	--rig_config_path %RIG_CONFIG%

colmap vocab_tree_matcher ^
	--database_path %DBFILE% ^
	--FeatureMatching.type ALIKED_LIGHTGLUE ^
	--FeatureMatching.rig_verification 1 ^
	--FeatureMatching.skip_image_pairs_in_same_frame 1

colmap mapper ^
	--database_path %DBFILE% ^
	--image_path %IMAGE_PATH% ^
	--Mapper.ba_refine_sensor_from_rig 1 ^
	--Mapper.ba_refine_focal_length 1 ^
	--Mapper.ba_refine_extra_params 1 ^
	--Mapper.ba_refine_principal_point 1 ^
	--output_path %SPARSE_DIR%

pause
