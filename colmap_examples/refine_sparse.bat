cd /D D:\pato_to_data\sparse

set REFINE_INPUT=0
set REFINE_OUTPUT=0_refined

mkdir %REFINE_OUTPUT%

colmap bundle_adjuster ^
	--input_path %REFINE_INPUT% ^
	--BundleAdjustmentCeres.max_num_iterations 400 ^
	--BundleAdjustment.refine_focal_length 1 ^
	--BundleAdjustment.refine_principal_point 1 ^
	--BundleAdjustment.refine_extra_params 1 ^
	--BundleAdjustment.refine_points3D 1 ^
	--BundleAdjustment.refine_sensor_from_rig 1 ^
	--BundleAdjustment.constant_rig_from_world_rotation 1 ^
	--output_path %REFINE_OUTPUT%

pause
