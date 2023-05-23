python run_mvd_vis.py ^
    --mask_ratio 0.9 ^
    --mask_type tube ^
    --model vit_small_patch16_224 ^
    --img_path E:\Job\ASUS\AICUP\Public\Private ^
	--save_path ./OUTPUT/mvd_vit_small_with_vit_base_teacher_k400_epoch_400/video/test ^
	--output_dir ./OUTPUT/mvd_vit_small_with_vit_base_teacher_k400_epoch_400/video/test ^
	--num_frames 16 ^
	--model_path ./OUTPUT/mvd_vit_small_with_vit_base_teacher_k400_epoch_400/20230504/finetune_on_custom/checkpoint-best.pth