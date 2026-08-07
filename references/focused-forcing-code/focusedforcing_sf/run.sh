# CUDA_VISIBLE_DEVICES=0 python inference.py --data_path prompts/test/test_0.txt --output_folder videos/test --num_output_frames 126 --method focusedforcing
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --nnodes=1 inference.py --data_path prompts/test/test_5.txt --output_folder videos/test --num_output_frames 126 --method focusedforcing
# 23min