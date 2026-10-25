# bash scripts/vis_dataset.sh

dataset_path=/home/cjj/AUV/avoidance_dataset/merged_avoidance.zarr

vis_cloud=0
cd Improved-3D-Diffusion-Policy
python vis_dataset.py --dataset_path $dataset_path \
                    --use_img 0\
                    --vis_cloud 1 \
                    --use_pc_color 0 \
                    --downsample 1 \