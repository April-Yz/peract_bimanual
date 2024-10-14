
CUDA_VISIBLE_DEVICES=0 python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint data/groundingdino_swint_ogc.pth \
  --sam_checkpoint data/sam_vit_h_4b8939.pth \
  --input_image /data1/zjyang/program/peract_bimanual/data2/train_data/dual_push_buttons/all_variations/episodes/episode10/nerf_data/8/images/5.png \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "arm" \
  --device "cuda"
