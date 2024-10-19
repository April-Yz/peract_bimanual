from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

# task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
# task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
rgb_id='0000'
eposide_id='episode0'
camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
# mask_path="/data1/zjyang/program/peract_bimanual/data2/train_data/${task_name}/all_variations/episodes/${eposide_id}/${camera_id}/rgb_${rgb_id}.png"
mask_path = f"/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/rgb_{rgb_id}.png"
output_name = f"{task_name}_{eposide_id}_{camera_id}_{rgb_id}.png"
# output_dir = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/${output_name}"
output_dir = f"/data1/zjyang/program/peract_bimanual/scripts/test_demo/{output_name}"
model_root = '/data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/'
# /data1/zjyang/program/test/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
# /data1/zjyang/program/peract_bimanual/third_part/GroundingDINO/weights/groundingdino_swint_ogc.pth
model = load_model(os.path.join(model_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"), \
                       os.path.join(model_root, "weights/groundingdino_swint_ogc.pth"))
# model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = mask_path # "weights/dog-3.jpeg"
TEXT_PROMPT = "object" # "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

# 图像中各个注释框的位置和大小，通常是一个包含多个矩形框坐标的列表。
# logits：与每个注释框相关的置信度分数，用于表示注释的可靠性。
# phrases：与每个注释框相关的注释文本，即注释的内容。
print("boxes: ", boxes)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite(output_dir, annotated_frame)