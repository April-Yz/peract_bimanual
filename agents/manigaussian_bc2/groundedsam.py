import cv2
import numpy as np
import json
import supervision as sv
import os 
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import matplotlib.pyplot as plt
from PIL import Image
import time
from torchvision import transforms

class GroundedSAM:
    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model paths
        model_root = '/data1/zjyang/program/peract_bimanual/third_part/Grounded-Segment-Anything/'
        self.grounding_dino_config_path = os.path.join(model_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        self.grounding_dino_checkpoint_path = os.path.join(model_root, "weights/groundingdino_swint_ogc.pth")
        self.sam_checkpoint_path = os.path.join(model_root, "weights/sam_vit_h_4b8939.pth")
        self.sam_encoder_version = "vit_h"
        
        # Initialize models
        # 但是这里的map location是cpu
        # self.grounding_dino_model = Model(model_config_path=self.grounding_dino_config_path, 
                                        #    model_checkpoint_path=self.grounding_dino_checkpoint_path)
        # # 另一边用的是其他的方法
        args = SLConfig.fromfile(self.grounding_dino_config_path)
        args.device = self.device
        self.model = build_model(args)
        self.checkpoint = torch.load(self.grounding_dino_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(clean_state_dict(self.checkpoint["model"]), strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.sam_model = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path)
        self.sam_model.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam_model) # 这是啥

        # # Task parameters
        # self.task_name = task_name
        # self.episode_id = episode_id
        # self.camera_id = camera_id
        # self.rgb_id = rgb_id
        
        # Detection parameters
        self.classes = ["object."]
        self.text_prompt = "object."
        self.box_threshold = 0.3 # 学长0.3原来0.35?
        self.text_threshold = 0.25
        self.nms_threshold = 0.8

  
    def load_image(self,image_path):
        # return cv2.imread(self.source_image_path)
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image (256, 256, 3)
        # print("image_pil.shape",image_pil.shape)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w (3, 800, 800)
        return image_pil, image    
    
    def trans_image(self,image): # [1,128,128,3]->[3,800,800]
        # 先改形状 转Tensor 归一化
        # return cv2.imread(self.source_image_path)
        # load image
        image_pil =  image.squeeze(0).permute(2,0,1)
        print("image_pil.shape",image_pil.shape) 
        # 定义调整大小的转换
        resize = transforms.Resize((800, 800))
        # 定义归一化的转换
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        image = resize(image_pil)
        print("image.shape",image.shape)
        image = normalize(image) # 3 h w
        return image_pil, image    
    
    def detect_objects(self, image):
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        return detections

    def annotate_image(self, image, detections):
        box_annotator = sv.BoxAnnotator()
        labels = [f"{self.classes[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        return annotated_frame

    def apply_nms(self, detections):
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        return detections

    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def process_image(self):
        image = self.load_image() 
        detections = self.detect_objects(image)
        
        # Annotate and save grounded DINO image
        annotated_frame = self.annotate_image(image, detections)
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

        # Apply NMS
        detections = self.apply_nms(detections)

        # Convert detections to masks
        detections.mask = self.segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)

        # Annotate and save grounded SAM image
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = self.annotate_image(annotated_image, detections)
        cv2.imwrite("grounded_sam_annotated_image.jpg", annotated_image)

    def get_grounding_output(self, image, with_logits=True,):
        # 文本预处理: 将 caption 转为小写，去除前后空白，并确保以句号结尾。
        time1 = time.perf_counter()
        caption = self.text_prompt
        # caption = caption.lower().strip()
        # if not caption.endswith("."):
        #     caption = caption + "."
        # 模型和图像移至指定设备
        model = self.model # .to(self.device)
        image = image.to(self.device)
        time2 = time.perf_counter()
        time_step1 = time2 - time1
        print(f"2 @@@ time2 = {time2} step1 = {time_step1:.2f}s model到GPU") # 0.50s -> 0.00s

        # 不计算梯度: 使用 torch.no_grad() 以禁用梯度计算，加快推理过程并节省内存。
        # 模型推理: 将图像和描述输入模型，获得输出。
        with torch.no_grad():  # 框
            outputs = model(image[None], captions=[caption])
    
        time3 = time.perf_counter()
        time_step2 = time3 - time2
        print(f"3 @@@ time3 = {time3} step2 = {time_step2:.2f}s output time")  # 0.65s -> 0.88s /0.12

        # 处理模型输出
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        # 过滤低置信度的输出: 创建克隆以避免修改原始数据，然后根据 box_threshold 过滤 logits 和 boxes，仅保留高于阈值的项。
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        time4 = time.perf_counter()
        time_step3 = time4 - time3
        print(f"4 @@@ time4 = {time4} step3 = {time_step3:.2f}s 去除第置信度") # 0.03s


        # get phrase 标记化: 使用模型的 tokenizer 对描述进行标记化
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred 构建预测短语
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits: #  如果 with_logits 为真，附加最大置信度得分
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        time5 = time.perf_counter()
        time_step4 = time5 - time4
        print(f"5 @@@ time5 = {time5} step4 = {time_step4:.2f}s 标签 总时间{time5-time1:.2f}s") # 0.0s
        return boxes_filt, pred_phrases    
    
    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)
    
    def save_mask_data(self, output_dir, mask_list, box_list, label_list,):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)

    def save_mask_data1(self, output_dir, mask_list, box_list, label_list, nerf_id="mask"):
        #  9开始左边的box是
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{nerf_id}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        # json_data = [{
        #     'value': value,
        #     'label': 'background'
        # }]
        # for label, box in zip(label_list, box_list):
        #     value += 1
        #     name, logit = label.split('(')
        #     logit = logit[:-1] # the last is ')'
        #     json_data.append({
        #         'value': value,
        #         'label': name,
        #         'logit': float(logit),
        #         'box': box.numpy().tolist(),
        #     })
        # with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        #     json.dump(json_data, f)



    def test(self):
        # cfg
        # sam_hq_checkpoint = args.sam_hq_checkpoint
        # use_sam_hq = args.use_sam_hq #False
        time1 = time.perf_counter()
        # time_step1 = time2 - time1
        print(f"1 ### time1 = {time1}")   
        output_dir = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/4/10"
        # device = args.device
        # Source image path
        task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
        eposide_id='episode0'
        camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
        rgb_id='0000'
        # image_path = f"/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/rgb_{rgb_id}.png"
        image_path="/data1/zjyang/program/peract_bimanual/scripts/test_demo/real_1.png"

        # sam_model = GroundedSAM()
        # make dir
        os.makedirs(output_dir, exist_ok=True)
        # load image 在这里有用（到时候直接传输）
        image_pil, image = self.load_image(image_path) # [3,h,w]
        # load model
        # model = load_model(config_file, grounded_checkpoint, device=device)

        # visualize raw image
        image_pil.save(os.path.join(output_dir, "raw_image1.jpg"))

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            image,
        )
        time2 = time.perf_counter()
        time_step1 = time2 - time1
        print(f"2 ### time2 = {time2} step1 = {time_step1:.2f}s Groundeddino time") # 1.08s -> 1.95s/2.66s ->1.15s ->0.77s/0.15s

        # initialize SAM  是否使用高质量sam模型？ 未提，估计是下面那个
        # 加载sam？改到前面去   self.sam_predictor = SamPredictor(self.sam_model)
        # predictor = SamPredictor(sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path).to(self.device))
        # 读取图像文件
        image = cv2.imread(image_path)
        #  将图像从 BGR 格式转换为 RGB 格式，OpenCV 默认使用 BGR，而大多数其他库使用 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像传递给 SAM 预测器进行处理
        self.sam_predictor.set_image(image)
        time3 = time.perf_counter()
        time_step2 = time3 - time2
        print(f"3 ### time3 = {time3} step2 = {time_step2:.2f}s 加载sam") # 8.39s ->0.54s(使用init)/ 0.91 / 0.47/ 0.45

        size = image_pil.size
        H, W = size[1], size[0]
        # 遍历过滤后的边框
        for i in range(boxes_filt.size(0)):
            # 将边框的坐标缩放到原始图像的尺寸
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            # 调整边框坐标，使其中心与原始坐标对齐
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        # 使用预测器进行掩码预测，不传入点坐标和标签    
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,  # # 不传入坐标
            point_labels = None,    # 不传入标签
            boxes = transformed_boxes.to(self.device),  # 传入变换后的边界框
            multimask_output = False,    # 不使用多掩码输出
        )
        
        time4 = time.perf_counter()
        time_step3 = time4 - time3
        print(f"4 ### time4 = {time4} step3 = {time_step3:.2f}s sam 预测") # 0.03s ->0.05s

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # print('masks', masks,masks.shape)
        # print('plt.gca()',plt.gca())
        for mask in masks:
            # print('mask', mask.shape)
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        self.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
        time5 = time.perf_counter()
        time_step4 = time5 - time4
        print(f"5 ### time5 = {time5} step4 = {time_step4:.2f}s final保存结束  总时间{time5-time1:.2f}s") # 0.03s -> 0.05s

    def getsam(self):
        time1 = time.perf_counter()
        # time_step1 = time2 - time1
        print(f"1 ### time1 = {time1}")   
        output_dir = "/data1/zjyang/program/peract_bimanual/scripts/test_demo/4/2"
        # device = args.device
        # Source image path
        task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
        eposide_id='episode0'
        camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
        rgb_id='0000'
        image_path = f"/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/rgb_{rgb_id}.png"
        image_pil = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        # 将图像转换为 Tensor
        image_pil = transform(image_pil)


    
        # sam_model = GroundedSAM()
        # make dir
        os.makedirs(output_dir, exist_ok=True)
        # load image 在这里有用（到时候直接传输）
        image_pil, image = self.trans_image(image_pil) # [3,h,w] 这个要变成BGR吗
        # load model
        # model = load_model(config_file, grounded_checkpoint, device=device)

        # visualize raw image
        image_pil.save(os.path.join(output_dir, "raw_image1.jpg"))

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            image,
        )
        time2 = time.perf_counter()
        time_step1 = time2 - time1
        print(f"2 ### time2 = {time2} step1 = {time_step1:.2f}s Groundeddino time") # 1.08s -> 1.95s/2.66s ->1.15s ->0.77s/0.15s

        # # 读取图像文件
        # image = cv2.imread(image_path)
        # #  将图像从 BGR 格式转换为 RGB 格式，OpenCV 默认使用 BGR，而大多数其他库使用 RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像传递给 SAM 预测器进行处理
        self.sam_predictor.set_image(image)
        time3 = time.perf_counter()
        time_step2 = time3 - time2
        print(f"3 ### time3 = {time3} step2 = {time_step2:.2f}s 加载sam") # 8.39s ->0.54s(使用init)/ 0.91 / 0.47/ 0.45

        size = image_pil.size
        H, W = size[1], size[0]
        # 遍历过滤后的边框
        for i in range(boxes_filt.size(0)):
            # 将边框的坐标缩放到原始图像的尺寸
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            # 调整边框坐标，使其中心与原始坐标对齐
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        # 使用预测器进行掩码预测，不传入点坐标和标签    
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,  # # 不传入坐标
            point_labels = None,    # 不传入标签
            boxes = transformed_boxes.to(self.device),  # 传入变换后的边界框
            multimask_output = False,    # 不使用多掩码输出
        )
        
        time4 = time.perf_counter()
        time_step3 = time4 - time3
        print(f"4 ### time4 = {time4} step3 = {time_step3:.2f}s sam 预测") # 0.03s ->0.05s

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # print('masks', masks,masks.shape)
        # print('plt.gca()',plt.gca())
        for mask in masks:
            # print('mask', mask.shape)
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        self.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
        time5 = time.perf_counter()
        time_step4 = time5 - time4
        print(f"5 ### time5 = {time5} step4 = {time_step4:.2f}s final保存结束  总时间{time5-time1:.2f}s") # 0.03s -> 0.05s

    def gen_mask(self,image_path, output_dir, nerf_id):
        # time1 = time.perf_counter()
        # # time_step1 = time2 - time1
        # print(f"1 ### time1 = {time1}")   

        # load image 在这里有用（到时候直接传输）
        image_pil, image = self.load_image(image_path) # [3,h,w]

        # visualize raw image
        # image_pil.save(os.path.join(output_dir, "raw_image1.jpg"))

        # run grounding dino model
        boxes_filt, pred_phrases = self.get_grounding_output(
            image,
        )

        # time2 = time.perf_counter()
        # time_step1 = time2 - time1
        # print(f"2 ### time2 = {time2} step1 = {time_step1:.2f}s Groundeddino time") # 1.08s -> 1.95s/2.66s ->1.15s ->0.77s/0.15s

        # initialize SAM  是否使用高质量sam模型？ 未提，估计是下面那个
        # 加载sam？改到前面去   self.sam_predictor = SamPredictor(self.sam_model)
        # predictor = SamPredictor(sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path).to(self.device))
        # 读取图像文件
        image = cv2.imread(image_path)
        #  将图像从 BGR 格式转换为 RGB 格式，OpenCV 默认使用 BGR，而大多数其他库使用 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # 将图像传递给 SAM 预测器进行处理
        self.sam_predictor.set_image(image)
        # time3 = time.perf_counter()
        # time_step2 = time3 - time2
        # print(f"3 ### time3 = {time3} step2 = {time_step2:.2f}s 加载sam") # 8.39s ->0.54s(使用init)/ 0.91 / 0.47/ 0.45

        size = image_pil.size
        H, W = size[1], size[0]
        # 遍历过滤后的边框
        for i in range(boxes_filt.size(0)):
            # 将边框的坐标缩放到原始图像的尺寸
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            # 调整边框坐标，使其中心与原始坐标对齐
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2   # 将边框的左上角坐标调整为中心坐标
            boxes_filt[i][2:] += boxes_filt[i][:2]       # 将边框的右下角坐标调整为以新的左上角为基准   

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        # 使用预测器进行掩码预测，不传入点坐标和标签    
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,  # # 不传入坐标
            point_labels = None,    # 不传入标签
            boxes = transformed_boxes.to(self.device),  # 传入变换后的边界框
            multimask_output = False,    # 不使用多掩码输出
        )
        
        # time4 = time.perf_counter()
        # time_step3 = time4 - time3
        # print(f"4 ### time4 = {time4} step3 = {time_step3:.2f}s sam 预测") # 0.03s ->0.05s

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # print('masks', masks,masks.shape)
        # print('plt.gca()',plt.gca())
        for mask in masks:
            # print('mask', mask.shape)
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, f"grounded_sam_output{nerf_id}.jpg"),
            bbox_inches="tight",# 自动调整图像边界，使其紧凑
            dpi=300, # 设置图像分辨率为 300 DPI（每英寸点数）
            pad_inches=0.0 # 设置图像周围的填充为 0
        )

        self.save_mask_data1(output_dir, masks, boxes_filt, pred_phrases, nerf_id)

        # time5 = time.perf_counter()
        # time_step4 = time5 - time4
        # print(f"5 ### time5 = {time5} step4 = {time_step4:.2f}s final保存结束  总时间{time5-time1:.2f}s") # 0.03s -> 0.05s


if __name__ == "__main__":
    time1 = time.perf_counter()
    sam_model = GroundedSAM()
    time2 = time.perf_counter()
    time_step1 = time2 - time1
    print(f"time2 = {time2} step1 = {time_step1:.2f}s 模型创建时间") # 10.84s

    sam_model.test()
    time3 = time.perf_counter()
    time_step2 = time3 - time2
    print(f"time3 = {time3} step2 = {time_step2:.2f}s 运行时间") # 4.03s /13s
    # sam_model.test()
    # time4 = time.perf_counter()
    # time_step3 = time4 - time3
    # print(f"time4 = {time4} step3 = {time_step3:.2f}s 运行时间2") # 3.65s /4.55s
    # for i in range(10):
    #     sam_model.test()
    #     time5 = time.perf_counter()
    #     time_step4 = time5 - time4
    #     print(f"#################5 ### time5 = {time5} step3 = {time_step4:.2f}s 运行时间3") # 12 6 5 6 6 
        # time4 = time5

    # sam_model.getsam()


    # device = args.device
    # Source image path
    # task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
    # task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
    # for
        # task_name= 'coordinated_take_tray_out_of_oven' # 'handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
        # eposide_id='episode0'
        # camera_id='nerf_data' #'overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
        # rgb_id='0000'
        # # lunshu='0'
        # for lunshu in range(0,319,60):
        #     output_dir = f"/data1/zjyang/program/peract_bimanual/scripts/test_demo/yulan/{task_name}/{lunshu}"
        #     # output_dir= f"/data1/zjyang/program/peract_bimanual/data2\
        #             # /train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/{lunshu}/masks/" #{nerf_id}.png"
        #     # make dir
        #     os.makedirs(output_dir, exist_ok=True)

        #     for nerf_id in range(0, 21):
        #         image_path = f"/data1/zjyang/program/peract_bimanual/data2/train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/{lunshu}/images/{nerf_id}.png"
        #         sam_model.gen_mask(image_path, output_dir, nerf_id)

        # time10 = time.perf_counter()
        # time_step10 = time10 - time1
        # print(f"总时间= {time_step10:.2f}s") # 10.84s