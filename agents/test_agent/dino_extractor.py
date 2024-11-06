'''
修改自XXX 开放词汇文本来分割3D场景
Modified from: https://github.com/Kunhao-Liu/3D-OVS/blob/main/models/DINO_extractor.py
VitExtractor类是一个用于特征提取的工具,它使用了一个预训练的视觉变换器(Vision Transformer,Vit)模型来从输入图像中提取特征。这个类可以用于图像识别、分类或其他计算机视觉任务中的特征提取步骤。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class VitExtractor(nn.Module):
    """继承自nn.Module,表示它是一个神经网络模块。"""
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        # 加载了一个预训练的模型
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval() # 将模型设置为评估模式，这通常用于关闭模型中的某些特定于训练的行为，如Dropout。
        self.patch_size = 14    # 模型的patch大小
        self.feature_dims = 1024 # 特征维度
        # 定义了图像预处理步骤，这里使用了T.Compose来组合多个变换，包括标准化。
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
        ])

        self._freeze()

    def _freeze(self):
        """私有方法，用于冻结模型的参数，使其在训练时不会更新"""
        super().train(mode=False)   # 调用了父类的train方法，将模型设置为非训练模式。
        # 循环遍历模型的所有参数，并将requires_grad属性设置为False，这意味着在反向传播时不会计算这些参数的梯度。
        for p in self.parameters():
            p.requires_grad = False

    # 装饰器应用于forward方法，确保在前向传播时不计算梯度。
    @torch.no_grad()
    def forward(self, input_img):
        """模型的前向传播逻辑：
        获取输入图像的维度。
        使用预处理步骤处理输入图像。
        使用加载的模型提取特征。
        调整特征的维度，以符合特定的输出要求。
        返回处理后的图像特征。"""
        B, C, H, W = input_img.shape
        input_img = self.preprocess(input_img)
        dino_ret = self.model.forward_features(input_img)['x_norm_patchtokens']
        dino_ret = dino_ret.transpose(1, 2).reshape([B, -1, H//self.patch_size, W//self.patch_size])    # [B, 1024, 128, 128]
        return dino_ret
