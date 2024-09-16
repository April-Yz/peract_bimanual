from torch import nn
import torch

import torch.autograd.profiler as profiler
import agents.manigaussian_bc2.utils as utils
from termcolor import colored
from .attention import Visual3DLangTransformer

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_latent=0,
        d_lang=0,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type="average",
        use_spade=False,
    ):
        """
        FC全连接?
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            nn.init.constant_(self.lin_in.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_lang = d_lang
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()
        

        

    def forward(self, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None, ret_last_feat=False,language_embed=None, batch_size=None):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims 用于多视图输入的组合尺寸。
                张量将被重塑为（-1, combine_inner_dims,...），并在 combine_layer 处使用 combine_type 对维 1 进行缩减。
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in, f"{zx.size(-1)} != {self.d_latent} + {self.d_in}"

            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx

            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    x = utils.combine_interleaved(
                        x, combine_inner_dims, self.combine_type
                    )

                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x)
            out = self.lin_out(self.activation(x))
            if not ret_last_feat:
                return out, x
            else:
                return torch.cat([out, x], dim=-1), x

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction 构建 PyHocon
        return cls(
            d_in,
            n_blocks=conf.n_blocks,
            d_hidden=conf.d_hidden,
            beta=conf.beta,
            combine_layer=conf.combine_layer,
            combine_type=conf.combine_type,  # average | max
            use_spade=conf.use_spade,
            **kwargs
        )

# class ResnetFC_follower(nn.Module):
#     """因为两个一样的维度就会产生超显存的问题"""
#     def __init__(
#         self,
#         d_in,
#         d_out=4,
#         n_blocks=5,
#         d_latent=0,
#         d_lang=0,
#         d_hidden=128,
#         beta=0.0,
#         combine_layer=1000,
#         combine_type="average",
#         use_spade=False,
#     ):
#         """
#         FC全连接?
#         :param d_in input size
#         :param d_out output size
#         :param n_blocks number of Resnet blocks
#         :param d_latent latent size, added in each resnet block (0 = disable)
#         :param d_hidden hiddent dimension throughout network
#         :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
#         """
#         super().__init__()
#         if d_in > 0: # 输入维度>0
#             # 一个全连接层，它将输入特征从 d_in 维映射到 d_hidden维。d_hidden 是网络的隐藏层维度。
#             self.lin_in = nn.Linear(d_in, d_hidden)
#             # 将 self.lin_in 层的偏置参数初始化为0
#             nn.init.constant_(self.lin_in.bias, 0.0)
#             # 初始化 self.lin_in 层的权重
#             nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

#         self.lin_out = nn.Linear(d_hidden, d_out)
#         nn.init.constant_(self.lin_out.bias, 0.0)
#         nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

#         self.n_blocks = n_blocks
#         self.d_latent = d_latent
#         self.d_lang = d_lang
#         self.d_in = d_in
#         self.d_out = d_out
#         self.d_hidden = d_hidden

#         self.combine_layer = combine_layer
#         self.combine_type = combine_type
#         self.use_spade = use_spade

#         # 包含了 n_blocks 个 ResnetBlockFC 残差块实例。每个块的隐藏层维度都是 d_hidden，并且使用了 beta 参数。
#         self.blocks = nn.ModuleList(
#             [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]
#         )

#         # 判断检查潜在特征的维度 d_latent 是否不为0
#         if d_latent != 0:
#             # n_lin_z 计算需要多少个线性层来处理潜在特征，取 combine_layer 和 n_blocks 的最小值。
#             n_lin_z = min(combine_layer, n_blocks)
#             # 是一个 nn.ModuleList，包含了 n_lin_z 个 nn.Linear 层，每个层将潜在特征从 d_latent 维映射到 d_hidden 维。
#             self.lin_z = nn.ModuleList(
#                 [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
#             )
#             # 初始化
#             for i in range(n_lin_z):
#                 nn.init.constant_(self.lin_z[i].bias, 0.0)
#                 nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

#             if self.use_spade:
#                 # 与 self.lin_z 相同数量的 nn.Linear 层，用于实现 SPADE（Spatially Adaptive DENormalization）或其他缩放操作。
#                 self.scale_z = nn.ModuleList(
#                     [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
#                 )
#                 for i in range(n_lin_z):
#                     nn.init.constant_(self.scale_z[i].bias, 0.0)
#                     nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

#         # 根据 beta 参数的值来决定使用哪种激活函数。如果 beta 大于0，使用 nn.Softplus，其平滑度由 beta 控制。如果 beta 小于等于0，使用 nn.ReLU 作为激活函数。
#         if beta > 0:
#             self.activation = nn.Softplus(beta=beta)
#         else:
#             self.activation = nn.ReLU()
        

        

#     def forward(self, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None, ret_last_feat=False,language_embed=None, batch_size=None):
#         """
#         :param zx (..., d_latent + d_in)
#         :param combine_inner_dims Combining dimensions for use with multiview inputs.
#         Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
#         on dim 1, at combine_layer
#         """
#         with profiler.record_function("resnetfc_infer"):
#             assert zx.size(-1) == self.d_latent + self.d_in, f"{zx.size(-1)} != {self.d_latent} + {self.d_in}"

#             if self.d_latent > 0:
#                 z = zx[..., : self.d_latent]
#                 x = zx[..., self.d_latent :]
#             else:
#                 x = zx

#             if self.d_in > 0:
#                 x = self.lin_in(x)
#             else:
#                 x = torch.zeros(self.d_hidden, device=zx.device)

#             for blkid in range(self.n_blocks):
#                 if blkid == self.combine_layer:
#                     x = utils.combine_interleaved(
#                         x, combine_inner_dims, self.combine_type
#                     )

#                 if self.d_latent > 0 and blkid < self.combine_layer:
#                     tz = self.lin_z[blkid](z)
#                     if self.use_spade:
#                         sz = self.scale_z[blkid](z)
#                         x = sz * x + tz
#                     else:
#                         x = x + tz

#                 x = self.blocks[blkid](x)
#             out = self.lin_out(self.activation(x))
#             if not ret_last_feat:
#                 return out, x
#             else:
#                 return torch.cat([out, x], dim=-1), x

#     @classmethod
#     def from_conf(cls, conf, d_in, **kwargs):
#         # PyHocon construction
#         return cls(
#             d_in,
#             n_blocks=conf.n_blocks,
#             d_hidden=conf.d_hidden,
#             beta=conf.beta,
#             combine_layer=conf.combine_layer,
#             combine_type=conf.combine_type,  # average | max
#             use_spade=conf.use_spade,
#             **kwargs
#         )
