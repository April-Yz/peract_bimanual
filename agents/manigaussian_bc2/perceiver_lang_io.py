# Perceiver IO implementation adpated for manipulation
# 感知器 IO 实现，用于操作
# Source: https://github.com/lucidrains/perceiver-pytorch
# License: https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE

from math import pi, log
from functools import wraps
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce
from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, \
            Conv3DUpsampleBlock, MultiLayer3DEncoderShallow
from termcolor import colored, cprint

# new
from perceiver_pytorch.perceiver_pytorch import cache_fn
from perceiver_pytorch.perceiver_pytorch import PreNorm, FeedForward, Attention
import logging

# helpers

# def exists(val):
#     return val is not None


# def default(val, d):
#     return val if exists(val) else d


# def cache_fn(f):
#     cache = None

#     @wraps(f)
#     def cached_fn(*args, _cache=True, **kwargs):
#         if not _cache:
#             return f(*args, **kwargs)
#         nonlocal cache
#         if cache is not None:
#             return cache
#         cache = f(*args, **kwargs)
#         return cache

#     return cached_fn

# def fourier_encode(x, max_freq, num_bands = 4):
#     x = x.unsqueeze(-1)
#     device, dtype, orig_x = x.device, x.dtype, x

#     scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
#     scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

#     x = x * scales * pi
#     x = torch.cat([x.sin(), x.cos()], dim = -1)
#     x = torch.cat((x, orig_x), dim = -1)
#     return x

# helper classes

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn, context_dim=None):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)
#         self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

#     def forward(self, x, **kwargs):
#         x = self.norm(x)

#         if exists(self.norm_context):
#             context = kwargs['context']
#             normed_context = self.norm_context(context)
#             kwargs.update(context=normed_context)

#         return self.fn(x, **kwargs)

#     def get_attention_matrix(self, x, **kwargs):
#         x = self.norm(x)

#         if exists(self.norm_context):
#             context = kwargs['context']
#             normed_context = self.norm_context(context)
#             kwargs.update(context=normed_context)
#         kwargs['return_attention_weights'] = True
#         return self.fn(x, **kwargs) 

# class GEGLU(nn.Module):
#     def forward(self, x):
#         x, gates = x.chunk(2, dim=-1)
#         return x * F.gelu(gates)


# class FeedForward(nn.Module):
#     def __init__(self, dim, mult=4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, dim * mult * 2),
#             GEGLU(),
#             nn.Linear(dim * mult, dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
#         super().__init__()
#         inner_dim = dim_head * heads
#         context_dim = default(context_dim, query_dim)
#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, query_dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, context=None, mask=None, return_attention_weights=False):
#         h = self.heads

#         q = self.to_q(x)
#         context = default(context, x)
#         k, v = self.to_kv(context).chunk(2, dim=-1)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         attn = sim.softmax(dim=-1)
        
#         if return_attention_weights:
#             return attn

#         # dropout
#         attn = self.dropout(attn)

#         out = einsum('b i j, b j d -> b i d', attn, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
#         return self.to_out(out)



# PerceiverIO adapted for 6-DoF manipulation
class PerceiverVoxelLangEncoder(nn.Module):
    def __init__(
            self,
            depth,                    # number of self-attention layers
            iterations,               # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,               # N voxels per side (size: N*N*N)
            initial_dim,              # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,             # 4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,       # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,             # 3D tensors have 3 axes
            num_latents=512, # 2048,         #!!不同  # number of latent vectors
            im_channels=64,           # intermediate channel size
            latent_dim=512,           # dimensions of latent vectors
            cross_heads=1,            # number of cross-attention heads
            latent_heads=8,           # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            pos_encoding_with_lang=True,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            lang_fusion_type='seq',
            voxel_patch_size=9,
            voxel_patch_stride=8,
            no_skip_connection=False,
            no_perceiver=False,
            no_language=False,
            final_dim=64,
            cfg=None,   # 多的
    ):
        super().__init__()
        self.cfg = cfg # 多的
        
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = no_language

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features (+ 64 lang goal features if concattenated)
        self.input_dim_before_seq = self.im_channels * 3 if self.lang_fusion_type == 'concat' else self.im_channels * 2

        # !!(中间参数不同)CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, cfg.method.language_model_dim, 77

        # learnable positional encoding
        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         lang_max_seq_len + spatial_size ** 3,
                                                         self.input_dim_before_seq))
        else:
            # assert self.lang_fusion_type == 'concat', 'Only concat is supported for pos encoding without lang.'
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         spatial_size, spatial_size, spatial_size,
                                                         self.input_dim_before_seq))

        # --------------------好像不太一样-------------------------------------
        # 多了1个，少了两行
        # voxel input preprocessing 1x1 conv encoder（bimanual中的）
        self.input_preprocess = Conv3DBlock(  # 双臂时的网络（[1,64,100,100]）
            self.init_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )

        # voxel input preprocessing （mani中的）
        # self.encoder_3d = MultiLayer3DEncoderShallow(in_channels=self.init_dim, out_channels=self.im_channels)   # return x, voxel_list [1,128,100,100]

        # patchify conv
        # self.patchify = Conv3DBlock(
        #     self.encoder_3d.out_channels, 
        #     self.im_channels,
        #     kernel_sizes=self.voxel_patch_size, 
        #     strides=self.voxel_patch_stride,
        #     norm=None, 
        #     activation=activation)

        # V bimanual新增
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels,
            self.im_channels,
            kernel_sizes=self.voxel_patch_size,
            strides=self.voxel_patch_stride,
            norm=None,
            activation=activation,
        )        
        # --------------------好像不太一样-------------------------------------

        # language preprocess
        if self.lang_fusion_type == 'concat':
            # print("---concat----------------------------------")
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)
        elif self.lang_fusion_type == 'seq':
            # print("seqseq----------------------------------",lang_emb_dim,"     ",self.im_channels)
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 2)

        # proprioception
        if self.low_dim_size > 0:
            # print(" self.proprio_preprocess = DenseBlock错误在这里 in out",self.low_dim_size, self.im_channels )
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size, 
                self.im_channels, 
                norm=None, 
                activation=activation,
            )
            # print(" self.proprio_preprocess = DenseBlock after")

        # pooling functions 上面一行bimanual新增
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1) # new
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        # print("---num_latents---------------",num_latents, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        # self.cross_attend_blocks = nn.ModuleList([
        #     PreNorm(latent_dim, Attention(latent_dim,
        #                                   self.input_dim_before_seq,
        #                                   heads=cross_heads,
        #                                   dim_head=cross_dim_head,
        #                                   dropout=input_dropout),
        #             context_dim=self.input_dim_before_seq),
        #     PreNorm(latent_dim, FeedForward(latent_dim)),
        #     PreNorm(latent_dim, FeedForward(latent_dim)), # new for bimanual
        # ])
        # print("------self.cross_attend_blocks ")
        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim,
                        self.input_dim_before_seq,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        dropout=input_dropout,
                    ),
                    context_dim=self.input_dim_before_seq,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, 
                                                    heads=latent_heads,
                                                    dim_head=latent_dim_head, 
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),get_latent_ff(**cache_args),
                get_latent_attn(**cache_args), get_latent_ff(**cache_args) # new for bimanual
            ]))
        # new for bimanual (下面两行新增)    
        self.combined_latent_attn = get_latent_attn(**cache_args)
        self.combined_latent_ff = get_latent_ff(**cache_args)

        # decoder cross attention
        # self.decoder_cross_attn = PreNorm(self.input_dim_before_seq, Attention(self.input_dim_before_seq,
        #                                                                        latent_dim,
        #                                                                        heads=cross_heads,
        #                                                                        dim_head=cross_dim_head,
        #                                                                        dropout=decoder_dropout),
        #                                   context_dim=latent_dim)
        self.decoder_cross_attn_right = PreNorm(
            self.input_dim_before_seq,
            Attention(
                self.input_dim_before_seq,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=decoder_dropout,
            ),
            context_dim=latent_dim,
        )

        self.decoder_cross_attn_left = PreNorm(
            self.input_dim_before_seq,
            Attention(
                self.input_dim_before_seq,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=decoder_dropout,
            ),
            context_dim=latent_dim,
        )

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, 
            self.final_dim,
            kernel_sizes=self.voxel_patch_size, 
            strides=self.voxel_patch_stride,
            norm=None, 
            activation=activation,
        )
        # voxel_patch_size: 5, voxel_patch_stride: 5


        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final 3D softmax
        self.final = Conv3DBlock(
            self.im_channels 
            if (self.no_perceiver or self.no_skip_connection) 
            else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        # self.trans_decoder = Conv3DBlock(
        #     self.final_dim, 1, kernel_sizes=3, strides=1,
        #     norm=None, activation=None,
        # )
        self.right_trans_decoder = Conv3DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        self.left_trans_decoder = Conv3DBlock(
            self.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        # rotation, gripper, and collision MLP layers
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size,
                self.im_channels)

            flat_size += self.im_channels * 4
        #     self.dense0 =  DenseBlock(flat_size, 256, None, activation)
        #     self.dense1 = DenseBlock(256, self.final_dim, None, activation)

        #     self.rot_grip_collision_ff = DenseBlock(self.final_dim,
        #                                             self.num_rotation_classes * 3 + \
        #                                             self.num_grip_classes + \
        #                                             self.num_collision_classes,
        #                                             None, None)
            self.right_dense0 = DenseBlock(flat_size, 256, None, activation)
            self.right_dense1 = DenseBlock(256, self.final_dim, None, activation)

            # print("before self.left_dense0 self.left_dense1")
            self.left_dense0 = DenseBlock(flat_size, 256, None, activation)
            self.left_dense1 = DenseBlock(256, self.final_dim, None, activation)

            self.right_rot_grip_collision_ff = DenseBlock(
                self.final_dim,
                self.num_rotation_classes * 3
                + self.num_grip_classes
                + self.num_collision_classes,
                None,
                None,
            )

            self.left_rot_grip_collision_ff = DenseBlock(
                self.final_dim,
                self.num_rotation_classes * 3
                + self.num_grip_classes
                + self.num_collision_classes,
                None,
                None,
            )

    def encode_text(self, x):
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(x)

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()
        text_mask = torch.where(x==0, x, 1)  # [1, max_token_len]
        return text_feat, text_emb
    
    # --------------------好像不太一样-------------------------------------
    # def save_tensor(self, x, save_path):
    #     import pickle
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(x, f)
    #     print(f'save tensor with shape {x.shape} to {save_path}')
    
    # def counter(self):
    #     if not hasattr(self, '_counter'):
    #         self._counter = 0
    #     else:
    #         self._counter += 1
    #     return self._counter
     # --------------------好像不太一样-------------------------------------

    def forward(
            self,
            ins,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_layer_bounds,
            mask=None,
    ):  

        # preprocess input（多了东西）new bimanual
        # d0, multi_scale_voxel_list = self.encoder_3d(ins) # yzj new [B,10,100,100,100] -> [B,128,100,100,100]
        # print("PerceiverVoxelLangEncoder d1.shape: ", d1.shape)
        # print("multi_scale_voxel_list.shape: ", multi_scale_voxel_list.shape)
        d0 = self.input_preprocess(ins)  # new bimanual # [B,10,100,100,100] -> [B,64,100,100,100]
        # print("ins type======",type(ins),ins.shape)
        # print("PerceiverVoxelLangEncoder d0.shape: ", d0.shape)
        # d0: [1, 128, 100, 100, 100]
        # multi_scale_voxel_list: [torch.Size([1, 10, 100, 100, 100]), torch.Size([1, 32, 25, 25, 25]), torch.Size([1, 16, 50, 50, 50])]
        
        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]
        # feats: [torch.Size([1, 384]), torch.Size([1, 128])]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)                               # [B,128,100,100,100] -> [B,128,20,20,20]
        # voxel_grid = ins.clone() # not working

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio 本体感知信息拼接
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)              # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)                  # [B,128,20,20,20]

        # language ablation
        if self.no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)
            lang_token_embs = torch.zeros_like(lang_token_embs)
        # option 1: tile and concat lang goal to input
        if self.lang_fusion_type == 'concat':
            lang_emb = lang_goal_emb
            lang_emb = lang_emb.to(dtype=ins.dtype)
            l = self.lang_preprocess(lang_emb)
            l = l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, l], dim=1)

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')            # [B,20,20,20,128]

        # add pos encoding to grid
        if not self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        # voxel_grid = ins.clone()

        ######################## NOTE #############################
        # NOTE: If you add positional encodings ^here the lang embs
        # won't have positional encodings. I accidently forgot
        # to turn this off for all the experiments in the paper.
        # So I guess those models were using language embs
        # as a bag of words :( But it doesn't matter much for
        # RLBench tasks since we don't test for novel instructions
        # at test time anyway. The recommend way is to add
        # positional encodings to the final input sequence
        # fed into the Perceiver Transformer, as done below
        # (and also in the Colab tutorial).
        ###########################################################

        # concat to channels of and flatten axis
        queries_orig_shape = ins.shape

        # rearrange input to be channel last
        ins = rearrange(ins, 'b ... d -> b (...) d')          # [B,8000,128]
        ins_wo_prev_layers = ins

        # option 2: add lang token embs as a sequence
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs.float())         # [B,77,512] -> [B,77,256]
            ins = torch.cat((l, ins), dim=1)                  # [B,8077,256]

        # add pos encoding to language + flattened grid (the recommended way)
        if self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        # bimanual ---
        cross_attn, cross_ff_right, cross_ff_left = self.cross_attend_blocks
        # bimanual ---
        # 单臂------------------------
        # cross_attn, cross_ff = self.cross_attend_blocks
        # 单臂------------------------

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=mask) + x # x: 1,2048,512, ins: 1,8077,128
            
            # 单臂------------------------
            # x = cross_ff(x) + x
            # 单臂------------------------
            # bimanual ---
            x_right, x_left = x.chunk(2, dim=1)

            x_right = cross_ff_right(x_right) + x_right
            x_left = cross_ff_left(x_left) + x_left           
            # bimanual ---

            # bimanual ---
            for self_attn_right, self_ff_right, self_attn_left, self_ff_left in self.layers:

                x_right = self_attn_right(x_right) + x_right
                x_right = self_ff_right(x_right) + x_right

                x_left = self_attn_left(x_left) + x_left
                x_left = self_ff_left(x_left) + x_left
            # bimanual ---
            # 单臂------------------------
            # self-attention layers
            # layer_counter = 1
            # for self_attn, self_ff in self.layers:
            #     x = self_attn(x) + x
            #     layer_counter += 1
                
            #     x = self_ff(x) + x
            # 单臂------------------------
                
        # bimanual ---
            x = torch.concat([x_right, x_left], dim=1)
            x = self.combined_latent_attn(x) + x
            x = self.combined_latent_ff(x) + x

        x_right, x_left = x.chunk(2, dim=1)

        # decoder cross attention
        latents_right = self.decoder_cross_attn_right(ins, context=x_right)
        latents_left = self.decoder_cross_attn_left(ins, context=x_left)

        # bimanual ---

        # decoder cross attention
        # ins: [B,8077,128], x: [B,2048,512]
        # latents = self.decoder_cross_attn(ins, context=x) # 单臂
        # latents: [B,8077,128]

        # crop out the language part of the output sequence
        if self.lang_fusion_type == 'seq':
            # latents = latents[:, l.shape[1]:]
            latents_right = latents_right[:, l.shape[1] :]
            latents_left = latents_left[:, l.shape[1] :]

        # reshape back to voxel grid
        # latents = latents.view(b, *queries_orig_shape[1:-1], latents.shape[-1]) # [B,20,20,20,128]
        # latents = rearrange(latents, 'b ... d -> b d ...')                      # [B,128,20,20,20]
      # reshape back to voxel grid
        latents_right = latents_right.view(
            b, *queries_orig_shape[1:-1], latents_right.shape[-1]
        )  # [B,20,20,20,64]
        latents_right = rearrange(latents_right, "b ... d -> b d ...")  # [B,64,20,20,20]

        # reshape back to voxel grid
        latents_left = latents_left.view(
            b, *queries_orig_shape[1:-1], latents_left.shape[-1]
        )  # [B,20,20,20,64]
        latents_left = rearrange(latents_left, "b ... d -> b d ...")  # [B,64,20,20,20]



        # aggregated features from 2nd softmax and maxpool for MLP decoders
        # feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])
        feats_right = feats.copy()
        feats_left = feats
        feats_right.extend(
            [self.ss1(latents_right.contiguous()), self.global_maxp(latents_right).view(b, -1)]
        )
        feats_left.extend(
            [self.ss1(latents_left.contiguous()), self.global_maxp(latents_left).view(b, -1)]
        )


        # upsample
        # latents = self.up0(latents) # [B,256,20,20,20] -> [B,128,100,100,100]
        u0_right = self.up0(latents_right)
        u0_left = self.up0(latents_left)

        # ablations
        # if self.no_skip_connection:
        #     latents = self.final(latents)
        # elif self.no_perceiver:
        #     latents = self.final(d0)
        # else:
        #     latents = self.final(torch.cat([d0, latents], dim=1)) # [1, 128, 100, 100, 100]
        if self.no_skip_connection:
            u_right = self.final(u0_right)
            u_left = self.final(u0_left)
        elif self.no_perceiver:
            u_right = self.final(d0)
            u_left = self.final(d0)
        else:
            u_right = self.final(torch.cat([d0, u0_right], dim=1)) # [1, 64, 100, 100, 100]
            u_left = self.final(torch.cat([d0, u0_left], dim=1))   # [1, 64, 100, 100, 100]

        # print("u_right.shape", u_right.shape,u_left.shape)

        # translation decoder
        # trans = self.trans_decoder(latents) # [1, 1, 100, 100, 100]
        right_trans = self.right_trans_decoder(u_right) # [1, 1, 100, 100, 100]
        left_trans = self.left_trans_decoder(u_left)
        # print("right_trans.shape left_trans.shape",right_trans.shape,left_trans.shape)
        # rotation, gripper, and collision MLPs
        # rot_and_grip_out = None
        # if self.num_rotation_classes > 0:
        #     feats.extend([self.ss_final(latents.contiguous()), self.global_maxp(latents).view(b, -1)])
            
        #     feats = self.dense0(torch.cat(feats, dim=1))   # [1,2048]->[1,256]
        #     feats = self.dense1(feats)                     # [B,72*3+2+2]

        #     rot_and_grip_collision_out = self.rot_grip_collision_ff(feats) # [1,220]
        #     rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]  # [1,218]
        #     collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:] # [1,2]
        # return trans, rot_and_grip_out, collision_out, d0, multi_scale_voxel_list, l

        rot_and_grip_out = None
        if self.num_rotation_classes > 0:
            feats_right.extend(
                [self.ss_final(u_right.contiguous()), self.global_maxp(u_right).view(b, -1)]
            )

            right_dense0 = self.right_dense0(torch.cat(feats_right, dim=1))
            right_dense1 = self.right_dense1(right_dense0)  # [B,72*3+2+2]

            right_rot_and_grip_collision_out = self.right_rot_grip_collision_ff(
                right_dense1
            )
            right_rot_and_grip_out = right_rot_and_grip_collision_out[
                :, : -self.num_collision_classes
            ]
            right_collision_out = right_rot_and_grip_collision_out[
                :, -self.num_collision_classes :
            ]

            feats_left.extend(
                [self.ss_final(u_left.contiguous()), self.global_maxp(u_left).view(b, -1)]
            )

            left_dense0 = self.left_dense0(torch.cat(feats_left, dim=1))
            left_dense1 = self.left_dense1(left_dense0)  # [B,72*3+2+2]

            left_rot_and_grip_collision_out = self.left_rot_grip_collision_ff(
                left_dense1
            )
            left_rot_and_grip_out = left_rot_and_grip_collision_out[
                :, : -self.num_collision_classes
            ]
            left_collision_out = left_rot_and_grip_collision_out[
                :, -self.num_collision_classes :
            ]

        # return trans, rot_and_grip_out, collision_out, d0, multi_scale_voxel_list, l # 原来的单臂操作
        # concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)
        # logging.info(" right_trans",  right_trans.shape)
        # logging.info(f" right_trans:{right_trans.shape} right_rot_and_grip_out shape: {right_rot_and_grip_out.shape} right_collision_out: {right_collision_out.shape}")
        # print(" right_rot_and_grip_out", right_rot_and_grip_out.shape)
        # print(" right_collision_out", right_collision_out.shape)
        # print("d0",d0.shape)
        # print("d1",d1.shape)
        # print("试试在这里拼接?现在有了")
        # d0=torch.cat((d0,d0),dim=1)
        # print("--------------------Perceiver output:--------------- ")
        return (
            right_trans,
            right_rot_and_grip_out,
            right_collision_out,
            left_trans,
            left_rot_and_grip_out,
            left_collision_out), \
            d0, \
            l
        # 暂时移除multi_scale_voxel_list, \
        #     return (
        #     right_trans,
        #     right_rot_and_grip_out,
        #     right_collision_out,
        #     left_trans,
        #     left_rot_and_grip_out,
        #     left_collision_out,
        #     d0, # d1,
        #     multi_scale_voxel_list,
        #     l,
        # )
        # return (  # 原来的双臂操作
        #     right_trans,
        #     right_rot_and_grip_out,
        #     right_collision_out,
        #     left_trans,
        #     left_rot_and_grip_out,
        #     left_collision_out,
        # )