import math
import torch
import torch.nn as nn
from math import ceil
from typing import List, Optional, Union, Tuple

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2ForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava.model.llava_arch import LlavaMetaForCausalLM
from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
import os
DEFAULT_LATENT_TOKEN="<|placeholder_0|>"
DEFAULT_LATENT_INDEX=151647

class StreamVLNModel(LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(StreamVLNModel, self).__init__(config)

        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

        self.num_history = getattr(config, 'num_history', None)

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)   # 残差连接
class StreamVLNForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        config.delay_load = True

        self.model = StreamVLNModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        # === 新增：推理时的 latent cache ===

        self.max_latent_num = int(os.environ.get("MAX_LATENT_NUM", "-1"))
        if self.max_latent_num>=0:
            self._inference_latent_cache = None
            print(f"=====================================latent mode in StreamVLNForCausalLM ===================================== Using latent_num {self.max_latent_num} due to MAX_LATENT_NUM env var.")
            # latent_proj 在 __init__ 中先随机初始化（from_pretrained 会自动覆盖）
            # 在模型中：
            # self.latent_proj = nn.Sequential(
            #     nn.Linear(config.hidden_size, config.hidden_size),
            #     nn.GELU(),
            #     nn.Linear(config.hidden_size, config.hidden_size),
            #     SimpleResBlock(config.hidden_size),  # 带 LayerNorm + 残差的精炼块
            # )
            self.latent_proj = nn.Sequential(
                nn.LayerNorm(config.hidden_size),               # 稳定输入分布
                nn.Linear(config.hidden_size, config.hidden_size),  # 同维线性变换
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),  # 第二层精炼
            )

    def get_model(self):
        return self.model

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side # 27

        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()

        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [ceil(height / stride), ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(3, 4)
        image_feature = torch.cat((image_feature, self.model.image_newline[:,None, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(2, 3).permute(1, 2, 0).contiguous()
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_rgbd(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None):
        batch_size, num_view, _, H, W = images.shape
        image_features = self.get_model().get_vision_tower()(images.flatten(0,1))

        num_patches_per_side = self.get_model().get_vision_tower().num_patches_per_side
        # (B, V, C, num_patch, num_patch)
        image_features = image_features.permute(0, 2, 1).reshape(batch_size, num_view, -1, num_patches_per_side, num_patches_per_side)

        # batch_size, num_view, H, W = depths.shape
        if num_view != 1:
            memory_features = []
            image_features_ = []
            for b in range(batch_size):
                if time_ids[b] is not None:
                    start_idx = time_ids[b][0]
                else:
                    start_idx = 0
                if start_idx == 0:
                    memory_features.append(None)
                    image_features_.append(image_features[b])
                    continue
                else:
                    history_idx = self.model.num_history
                    image_features_.append(image_features[b, history_idx:])
                his_image_feature = image_features[b, :history_idx].flatten(2,3).permute(0,2,1)
                his_image_feature = self.get_model().mm_projector(his_image_feature)
                his_image_feature = self.get_2dPool(his_image_feature, 2) # [N, 196, 1152]

                memory_features.append(his_image_feature.flatten(0,1).unsqueeze(0))
            image_features = image_features_
        else:
            memory_features = [None] * batch_size

        image_features_=[]
        for j, image_feature in enumerate(image_features):
            image_feature = image_feature.flatten(2,3).permute(0,2,1)
            image_feature = self.get_model().mm_projector(image_feature)
            image_feature = self.get_2dPool(image_feature, 2)
            image_features_.append(image_feature)
        image_features = image_features_
        return image_features, memory_features


    def encode_extra_images_to_single_token(self, extra_images):
        """
        extra_images: [B, N, 3, H, W]  —— N 张额外图像
        return: [B, N, D_llm]  —— 每张图 1 个 token
        """
        B, N, C, H, W = extra_images.shape

        # Step1: 一次性过 vision tower（与 encode_rgbd 完全相同路径）
        feat = self.get_model().get_vision_tower()(extra_images.flatten(0, 1))
        # → [B*N, 729, C_vit]

        # Step2: mm_projector 投影（复用已有参数）
        feat = self.get_model().mm_projector(feat)
        # → [B*N, 729, D_llm]

        # Step3: 全局 mean pooling → 每张图 1 token（无新增参数）
        feat = feat.mean(dim=1)
        # → [B*N, D_llm]

        feat = feat.view(B, N, -1)
        # → [B, N, D_llm]

        return feat

    def _replace_latent_tokens_in_segment(self, seg_embed, seg_ids, future_feats_for_sample, cur_latent_id):
        """
        将 seg_embed 中 DEFAULT_LATENT_INDEX 对应位置替换为 future_feats_for_sample[cur_latent_id]。

        Args:
            seg_embed:              当前文本段的 embed，shape [L, D_llm]，可能包含 latent 占位符
            seg_ids:                对应的 token id，shape [L]
            future_feats_for_sample: 本 sample 的 future latent feats，shape [N_rounds, D_llm]
            cur_latent_id:          当前已消耗的 latent token 计数

        Returns:
            seg_embed (替换后):     shape [L, D_llm]
            cur_latent_id (更新后): int
        """
        latent_positions = (seg_ids == DEFAULT_LATENT_INDEX).nonzero(as_tuple=True)[0]
        if len(latent_positions) == 0:
            return seg_embed, cur_latent_id

        seg_embed = seg_embed.clone()
        for pos in latent_positions:
            # 越界保护：pad 帧对应 cur_latent_id 不会出现（因为 DEFAULT_LATENT_INDEX 数量 == 真实round数）
            # 但保险起见加个 assert/clamp
            if cur_latent_id >= future_feats_for_sample.shape[0]:
                break  # pad 越界，直接停止（理论上不会触发）
            seg_embed[pos] = future_feats_for_sample[cur_latent_id]
            cur_latent_id += 1

        return seg_embed, cur_latent_id

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes, depths, poses, intrinsics, time_ids=None, task_ids=None,future_1_imgs=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        image_features, memory_features = self.encode_rgbd(images, depths, poses, intrinsics, time_ids, task_ids)

        # ===== 新增：编码 future_1_imgs =====
        future_latent_feats = None  # shape: [B, N_pad, D_llm] or None
        if self.max_latent_num >= 0 and future_1_imgs is not None:
            with torch.no_grad():#我认为在推理阶段不需要反向传播到 vision tower，所以加上 no_grad 还能同时节省显存和计算资源
                future_latent_feats = self.encode_extra_images_to_single_token(future_1_imgs)
            future_latent_feats = future_latent_feats.detach()
            # future_1_imgs: [B, N_pad, C, H, W] → future_latent_feats: [B, N_pad, D_llm]
            # print(f"Encoded future_1_imgs to future_latent_feats with shape {future_latent_feats.shape}")
        elif self.max_latent_num >= 0 and self.max_latent_num !=10 and future_1_imgs is None:
            if self._inference_latent_cache is None:
                print('------first inference and latent mode------')
                # 第一次推理：用当前帧 image_features 做 mean pooling 作为伪 latent feats
                # image_features: list[B] of [N_views, 196, D_llm]
                # 取每张图全空间 mean → [N_views, D_llm]，拼成 [B, N_views, D_llm]
                pseudo = torch.stack(
                    [feats.mean(dim=1) for feats in image_features], dim=0
                )  # [B, N_views, D_llm]
                future_latent_feats = pseudo
            else:
                # 非第一次：cache shape [N_latent, D_llm]，扩展为 [B, N_latent, D_llm]
                cache = self._inference_latent_cache.unsqueeze(0).expand(
                    len(image_features), -1, -1
                )  # [B, N_latent, D_llm]
                future_latent_feats = cache
        # =====================================

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = [] if labels is not None else None

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_memories = (cur_input_ids == MEMORY_TOKEN_INDEX).sum()
            if self.max_latent_num>=0:
                num_latent= (cur_input_ids == DEFAULT_LATENT_INDEX).sum()
                assert num_latent> 0
            # print(batch_idx, num_images, num_memories)
            num_specials = num_images + num_memories
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + memory_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            cur_img_id = 0
            cur_mem_id = 0
            cur_latent_id = 0  # ← 新增：记录已消耗的 future latent token 数
            latent_abs_positions = []  # ← 新增记录 latent token 在 new_input_embeds 中的绝对位置
            cur_pos = 0  # 当前已追加的 token 数

            for i in range(num_specials + 1):
                cur_seg_embed = cur_input_embeds_no_im[i]
                seg_ids = cur_input_ids_noim[i]

                if future_latent_feats is not None:
                    # 找到本段内 latent 的相对位置
                    latent_rel_positions = (seg_ids == DEFAULT_LATENT_INDEX).nonzero(as_tuple=True)[0]
                    for rel_pos in latent_rel_positions:
                        latent_abs_positions.append(cur_pos + rel_pos.item())

                    cur_seg_embed, cur_latent_id = self._replace_latent_tokens_in_segment(
                        cur_seg_embed, seg_ids,
                        future_latent_feats[batch_idx], cur_latent_id
                    )

                cur_new_input_embeds.append(cur_seg_embed)
                cur_new_labels.append(cur_labels_noim[i])
                cur_pos += cur_seg_embed.shape[0]  # 更新已追加长度
                # =====================================
                if i < num_specials:
                    # print(f"Batch Index: {batch_idx}\n, Current Image Index: {cur_image_idx}\n, Num Images: {num_images}")
                    special_token = special_tokens[i]

                    if special_token == IMAGE_TOKEN_INDEX:
                        cur_image_feature = image_features[batch_idx][cur_img_id]
                        cur_img_id += 1
                        # print(batch_idx, i, 'cur_image_feature shape:', cur_image_feature.shape)
                        cur_new_input_embeds.append(cur_image_feature)
                        cur_new_labels.append(torch.full((cur_image_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_pos += cur_image_feature.shape[0]  # 更新
                    elif special_token == MEMORY_TOKEN_INDEX:
                        # 跳过 memory_features 为 None 的情况（首步或单视角）
                        if memory_features[batch_idx] is None:
                            continue
                        cur_memory_feature = memory_features[batch_idx][cur_mem_id]
                        cur_mem_id += 1
                        # print(batch_idx, i, 'cur_memory_feature shape:', cur_memory_feature.shape)
                        cur_new_input_embeds.append(cur_memory_feature)
                        cur_new_labels.append(torch.full((cur_memory_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_pos += cur_memory_feature.shape[0]  # 更新
                    else:
                        raise NotImplementedError

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # assert len(cur_new_input_embeds) <= 4096
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # # 用于验证 latent_abs_positions求得是否正确
        # if latent_abs_positions and future_1_imgs is not None:
        #     for i, pos in enumerate(latent_abs_positions):
        #         # new_input_embeds shape: [1, seq_len, D_llm]（batch_size=1）
        #         actual_embed = new_input_embeds[0, pos, :]        # 取出该位置的 embedding
        #         expected_embed = future_latent_feats[0, i, :]  # 对应的 future feat
        #         diff = (actual_embed - expected_embed).abs().max().item()
        #         # print(f"[VERIFY] latent[{i}] @ pos={pos}, max_diff={diff:.6f}")
        #         # 若替换正确，diff 应该 == 0.0（完全一致）

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, latent_abs_positions,future_latent_feats

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: torch.FloatTensor = None,
        depths: torch.FloatTensor = None,
        poses: torch.FloatTensor = None,
        intrinsics: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        tokenizer = kwargs.get("tokenizer", None)
        input_ids_ = input_ids
        time_ids = kwargs.get("time_ids", None)
        task_ids = kwargs.get("task_type", None)
        future_1_imgs = kwargs.get("future_1_imgs", None)
        # 默认值：当 inputs_embeds is not None 时（KV-cache 推理路径）不会进入loss分支
        future_latent_feats = None      # ← 加这两行
        latent_abs_positions = None     # ← 加这两行
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latent_abs_positions,future_latent_feats   # 新增返回值
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
                future_1_imgs
            )


        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # ===== 新增 MSE loss =====
        if (future_latent_feats is not None
                and latent_abs_positions is not None
                and len(latent_abs_positions) > 0
                and outputs.loss is not None):

            last_hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_dim)
            latent_pos_tensor = torch.tensor(
                latent_abs_positions, dtype=torch.long, device=last_hidden.device
            )
            latent_hidden = last_hidden[0, latent_pos_tensor]    # (N_latent, hidden_dim)
            latent_pred   = self.latent_proj(latent_hidden)      # (N_latent, hidden_dim)
            latent_target_shifted = future_latent_feats[0, 1:, :]    # (N_latent, D_llm), detached
            latent_pred_shifted   = latent_pred[:-1, :]
            mse_loss = nn.functional.mse_loss(latent_pred_shifted, latent_target_shifted)
            latent_loss_weight = float(os.environ.get("LATENT_LOSS_WEIGHT", "0.1"))
            outputs.loss = outputs.loss + latent_loss_weight * mse_loss

        # 推理时：更新 _inference_latent_cache
        # 条件：latent 位置已记录 + 不是训练（future_1_imgs=None）
        if (self.max_latent_num >= 0 and self.max_latent_num !=10
                and latent_abs_positions is not None
                and len(latent_abs_positions) > 0
                and future_1_imgs is None):          # future_1_imgs 需要从 kwargs 透传过来
            last_hidden = outputs.hidden_states[-1]   # (B, seq_len, D)
            latent_pos_tensor = torch.tensor(
                latent_abs_positions, dtype=torch.long, device=last_hidden.device
            )
            latent_hidden = last_hidden[0, latent_pos_tensor]  # (N_latent, D)
            self._inference_latent_cache = self.latent_proj(latent_hidden).detach()
            # detach：推理不需要梯度，且防止 graph 泄漏
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        time_ids = kwargs.pop("time_ids", None)
        task_ids = kwargs.pop("task_type", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                latent_abs_positions, _   # ← 保留 latent_abs_positions
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
                None
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        env_id = kwargs.pop("env_id", None)
        if self.curr_t[env_id] == 0:
            self.cache[env_id]["inputs_embeds"] = inputs_embeds
        else:
            self.cache[env_id]["inputs_embeds"] = torch.cat([self.cache[env_id]["inputs_embeds"], inputs_embeds],dim=1)
        self.curr_t[env_id] += 1

        outputs = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=self.cache[env_id]["inputs_embeds"],
            output_hidden_states=True,
            **kwargs
        )
        # 紧接在 outputs = super().generate(...) 之后添加：
        if (self.max_latent_num >= 0 and self.max_latent_num != 10
                and latent_abs_positions is not None
                and len(latent_abs_positions) > 0):
            # outputs.hidden_states[0] = 预填充阶段各层隐状态的元组
            # outputs.hidden_states[0][-1] = 最后一层，shape [B, seq_len, D]
            last_hidden = outputs.hidden_states[0][-1]
            latent_pos_tensor = torch.tensor(
                latent_abs_positions, dtype=torch.long, device=last_hidden.device
            )
            latent_hidden = last_hidden[0, latent_pos_tensor]  # [N_latent, D]
            self._inference_latent_cache = self.latent_proj(latent_hidden).detach()

        return outputs


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # print('inputs_embeds', inputs_embeds.shape)
        # print('input_ids', input_ids, cache_position)
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        # print('input_ids', input_ids, cache_position, cache_position.shape)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # print('cache_position_prepare:', cache_position, len(cache_position))
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        elif inputs_embeds is not None and len(cache_position) > 1:
            model_inputs = {"inputs_embeds": inputs_embeds[:, -len(cache_position):], "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": None, #position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if images is not None:
            model_inputs['images'] = images
        if image_sizes is not None:
            model_inputs['image_sizes'] = image_sizes
        return model_inputs

    def reset(self, env_num):
        self.curr_t = [0] * env_num
        self.cache = [dict()] * env_num
        self._inference_latent_cache = None   # 新增

    def reset_for_env(self, env_idx, clear_latent_cache=True):
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = dict()
        if clear_latent_cache:
            self._inference_latent_cache = None
