import os
import torch
import json
import copy
import io
import base64
import urllib.request
import urllib.error
import random
import time
import tokenizers
import numpy as np
import transformers
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, Optional, Sequence, List
from PIL import Image
from packaging import version
import cv2

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import *

from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX
from streamvln.args import DataArguments

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


DEFAULT_LATENT_TOKEN="<|placeholder_0|>"
Future_num=1

# 关键参数说明
# 参数	含义
# num_future_steps	每个 round 预测的动作数（控制 round 粒度）
# num_frames	窗口大小（一次 forward 处理的步数上限）
# num_history	history 帧采样数（<memory> 展开的帧数



def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    # stop_token_index = tokenizer.convert_tokens_to_ids("Ġstop")

    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

class VLNActionDataset(Dataset):
    def __init__(self, tokenizer, data_args, task_id):
        super(VLNActionDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        self.video_folder = data_args.video_folder.split(',')

        self.nav_data = []
        for vf in self.video_folder:
            anno_json = json.load(open(os.path.join(vf, 'annotations.json'), 'r'))
            for tdata in anno_json:
                tdata['video'] = os.path.join(vf, tdata['video'])
            self.nav_data += anno_json

        self.data_list = []

        for ep_id, item in enumerate(self.nav_data):
            instructions = item['instructions']
            actions = item['actions']
            actions_len_raw = len(actions)
            if actions_len_raw < 4:
                continue

            if not isinstance(instructions, list):
                instructions = [instructions]

            # ===================== [MODIFIED] read deviation flags =====================
            has_deviation = bool(item.get("has_deviation", False))
            deviation_step = item.get("deviation_step", -1)
            try:
                deviation_step = int(deviation_step)
            except Exception:
                deviation_step = -1
            # ==========================================================================

            for ins_id in range(len(instructions)):
                valid_idx = 0
                if self.remove_init_turns:
                    valid_idx = self.clean_initial_rotations(instructions[ins_id], actions)

                # actions_used = actions[1+valid_idx:] + [0]
                # len(actions_used) = (actions_len_raw - (1+valid_idx)) + 1 = actions_len_raw - valid_idx
                actions_len_used = actions_len_raw - valid_idx

                # 原本就有的短轨迹过滤（逻辑等价于你原始的）
                if actions_len_used < 4:
                    continue

                # ===================== [MODIFIED] map deviation_step -> start_offset in actions_used space =====================
                use_deviation = (has_deviation and deviation_step >= 0)
                start_offset = 0
                base_offset = 0

                if use_deviation:
                    # deviation_step assumed in ORIGINAL actions index space
                    base_offset = deviation_step - (1 + valid_idx)
                    # base_offset must be within [0, actions_len_used-1]
                    if base_offset < 0 or base_offset >= actions_len_used:
                        use_deviation = False
                        base_offset = 0

                start_offset = base_offset  # start building windows from here

                # remaining length from start_offset must be enough
                if actions_len_used - start_offset < 4:
                    continue

                # ===================== [MODIFIED] build windows safely: start_idx always < actions_len_used =====================
                for start_idx in range(start_offset, actions_len_used, self.num_frames):
                    # store extra info for debug printing
                    self.data_list.append((ep_id, ins_id, start_idx, valid_idx, use_deviation, deviation_step, base_offset))
                # =====================================================================================

        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

        self.act_conjunctions = [
            'and then ',
            'after that ',
            'next ',
            'the next action is ',
            'followed by ',
            'leading to ',
            'continuing ',
            'subsequently ',
            'proceeding to '
        ]

        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        # ===================== [MODIFIED] prevent print spam =====================
        self._printed_deviation_eps = set()
        self._printed_expert_error = False
        # ========================================================================
        self.use_conav_expert = os.environ.get("conav", "0") == "1"
        self.conav_expert_api = os.environ.get("CONAV_EXPERT_API", "http://127.0.0.1:8005/v1/chat/completions")
        self.conav_expert_model = os.environ.get("CONAV_EXPERT_MODEL", "./checkpoints/InternVL3_5-4B")
        self.conav_expert_prompt = os.environ.get("CONAV_EXPERT_PROMPT", "Describe the image.")
        self.conav_expert_timeout = float(os.environ.get("CONAV_EXPERT_TIMEOUT", "8"))
        self.conav_expert_max_chars = int(os.environ.get("CONAV_EXPERT_MAX_CHARS", "1024"))
        self.conav_expert_wait_seconds = float(os.environ.get("CONAV_EXPERT_WAIT_SECONDS", "0.08"))
        self._conav_expert_cache = {}
        if self.use_conav_expert:
            print(f"[VLNActionDataset] conav=1, expert API enabled: {self.conav_expert_api}")
        self.add_step_info = data_args.add_step_info
        self.max_latent_num = int(os.environ.get("MAX_LATENT_NUM", "-1"))
        if self.max_latent_num>=0:
            self.latent_max_latentpadNum=self.max_latent_num
            print(f"=============self.add_step_info==========”{self.add_step_info}“==============latent mode in VLNActionDataset ===================================== Using latent_num {self.max_latent_num} due to MAX_LATENT_NUM env var.")



    def __len__(self):
        return len(self.data_list)

    def _load_frame_from_mp4(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        else:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

    def _load_raw_frame(self, image_file, is_mp4, mp4_file_path):
        if is_mp4:
            frame_idx = int(os.path.basename(image_file))
            return self._load_frame_from_mp4(mp4_file_path, frame_idx)
        return Image.open(image_file).convert('RGB')

    def _query_conav_expert(self, image: Image.Image):
        if not self.use_conav_expert:
            return ""
        try:
            if self.conav_expert_wait_seconds > 0:
                time.sleep(self.conav_expert_wait_seconds)
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            payload = {
                "model": self.conav_expert_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.conav_expert_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
            }
            req = urllib.request.Request(
                self.conav_expert_api,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.conav_expert_timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(text, str):
                text = str(text)
            text = " ".join(text.strip().split())
            if self.conav_expert_max_chars > 0:
                text = text[: self.conav_expert_max_chars]
            return text
        except Exception as e:
            if not self._printed_expert_error:
                print(f"[VLNActionDataset][conav] expert query failed once: {repr(e)}")
                self._printed_expert_error = True
            return ""

    def _build_conav_expert_infos(self, sample_frames, is_mp4, mp4_file_path):
        infos = []
        for image_file in sample_frames:
            cache_key = f"{mp4_file_path}:{os.path.basename(image_file)}" if is_mp4 else image_file
            if cache_key in self._conav_expert_cache:
                infos.append(self._conav_expert_cache[cache_key])
                continue
            img = self._load_raw_frame(image_file, is_mp4, mp4_file_path)
            expert_text = self._query_conav_expert(img)
            self._conav_expert_cache[cache_key] = expert_text
            infos.append(expert_text)
        return infos

    @property
    def task(self):
        return self.task_id

    def actions2text(self, actions):
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions[str(action)]
            if type(act_text) == list:
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        return ''.join(converted_sequence)

    def prepare_conversation(self, conversation, actions, start_step=0, expert_infos=None):  # 加 start_step
        i = 0
        sources = []
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i + self.num_future_steps]
            answer = self.actions2text(step_actions)
            current_step = start_step + i  # ← 补上这行

            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if self.max_latent_num >= 0:
                source[0]["value"] += f' You can also access to latent observations{DEFAULT_LATENT_TOKEN*self.latent_max_latentpadNum} if needed.'
            round_id = i // self.num_future_steps
            if expert_infos is not None and round_id < len(expert_infos) and expert_infos[round_id]:
                source[0]["value"] += f' You can also access information from experts of spatial understanding: {expert_infos[round_id]}'
            if self.add_step_info:
                step_text = f' You are currently at step {current_step}.'

                step_text += (
                    ' Note: navigation is usually completed within 100 steps.'
                    ' Since you have exceeded this limit, you may have taken a wrong path.'
                    ' Please re-orient yourself and find the correct route to complete the navigation.'
                )
                source[0]["value"] += step_text
            source[1]["value"] = answer
            i += len(step_actions)
            sources.extend(source)
        return sources

    def _load_and_preprocess_frames(self, frame_paths, video_path, is_mp4, mp4_file_path):
        """
        给定帧路径列表，加载并预处理为 tensor，返回 [N, C, H, W]。
        """
        images = []
        for image_file in frame_paths:
            if is_mp4:
                frame_idx = int(os.path.basename(image_file))
                image = self._load_frame_from_mp4(mp4_file_path, frame_idx)
            else:
                image = Image.open(image_file).convert('RGB')

            if self.transforms is not None:
                image = self.transforms(image)

            image = self.image_processor.preprocess(
                images=image, return_tensors='pt'
            )['pixel_values'][0]
            images.append(image)

        if len(images) == 0:
            raise ValueError(f"no images collected from {frame_paths}")

        return torch.stack(images)

    def __getitem__(self, i):
        # ===================== [MODIFIED] robust skip logic =====================
        max_retry = 30
        idx = i
        last_err = None
        # ========================================================================

        def _even_subsample(paths, target_len):
            """Evenly subsample to target_len (target_len > 0)."""
            if len(paths) == target_len:
                return paths
            if len(paths) < target_len:
                # pad by repeating last
                if len(paths) == 0:
                    return paths
                return paths + [paths[-1]] * (target_len - len(paths))
            # len(paths) > target_len
            pick = np.linspace(0, len(paths) - 1, target_len)
            pick = np.round(pick).astype(int).tolist()
            return [paths[p] for p in pick]

        for retry in range(max_retry):
            try:
                # ===================== [MODIFIED] unpack extra fields =====================
                ep_id, ins_id, start_idx, valid_idx, use_deviation, deviation_step, base_offset = self.data_list[idx]
                # ========================================================================

                data = self.nav_data[ep_id]
                video_path = data['video']

                # ===================== [MODIFIED] print once per deviation episode =====================
                if use_deviation and (ep_id not in self._printed_deviation_eps):
                    # print(
                    #     f"[VLNActionDataset] Using deviation start for ep_id={ep_id}: "
                    #     f"has_deviation={data.get('has_deviation', None)}, deviation_step={deviation_step}, "
                    #     f"base_offset(actions_used)={base_offset}, mapped_start_idx(window_start)={start_idx}, valid_idx={valid_idx}"
                    # )
                    self._printed_deviation_eps.add(ep_id)
                # ========================================================================

                # ---------- load frame index list ----------
                rgb_path = os.path.join(video_path, 'rgb')
                is_mp4 = False
                mp4_file_path = None

                if not os.path.isdir(rgb_path):
                    raise ValueError(f"RGB path does not exist: {rgb_path}")

                rgb_contents = os.listdir(rgb_path)
                mp4_files = [f for f in rgb_contents if f.endswith('.mp4')]
                if mp4_files:
                    is_mp4 = True
                    mp4_file_path = os.path.join(rgb_path, mp4_files[0])
                    cap = cv2.VideoCapture(mp4_file_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    video_frames = [str(k) for k in range(total_frames)]
                else:
                    video_frames = sorted(rgb_contents)

                if len(video_frames) == 0:
                    raise ValueError(f"Empty rgb folder: {rgb_path}")

                # ---------- instructions ----------
                instructions = data.get("instructions", None)
                if not isinstance(instructions, list):
                    instructions = [instructions]

                # ---------- actions window ----------
                actions = data['actions'][1 + valid_idx:] + [0]
                actions_len = len(actions)
                if start_idx >= actions_len:
                    raise ValueError(f"start_idx out of range: start_idx={start_idx}, actions_len={actions_len}")

                time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
                if len(time_ids) <= 0:
                    raise ValueError(f"empty time_ids: start_idx={start_idx}, actions_len={actions_len}, num_frames={self.num_frames}")

                actions = np.array(actions)[time_ids]
                L = int(len(actions))
                if L <= 0:
                    raise ValueError("empty actions after slicing by time_ids")

                # ===================== [MODIFIED] expected #rounds == expected #<image> tokens =====================
                # prepare_conversation makes one round per chunk of size num_future_steps
                expected_rounds = (L + self.num_future_steps - 1) // self.num_future_steps
                if expected_rounds <= 0:
                    raise ValueError(f"expected_rounds invalid: {expected_rounds}, L={L}, num_future_steps={self.num_future_steps}")
                # ================================================================================================

                # ---------- sample frames (current observations for each round) ----------
                start_frame_idx = int(time_ids[0]) + valid_idx
                end_frame_idx = int(time_ids[-1]) + 1 + valid_idx
                interval = self.num_future_steps

                sample_step_ids = np.arange(start_frame_idx, end_frame_idx, interval, dtype=np.int32)
                sample_frames = [
                    os.path.join(video_path, 'rgb', video_frames[min(int(j), len(video_frames) - 1)])
                    for j in sample_step_ids
                ]
                # ===================== [MODIFIED] force sample_frames length == expected_rounds =====================
                if len(sample_frames) == 0:
                    # fallback to current start frame
                    fallback_j = min(max(start_frame_idx, 0), len(video_frames) - 1)
                    sample_frames = [os.path.join(video_path, 'rgb', video_frames[fallback_j])]
                sample_frames = _even_subsample(sample_frames, expected_rounds)
                # ================================================================================================

                # ---------- history frames (for <memory>) ----------
                has_history = (int(time_ids[0]) != 0)
                if has_history:
                    history_step_ids = np.arange(
                        0 + valid_idx,
                        int(time_ids[0]) + valid_idx,
                        max(int(time_ids[0]) // self.num_history, 1),
                        dtype=np.int32
                    )
                    history_frames = [
                        os.path.join(video_path, 'rgb', video_frames[min(int(j), len(video_frames) - 1)])
                        for j in history_step_ids
                    ]

                    # ===================== [MODIFIED] CRITICAL: force history length == self.num_history =====================
                    # to match model-side fixed splitting behavior (very likely).
                    if len(history_frames) == 0:
                        # fallback to earliest available frame (valid_idx or 0)
                        fb = min(max(valid_idx, 0), len(video_frames) - 1)
                        history_frames = [os.path.join(video_path, 'rgb', video_frames[fb])]
                    history_frames = _even_subsample(history_frames, self.num_history)
                    # ================================================================================================
                else:
                    history_frames = []



                # ---------- future_1 frames（current 帧的正后一帧）----------
                # sample_step_ids: current 帧在 video_frames 中的索引（已含 valid_idx 偏移）
                # future_1_frames = []
                # for j in sample_step_ids:
                #     next_j = int(j) + 1   # 正后一帧（步长为1）
                #     # 若越界则复用当前帧
                #     next_j = min(next_j, len(video_frames) - 1)
                #     future_1_frames.append(
                #         os.path.join(video_path, 'rgb', video_frames[next_j])
                #     )
                # # future_1_frames 长度 == len(sample_frames) == expected_rounds


                # ---------- load & preprocess images ----------
                images = self._load_and_preprocess_frames(
                    history_frames + sample_frames,
                    video_path, is_mp4, mp4_file_path
                )

                # ---------- future_1_imgs（current 帧的正后一帧）----------
                future_1_frames = []
                for j in sample_step_ids:
                    next_j = min(int(j) + Future_num, len(video_frames) - 1)
                    future_1_frames.append(
                        os.path.join(video_path, 'rgb', video_frames[next_j])
                    )

                future_1_imgs = self._load_and_preprocess_frames(
                    future_1_frames,
                    video_path, is_mp4, mp4_file_path
                )
                # future_1_imgs.shape == [expected_rounds, C, H, W]

                # ---------- build conversation & tokenize ----------
                sources = copy.deepcopy(self.conversations)
                if has_history:
                    sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

                expert_infos = None
                if self.use_conav_expert:
                    expert_infos = self._build_conav_expert_infos(sample_frames, is_mp4, mp4_file_path)

                sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instructions[ins_id])
                interleave_sources = self.prepare_conversation(
                    sources,
                    list(actions),
                    start_step=start_idx,
                    expert_infos=expert_infos,
                )
                data_dict = preprocess([interleave_sources], self.tokenizer, True)

                input_ids = data_dict["input_ids"][0]
                labels = data_dict["labels"][0]

                # ===================== [MODIFIED] strong consistency checks =====================
                num_img_tok = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
                num_mem_tok = int((input_ids == MEMORY_TOKEN_INDEX).sum().item()) if 'MEMORY_TOKEN_INDEX' in globals() else 0

                expected_total_imgs = (self.num_history if has_history else 0) + expected_rounds

                if num_img_tok != expected_rounds:
                    print(
                        f"[WARN][SkipSample] idx={idx} ep_id={ep_id} "
                        f"num_image_tokens={num_img_tok} != expected_rounds={expected_rounds}. Skipping."
                    )
                    idx = (idx + 1) % len(self.data_list)
                    continue

                if has_history and num_mem_tok < 1:
                    # tokenizer might drop/alter it depending on template; skip to be safe
                    print(
                        f"[WARN][SkipSample] idx={idx} ep_id={ep_id} has_history=True but num_mem_tok={num_mem_tok}. Skipping."
                    )
                    idx = (idx + 1) % len(self.data_list)
                    continue

                if int(images.size(0)) != expected_total_imgs:
                    print(
                        f"[WARN][SkipSample] idx={idx} ep_id={ep_id} "
                        f"images={int(images.size(0))} != expected_total_imgs={expected_total_imgs} "
                        f"(history={self.num_history if has_history else 0}, rounds={expected_rounds}). Skipping."
                    )
                    idx = (idx + 1) % len(self.data_list)
                    continue
                # ========================================================================

                return (
                    input_ids,
                    labels,
                    images,
                    torch.tensor(time_ids),
                    self.task,
                    future_1_imgs,   # 新增，shape [expected_rounds, C, H, W]
                )

            except Exception as e:
                last_err = e
                try:
                    print(
                        f"[WARN][SkipSampleException] idx={idx} retry={retry+1}/{max_retry} "
                        f"ep_id={locals().get('ep_id', 'NA')} ins_id={locals().get('ins_id', 'NA')} "
                        f"start_idx={locals().get('start_idx', 'NA')} valid_idx={locals().get('valid_idx', 'NA')} | "
                        f"err={repr(e)}"
                    )
                except Exception:
                    print(f"[WARN][SkipSampleException] idx={idx} retry={retry+1}/{max_retry} err={repr(e)}")

                idx = (idx + 1) % len(self.data_list)
                continue

        raise RuntimeError(f"Too many bad samples starting from index={i}. Last error: {repr(last_err)}")
def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def collate_fn(batch, tokenizer):
    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch, future_1_imgs_batch = zip(*batch)
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_batch = pad_sequence(labels_batch, batch_first=True, padding_value=IGNORE_INDEX)

    input_ids_batch = input_ids_batch[:, :tokenizer.model_max_length]
    labels_batch = labels_batch[:, :tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)

    img_lens = np.array([i.size(0) for i in image_batch])

    if time_ids_batch[0] is not None:
        time_ids_batch = pad_sequence(time_ids_batch, batch_first=True, padding_value=-1)

    image_batch = pad_tensors(image_batch, img_lens)

    # 新增: future_1_imgs_batch pad
    # future_1_imgs 每个 shape 是 [expected_rounds, C, H, W]
    # 同一 batch 内 expected_rounds 可能不同，需要 pad
    future_1_lens = np.array([f.size(0) for f in future_1_imgs_batch])
    future_1_imgs_batch = pad_tensors(future_1_imgs_batch, future_1_lens)

    return {'images': image_batch,
            'time_ids': time_ids_batch,
            'attention_mask': attention_mask,
            'input_ids': input_ids_batch,
            'labels': labels_batch,
            'task_type': task_type_batch,
            'future_1_imgs': future_1_imgs_batch,   # 新增
            }