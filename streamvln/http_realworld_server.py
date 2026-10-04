import argparse
import numpy as np
import json
import time
import torch
import sys
import os
import transformers
import importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from streamvln.streamvln_agent import VLNEvaluator
if os.environ.get("conav", "0") == "1":
    StreamVLNForCausalLM = importlib.import_module("model.3d_expert_vln").StreamVLNForCausalLM
else:
    StreamVLNForCausalLM = importlib.import_module("model.stream_video_vln").StreamVLNForCausalLM

app = Flask(__name__)
action_seq = np.zeros(4)
idx = 0
terminate = False
total_generate_time = 0.0
start_time = time.time()
output_dir = ''
current_instruction = "Walk forward and immediately stop when you exit the room."
pending_instruction = None
SAVE_ROOT = './realworld_save_img'
os.makedirs(SAVE_ROOT, exist_ok=True)

_input_thread_running = False   # 防止重复启动输入线程

def _ask_next_instruction():
    global pending_instruction, _input_thread_running
    print(f"\n{'='*50}")
    print(f"[STOP] 任务完成！当前指令: \"{current_instruction}\"")
    user_input = input("[STOP] 请输入下一条导航指令（回车保持不变）: ").strip()
    pending_instruction = user_input if user_input else current_instruction
    print(f"[STOP] 下一条指令已就绪: \"{pending_instruction}\"")
    print(f"{'='*50}\n")
    _input_thread_running = False
def _wrap_text(text, font, draw, max_width):
    """将文字按 max_width 自动换行，返回行列表"""
    words = text.split(' ')
    lines = []
    current = ''
    for word in words:
        test = (current + ' ' + word).strip()
        w = draw.textbbox((0, 0), test, font=font)[2]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]

def annotate_image(idx, image, start_time, total_generate_time, llm_output, output_dir):
    Image.fromarray(image).save(f'{output_dir}/rgb_{idx}_raw.jpg')
    image = Image.fromarray(image)
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)
    font_size = 11
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    line_h = 14
    padding = 4
    box_x = 10
    max_text_w = img_w - box_x * 2 - padding * 2

    # ── 前4行：固定单行信息 ──────────────────────────
    info_lines = [
        f"Frame    Id  : {idx}",
        f"Running  time: {time.time() - start_time:.2f} s",
        f"Generate time: {total_generate_time:.2f} s",
        f"Actions      : {llm_output}",
    ]

    # -- Instruction: auto wrap --
    instr_prefix = "Instruction  : "
    instr_wrapped = _wrap_text(instr_prefix + current_instruction, font, draw, max_text_w)

    all_lines = info_lines + instr_wrapped
    total_h = line_h * len(all_lines) + padding * 2

    box_w = img_w - box_x * 2
    draw.rectangle([box_x, 10, box_x + box_w, 10 + total_h], fill='black')

    # ── 逐行写字 ──────────────────────────────────────
    y = 10 + padding
    for line in all_lines:
        draw.text((box_x + padding, y), line, fill='white', font=font)
        y += line_h

    image.save(f'{output_dir}/rgb_{idx}_annotated.png')

@app.route("/eval_vln",methods=['POST'])
def eval_vln():
    global action_seq, idx, terminate, total_generate_time, output_dir, start_time, current_instruction, pending_instruction, _input_thread_running

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = current_instruction


    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        total_generate_time = 0.0
        terminate = False
        idx = 0
        output_dir = os.path.join(SAVE_ROOT, 'runs' + datetime.now().strftime('%m-%d-%H%M'))
        os.makedirs(output_dir, exist_ok=True)
        # 使用 STOP 时预输入的指令（若有），否则保持当前指令
        if pending_instruction is not None:
            current_instruction = pending_instruction
            pending_instruction = None
        print(f"[指令] 执行指令: \"{current_instruction}\"")
        print("init reset model!!!")
        evaluator.reset_memory()
    idx += 1

    if terminate:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})

    for i in range(4):
        t1 = time.time()
        depth = np.zeros((image.shape[0], image.shape[1], 1))
        return_action, generate_time, return_llm_output = evaluator.step(0,
                                        image,
                                        #depth,
                                        #camera_pose,
                                        instruction,
                                        run_model=(evaluator.step_id % 4 == 0))
        llm_output = return_llm_output if return_llm_output is not None else llm_output
        print(f"one evalute cost {time.time() - t1}")
        # total_generate_time += generate_time

        if generate_time > 0:
            total_generate_time = generate_time
        action_seq = action_seq if return_action is None else return_action
        if 0 in action_seq:
            terminate = True
            if not _input_thread_running:
                _input_thread_running = True
                import threading
                threading.Thread(target=_ask_next_instruction, daemon=True).start()
        evaluator.step_id += 1

    str_action = [str(i) for i in action_seq]
    str_action = ''.join(str_action)
    str_action = str_action.replace('1', '↑')  # 前箭头
    str_action = str_action.replace('2', '←')  # 左箭头
    str_action = str_action.replace('3', '→')  # 右箭头
    str_action = str_action.replace('0', 'STOP')  # 停止
    if idx > 1 and total_generate_time > 0.5:
        total_generate_time -= 0.3

    annotate_image(idx, image, start_time, total_generate_time, str_action, output_dir)

    if len(action_seq) == 0:
        print("!!!!!!!!!!!!!!!!!task finish!!!!!!!!!!!!!!!!!!!!!")
        return jsonify({'action': [0]})
    action_list = list(action_seq)
    if 0 in action_list:
        action_list = action_list[:action_list.index(0)]  # 截断 STOP 及之后
    return jsonify({'action': action_list})

if __name__ == '__main__':
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                )
    model.model.num_history = args.num_history
    vision_tower = model.get_vision_tower()
    if isinstance(vision_tower, list):
        vision_tower = vision_tower[0]
    if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
        print("[realworld] vision tower is not loaded, forcing load_model()")
        vision_tower.load_model()
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    vision_tower.to(args.device)
    model.eval()


    vln_sensor_config = {
        "rgb_height" : 1.25,
        "camera_intrinsic" : np.array([[192.        ,   0.        , 191.42857143,   0.        ],
            [  0.        , 192.        , 191.42857143,   0.        ],
            [  0.        ,   0.        ,   1.        ,   0.        ],
            [  0.        ,   0.        ,   0.        ,   1.        ]]),
    }

    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )


    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    app.run(host='0.0.0.0', port=
            5801)
