#!/usr/bin/env python3
# coding: utf-8
"""
go2_vln_client_openloop.py
==========================
Go2 开环版本（对标 limo_client.py）
- 无 ROS2，无 PID，无里程计
- 底盘控制：unitree_sdk2py SportClient
- 相机：    unitree_sdk2py VideoClient.GetImageSample()
- 运动方式：开环时间控制（同 LIMO）
- 线程结构：camera_thread + planning_thread + control_thread

Action 编码：
  0 → STOP
  1 → 前进 FORWARD_DIST 米
  2 → 左转 TURN_ANGLE 度
  3 → 右转 TURN_ANGLE 度

Usage:
  python3 go2_vln_client_openloop.py --iface eth0
"""

import argparse
import io
import json
import math
import threading
import time

import cv2
import numpy as np
import PIL.Image as PIL_Image
import requests

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.video.video_client import VideoClient

# ─────────────────────────────────────────────
# 参数配置
# ─────────────────────────────────────────────
SERVER_URL   = "http://localhost:5801/eval_vln"

# 运动参数（根据实际场地调整）
FORWARD_V    = 0.3    # 前进线速度 m/s
FORWARD_DIST = 0.25   # 每步前进距离 m  → 执行时间 = FORWARD_DIST / FORWARD_V
TURN_V       = 0.0    # 转弯时线速度（原地转 = 0）
TURN_W       = 0.5    # 转弯角速度 rad/s
TURN_ANGLE   = 15.0   # 每步转弯角度 deg → 执行时间 = radians(TURN_ANGLE) / TURN_W
STOP_COAST   = 0.05   # 每步结束后惯性滑行等待时间 s

# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
policy_init  = True         # 下一次请求是否重置服务器状态
action_queue = []           # 待执行的动作序列（planning 写，control 读）
queue_lock   = threading.Lock()
should_plan  = threading.Event()
terminate    = False

rgb_image    = None         # 最新 RGB 帧（numpy BGR）
rgb_lock     = threading.Lock()

sport: SportClient = None   # 运动客户端（main 中初始化）
video: VideoClient = None   # 视频客户端（main 中初始化）


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def forward_duration() -> float:
    return FORWARD_DIST / FORWARD_V

def turn_duration() -> float:
    return math.radians(TURN_ANGLE) / TURN_W

def eval_vln_request(image_bgr: np.ndarray) -> list:
    """将 BGR numpy 图像 POST 到服务器，返回 action 列表"""
    global policy_init
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = PIL_Image.fromarray(image_rgb)
    buf       = io.BytesIO()
    pil_img.save(buf, format="jpeg", quality=85)
    buf.seek(0)

    payload     = json.dumps({"reset": policy_init})
    policy_init = False

    t0 = time.time()
    try:
        resp    = requests.post(
            SERVER_URL,
            files={"image": ("rgb_image", buf, "image/jpeg")},
            data={"json": payload},
            timeout=150,
        )
        print(f"[VLN] latency: {time.time()-t0:.2f}s  |  response: {resp.text}")
        actions = json.loads(resp.text)["action"]
    except Exception as e:
        print(f"[VLN] request failed: {e}")
        actions = [0]   # 失败则停止
    return actions


# ─────────────────────────────────────────────
# 相机采集线程
# ─────────────────────────────────────────────
def camera_thread():
    """持续通过 VideoClient.GetImageSample() 获取最新帧，写入全局 rgb_image"""
    global rgb_image
    print("[Camera] started")
    frame_cnt = 0
    while not terminate:
        try:
            code, data = video.GetImageSample()
            if code != 0 or not data:
                time.sleep(0.05)
                continue
            arr   = np.frombuffer(bytes(data), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(0.05)
                continue
            frame_cnt += 1
            if frame_cnt == 1:
                print(f"[Camera] first frame: {frame.shape}")
            with rgb_lock:
                rgb_image = frame
        except Exception as e:
            print(f"[Camera] error: {e}")
            time.sleep(0.1)
        time.sleep(0.1)   # ~10 Hz，与 v2 版本 video_thread 保持一致


# ─────────────────────────────────────────────
# 决策线程（planning）
# ─────────────────────────────────────────────
def planning_thread():
    """等待 should_plan 信号 → 截图 → 请求服务器 → 写入 action_queue"""
    global terminate
    print("[Planning] started")
    while not terminate:
        should_plan.wait()      # 阻塞直到 control_thread 触发
        should_plan.clear()

        with rgb_lock:
            frame = rgb_image.copy() if rgb_image is not None else None

        if frame is None:
            print("[Planning] waiting for camera frame...")
            time.sleep(0.1)
            should_plan.set()   # 还没拿到图像，继续等
            continue

        print("[Planning] requesting server...")
        actions = eval_vln_request(frame)
        print(f"[Planning] actions = {actions}")

        if not actions:
            print("[Planning] task finished (empty actions), stopping.")
            terminate = True
            return

        if 0 in actions:
            terminate = True

        with queue_lock:
            action_queue.extend(actions)


# ─────────────────────────────────────────────
# 执行线程（control）—— 开环时间控制
# ─────────────────────────────────────────────
def control_thread():
    """从 action_queue 取出动作，逐步开环执行"""
    global terminate
    print("[Control] started, waiting for first plan...")

    # 触发第一次规划
    should_plan.set()

    while not terminate:
        with queue_lock:
            if not action_queue:
                # 队列空了，触发新一轮规划
                should_plan.set()
                time.sleep(0.05)
                continue
            action = action_queue.pop(0)

        execute_action(action)

    print("[Control] STOP received, halting robot.")
    try:
        sport.StopMove()
    except Exception:
        pass


def execute_action(action: int):
    """
    开环执行单步动作：
      0 → STOP
      1 → 前进 FORWARD_DIST 米
      2 → 左转 TURN_ANGLE 度
      3 → 右转 TURN_ANGLE 度
    """
    if action == 0:
        sport.Move(0.0, 0.0, 0.0)
        time.sleep(0.1)

    elif action == 1:
        duration = forward_duration()
        print(f"[Control] FORWARD {FORWARD_DIST}m ({duration:.2f}s)")
        sport.Move(FORWARD_V, 0.0, 0.0)
        time.sleep(duration)
        sport.Move(0.0, 0.0, 0.0)
        time.sleep(STOP_COAST)

    elif action == 2:
        duration = turn_duration()
        print(f"[Control] TURN LEFT {TURN_ANGLE}° ({duration:.2f}s)")
        sport.Move(TURN_V, 0.0, TURN_W)
        time.sleep(duration)
        sport.Move(0.0, 0.0, 0.0)
        time.sleep(STOP_COAST)

    elif action == 3:
        duration = turn_duration()
        print(f"[Control] TURN RIGHT {TURN_ANGLE}° ({duration:.2f}s)")
        sport.Move(TURN_V, 0.0, -TURN_W)
        time.sleep(duration)
        sport.Move(0.0, 0.0, 0.0)
        time.sleep(STOP_COAST)

    else:
        print(f"[Control] unknown action {action}, skipping")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface",     type=str, default=None,
                        help="network interface for SDK2 DDS (e.g. eth0)")
    parser.add_argument("--domain_id", type=int, default=0,
                        help="SDK2 DDS domain id (robot=0, sim=1)")
    args = parser.parse_args()

    print("=" * 50)
    print("  Go2 VLN Client (Open-Loop)")
    print(f"  Server : {SERVER_URL}")
    print(f"  Forward: {FORWARD_DIST}m @ {FORWARD_V}m/s")
    print(f"  Turn   : {TURN_ANGLE}° @ {TURN_W}rad/s")
    print("=" * 50)

    # 初始化 SDK2
    if args.iface:
        ChannelFactoryInitialize(args.domain_id, args.iface)
    else:
        ChannelFactoryInitialize(args.domain_id)

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    sport.Move(0.0, 0.0, 0.0)
    print("[SDK2] SportClient ready")

    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    print("[SDK2] VideoClient ready")

    # 启动线程
    t_cam     = threading.Thread(target=camera_thread,   daemon=True)
    t_plan    = threading.Thread(target=planning_thread,  daemon=True)
    t_control = threading.Thread(target=control_thread,   daemon=False)

    t_cam.start()
    time.sleep(1.0)   # 等相机预热，确保能拿到第一帧

    t_plan.start()
    t_control.start()

    try:
        t_control.join()   # 等待 control 线程结束（收到 STOP 后）
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, stopping...")
        terminate = True
    finally:
        try:
            sport.StopMove()
        except Exception:
            pass
        time.sleep(0.1)
        print("[Main] done.")