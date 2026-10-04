#!/usr/bin/env python3
# coding: utf-8
"""
LIMO VLN Client
===============
与 Go2 版本的核心差异：
  - 底盘控制：pylimo.limo.LIMO  替代 unitree ROS2 API
  - 里程计：LIMO 无高精度里程计反馈，改用「开环时间控制」执行每步动作
  - 相机：通过 OpenCV/RealSense SDK 直接读取 RGB 帧（无需 ROS2）
           若有 ROS2 也可切换为 ROS2 subscriber（见注释）
  - 线程结构保持与 Go2 版本一致：planning_thread + control_thread

Action 编码（与服务器 http_realworld_server.py 一致）：
  0 → STOP
  1 → 前进 FORWARD_DIST 米
  2 → 左转 TURN_ANGLE 度
  3 → 右转 TURN_ANGLE 度
"""

import io
import json
import math
import threading
import time

import cv2
import numpy as np
import PIL.Image as PIL_Image
import requests
import serial
from pylimo import limo

# ── 静默 pylimo 内部串口线程的 SerialException（读线程崩溃不影响指令下发）──
_orig_excepthook = threading.excepthook
def _limo_thread_excepthook(args):
    if args.exc_type is serial.SerialException and args.thread is not None \
            and 'Serial' in (args.thread.name or ''):
        pass   # 忽略 pylimo 后台读线程的串口断联异常
    else:
        _orig_excepthook(args)
threading.excepthook = _limo_thread_excepthook

# ─────────────────────────────────────────────
# 参数配置
# ─────────────────────────────────────────────
SERVER_URL    = "http://localhost:5801/eval_vln"  # SSH隧道本地端口，无需修改

# 运动参数（根据实际场地调整）
FORWARD_V     = 0.3    # 前进线速度 m/s
FORWARD_DIST  = 0.25   # 每步前进距离 m  → 执行时间 = FORWARD_DIST / FORWARD_V
TURN_V        = 0.0    # 转弯时线速度（原地转 = 0）
TURN_W        = 0.5    # 转弯角速度 rad/s
TURN_ANGLE    = 15.0   # 每步转弯角度 deg → 执行时间 = radians(TURN_ANGLE) / TURN_W
STOP_COAST    = 0.05   # 每步结束后的惯性滑行等待时间 s

# 相机参数
CAMERA_ID     = 0      # OpenCV 相机索引，RealSense 通常是 0 或 4
CAMERA_W      = 640
CAMERA_H      = 480

# ─────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────
policy_init   = True        # 下一次请求是否重置服务器状态
action_queue  = []          # 待执行的动作序列（planning 写，control 读）
queue_lock    = threading.Lock()
should_plan   = threading.Event()
terminate     = False

rgb_image     = None        # 最新 RGB 帧（numpy BGR）
rgb_lock      = threading.Lock()

bot           = None        # LIMO 实例（main 中初始化）
cap           = None        # OpenCV VideoCapture


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

    payload   = json.dumps({"reset": policy_init})
    policy_init = False

    t0 = time.time()
    try:
        resp = requests.post(
            SERVER_URL,
            files={"image": ("rgb_image", buf, "image/jpeg")},
            data={"json": payload},
            timeout=150,
        )
        print(f"[VLN] server latency: {time.time()-t0:.2f}s  |  response: {resp.text}")
        actions = json.loads(resp.text)["action"]
    except Exception as e:
        print(f"[VLN] request failed: {e}")
        actions = [0]   # 失败则停止
    return actions


# ─────────────────────────────────────────────
# 相机采集线程
# ─────────────────────────────────────────────
def camera_thread():
    """持续读取摄像头最新帧，写入全局 rgb_image"""
    global rgb_image, cap
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 只保留最新帧

    if not cap.isOpened():
        print("[Camera] ERROR: cannot open camera, check CAMERA_ID")
        return

    print("[Camera] started")
    while not terminate:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with rgb_lock:
            rgb_image = frame
    cap.release()


# ─────────────────────────────────────────────
# 决策线程（planning）
# ─────────────────────────────────────────────
def planning_thread():
    """等待 should_plan 信号 → 截图 → 请求服务器 → 写入 action_queue"""
    global terminate
    print("[Planning] started")
    while not terminate:
        should_plan.wait()          # 阻塞直到 control_thread 触发
        should_plan.clear()

        with rgb_lock:
            frame = rgb_image.copy() if rgb_image is not None else None

        if frame is None:
            print("[Planning] waiting for camera frame...")
            time.sleep(0.1)
            should_plan.set()       # 还没拿到图像，继续等
            continue

        print("[Planning] requesting server...")
        actions = eval_vln_request(frame)

        print(f"[Planning] actions = {actions}")

        if not actions:
            # 服务器通知任务完成但不含 STOP，机器人停下等待人工启动下一任务
            print("[Planning] task finished, waiting for next task trigger...")
            # 不 extend action_queue，不 set should_plan，直接 break 或 while 等待
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
                # 队列空了，说明当前 batch 执行完，触发新一轮规划
                should_plan.set()
                time.sleep(0.05)
                continue
            action = action_queue.pop(0)

        execute_action(action)

    # 收到 STOP
    print("[Control] STOP received, halting robot.")
    bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)


def execute_action(action: int):
    """
    开环执行单步动作：
      0 → STOP（不移动，等待 0.1s）
      1 → 前进 FORWARD_DIST 米
      2 → 左转 TURN_ANGLE 度
      3 → 右转 TURN_ANGLE 度
    """
    if action == 0:
        bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
        time.sleep(0.1)

    elif action == 1:
        duration = forward_duration()
        print(f"[Control] FORWARD {FORWARD_DIST}m ({duration:.2f}s)")
        bot.SetMotionCommand(linear_vel=FORWARD_V, angular_vel=0.0)
        time.sleep(duration)
        bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
        time.sleep(STOP_COAST)

    elif action == 2:
        duration = turn_duration()
        print(f"[Control] TURN LEFT {TURN_ANGLE}° ({duration:.2f}s)")
        bot.SetMotionCommand(linear_vel=TURN_V, angular_vel=TURN_W)
        time.sleep(duration)
        bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
        time.sleep(STOP_COAST)

    elif action == 3:
        duration = turn_duration()
        print(f"[Control] TURN RIGHT {TURN_ANGLE}° ({duration:.2f}s)")
        bot.SetMotionCommand(linear_vel=TURN_V, angular_vel=-TURN_W)
        time.sleep(duration)
        bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
        time.sleep(STOP_COAST)

    else:
        print(f"[Control] unknown action {action}, skipping")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  LIMO VLN Client")
    print(f"  Server : {SERVER_URL}")
    print(f"  Forward: {FORWARD_DIST}m @ {FORWARD_V}m/s")
    print(f"  Turn   : {TURN_ANGLE}° @ {TURN_W}rad/s")
    print("=" * 50)

    # 初始化 LIMO
    bot = limo.LIMO()
    bot.EnableCommand()
    bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
    print("[LIMO] initialized")

    # 启动线程
    t_cam     = threading.Thread(target=camera_thread,  daemon=True)
    t_plan    = threading.Thread(target=planning_thread, daemon=True)
    t_control = threading.Thread(target=control_thread,  daemon=False)

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
        bot.SetMotionCommand(linear_vel=0.0, angular_vel=0.0)
        time.sleep(0.1)
        print("[Main] done.")