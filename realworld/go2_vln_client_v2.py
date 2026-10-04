"""
go2_vln_client_v2.py
====================
Architecture:
  Main process  → ROS2 (rclpy): camera + odom subscription, planning
  Child process → SDK2 (SportClient): motion control only

  Communication: multiprocessing.Queue  (vx, vy, vyaw) tuples
                 sentinel None → StopMove + exit

Usage:
  python3 go2_vln_client_v2.py --iface eth0
"""

import multiprocessing as mp
import threading
import PIL.Image as PIL_Image
import io
import json
import requests
import time
import numpy as np
import math
import argparse
from typing import Optional

import cv2

# unitree related (state from ROS2) -- imported after fork inside __main__

# These are needed at class-definition time (top-level), but rclpy.init() is called after fork
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_go.msg import SportModeState

# user-specific
from pid_controller import *
from utils import ReadWriteLock

# ─────────────────────────────────────────────
# SDK2 子进程：与 ROS2 完全隔离
# ─────────────────────────────────────────────
def sdk2_worker(cmd_queue, net_if, domain_id, shared_img, img_meta, img_lock):
    """
    SDK2 worker process: SportClient for motion + VideoClient.GetImageSample() for video.
    VideoClient uses RPC (request-response), not DDS pub/sub.
    """
    import os as _os
    _os.environ["PYTHONUNBUFFERED"] = "1"
    import cv2 as _cv2
    import numpy as _np
    import queue as _queue
    import threading as _threading
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient

    if net_if:
        ChannelFactoryInitialize(domain_id, net_if)
    else:
        ChannelFactoryInitialize(domain_id)

    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()
    print(f"[SDK2 worker] SportClient ready (iface={net_if}, domain={domain_id})", flush=True)

    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()
    print("[SDK2 worker] VideoClient ready", flush=True)

    _frame_cnt = [0]
    _stop_event = _threading.Event()

    def video_thread():
        """独立线程：轮询 GetImageSample，写入共享内存，~10Hz"""
        while not _stop_event.is_set():
            try:
                code, data = video.GetImageSample()
                if code != 0:
                    _stop_event.wait(0.1)
                    continue
                data = bytes(data)
                if not data:
                    _stop_event.wait(0.1)
                    continue
                arr = _np.frombuffer(data, dtype=_np.uint8)
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is None:
                    _stop_event.wait(0.1)
                    continue
                h, w = frame.shape[:2]
                flat = frame.flatten()
                n = len(flat)
                import ctypes as _ctypes
                _ctypes.memmove(shared_img, flat.ctypes.data, min(n, len(shared_img)))
                with img_lock:
                    img_meta[0] = h
                    img_meta[1] = w
                    img_meta[2] = 1
                _frame_cnt[0] += 1
                if _frame_cnt[0] == 1:
                    print(f"[SDK2 video] first frame: {h}x{w}, {len(data)}B", flush=True)
            except Exception as e:
                import traceback as _tb
                print(f"[SDK2 video] error: {e}", flush=True)
                _tb.print_exc()
            _stop_event.wait(0.1)   # ~10 Hz

    vt = _threading.Thread(target=video_thread, daemon=True)
    vt.start()
    print("[SDK2 worker] video thread started", flush=True)

    while True:
        try:
            cmd = cmd_queue.get(timeout=0.05)
        except _queue.Empty:
            continue
        if cmd is None:
            _stop_event.set()
            try:
                sport.StopMove()
            except Exception:
                pass
            print("[SDK2 worker] StopMove & exit")
            break
        vx, vy, vyaw = cmd
        try:
            sport.Move(float(vx), float(vy), float(vyaw))
        except Exception as e:
            print(f"[SDK2 worker] Move error: {e}")


# 全局变量（主进程）
# ─────────────────────────────────────────────
policy_init = True
pid = PID_controller(Kp_trans=3.0, Kd_trans=0.5, Kp_yaw=3.0, Kd_yaw=0.5, max_v=1.0, max_w=1.2)
manager = None
cmd_queue: Optional[mp.Queue] = None   # 发送给 sdk2_worker

rgb_rw_lock  = ReadWriteLock()
odom_rw_lock = ReadWriteLock()


# ─────────────────────────────────────────────
# VLN 推理
# ─────────────────────────────────────────────
def eval_vln(image, depth, camera_pose, instruction, url='http://localhost:5801/eval_vln'):
    global policy_init
    image = PIL_Image.fromarray(image)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpeg')
    image_bytes.seek(0)

    data = {"reset": policy_init}
    json_data = json.dumps(data)
    policy_init = False

    files = {'image': ('rgb_image', image_bytes, 'image/jpg')}
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=150)
    print(f"total time(delay + policy): {time.time() - start}")
    print(response.text)

    action = json.loads(response.text)['action']
    return action


# ─────────────────────────────────────────────
# 控制线程：PID → 发指令给子进程
# ─────────────────────────────────────────────
def control_thread():
    while True:
        if manager is None or cmd_queue is None:
            time.sleep(0.05)
            continue

        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel       = manager.vel.copy()       if manager.vel       is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            cmd_queue.put((v, 0.0, w))   # 发给 sdk2_worker

        if abs(e_p) < 0.1 and abs(e_r) < 0.1:
            manager.trigger_replan()

        time.sleep(0.1)


# ─────────────────────────────────────────────
# 规划线程：获取图像 → 请求 VLN → 更新目标
# ─────────────────────────────────────────────
def planning_thread():
    while True:
        if manager is None:
            time.sleep(0.05)
            continue

        if not manager.should_plan:
            time.sleep(0.01)
            continue

        rgb_rw_lock.acquire_read()
        rgb_image = manager.rgb_image
        rgb_rw_lock.release_read()

        if rgb_image is None:
            print("[Planning] rgb_image is None, waiting for camera...")
            manager.should_plan = False  # reset to avoid busy loop
            time.sleep(0.5)
            continue

        actions = eval_vln(rgb_image, None, None, None)

        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()

        time.sleep(0.1)


# ─────────────────────────────────────────────
# ROS2 Image → numpy
# ─────────────────────────────────────────────
def imgmsg_to_numpy(msg, desired_encoding):
    dtype_map = {
        'bgr8':  (np.uint8,   3),
        'rgb8':  (np.uint8,   3),
        '16UC1': (np.uint16,  1),
        '32FC1': (np.float32, 1),
        'mono8': (np.uint8,   1),
    }
    dtype, channels = dtype_map[desired_encoding]
    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        arr = arr.reshape(msg.height, msg.width)
    else:
        arr = arr.reshape(msg.height, msg.width, channels)
    return arr


# ─────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────
class Go2VlnManager(Node):
    def __init__(self, shared_img, img_meta, img_lock):
        super().__init__('go2_manager')
        self._shared_img = shared_img
        self._img_meta   = img_meta
        self._img_lock   = img_lock

        # video is handled by SDK2 worker via shared memory
        self.odom_sub = self.create_subscription(SportModeState, "/sportmodestate",                self.odom_callback, 10)

        self.rgb_image  = None
        self.homo_goal  = None
        self.homo_odom  = None
        self.vel        = None

        self.request_cnt = 0
        self.odom_cnt    = 0

        self.should_plan    = False
        self.last_plan_time = 0.0

        # poll shared memory for video frames at ~5Hz
        self.create_timer(0.2, self.poll_shared_image)

    def poll_shared_image(self):
        """由 ROS2 timer 定期从共享内存读取 SDK2 子进程写入的图像帧"""
        with self._img_lock:
            if self._img_meta[2] == 0:
                return  # not ready yet
            h, w = int(self._img_meta[0]), int(self._img_meta[1])
            n = h * w * 3
            frame = np.frombuffer(self._shared_img, dtype=np.uint8, count=n).reshape(h, w, 3).copy()
            self._img_meta[2] = 0  # consume
        rgb_rw_lock.acquire_write()
        if self.rgb_image is None:
            print(f"[RGB] First frame from shared mem: {frame.shape}")
        self.rgb_image = frame
        rgb_rw_lock.release_write()

    def odom_callback(self, msg):
        DOWNSAMPLE_RATIO = 5
        self.odom_cnt += 1
        if self.odom_cnt == 1:
            print(f"[Odom] First odom received")
        if self.odom_cnt % DOWNSAMPLE_RATIO != 0:
            return

        odom_rw_lock.acquire_write()
        yaw = msg.imu_state.rpy[2]
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw),  np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3]  = [msg.position[0], msg.position[1]]
        self.vel = [msg.velocity[0], msg.yaw_speed]

        if self.odom_cnt == DOWNSAMPLE_RATIO:
            self.homo_goal = self.homo_odom.copy()

        odom_rw_lock.release_write()

    def trigger_replan(self):
        if not self.should_plan:
            print("[Replan] triggered")
        self.should_plan = True

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_goal

        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rot = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle),  math.cos(angle), 0],
                                [0,                0,               1]])
                homo_goal[:3, :3] = np.dot(rot, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rot = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle),  math.cos(angle), 0],
                                [0,                0,               1]])
                homo_goal[:3, :3] = np.dot(rot, homo_goal[:3, :3])

        self.homo_goal = homo_goal


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    mp.set_start_method('fork')    # inherit conda sys.path

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--iface',      type=str,   default=None,
                        help='network interface for SDK2 DDS (e.g. eth0)')
    parser.add_argument('--domain_id',  type=int,   default=0,
                        help='SDK2 DDS domain id (robot=0, sim=1)')
    args, ros_args = parser.parse_known_args()

    # 0) 共享内存（视频帧）：最大支持 720p = 720x1280x3
    _MAX_PIXELS = 1920 * 1080 * 3  # support up to 1080p
    shared_img = mp.RawArray('B', _MAX_PIXELS)   # fork-safe shared memory
    img_meta   = mp.RawArray('i', 3)            # [h, w, ready_flag]
    img_lock   = mp.Lock()

    # 1) 创建进程间队列
    cmd_queue = mp.Queue(maxsize=5)

    # 2) 启动 SDK2 子进程（纯 DDS，不含 ROS2）
    sdk2_proc = mp.Process(
        target=sdk2_worker,
        args=(cmd_queue, args.iface, args.domain_id, shared_img, img_meta, img_lock),
        daemon=True
    )
    sdk2_proc.start()
    print(f"[Main] SDK2 worker PID={sdk2_proc.pid} started")

    # 3) 启动辅助线程
    control_thread_instance  = threading.Thread(target=control_thread,  daemon=True)
    planning_thread_instance = threading.Thread(target=planning_thread, daemon=True)

    # 4) 初始化 ROS2（主进程，不含 ChannelFactoryInitialize）
    rclpy.init(args=ros_args)

    try:
        manager = Go2VlnManager(shared_img, img_meta, img_lock)
        control_thread_instance.start()
        planning_thread_instance.start()
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        cmd_queue.put(None)        # 通知子进程 StopMove + 退出
        sdk2_proc.join(timeout=3)

        if manager is not None:
            manager.destroy_node()
        rclpy.shutdown()