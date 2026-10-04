"""
go2_vln_client_v2.py
====================
Architecture:
  Main process  → ROS2 (rclpy): odom subscription + planning/control threads
  Child process → SDK2 (SportClient + VideoClient): motion control + video capture only

Communication:
  - multiprocessing.Queue      : (vx, vy, vyaw) control commands
  - multiprocessing.RawArray   : shared image buffer
  - multiprocessing.RawArray   : image metadata [h, w, ready_flag]
  - sentinel None              : StopMove + exit

Usage:
  python3 go2_vln_client_v2.py --iface eth0
"""

import multiprocessing as mp
import threading
import queue
import traceback
import PIL.Image as PIL_Image
import io
import json
import requests
import time
import numpy as np
import math
import argparse
from typing import Optional

# user-specific
from pid_controller import *
from utils import ReadWriteLock


# ─────────────────────────────────────────────
# 全局变量（主进程）
# ─────────────────────────────────────────────
policy_init = True
pid = PID_controller(
    Kp_trans=3.0,
    Kd_trans=0.5,
    Kp_yaw=3.0,
    Kd_yaw=0.5,
    max_v=1.0,
    max_w=1.2,
)

manager = None
cmd_queue: Optional[mp.Queue] = None

rgb_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
shutdown_event = threading.Event()


# ─────────────────────────────────────────────
# SDK2 子进程：与 ROS2 完全隔离
# ─────────────────────────────────────────────
def sdk2_worker(cmd_queue, net_if, domain_id, shared_img, img_meta, img_lock):
    """
    SDK2 worker process:
      - SportClient for motion
      - VideoClient.GetImageSample() for video
    No ROS2 here.
    """
    import os as _os
    _os.environ["PYTHONUNBUFFERED"] = "1"

    import cv2 as _cv2
    import numpy as _np
    import queue as _queue
    import threading as _threading
    import ctypes as _ctypes

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.video.video_client import VideoClient

    print(f"[SDK2 worker] starting (iface={net_if}, domain={domain_id})", flush=True)

    try:
        if net_if:
            ChannelFactoryInitialize(domain_id, net_if)
        else:
            ChannelFactoryInitialize(domain_id)
            print("[SDK2 worker] WARNING: --iface not specified, SDK2 will auto-select NIC", flush=True)

        sport = SportClient()
        sport.SetTimeout(10.0)
        sport.Init()
        print(f"[SDK2 worker] SportClient ready (iface={net_if}, domain={domain_id})", flush=True)

        video = VideoClient()
        video.SetTimeout(3.0)
        video.Init()
        print("[SDK2 worker] VideoClient ready", flush=True)

    except Exception as e:
        print(f"[SDK2 worker] init failed: {e}", flush=True)
        traceback.print_exc()
        return

    frame_cnt = 0
    stop_event = _threading.Event()
    shared_capacity = len(shared_img)

    def video_thread():
        nonlocal frame_cnt
        while not stop_event.is_set():
            try:
                code, data = video.GetImageSample()
                if code != 0:
                    stop_event.wait(0.1)
                    continue

                data = bytes(data)
                if not data:
                    stop_event.wait(0.1)
                    continue

                arr = _np.frombuffer(data, dtype=_np.uint8)
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is None:
                    stop_event.wait(0.1)
                    continue

                if frame.ndim != 3 or frame.shape[2] != 3:
                    print(f"[SDK2 video] unexpected frame shape: {frame.shape}", flush=True)
                    stop_event.wait(0.1)
                    continue

                h, w = frame.shape[:2]
                flat = frame.reshape(-1)
                n = int(flat.size)

                if n > shared_capacity:
                    print(
                        f"[SDK2 video] frame too large for shared buffer: "
                        f"{h}x{w}x3={n} > {shared_capacity}",
                        flush=True,
                    )
                    stop_event.wait(0.1)
                    continue

                with img_lock:
                    img_meta[2] = 0
                    _ctypes.memmove(shared_img, flat.ctypes.data, n)
                    img_meta[0] = h
                    img_meta[1] = w
                    img_meta[2] = 1

                frame_cnt += 1
                if frame_cnt == 1:
                    print(f"[SDK2 video] first frame: {h}x{w}, jpeg_bytes={len(data)}", flush=True)

            except Exception as e:
                print(f"[SDK2 video] error: {e}", flush=True)
                traceback.print_exc()

            stop_event.wait(0.1)   # ~10 Hz

    vt = _threading.Thread(target=video_thread, daemon=True)
    vt.start()
    print("[SDK2 worker] video thread started", flush=True)

    while True:
        try:
            cmd = cmd_queue.get(timeout=0.05)
        except _queue.Empty:
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[SDK2 worker] queue error: {e}", flush=True)
            traceback.print_exc()
            continue

        if cmd is None:
            stop_event.set()
            try:
                sport.StopMove()
            except Exception:
                pass
            print("[SDK2 worker] StopMove & exit", flush=True)
            break

        vx, vy, vyaw = cmd
        try:
            sport.Move(float(vx), float(vy), float(vyaw))
        except Exception as e:
            print(f"[SDK2 worker] Move error: {e}", flush=True)
            traceback.print_exc()


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
    elapsed = time.time() - start
    print(f"[VLN] total time(delay + policy): {elapsed:.3f}s")

    response.raise_for_status()
    print(response.text)

    result = json.loads(response.text)
    action = result['action']
    if not isinstance(action, list):
        raise ValueError(f"Invalid action format: {action}")
    return action


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def safe_queue_put_latest(q: mp.Queue, item):
    """
    控制队列满了时，丢掉一个旧指令，尽量保留最新指令。
    """
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    except Exception:
        return

    try:
        _ = q.get_nowait()
    except Exception:
        pass

    try:
        q.put_nowait(item)
    except Exception:
        pass


# ─────────────────────────────────────────────
# 控制线程：PID → 发指令给子进程
# ─────────────────────────────────────────────
def control_thread():
    while not shutdown_event.is_set():
        try:
            if manager is None or cmd_queue is None:
                time.sleep(0.05)
                continue

            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            # 状态没准备好时，不允许触发 replan
            if homo_odom is None or vel is None or homo_goal is None:
                time.sleep(0.05)
                continue

            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            safe_queue_put_latest(cmd_queue, (v, 0.0, w))

            # 只有在状态齐全并且到达当前局部目标后才触发 replan
            if abs(e_p) < 0.1 and abs(e_r) < 0.1:
                manager.trigger_replan()

        except Exception as e:
            print(f"[Control] error: {e}")
            traceback.print_exc()

        time.sleep(0.1)


# ─────────────────────────────────────────────
# 规划线程：获取图像 → 请求 VLN → 更新目标
# ─────────────────────────────────────────────
def planning_thread():
    last_wait_print = 0.0

    while not shutdown_event.is_set():
        try:
            if manager is None:
                time.sleep(0.05)
                continue

            if not manager.should_plan:
                time.sleep(0.01)
                continue

            # 先检查 odom / vel / goal 是否就绪
            odom_rw_lock.acquire_read()
            goal_ready = (
                manager.homo_goal is not None and
                manager.homo_odom is not None and
                manager.vel is not None
            )
            odom_rw_lock.release_read()

            if not goal_ready:
                now = time.time()
                if now - last_wait_print > 1.0:
                    print("[Planning] goal/odom/vel not ready yet, waiting...")
                    last_wait_print = now
                time.sleep(0.1)
                continue

            # 再检查图像
            rgb_rw_lock.acquire_read()
            rgb_image = None if manager.rgb_image is None else manager.rgb_image.copy()
            rgb_rw_lock.release_read()

            if rgb_image is None:
                now = time.time()
                if now - last_wait_print > 1.0:
                    print("[Planning] rgb_image is None, waiting for camera...")
                    last_wait_print = now
                time.sleep(0.1)
                continue

            actions = eval_vln(rgb_image, None, None, None)

            odom_rw_lock.acquire_write()
            try:
                if manager.homo_goal is None:
                    print("[Planning] homo_goal is None after eval_vln, skip this plan")
                else:
                    manager.request_cnt += 1
                    manager.incremental_change_goal(actions)
                    manager.should_plan = False
            finally:
                odom_rw_lock.release_write()

        except requests.RequestException as e:
            print(f"[Planning] HTTP error: {e}")
            traceback.print_exc()
            time.sleep(0.5)
        except Exception as e:
            print(f"[Planning] error: {e}")
            traceback.print_exc()
            time.sleep(0.5)

        time.sleep(0.1)


# ─────────────────────────────────────────────
# ROS2 Node 工厂：避免在 fork/spawn 前导入 rclpy
# ─────────────────────────────────────────────
def build_go2_manager_class():
    from rclpy.node import Node
    from unitree_go.msg import SportModeState

    class Go2VlnManager(Node):
        def __init__(self, shared_img, img_meta, img_lock):
            super().__init__('go2_manager')

            self._shared_img = shared_img
            self._img_meta = img_meta
            self._img_lock = img_lock
            self._shared_capacity = len(shared_img)

            # state
            self.rgb_image = None
            self.homo_goal = None
            self.homo_odom = None
            self.vel = None

            self.request_cnt = 0
            self.odom_cnt = 0

            self.should_plan = False
            self.last_plan_time = 0.0

            self._last_shared_warn = 0.0

            self.odom_sub = self.create_subscription(
                SportModeState,
                "/sportmodestate",
                self.odom_callback,
                10
            )

            # poll shared memory for video frames at ~5Hz
            self.create_timer(0.2, self.poll_shared_image)

        def poll_shared_image(self):
            with self._img_lock:
                if self._img_meta[2] == 0:
                    return

                h = int(self._img_meta[0])
                w = int(self._img_meta[1])

                if h <= 0 or w <= 0:
                    self._img_meta[2] = 0
                    return

                n = h * w * 3
                if n <= 0 or n > self._shared_capacity:
                    now = time.time()
                    if now - self._last_shared_warn > 1.0:
                        print(
                            f"[RGB] invalid shared frame size: h={h}, w={w}, n={n}, cap={self._shared_capacity}"
                        )
                        self._last_shared_warn = now
                    self._img_meta[2] = 0
                    return

                frame = np.frombuffer(
                    self._shared_img,
                    dtype=np.uint8,
                    count=n
                ).reshape(h, w, 3).copy()

                self._img_meta[2] = 0  # consume

            rgb_rw_lock.acquire_write()
            try:
                if self.rgb_image is None:
                    print(f"[RGB] First frame from shared mem: {frame.shape}")
                self.rgb_image = frame
            finally:
                rgb_rw_lock.release_write()

        def odom_callback(self, msg):
            DOWNSAMPLE_RATIO = 5
            self.odom_cnt += 1

            if self.odom_cnt == 1:
                print("[Odom] First odom received")

            if self.odom_cnt % DOWNSAMPLE_RATIO != 0:
                return

            odom_rw_lock.acquire_write()
            try:
                yaw = msg.imu_state.rpy[2]
                R0 = np.array([
                    [np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]
                ])

                self.homo_odom = np.eye(4)
                self.homo_odom[:2, :2] = R0
                self.homo_odom[:2, 3] = [msg.position[0], msg.position[1]]
                self.vel = [msg.velocity[0], msg.yaw_speed]

                # 初始化局部目标
                if self.homo_goal is None:
                    self.homo_goal = self.homo_odom.copy()
                    print("[Odom] homo_goal initialized from first valid odom")

            finally:
                odom_rw_lock.release_write()

        def trigger_replan(self):
            if not self.should_plan:
                print("[Replan] triggered")
            self.should_plan = True

        def incremental_change_goal(self, actions):
            if self.homo_goal is None:
                print("[Planning] homo_goal is None, skip incremental_change_goal")
                return

            homo_goal = self.homo_goal.copy()

            for each_action in actions:
                if each_action == 0:
                    # STOP / no-op
                    pass

                elif each_action == 1:
                    # FORWARD
                    yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                    homo_goal[0, 3] += 0.25 * np.cos(yaw)
                    homo_goal[1, 3] += 0.25 * np.sin(yaw)

                elif each_action == 2:
                    # LEFT
                    angle = math.radians(15.0)
                    rot = np.array([
                        [math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle),  math.cos(angle), 0],
                        [0,                0,               1]
                    ])
                    homo_goal[:3, :3] = np.dot(rot, homo_goal[:3, :3])

                elif each_action == 3:
                    # RIGHT
                    angle = -math.radians(15.0)
                    rot = np.array([
                        [math.cos(angle), -math.sin(angle), 0],
                        [math.sin(angle),  math.cos(angle), 0],
                        [0,                0,               1]
                    ])
                    homo_goal[:3, :3] = np.dot(rot, homo_goal[:3, :3])

                else:
                    print(f"[Planning] unknown action: {each_action}")

            self.homo_goal = homo_goal

    return Go2VlnManager


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # 对 ROS2 / DDS / 原生扩展更稳，避免 fork 带来的奇怪 native 问题
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--iface', type=str, default=None,
                        help='network interface for SDK2 DDS (e.g. eth0)')
    parser.add_argument('--domain_id', type=int, default=0,
                        help='SDK2 DDS domain id (robot=0, sim=1)')
    args, ros_args = parser.parse_known_args()

    if args.iface is None:
        print("[Main] WARNING: --iface is None. If you have multiple NICs, SDK2 may choose the wrong one.")

    # 0) 共享内存（视频帧）：最大支持 1080p = 1920x1080x3
    MAX_BYTES = 1920 * 1080 * 3
    shared_img = mp.RawArray('B', MAX_BYTES)
    img_meta = mp.RawArray('i', 3)     # [h, w, ready_flag]
    img_lock = mp.Lock()

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

    # 3) 现在再导入/初始化 ROS2
    import rclpy
    Go2VlnManager = build_go2_manager_class()

    # 4) 启动辅助线程
    control_thread_instance = threading.Thread(target=control_thread, daemon=True)
    planning_thread_instance = threading.Thread(target=planning_thread, daemon=True)

    rclpy.init(args=ros_args)

    try:
        manager = Go2VlnManager(shared_img, img_meta, img_lock)

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt")
    except Exception as e:
        print(f"[Main] fatal error: {e}")
        traceback.print_exc()
    finally:
        shutdown_event.set()

        try:
            if cmd_queue is not None:
                cmd_queue.put_nowait(None)
        except Exception:
            pass

        try:
            sdk2_proc.join(timeout=3)
        except Exception:
            pass

        try:
            if manager is not None:
                manager.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        print("[Main] shutdown complete")