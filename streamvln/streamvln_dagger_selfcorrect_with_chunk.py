"""
streamvln_dagger_multicard.py

在 streamvln_dagger_selfcorrect.py 的基础上，新增基于文件池（ChunkManager）的
多卡/多机协同数据收集功能。

新增参数：
  --use_chunk_pool        : 开启文件池模式（默认关闭，关闭时完全等价于原始脚本逻辑）
  --chunk_size            : 每个 chunk 包含的 episode 数量（默认 200）
  --chunk_pool_dir        : chunk 状态文件（.lock / .done）存储目录（默认 dagger_output_path/chunk_pool）
  --chunk_lock_stale_sec  : lock 文件过期时间，秒（默认 3600）

使用方式：
  # 原有方式（不变）：
  torchrun --nproc_per_node=8 streamvln/streamvln_dagger_multicard.py ...

  # 多机/多卡文件池模式（不同机器各自独立启动，自动认领 chunk）：
  # 机器A（4卡）：
  torchrun --nproc_per_node=4 streamvln/streamvln_dagger_multicard.py \
      --use_chunk_pool --chunk_size 200 --chunk_pool_dir /shared/pool ...
  # 机器B（4卡）同时启动，自动认领不同 chunk：
  torchrun --nproc_per_node=4 streamvln/streamvln_dagger_multicard.py \
      --use_chunk_pool --chunk_size 200 --chunk_pool_dir /shared/pool ...
"""

import os
import sys
import torch
import json
import argparse
import transformers

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dist import *
import torch.distributed as dist
from streamvln_eval import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM

import os
import random
import numpy as np
import torch
import tqdm
import copy
import json
import random
import habitat
import time
import socket
import uuid

from PIL import Image
from omegaconf import OmegaConf
from typing import List, Dict
from pathlib import Path

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image, append_text_underneath_image
from depth_camera_filtering import filter_depth

from utils.dist import *
from utils.utils import dict_to_cuda
from utils.utils import DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN
from habitat_extensions.maps import image_resize

DEFAULT_EPISODE_LENGTH = 60
MIDGOAL_RADIUS = 2
GOAL_RADIUS = 2
RELATIVE_PATH_LENGTH_THRESHOLD = 0.93
SUCCESS_RELATIVE_PATH_LENGTH_THRESHOLD = 0.9
DEVIATION_THRESHOLD = 2.3


# ============================================================
# ChunkManager：基于文件系统的分布式任务池
# ============================================================
class ChunkManager:
    """
    管理 episode 列表的分片认领与完成状态。
    每个 chunk 对应若干个 episode（由 chunk_size 控制）。
    状态文件存放在 pool_dir 下：
      {cid:06d}.lock  —— 某进程正在处理此 chunk
      {cid:06d}.done  —— 此 chunk 已处理完成
    支持跨机器（共享文件系统）的并发认领，lock 超期自动释放。
    """

    def __init__(self, total_episodes: int, chunk_size: int, pool_dir: str,
                 rank: int, lock_stale_sec: int = 3600):
        self.total_episodes = total_episodes
        self.chunk_size = chunk_size
        self.total_chunks = (total_episodes + chunk_size - 1) // chunk_size
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.lock_stale_sec = lock_stale_sec
        self.node = socket.gethostname()
        self.proc_uuid = str(uuid.uuid4())[:8]

    def _try_acquire_lock(self, lock_path: Path) -> bool:
        """原子创建锁文件，返回是否成功抢到锁"""
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(lock_path), flags, 0o644)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({
                    "node": self.node,
                    "rank": self.rank,
                    "pid": os.getpid(),
                    "uuid": self.proc_uuid,
                    "ts": time.time(),
                }))
            return True
        except FileExistsError:
            return False

    def _is_lock_stale(self, lock_path: Path) -> bool:
        """检查锁文件是否已超期"""
        try:
            mtime = lock_path.stat().st_mtime
            return (time.time() - mtime) >= self.lock_stale_sec
        except FileNotFoundError:
            return False

    def all_done(self) -> bool:
        """检查所有 chunk 是否均已完成"""
        for i in range(1, self.total_chunks + 1):
            if not (self.pool_dir / f"{i:06d}.done").exists():
                return False
        return True

    def claim_next_chunk(self):
        """
        扫描所有 chunk，认领第一个未完成且未被占用的 chunk。
        返回 (chunk_id, start_idx, end_idx, lock_path) 或 None。
        """
        for cid in range(1, self.total_chunks + 1):
            lock_path = self.pool_dir / f"{cid:06d}.lock"
            done_file = self.pool_dir / f"{cid:06d}.done"

            if done_file.exists():
                continue  # 已完成，跳过

            if lock_path.exists():
                if not self._is_lock_stale(lock_path):
                    continue  # 被其他进程持有且未超期，跳过
                # 超期锁，尝试删除后重新抢
                try:
                    lock_path.unlink()
                except Exception:
                    continue

            if not self._try_acquire_lock(lock_path):
                continue  # 抢锁失败（其他进程刚好抢到），跳过

            start_idx = (cid - 1) * self.chunk_size
            end_idx = min(cid * self.chunk_size, self.total_episodes)
            return cid, start_idx, end_idx, lock_path

        return None

    def mark_done(self, chunk_id: int, lock_path: Path):
        """标记 chunk 完成，释放锁"""
        done_file = self.pool_dir / f"{chunk_id:06d}.done"
        done_file.touch()
        try:
            lock_path.unlink()
        except Exception:
            pass


# ============================================================
# StreamVLNDAggerCollector（完全保留原始逻辑，新增 chunk pool 支持）
# ============================================================
class StreamVLNDAggerCollector:
    def __init__(self, args, rank, world_size):
        self.device = torch.device("cuda")
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.dataset = self.args.dagger_dataset.lower()
        self.output_path = self.args.dagger_output_path
        self.data_path = self.args.dagger_data_path
        self.config = get_habitat_config(args.habitat_config_path)
        print(OmegaConf.to_yaml(self.config))

        with open(self.args.dagger_gt_annotations_path, "r") as f:
            self.gt_annotations = json.load(f)

        with read_write(self.config):
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.dagger_config = OmegaConf.create({
            "p": self.args.dagger_p,
            "update_size": self.args.dagger_update_size,
            "commit_freq": self.args.dagger_commit_freq,
        })
        print(self.dagger_config)

        sim_sensors_cfg = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self._camera_height = sim_sensors_cfg.rgb_sensor.position[1]
        self._min_depth = sim_sensors_cfg.depth_sensor.min_depth
        self._max_depth = sim_sensors_cfg.depth_sensor.max_depth
        camera_fov_rad = np.deg2rad(sim_sensors_cfg.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = sim_sensors_cfg.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))

    def config_env(self, scene=None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.data_path = self.data_path
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def get_intrinsic_matrix(self, sensor_cfg):
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [0.0,  fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return intrinsic_matrix

    def get_axis_align_matrix(self):
        ma = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw),  np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
        return transformation_matrix

    def compute_distance_to_path(self, agent_position, gt_path):
        if len(gt_path) < 2:
            return np.linalg.norm(agent_position - np.array(gt_path[0])), 0
        min_distance = float('inf')
        closest_segment_idx = -1
        for i in range(len(gt_path) - 1):
            p1 = np.array(gt_path[i])
            p2 = np.array(gt_path[i + 1])
            segment_vec = p2 - p1
            to_agent = agent_position - p1
            segment_length_sq = np.dot(segment_vec, segment_vec)
            if segment_length_sq < 1e-8:
                distance = np.linalg.norm(to_agent)
            else:
                t = np.clip(np.dot(to_agent, segment_vec) / segment_length_sq, 0.0, 1.0)
                closest_point = p1 + t * segment_vec
                distance = np.linalg.norm(agent_position - closest_point)
            if distance < min_distance:
                min_distance = distance
                closest_segment_idx = i
        return min_distance, closest_segment_idx

    def generate(self, env: habitat.Env, evaluator=None,
                 save_video: bool = True, force_expert: bool = False) -> Dict:

        beta = 0 if self.dagger_config.p == 0 else self.dagger_config.p ** self.args.dagger_data_it

        os.makedirs(os.path.join(self.output_path), exist_ok=True)

        episode = env.current_episode
        agent = ShortestPathFollower(sim=env.sim, goal_radius=1.8, return_one_hot=False)
        scene_id = episode.scene_id.split('/')[-2]
        episode_id = int(episode.episode_id)
        trajectory_id = episode.trajectory_id
        instructions = episode.instruction.instruction_text

        observation = env.reset()

        start_position = env.sim.get_agent_state().position
        ref_path = [start_position] + episode.reference_path
        if len(episode.goals) > 0:
            goal_position = episode.goals[0].position
            if len(ref_path) == 0 or np.linalg.norm(np.array(ref_path[-1]) - np.array(goal_position)) > 0.1:
                ref_path.append(goal_position)

        print(f"GT path length: {len(ref_path)} points (including start and goal)")
        annotation = []
        rgb_data_list = []
        depth_data_list = []
        step_id = 0
        actions = [-1]
        next_waypoint_id = 1

        if save_video:
            os.makedirs(os.path.join(self.output_path, 'videos'), exist_ok=True)

        initial_height = env.sim.get_agent_state().position[1]
        intrinsic_matrix = self.get_intrinsic_matrix(
            self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        intrinsic_matrix = np.around(intrinsic_matrix, decimals=4)

        mem_ids = []
        vis_frames = []
        left_expert_actions_num = 0
        from_expert = True if force_expert else False
        force_episode_end = False
        model_success = True
        action_seq, action_mask = [], []
        rgb_list, depth_list, pose_list, intrinsic_list, time_ids = [], [], [], [], []
        past_key_values, output_ids = None, None
        metrics = None
        accumulated_error = 0

        has_deviation = False
        deviation_step = -1
        distances_to_path = []
        in_correction_mode = False
        correction_start_step = -1
        correction_end_step = -1

        ref_actions_len = next(
            (len(annot["actions"]) for annot in self.gt_annotations if int(episode_id) == annot["id"]),
            DEFAULT_EPISODE_LENGTH
        )
        print(f"ref_actions_len: {ref_actions_len}")

        if evaluator is not None:
            evaluator.model.eval()

        while not env.episode_over:
            time_ids.append(step_id)
            rgb = observation["rgb"]
            depth = observation["depth"]
            x, y = observation["gps"]
            camera_yaw = observation["compass"][0]
            depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
            filled_depth = depth * (self._max_depth - self._min_depth) + self._min_depth
            depth = filled_depth * 1000

            height = env.sim.get_agent_state().position[1] - initial_height
            camera_position = np.array([x, -y, self._camera_height + height])
            tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
            tf_camera_to_episodic = tf_camera_to_episodic @ self.get_axis_align_matrix()
            tf_camera_to_episodic = np.around(tf_camera_to_episodic, decimals=4)

            rgb_data_list.append(rgb)
            depth_data_list.append(depth)

            agent_position = env.sim.get_agent_state().position
            distance_to_path, _ = self.compute_distance_to_path(agent_position, ref_path)
            distances_to_path.append(distance_to_path)

            print("###########distances_to_path#############", distance_to_path)

            if not has_deviation and distance_to_path > DEVIATION_THRESHOLD:
                has_deviation = True
                deviation_step = step_id
                in_correction_mode = True
                correction_start_step = step_id
                print(f"[Deviation Detected] Step {step_id}: distance = {distance_to_path:.2f}m, GT taking over...")

            if evaluator is not None:
                if in_correction_mode and distance_to_path < DEVIATION_THRESHOLD * 0.5:
                    in_correction_mode = False
                    correction_end_step = step_id
                    print(f"[Correction Complete] Step {step_id}: distance = {distance_to_path:.2f}m, agent back on track")

                image = Image.fromarray(rgb).convert('RGB')
                image_size = image.size
                image = evaluator.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                depth_image, resize_shape = evaluator.preprocess_depth_image(
                    Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)

                intrinsic = evaluator.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                intrinsic = torch.from_numpy(intrinsic).float()

                rgb_list.append(image)
                depth_list.append(torch.from_numpy(depth_image).float())
                pose_list.append(torch.from_numpy(tf_camera_to_episodic))
                intrinsic_list.append(intrinsic)

                if in_correction_mode:
                    from_expert = True
                elif len(action_seq) == 0 and left_expert_actions_num == 0:
                    from_expert = True if force_expert else random.random() < beta

                if len(action_seq) == 0:
                    if left_expert_actions_num > 0:
                        action = agent.get_next_action(ref_path[next_waypoint_id])
                        action_seq = [action]
                        left_expert_actions_num -= 1
                    else:
                        if from_expert:
                            action = agent.get_next_action(ref_path[next_waypoint_id])
                            action_seq = [action]
                            left_expert_actions_num = self.args.num_future_steps - 1
                        else:
                            if output_ids is None:
                                sources = copy.deepcopy(evaluator.conversation)
                                sources[0]["value"] = sources[0]["value"].replace(
                                    ' Where should you go next to stay on track?',
                                    ' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                                if step_id != 0:
                                    sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
                                # # 在这之后加：
                                # if args.add_step_info:
                                #     step_text = f' You are currently at step {step_id}.'
                                #     step_text += (
                                #         ' Note: navigation is usually completed within 100 steps.'
                                #         ' Since you have exceeded this limit, you may have taken a wrong path.'
                                #         ' Please re-orient yourself and find the correct route to complete the navigation.'
                                #     )
                                #     sources[0]["value"] += step_text

                                sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + '\n', '')
                                sources[0]["value"] = sources[0]["value"].replace(
                                    '<instruction>.',
                                    episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                                add_system = True
                            else:
                                sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                                add_system = False

                            input_ids, conversations = evaluator.preprocess_qwen(
                                [sources], evaluator.tokenizer, True, add_system=add_system)
                            if output_ids is not None:
                                input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

                            images = rgb_list[-1:]
                            depths = depth_list[-1:]
                            poses = pose_list[-1:]
                            intrinsics = intrinsic_list[-1:]

                            add_mem_or_not = False
                            mem_ids.append(step_id)
                            if len(mem_ids) > 1:
                                add_mem_or_not = ((mem_ids[-1] // evaluator.num_frames) - (mem_ids[-2] // evaluator.num_frames) >= 1)
                            if step_id != 0 and (step_id % evaluator.num_frames == 0 or add_mem_or_not):
                                if evaluator.num_history is None:
                                    history_ids = slice(0, time_ids[0], evaluator.num_future_steps)
                                else:
                                    history_ids = slice(0, time_ids[0], ((time_ids[0]) // evaluator.num_history))
                                images = rgb_list[history_ids] + images
                                depths = depth_list[history_ids] + depths
                                poses = pose_list[history_ids] + poses
                                intrinsics = intrinsic_list[history_ids] + intrinsics

                            input_dict = {
                                'images': torch.stack(images).unsqueeze(0),
                                'depths': torch.stack(depths).unsqueeze(0),
                                'poses': torch.stack(poses).unsqueeze(0),
                                'intrinsics': torch.stack(intrinsics).unsqueeze(0),
                                'inputs': input_ids,
                                'env_id': self.rank,
                                'time_ids': [time_ids],
                                'task_type': [0]
                            }
                            input_dict = dict_to_cuda(input_dict, self.device)
                            for key in ['images', 'depths', 'poses', 'intrinsics']:
                                if key in input_dict:
                                    input_dict[key] = input_dict[key].to(torch.bfloat16)

                            outputs = evaluator.model.generate(
                                **input_dict, do_sample=False, num_beams=1,
                                max_new_tokens=10000, use_cache=True,
                                return_dict_in_generate=True,
                                past_key_values=past_key_values)
                            output_ids = outputs.sequences
                            past_key_values = outputs.past_key_values
                            llm_outputs = evaluator.tokenizer.batch_decode(
                                output_ids, skip_special_tokens=True)[0].strip()
                            action_seq = evaluator.parse_actions(llm_outputs)
            else:
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_seq = [action]

            action_source = "expert" if from_expert else "model"

            if len(action_seq) == 0:
                action_seq = [0]

            action = action_seq.pop(0)
            if action != agent.get_next_action(ref_path[next_waypoint_id]):
                accumulated_error += 1

            while agent.get_next_action(ref_path[next_waypoint_id]) == 0:
                next_waypoint_id += 1
                force_expert = False
                left_expert_actions_num = 0
                if next_waypoint_id == len(ref_path) - 1:
                    agent = ShortestPathFollower(sim=env.sim, goal_radius=GOAL_RADIUS, return_one_hot=False)
                if next_waypoint_id >= len(ref_path):
                    force_episode_end = True
                    action = 0
                    action_source = "expert"
                    break

            metrics = env.get_metrics()
            wp_id_available = next_waypoint_id < len(ref_path)
            error_not_toleranted = (
                (from_expert == False and action == 0 and metrics["distance_to_goal"] >= 3.0) or
                (accumulated_error / max(1, int(ref_actions_len / (len(ref_path) - 1))) > 0.8) or
                accumulated_error > 12
            )
            if wp_id_available and error_not_toleranted:
                model_success = False
                force_expert = True
                accumulated_error = 0
                action = agent.get_next_action(ref_path[next_waypoint_id])
                action_source = "expert"
                action_seq = []

            if action == 0 and not force_episode_end:
                action = agent.get_next_action(ref_path[next_waypoint_id])

            observation = env.step(action)
            metrics = env.get_metrics()

            if save_video:
                metrics = env.get_metrics()
                if metrics['top_down_map'] is not None:
                    resized_rgb = np.array(image_resize(
                        img=observation['rgb'],
                        size=(int(observation['rgb'].shape[0] * 1.6), int(observation['rgb'].shape[1] * 1.6)),
                        channels_last=True))
                    frame = observations_to_image({'rgb': resized_rgb}, metrics)
                    frame = append_text_underneath_image(
                        frame,
                        episode.instruction.instruction_text if isinstance(episode.instruction.instruction_text, str) else episode.instruction.instruction_text[0])
                    frame = append_text_underneath_image(frame, action_source)
                    frame = append_text_underneath_image(frame, f"force_expert is {force_expert}")
                    frame = append_text_underneath_image(frame, f"step: {step_id}")
                    frame = append_text_underneath_image(frame, f"next wp id: {next_waypoint_id} / {len(ref_path) - 1}")
                    vis_frames.append(frame)

            if env.episode_over or force_episode_end:
                break
            actions.append(action)
            step_id += 1
            if step_id % evaluator.num_frames == 0:
                evaluator.model.reset_for_env(self.rank)
                output_ids = None
                past_key_values = None
                time_ids = []

        assert len(rgb_data_list) == len(actions), \
            f"Length of rgbs and actions mismatch, rgb_data_list: {len(rgb_data_list)}, actions: {(actions)}"

        annotation.append({
            "id": episode_id,
            "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
            "instructions": instructions if isinstance(instructions, list) else [instructions],
            "actions": actions,
            "has_deviation": has_deviation,
            "deviation_step": deviation_step,
            "correction_range": (correction_start_step, correction_end_step) if has_deviation else None,
            "distances_to_path": [float(d) for d in distances_to_path],
        })
        print("====distance_to_goal====", metrics["distance_to_goal"])
        print("====spl====", metrics["spl"])

        if has_deviation:
            episode_save = True
        else:
            success_condition = (metrics["distance_to_goal"] < MIDGOAL_RADIUS) and (metrics["spl"] > 0.75)
            episode_save = success_condition and (random.random() < self.args.dagger_no_deviation_save_rate)

        if episode_save:
            os.makedirs(os.path.join(self.output_path, "images",
                                     f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb"), exist_ok=True)
            video_path = os.path.join(self.output_path, "images",
                                      f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb", "frames.mp4")
            import cv2
            if len(rgb_data_list) > 0:
                h, w = rgb_data_list[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 6, (w, h))
                for rgb_frame in rgb_data_list:
                    out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                out.release()
                file_size_mb = os.path.getsize(video_path) / 1024 / 1024
                print(f"Saved {len(rgb_data_list)} frames to MP4: {video_path}, file size: {file_size_mb:.2f} MB")
            else:
                print(f"Warning: No RGB frames to save for episode {episode_id}")

        if save_video:
            prefix = 'save' if episode_save else 'notsave'
            images_to_video(vis_frames, os.path.join(self.output_path, 'videos'),
                            f'{prefix}_{scene_id}_{self.dataset}_{episode_id:06d}', fps=6, quality=10)
            vis_frames.clear()

        metrics.update({
            "step_id": step_id,
            "ref_actions_len": ref_actions_len,
            "accumulated_error": accumulated_error,
            "save": int(episode_save),
            "model_success": model_success,
            "force_episode_end": force_episode_end,
        })

        return dict(anno=annotation, metrics=metrics)

    # ----------------------------------------------------------
    # 内部辅助：将收集到的 annotations 写入磁盘（去重）
    # ----------------------------------------------------------
    def _commit_annotations(self, annotations: list, anno_path: str):
        if os.path.exists(anno_path):
            try:
                with open(anno_path, "r") as f:
                    content = f.read().strip()
                merged = json.loads(content) if content else []
            except json.JSONDecodeError:
                print(f"Warning: skip corrupted {anno_path}, re-init empty list")
                merged = []
        else:
            merged = []

        merged.extend(annotations)
        # 去重（保留每个 video 的最后一条）
        seen = set()
        deduped = []
        for item in reversed(merged):
            if item["video"] not in seen:
                seen.add(item["video"])
                deduped.append(item)
        deduped.reverse()

        with open(anno_path, "w") as f:
            json.dump(deduped, f, indent=4)

    # ----------------------------------------------------------
    # 原始模式：与原脚本完全等价
    # ----------------------------------------------------------
    def update_dataset(self, evaluator, dataset=None):
        """原始多卡收集逻辑（与 streamvln_dagger_selfcorrect.py 完全等价）"""
        seed = self.rank
        random.seed(seed)
        np.random.seed(seed)

        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        env = self.config_env()
        scene_episode_dict = {}
        episode_uuids = []
        start = time.time()
        for episode in env.episodes:
            episode_uuid = (episode.scene_id, episode.episode_id, episode.trajectory_id)
            episode_uuids.append(episode_uuid)
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        sampled_episodes_uuids = episode_uuids
        sampled_episodes_by_scene = {}
        for scene_id in sorted(scene_episode_dict.keys()):
            sampled_episodes_traj_ids = [
                (ep_uuid[1], ep_uuid[2])
                for ep_uuid in sampled_episodes_uuids if ep_uuid[0] == scene_id
            ]
            sampled_episodes_by_scene[scene_id] = [
                ep for ep in scene_episode_dict[scene_id]
                if (ep.episode_id, ep.trajectory_id) in sampled_episodes_traj_ids
            ]

        num_collect_episodes = 0
        annotations = []
        with tqdm.tqdm(
            total=min(self.dagger_config.update_size, len(sampled_episodes_uuids)) // self.world_size,
            dynamic_ncols=True
        ) as pbar, torch.no_grad():
            for scene_id in sorted(scene_episode_dict.keys()):
                episodes = sampled_episodes_by_scene[scene_id]
                if len(episodes) == 0:
                    continue
                print(f"scene_id: {scene_id}, len of episodes: {len(episodes)}")
                for episode in episodes[self.rank::self.world_size]:
                    if random.random() > self.args.dagger_sample_rate:
                        pbar.update()
                        continue
                    assert scene_id == episode.scene_id
                    scan = episode.scene_id.split('/')[-2]
                    env.current_episode = episode
                    env.current_episode.goals[0].radius = MIDGOAL_RADIUS
                    if evaluator is not None:
                        evaluator.model.reset_for_env(self.rank)
                    episode_dagger = self.generate(
                        env=env, evaluator=evaluator,
                        save_video=self.args.dagger_save_video,
                        force_expert=self.args.force_expert)

                    with open(os.path.join(self.output_path, "result.json"), "a") as f:
                        result = {
                            "scene": scan,
                            "episode_id": episode.episode_id,
                            "trajectory_id": episode.trajectory_id,
                            "save": episode_dagger["metrics"]["save"],
                            "model_success": episode_dagger["metrics"]["model_success"],
                            "success": episode_dagger["metrics"]["success"],
                            "relative_pl": episode_dagger["metrics"]["pl"],
                            "step_id": episode_dagger["metrics"]["step_id"],
                            "ref_actions": episode_dagger["metrics"]["ref_actions_len"],
                            "accumulated_error": episode_dagger["metrics"]["accumulated_error"],
                            "force_episode_end": episode_dagger["metrics"]["force_episode_end"],
                        }
                        f.write(json.dumps(result) + "\n")

                    if not episode_dagger["metrics"]["save"]:
                        pbar.update()
                        continue

                    annotations.extend(episode_dagger['anno'])
                    pbar.update()
                    num_collect_episodes += 1

                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
                        self._commit_annotations(annotations, tgt_anno_path)
                        annotations = []

                    if num_collect_episodes >= self.dagger_config.update_size:
                        break
                if num_collect_episodes >= self.dagger_config.update_size:
                    break

            # 最终写入
            tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
            self._commit_annotations(annotations, tgt_anno_path)
            print(f"save scene_id {scene_id} with total episodes {num_collect_episodes} time cost {time.time() - start}")

        dist.barrier()
        if get_rank() == 0:
            self._merge_all_annotations()

    # ----------------------------------------------------------
    # Chunk Pool 模式：多机/多卡共享文件池，自动认领任务
    # ----------------------------------------------------------
    def update_dataset_chunk_pool(self, evaluator, dataset=None):
        """
        文件池模式：将全部 episode 按 chunk_size 分片，
        各进程（跨机器/跨卡）通过文件锁原子认领 chunk，互不干扰。

        优点：
          - 不同服务器、不同显卡数量均可独立启动，自动分摊任务
          - 某进程挂掉后，lock 超期自动释放，其他进程可重新认领
          - 与原始模式逻辑完全一致，只是任务分配方式不同
        """
        if evaluator is None:
            self.args.force_expert = True

        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        # 构建全局 episode 列表（所有进程相同，顺序一致）
        env = self.config_env()
        all_episodes = []
        for episode in env.episodes:
            all_episodes.append(episode)
        total_episodes = len(all_episodes)

        pool_dir = getattr(self.args, 'chunk_pool_dir', None) or \
                   os.path.join(self.output_path, "chunk_pool")
        chunk_size = getattr(self.args, 'chunk_size', 200)
        lock_stale_sec = getattr(self.args, 'chunk_lock_stale_sec', 3600)

        # 每个进程（rank）独立创建自己的 ChunkManager
        # 注意：使用全局唯一 rank（含机器编号）作为 ChunkManager 的 rank 标识
        chunk_mgr = ChunkManager(
            total_episodes=total_episodes,
            chunk_size=chunk_size,
            pool_dir=pool_dir,
            rank=self.rank,
            lock_stale_sec=lock_stale_sec,
        )

        print(f"[Rank {self.rank}] Chunk Pool Mode: {total_episodes} episodes, "
              f"chunk_size={chunk_size}, total_chunks={chunk_mgr.total_chunks}, "
              f"pool_dir={pool_dir}")

        poll_seconds = 10
        num_collect_episodes = 0
        annotations = []
        start = time.time()

        with torch.no_grad():
            while True:
                claim = chunk_mgr.claim_next_chunk()

                if claim is None:
                    if chunk_mgr.all_done():
                        print(f"[Rank {self.rank}] All chunks done.")
                        break
                    print(f"[Rank {self.rank}] No available chunk, waiting {poll_seconds}s...")
                    time.sleep(poll_seconds)
                    continue

                chunk_id, start_idx, end_idx, lock_path = claim
                chunk_episodes = all_episodes[start_idx:end_idx]
                print(f"[Rank {self.rank}] Claimed chunk {chunk_id:06d} "
                      f"[{start_idx}:{end_idx}] ({len(chunk_episodes)} episodes)")

                # 处理当前 chunk 内的所有 episode
                for episode in chunk_episodes:
                    # 按采样率随机跳过
                    if random.random() > self.args.dagger_sample_rate:
                        continue

                    scan = episode.scene_id.split('/')[-2]
                    env.current_episode = episode
                    env.current_episode.goals[0].radius = MIDGOAL_RADIUS
                    if evaluator is not None:
                        evaluator.model.reset_for_env(self.rank)

                    episode_dagger = self.generate(
                        env=env, evaluator=evaluator,
                        save_video=self.args.dagger_save_video,
                        force_expert=self.args.force_expert)

                    with open(os.path.join(self.output_path, "result.json"), "a") as f:
                        result = {
                            "scene": scan,
                            "episode_id": episode.episode_id,
                            "trajectory_id": episode.trajectory_id,
                            "save": episode_dagger["metrics"]["save"],
                            "model_success": episode_dagger["metrics"]["model_success"],
                            "success": episode_dagger["metrics"]["success"],
                            "relative_pl": episode_dagger["metrics"]["pl"],
                            "step_id": episode_dagger["metrics"]["step_id"],
                            "ref_actions": episode_dagger["metrics"]["ref_actions_len"],
                            "accumulated_error": episode_dagger["metrics"]["accumulated_error"],
                            "force_episode_end": episode_dagger["metrics"]["force_episode_end"],
                        }
                        f.write(json.dumps(result) + "\n")

                    if not episode_dagger["metrics"]["save"]:
                        continue

                    annotations.extend(episode_dagger['anno'])
                    num_collect_episodes += 1

                    # 按 commit_freq 周期性写入（以 rank 区分文件）
                    if num_collect_episodes % self.dagger_config.commit_freq == 0:
                        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
                        self._commit_annotations(annotations, tgt_anno_path)
                        annotations = []

                    if num_collect_episodes >= self.dagger_config.update_size:
                        chunk_mgr.mark_done(chunk_id, lock_path)
                        print(f"[Rank {self.rank}] Reached update_size limit, stopping.")
                        # 最终写入
                        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
                        self._commit_annotations(annotations, tgt_anno_path)
                        print(f"[Rank {self.rank}] Total collected: {num_collect_episodes}, "
                              f"time cost: {time.time() - start:.1f}s")
                        return

                # chunk 处理完毕，标记 done
                chunk_mgr.mark_done(chunk_id, lock_path)
                print(f"[Rank {self.rank}] Chunk {chunk_id:06d} done "
                      f"({num_collect_episodes} saved so far)")

        # 最终写入剩余 annotations
        tgt_anno_path = os.path.join(self.output_path, f"annotations_{self.rank}.json")
        self._commit_annotations(annotations, tgt_anno_path)
        print(f"[Rank {self.rank}] Total collected: {num_collect_episodes}, "
              f"time cost: {time.time() - start:.1f}s")

    # ----------------------------------------------------------
    # 合并所有 rank 的 annotations_x.json -> annotations.json
    # ----------------------------------------------------------
    def _merge_all_annotations(self):
        tgt_anno_path = os.path.join(self.output_path, "annotations.json")
        merged_anno = []
        sub_list = [
            os.path.join(self.output_path, f)
            for f in os.listdir(self.output_path)
            if f.startswith('annotations_') and f.endswith('.json')
        ]
        for sub_path in sub_list:
            if os.path.exists(sub_path):
                try:
                    with open(sub_path, "r") as f:
                        content = f.read().strip()
                    if content:
                        merged_anno.extend(json.loads(content))
                except json.JSONDecodeError:
                    print(f"Warning: skip corrupted {sub_path}")

        merged_anno = sorted(merged_anno, key=lambda x: x['id'])
        seen = set()
        deduped = []
        for item in reversed(merged_anno):
            if item["video"] not in seen:
                seen.add(item["video"])
                deduped.append(item)
        deduped.reverse()

        with open(tgt_anno_path, "w") as f:
            json.dump(deduped, f, indent=4)
        print(f"[Merge] annotations.json written: {len(deduped)} episodes")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":

    global local_rank

    parser = argparse.ArgumentParser()
    # ===== 原有参数（完全不变）=====
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_dagger.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--dagger_p", type=float, default=0.9)
    parser.add_argument("--dagger_update_size", type=int, default=1)
    parser.add_argument("--dagger_commit_freq", type=int, default=1)
    parser.add_argument("--dagger_dataset", type=str, default="DATASET")
    parser.add_argument("--force_expert", action="store_true", default=False)
    parser.add_argument("--dagger_data_it", type=int, default=0)
    parser.add_argument("--dagger_output_path", type=str, default="data/dagger")
    parser.add_argument("--dagger_data_path", type=str, default="data/datasets/vln_datasets/{split}.json.gz")
    parser.add_argument("--dagger_gt_annotations_path", type=str, default="data/datasets/vln_datasets/annotations.json")
    parser.add_argument("--dagger_save_video", action="store_true", default=False)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--dagger_sample_rate", type=float, default=1.0)
    parser.add_argument("--dagger_no_deviation_save_rate", type=float, default=1.0)
    parser.add_argument("--no_step_info", action="store_true", default=False,
                        help="Disable adding current step info to prompt.")
    # ===== 新增：文件池模式参数（默认关闭，不影响原有用法）=====
    parser.add_argument(
        "--use_chunk_pool", action="store_true", default=False,
        help="开启文件池模式：将全部 episode 分片，多机/多卡自动认领，互不干扰。"
             "关闭时完全等价于原始 streamvln_dagger_selfcorrect.py 的逻辑。")
    parser.add_argument(
        "--chunk_size", type=int, default=200,
        help="文件池模式：每个 chunk 包含的 episode 数量（默认 200）")
    parser.add_argument(
        "--chunk_pool_dir", type=str, default=None,
        help="文件池模式：chunk 状态文件存储目录（默认 dagger_output_path/chunk_pool）。"
             "多机共享时需指定为共享文件系统路径。")
    parser.add_argument(
        "--chunk_lock_stale_sec", type=int, default=3600,
        help="文件池模式：lock 文件过期时间（秒），超期后其他进程可重新认领（默认 3600）")

    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.model_max_length, padding_side="right")

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    # 修复 vision tower 未加载问题
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)
    model.eval()

    rank = get_rank()
    world_size = get_world_size()
    model.reset(world_size)

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args,
    )

    collector = StreamVLNDAggerCollector(args=args, rank=rank, world_size=world_size)

    if args.use_chunk_pool:
        # ===== 文件池模式：多机/多卡各自独立启动，自动认领 chunk =====
        collector.update_dataset_chunk_pool(evaluator=evaluator)
        # chunk pool 模式下，每台机器 rank0 各自合并本机收集的数据
        # 全局合并需在所有机器都完成后手动触发（或由调度系统触发）
        if rank == 0:
            collector._merge_all_annotations()
    else:
        # ===== 原始模式：与 streamvln_dagger_selfcorrect.py 完全等价 =====
        collector.update_dataset(evaluator=evaluator)