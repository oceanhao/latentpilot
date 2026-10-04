# streamvln/selfcorrect/deviation_judge.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image


class TrajectoryRecorder:
    """Record each episode as a separate Parquet file with embedded images.

    - Each episode saved as {output_dir}/{episode_num:05d}.parquet
    - RGB/Depth images encoded as base64 strings in Parquet
    - Immediately readable after each episode
    """

    def __init__(self, output_dir: str, save_images: bool = True, image_format: str = "jpg"):
        """
        Args:
            output_dir: Directory to save episode Parquet files
            save_images: Whether to save RGB/Depth images
            image_format: "jpg" (smaller) or "png" (lossless) for RGB
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.image_format = image_format.lower()
        self._episode_counter = 0

    def _encode_image(self, image_data: Any, image_type: str) -> Optional[str]:
        """Encode image to base64 string."""
        if not self.save_images or image_data is None:
            return None

        try:
            # Convert to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                if image_type == "depth":
                    # Depth: save as 16-bit PNG
                    if image_data.dtype != np.uint16:
                        depth_normalized = (image_data * 1000).astype(np.uint16)
                        img = Image.fromarray(depth_normalized, mode='I;16')
                    else:
                        img = Image.fromarray(image_data, mode='I;16')
                else:  # RGB
                    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                        image_data = (image_data * 255).astype(np.uint8)
                    img = Image.fromarray(image_data)
            else:
                img = image_data

            # Encode to base64
            buffer = BytesIO()
            if image_type == "depth":
                img.save(buffer, format="PNG")
            else:
                if self.image_format == "png":
                    img.save(buffer, format="PNG")
                else:
                    img.save(buffer, format="JPEG", quality=90)

            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')

        except Exception as e:
            print(f"Warning: Failed to encode {image_type} image: {e}")
            return None

    def record_episode(
        self,
        scene_id: str,
        episode_id: Any,
        instruction: str,
        trajectory: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        gt_path: Optional[Any] = None,
        episode_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Record episode as a separate Parquet file.

        Each step in trajectory should contain:
            - gps: [x, y]
            - position: [x, y, z]
            - compass: float
            - action: int
            - llm_output: str (optional)
            - rgb: numpy array or PIL Image (optional)
            - depth: numpy array or PIL Image (optional)

        Returns:
            Path to the created Parquet file
        """
        ts = datetime.now().isoformat()
        metrics = metrics or {}
        episode_meta = episode_meta or {}

        gt_list = None if gt_path is None else (
            gt_path.tolist() if hasattr(gt_path, "tolist") else gt_path
        )

        # Extract episode metadata
        start_pos = episode_meta.get("start_position")
        start_rot = episode_meta.get("start_rotation")
        goal_pos = episode_meta.get("goal_position")

        start_pos_list = None if start_pos is None else (
            start_pos.tolist() if hasattr(start_pos, "tolist") else list(start_pos)
        )
        start_rot_list = None if start_rot is None else (
            start_rot.tolist() if hasattr(start_rot, "tolist") else list(start_rot)
        )
        goal_pos_list = None if goal_pos is None else (
            goal_pos.tolist() if hasattr(goal_pos, "tolist") else list(goal_pos)
        )

        # Build records for this episode
        records = []
        for idx, step in enumerate(trajectory):
            gps = step.get("gps")
            pos = step.get("position")

            # Encode images
            rgb_encoded = self._encode_image(step.get("rgb"), "rgb")
            depth_encoded = self._encode_image(step.get("depth"), "depth")

            record = {
                "timestamp": ts,
                "scene_id": str(scene_id),
                "episode_id": str(episode_id),
                "instruction": instruction,
                "step_id": int(step.get("step_id", idx)),
                "gps_x": float(gps[0]) if gps else None,
                "gps_y": float(gps[1]) if gps else None,
                "pos_x": float(pos[0]) if pos else None,
                "pos_y": float(pos[1]) if pos else None,
                "pos_z": float(pos[2]) if pos else None,
                "compass": float(step.get("compass")) if step.get("compass") is not None else None,
                "action": int(step.get("action")) if step.get("action") is not None else None,
                "llm_output": step.get("llm_output"),
                "rgb_base64": rgb_encoded,
                "depth_base64": depth_encoded,
                "success": float(metrics.get("success", 0.0)),
                "spl": float(metrics.get("spl", 0.0)),
                "oracle_success": float(metrics.get("oracle_success", 0.0)),
                "distance_to_goal": float(metrics.get("distance_to_goal", 0.0)),
            }

            # Only include metadata on first step to save space
            if idx == 0:
                record.update({
                    "gt_path": json.dumps(gt_list) if gt_list else None,
                    "start_position": json.dumps(start_pos_list) if start_pos_list else None,
                    "start_rotation": json.dumps(start_rot_list) if start_rot_list else None,
                    "goal_position": json.dumps(goal_pos_list) if goal_pos_list else None,
                    "goal_radius": float(episode_meta.get("goal_radius", 0.0)),
                    "geodesic_distance": float(episode_meta.get("geodesic_distance", 0.0)),
                    "trajectory_id": int(episode_meta.get("trajectory_id", -1)),
                })
            else:
                # Set to None for subsequent steps
                record.update({
                    "gt_path": None,
                    "start_position": None,
                    "start_rotation": None,
                    "goal_position": None,
                    "goal_radius": None,
                    "geodesic_distance": None,
                    "trajectory_id": None,
                })

            records.append(record)

        # Convert to DataFrame and save as Parquet
        df = pd.DataFrame(records)

        # Generate filename: 5-digit sequential number
        parquet_path = self.output_dir / f"{self._episode_counter:05d}.parquet"
        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

        self._episode_counter += 1
        return parquet_path

    def close(self) -> None:
        """Finalize recording - print summary."""
        print(f"Saved {self._episode_counter} episodes to {self.output_dir}")
        print(f"Total Parquet files: {len(list(self.output_dir.glob('*.parquet')))}")


def decode_image(base64_string: str, image_type: str = "rgb") -> Optional[np.ndarray]:
    """Decode base64 string back to numpy array.

    Args:
        base64_string: Base64 encoded image
        image_type: "rgb" or "depth"

    Returns:
        Numpy array of the image
    """
    if not base64_string:
        return None

    try:
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_bytes))

        if image_type == "depth":
            # Convert back to float (meters)
            return np.array(img, dtype=np.float32) / 1000.0
        else:
            return np.array(img)

    except Exception as e:
        print(f"Warning: Failed to decode {image_type} image: {e}")
        return None
