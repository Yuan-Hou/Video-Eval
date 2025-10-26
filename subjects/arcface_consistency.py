# arcface_wrapper.py
import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Union, Optional
from insightface.app import FaceAnalysis
from utils import load_video, load_image_cv2

from tqdm import tqdm

from video import VideoData

# 全局对象
_arcface_app: Optional[FaceAnalysis] = None
_arcface_device: str = "cpu"

def get_arcface_model(name: str = "buffalo_l",
                      det_size: Tuple[int, int] = (640, 640),
                      device: str = "cuda", **kwargs) -> None:
    """
    加载 ArcFace 模型（带人脸检测与对齐）
    - name: InsightFace 组合配置名，"buffalo_l" 通用且精度好
    - det_size: 检测输入尺寸
    - device: "cuda" / "cpu"；GPU 时会自动选 0 号
    """
    global _arcface_app, _arcface_device
    _arcface_device = device

    # ctx_id: -1=CPU, >=0=GPU index
    if device.startswith("cuda") and torch.cuda.is_available():
        ctx_id = 0
    else:
        ctx_id = -1

    app = FaceAnalysis(name=name, **kwargs)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    _arcface_app = app


def _to_numpy_hw3(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    将输入统一成 numpy(H, W, 3)，且为 BGR（InsightFace 内部使用 BGR）。
    允许输入：
      - np.ndarray: (H,W,3) BGR
      - torch.Tensor: (3,H,W) 或 (B,3,H,W)，此函数只处理单张，批量在上层拆分
    """
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("np.ndarray 输入需为 (H, W, 3)")
        return arr.astype(np.uint8)
            

    if isinstance(img, torch.Tensor):
        if img.ndim == 3 and img.shape[0] == 3:
            # (3,H,W) RGB -> HWC BGR
            arr = img.detach().cpu().numpy()
            arr = np.transpose(arr, (1, 2, 0))  # HWC, RGB
            arr = (arr * 255.0) if arr.max() <= 1.0 else arr
            return arr[:, :, ::-1].astype(np.uint8).copy()
        else:
            raise ValueError("torch.Tensor 输入需为 (3,H,W)。批量请在上层拆分后逐张传入。")
    print(img)
    raise TypeError(f"不支持的输入类型：{type(img)}")


def extract_arcface_features(
    image: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
) -> List[torch.Tensor]:
    """
    提取 ArcFace 特征，输出与 DINOv3 extract_features 的“批处理风格”保持一致：
    - 支持传入单张或 List[图]。
    - 每张图可能有多张脸 -> 返回 list[Tensor]，每个 Tensor 形状为 (num_faces, 512)。
    - 所有向量均为 L2 归一化（InsightFace 提供的 normed_embedding）。

    返回：
    - 默认：List[Tensor]，长度=输入图数量；每个元素是 (num_faces, 512)
    """
    global _arcface_app
    if _arcface_app is None:
        raise ValueError("ArcFace 模型尚未加载。请先调用 get_arcface_model。")

    # 统一成列表处理
    img_list = image if isinstance(image, list) else [image]

    all_feats: List[torch.Tensor] = []

    for img in img_list:
        np_img = _to_numpy_hw3(img)  # HWC, BGR, uint8
        faces = _arcface_app.get(np_img)  # 检测+对齐+特征
        # faces[i].normed_embedding: (512,), 已 L2 归一化
        if len(faces) == 0:
            all_feats.append(torch.empty((0, 512), dtype=torch.float32))
            continue

        embs = []
        for f in faces:
            if getattr(f, "normed_embedding", None) is None:
                # 某些检测结果可能没有成功提特征
                continue
            embs.append(torch.from_numpy(f.normed_embedding.astype(np.float32)))
        if len(embs) == 0:
            all_feats.append(torch.empty((0, 512), dtype=torch.float32))
        else:
            all_feats.append(torch.stack(embs, dim=0))  # (num_faces, 512)
    return all_feats


def pairwise_cosine_sim(feats_a: torch.Tensor, feats_b: torch.Tensor) -> torch.Tensor:
    """
    计算两个特征集合的两两余弦相似度。
    要求 feats_* 为 (N, 512) 且 **已 L2 归一化**（ArcFace 默认即是）。
    返回 (N, M)。
    """
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise ValueError("feats_a/feats_b 应为二维张量 (N, D) / (M, D)")
    # 由于已归一化，直接点积即可；为稳妥再做一次轻微归一化
    feats_a = torch.nn.functional.normalize(feats_a, dim=1)
    feats_b = torch.nn.functional.normalize(feats_b, dim=1)
    # 如果任意一方为空，返回0
    if feats_a.shape[0] == 0 or feats_b.shape[0] == 0:
        return 0.0

    mat = feats_a @ feats_b.t()
    
    # 取所有相似度中的最大值作为最终相似度，仅返回一个浮点数
    return mat.max().item()

def evaluate(data_list: List[VideoData], device='cuda', batch_size=16, sampling=16, model_args: dict = {}) -> List['VideoData']:
    get_arcface_model(device=device, **model_args)
    for data in tqdm(data_list, desc="Evaluating ArcFace Consistency"):
        with torch.no_grad():
            video_frames = load_video(
                data.video_path,
                return_tensor=False,
                num_frames=sampling
            )  # (T, H, W, C) numpy RGB

            video_frames = video_frames[:, :, :, ::-1]  # 转 BGR

            # 逐帧提取 ArcFace 特征
            frame_features = []
            for i in range(0, video_frames.shape[0], batch_size):
                frame_batch = video_frames[i:i + batch_size]  # (B, H, W, C) numpy BGR
                feats_batch = extract_arcface_features(
                    [frame_batch[j] for j in range(frame_batch.shape[0])]
                )  # List[Tensor(num_faces, 512)]
                frame_features.extend(feats_batch)
            
            # 获取参考图特征
            if not data.has_image():
                raise ValueError("ArcFace Consistency 评测需要参考图像，请确保提供了参考图像路径。")
            ref_image = load_image_cv2(data.image_path)  # numpy BGR
            ref_feature = extract_arcface_features(ref_image)  # List[Tensor(num_faces, 512)]
            if len(ref_feature) == 0 or ref_feature[0].shape[0] == 0:
                raise ValueError("参考图像中未检测到人脸，无法进行 ArcFace Consistency 评测。")
            
            total_similarity = 0.0
            valid_frame_count = 0
            sim_list = []
            for feats in frame_features:
                if feats.shape[0] == 0:
                    continue  # 当前帧无检测到人脸，跳过
                sim = pairwise_cosine_sim(feats, ref_feature[0])  # 计算与参考图的最大相似度
                sim_list.append(sim)
                total_similarity += sim
                valid_frame_count += 1
            if valid_frame_count == 0:
                average_similarity = 0.0
            else:
                average_similarity = total_similarity / valid_frame_count
            results = {
                "arcface_consistency": average_similarity,
                "arcface_consistency_list": sim_list
            }
            data.register_result("arcface_consistency", results)
    return data_list