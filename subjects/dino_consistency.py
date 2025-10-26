import subprocess
import os
import sys
from utils import load_video, load_image
import torch
from tqdm import tqdm

from transformers import pipeline


import numpy as np

from video import VideoData

from transformers import AutoImageProcessor, AutoModel

dino_model = None
dino_processor = None

def get_dino_v3_model(model_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m", device='cuda', **kwargs):
    global dino_model, dino_processor
    dino_processor = AutoImageProcessor.from_pretrained(model_name)
    dino_model = AutoModel.from_pretrained(
        model_name, 
        device_map="auto", 
    )
    dino_model.to(device)

def extract_features(image: torch.Tensor) -> torch.Tensor:
    global dino_model, dino_processor
    if dino_model is None:
        raise ValueError("DINOv3模型尚未加载。请先调用get_dino_v3_model函数加载模型。")
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    if not isinstance(image, torch.Tensor):
        print("输入图像类型：", type(image))
        raise TypeError("输入图像应为torch.Tensor格式。")
    if image.ndim != 4:
        if image.ndim == 3:
            image = image.unsqueeze(0)  # 添加批次维度
        else:
            raise ValueError("输入图像应为3维或4维torch.Tensor。")
    image = dino_processor(images=image, return_tensors="pt")
    features = dino_model(**image).pooler_output  # (B, D)
    features = torch.tensor(features)
    return features

def evaluate(data_list: list[VideoData], device='cuda', batch_size=16, sampling=16, model_args: dict = {}):
    get_dino_v3_model(**model_args, device=device)
    for data in tqdm(data_list, desc="Evaluating"):
        with torch.no_grad():
            video_frames = load_video(
                data.video_path,
                return_tensor=True,
                num_frames=16
            )  # (T, C, H, W)
            video_frames = video_frames.to(device)

            # 逐帧提取特征
            frame_features = []
            for i in range(0, video_frames.shape[0], batch_size):
                frame_batch = video_frames[i:i + batch_size]
                frame_features.append(extract_features(frame_batch))  # (T, D)
            frame_features = torch.cat(frame_features, dim=0)  # (T, D)

            # 如果有参考图，取出参考图并提取特征
            if data.has_image():
                image = load_image(data.image_path).to(device)
                ref_features = extract_features(image)
            else:
                # 如果没有参考图，取视频第一帧
                ref_features = frame_features[0:1, :]
            # 计算视频帧与参考图的相似度（除去第一帧）
            similarities = torch.nn.functional.cosine_similarity(
                frame_features[1:], ref_features.repeat(frame_features.shape[0] - 1, 1), dim=1
            )  # (T-1,)
            # 计算相邻帧之间的余弦相似度
            temporal_consistency = torch.nn.functional.cosine_similarity(
                frame_features[:-1], frame_features[1:], dim=1
            )  # (T-1,)
            # 汇总结果：平均值
            results = {
                "similarity": similarities.mean().item(),
                "temporal_consistency": temporal_consistency.mean().item(),
                "similarity_list": similarities.cpu().tolist(),
                "temporal_consistency_list": temporal_consistency.cpu().tolist()
            }
            results["overall_score"] = (results["similarity"] + results["temporal_consistency"]) / 2.0
            data.register_result("dino_consistency", results)
    return data_list
        
# # 改用huggingface下载模型权重，预备废弃
# def prepare_dino_v3_repo():
#     """
#     克隆并准备DINOv3模型的代码库。
#     """
#     # 检测是否已经存在DINOv3代码库
#     current_dir = os.getcwd()
#     repo_dir = os.path.join(current_dir, "repos", "dinov3")
#     if os.path.exists(repo_dir):
#         print("DINOv3代码库已经存在，跳过克隆步骤。")
#         sys.path.append(repo_dir)
#         return repo_dir

#     current_dir = os.getcwd()
#     repo_dir = os.path.join(current_dir, "repos")
#     if not os.path.exists(repo_dir):
#         os.makedirs(repo_dir, exist_ok=True)
#         os.chdir(repo_dir)
#     DINO_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
#     subprocess.run(["git", "clone", DINO_REPO_URL])
#     dino_repo_path = os.path.join(repo_dir, "dinov3")
#     sys.path.append(dino_repo_path)
#     os.chdir(current_dir)
#     return dino_repo_path
    