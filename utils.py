import os
import json
import numpy as np
import logging
import subprocess
import random
import torch
import re
from pathlib import Path
from PIL import Image, ImageSequence
from decord import VideoReader, cpu
from torchvision import transforms
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def dino_transform(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def tag2text_transform(n_px):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    return Compose([ToPILImage(),Resize((n_px, n_px), antialias=False),ToTensor(),normalize])

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def load_image(image_path):
    """
    Load an image from the given path and convert it to a torch.Tensor.

    Parameters:
    - image_path (str): The file path to the image.

    Returns:
    - image_tensor (torch.Tensor): A tensor representation of the image with shape (1, C, H, W).
    """
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.uint8)
    img_tensor = torch.Tensor(img)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return img_tensor

def load_image_cv2(image_path) -> np.ndarray:
    """
    Load an image from the given path using OpenCV and convert it to BGR format.

    Parameters:
    - image_path (str): The file path to the image.

    Returns:
    - img_bgr (np.ndarray): A NumPy array representation of the image in BGR (H, W, C) format.
    """
    img = cv2.imread(image_path)  # BGR format
    if img is None:
        # fall back to PIL
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil).astype(np.uint8)
        img = img[:, :, ::-1]  # Convert RGB to BGR
        print(f"Warning: cv2.imread failed for {image_path}, used PIL fallback.")
    return img.astype(np.uint8)

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.
    
    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.
    
    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.
    
    Raises:
    - NotImplementedError: If the video format is not supported.
    
    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
    if num_frames is not None and num_frames == 0:
        num_frames = None  # Use all frames
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
            num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
        num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames
    

def read_frames_decord_by_fps(
        video_path, sample_fps=2, sample='rand', fix_start=None, 
        max_num_frames=-1,  trimmed30=False, num_frames=8
    ):
    import decord
    decord.bridge.set_bridge("torch")
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames
    
def load_dimension_info(json_dir, dimension, lang):
    """
    Load video list and prompt information based on a specified dimension and language from a JSON file.
    
    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation to filter the video prompts.
    - lang (str): The language key used to retrieve the appropriate prompt text.
    
    Returns:
    - video_list (list): A list of video file paths that match the specified dimension.
    - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding video list.
    
    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension and compiles a list of video paths and associated prompts in the specified language.
    
    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
    - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
    """
    video_list = []
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)
    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict['dimension'] and 'video_list' in prompt_dict:
            prompt = prompt_dict[f'prompt_{lang}']
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
            else:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
    return video_list, prompt_dict_ls

def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_json(path):
    """
    Load a JSON file from the given file path.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)