# Video-Eval 使用指南

本项目提供一套可扩展的视频生成模型客观评测框架。通过统一的入口脚本 `evaluate.py`，可以在同一批生成结果上串联执行多个评测主题（subject），并输出结构化的 JSON 报告。下文将详细说明项目结构、准备工作、整体流程以及各个评测主题的使用要点。

## 目录结构概览

```
Video-Eval/
├── evaluate.py           # 主评测脚本，负责装载数据并依次运行各个 subject
├── utils.py              # 视频/图片读写与通用工具函数
├── video.py              # `VideoData` 数据结构定义
└── subjects/             # 具体的评测主题实现
    ├── arcface_consistency.py
    ├── dino_consistency.py
    ├── latent_sync.py
    ├── qwen_vl_vllm.py
    └── video_quality.py
```

核心流程如下：

1. `evaluate.py` 根据指定的输入/输出目录构造 `VideoData` 列表。
2. 根据 `--evaluate_subjects` 参数依次导入 `subjects/` 目录下的模块，并调用它们的 `evaluate` 函数。
3. 每个 subject 在 `VideoData` 上写入评测结果（`register_result`）。
4. 所有结果最终保存为 `evaluation_results.json`。

`VideoData` 对象会自动关联与视频同名的参考图片（`.png`）、音频（`.wav`）、文本（`.txt`），并提供 `has_image/has_audio/has_text` 等便捷方法，供各个 subject 调用。

## 环境准备

- Python 3.10 及以上（建议与依赖库兼容的版本）。
- 常规依赖（均可通过 `pip install -r requirements.txt` 方式自行整理）：
  - `torch`, `torchvision`
  - `transformers`, `tqdm`, `numpy`
  - `decord`, `opencv-python`, `pillow`
- 各 subject 可能需要额外的第三方库或模型权重，详见后文。

## 快速上手

1. **准备评测数据**
   - `--model_input_dir`：存放参考资料的文件夹，需包含与视频同名的 `.png`（参考图）、`.wav`（参考音频）、`.txt`（文本提示）等，可按需缺省。
   - `--model_output_dir`：被评测模型生成的视频结果，扩展名需为 `.mp4`。

2. **选择评测主题**
   - 使用逗号分隔，例如：`--evaluate_subjects dino_consistency,audio_visual_synchrony`
   - `--model_args` 可按同样数量提供 JSON 字符串，为每个 subject 传入额外配置，详见各节。

3. **运行示例**

   ```bash
   python evaluate.py \
     --model_input_dir /path/to/inputs \
     --model_output_dir /path/to/outputs \
     --results_dir ./evaluation_results \
     --evaluate_subjects dino_consistency,arcface_consistency \
     --model_args '{"model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"}','{}'
   ```

   常用参数：
   - `--device`：默认 `cuda`，部分 subject 支持 `cpu`。
   - `--batch_size`：处理帧批大小，默认 `16`。
   - `--sampling`：单个视频采样帧数。默认采样 16 帧；将 `DEFAULT_ALL_FRAMES` 改为 `True` 后默认使用全部帧。

4. **结果查看**
   - 所有 `VideoData` 的结果会写入 `results_dir/evaluation_results.json`。
   - JSON 中每个条目包含视频路径和各 subject 的评估结果。

## 评测主题详解

### 1. `dino_consistency` — 图像语义一致性

- **作用**：使用 DINOv3 模型评估生成视频与参考图像之间的语义相似度，以及帧间时序一致性。
- **依赖**：`transformers`, `torch`, `numpy`。
- **关键参数**（通过 `--model_args` 传入 JSON）：
  - `model_name`：Hugging Face 上的 DINOv3 权重名称，默认 `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`。
- **注意事项**：
  - 若提供了参考图片，会使用参考图片提特征；否则退化为与首帧对比。
  - 会返回 `similarity`、`temporal_consistency` 及每帧的分布列表，并给出 `overall_score` 均值。

### 2. `arcface_consistency` — 人脸一致性

- **作用**：借助 InsightFace ArcFace 模型，衡量视频中人脸与参考图像人脸的相似度。
- **依赖**：`insightface`, `opencv-python`, `torch`（需 GPU 支持以提升检测速度）。
- **关键参数**：
  - `name`：InsightFace 的模型组合，默认 `buffalo_l`。
  - `det_size`：人脸检测输入尺寸，默认 `(640, 640)`。
- **使用要求**：
  - 必须提供对应的参考人脸图片（`.png`）。
  - 没有检测到人脸时会抛出异常或跳过该帧。
  - 结果包含 `arcface_consistency` 均值及逐帧相似度列表。

### 3. `latent_sync` — 视听同步度

- **作用**：集成 ByteDance 开源的 LatentSync 项目，对唇形与语音同步进行评估。
- **依赖**：
  - 系统需安装 `git`、`ffmpeg`（LatentSync 运行时需求）。
  - Python 包：`huggingface_hub`（自动下载权重），以及 LatentSync 自身依赖（在 clone 后按照其 README 安装）。
- **关键参数**：
  - `repo_dir`：LatentSync 代码存放位置，默认 `third_party/LatentSync`。
  - `force_clone`：`true` 时强制重新 clone。
  - `min_track`：人脸跟踪最小帧数，默认 `50`。
  - `syncnet_checkpoint`、`huggingface_repo_id`：自定义权重路径或来源。
  - `subject_name`：自定义写入结果的键名，默认 `latent_sync`。
- **输出**：每个视频包含 `confidence`、`av_offset`、`num_crops` 等字段，若失败会记录 `error`。

### 4. `qwen_vl_vllm` — 大模型主观问答

- **作用**：通过 vLLM 调用多模态大模型（默认 Qwen3-VL）读取视频并回答自定义提示问题，记录模型输出。
- **依赖**：`vllm` 及其硬件依赖（CUDA 环境、对应显存要求）。
- **关键参数**：
  - `model_name`：默认 `Qwen/Qwen3-VL-2B-Thinking`。
  - `prompt_template` 或 `prompt_template_path`：生成文本提示的模板，支持 `str.format` 占位符。
  - `template_variables`：模板可选默认变量。
  - `system_prompt`：额外的系统消息。
  - `include_video`：是否将视频传给模型，默认 `true`。
  - `sampling_options`：传给 `SamplingParams` 的设置，例如 `{"max_tokens":512,"temperature":0.0}`。
  - `engine_options` / 其他直通参数：传递给 `vllm.LLM` 的配置，如 `tensor_parallel_size`、`dtype` 等。
  - `store_messages`：为 `true` 时额外保存完整的消息与原始输出。
- **输入要求**：视频路径需对 vLLM 可见（脚本默认允许本地文件路径 `file://`）。

### 5. `video_quality` — FineVQ 视频质量评估

- **作用**：自动调用 FineVQ 项目，对视频视觉质量打分。
- **依赖**：
  - 系统需安装 `git`、`ffmpeg`（FineVQ 依赖环境）。
  - Python 包：运行时会按需安装 FineVQ 仓库中的依赖。
- **关键参数**：
  - `repo_dir`：FineVQ 仓库存放路径，默认 `third_party/FineVQ`。
  - `repo_url`：仓库地址，默认官方仓库。
  - `install_dependencies`：是否在首次运行时安装依赖，默认 `true`。
  - `force_clone`：强制重新 clone 仓库。
  - `per_device_batch_size`、`batch_size`、`nproc_per_node`：控制分布式推理的批量大小与进程数。
  - `use_bf16`：是否启用 bfloat16，默认 `true`。
  - `env`：附加环境变量（如 `CUDA_VISIBLE_DEVICES`）。
  - `extra_args`：以列表形式追加或覆盖 FineVQ CLI 参数。
- **执行流程**：
  1. 将目标视频软链到临时目录并生成 FineVQ 所需的 `meta.json`。
  2. 调用 FineVQ 官方 `torch.distributed.run` 推理脚本。
  3. 解析输出的 CSV 与指标文件，并写入 `video_quality` 结果字段。
- **输出字段**：`video_quality_score`（通常对应 `pred_score`）、详细原始指标及可选的整体指标 `aggregate_metrics`。

## 自定义扩展

- 要新增评测主题，可在 `subjects/` 中新增同名 Python 文件，导出 `evaluate(data_list, device, batch_size, sampling, model_args)` 函数。
- 每个 `VideoData` 的自定义结果通过 `register_result(subject_name, payload)` 写入，最终都会出现在汇总 JSON 中。

## 常见问题

- **内存占用过高**：可调整 `--sampling` 减少每个视频采样帧数，或修改 `DEFAULT_ALL_FRAMES=False`。
- **缺少依赖或模型**：根据报错信息安装所需 Python 包或下载权重，必要时手动准备 `checkpoints/` 目录。
- **评测顺序**：`evaluate.py` 会按照 `--evaluate_subjects` 中的顺序依次执行，某个 subject 修改的 `data_list` 会传给下一个 subject。

希望本说明能帮助你快速理解并使用 Video-Eval 框架。如需进一步自定义或排查问题，可直接查阅对应 subject 的源码并根据项目需求调整。
