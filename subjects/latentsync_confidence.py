"""LatentSync audio-visual synchrony evaluation subject.

This module integrates the Stable SyncNet evaluation pipeline that ships with
LatentSync (https://github.com/bytedance/LatentSync).  Instead of relying on the
legacy ``eval_sync_conf.py`` workflow, the implementation mirrors the logic of
``eval/eval_syncnet_acc.py`` which is tailored to the Stable SyncNet model that
ships with the public ``stable_syncnet.pt`` checkpoint.

Key features
============
* The LatentSync repository is cloned on demand (``third_party/LatentSync`` by
  default) and optionally its Python dependencies can be installed.
* The Stable SyncNet checkpoint is fetched automatically from the official
  Hugging Face repository without invoking the CLI helper used upstream.
* Each :class:`~video.VideoData` instance is evaluated individually and the
  resulting accuracy/similarity statistics are registered under the
  ``"latentsync_confidence"`` subject key by default.

At a high level the subject repeatedly samples short clips from the target
video, creates matched/mismatched audio-visual pairs following the upstream
``SyncNetDataset`` implementation and feeds them through ``StableSyncNet``.
Accuracies and cosine similarity statistics are aggregated per video to
quantify lip-sync quality.
"""

from __future__ import annotations

import contextlib
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterator, List

from video import VideoData

LATENTSYNC_REPO_URL = "https://github.com/bytedance/LatentSync.git"
DEFAULT_REPO_SUBDIR = Path("third_party") / "LatentSync"
DEFAULT_CHECKPOINT_REPO = "ByteDance/LatentSync-1.5"
# ``stable_syncnet.pt`` is the default SyncNet checkpoint shipped with the public
# LatentSync release hosted on Hugging Face.  Older revisions used the
# ``checkpoints/auxiliary/syncnet_v2.model`` path, which is now deprecated.  We
# default to the new filename while still allowing callers to override it via
# ``model_args["inference_ckpt_path"]`` when needed.
DEFAULT_INITIAL_MODEL = Path("stable_syncnet.pt")
LEGACY_SYNCNET_MODEL_PATH = Path("checkpoints") / "auxiliary" / "syncnet_v2.model"
DEFAULT_RESULTS_KEY = "latentsync_confidence"
DEFAULT_SYNCNET_BATCH_SIZE = 20
DEFAULT_SYNCNET_NUM_BATCHES = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.5


class LatentSyncDependencyError(RuntimeError):
    """Raised when a required LatentSync dependency is missing."""


def _prepare_latentsync_repo(
    repo_dir: Path | None = None,
    repo_url: str = LATENTSYNC_REPO_URL,
    install_dependencies: bool = False,
    force_clone: bool = False,
    requirements_path: str | None = None,
) -> Path:
    """Clone the LatentSync repository and (optionally) install dependencies."""

    repo_dir = (repo_dir or (Path(__file__).resolve().parent.parent / DEFAULT_REPO_SUBDIR)).resolve()
    if force_clone and repo_dir.exists():
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    if install_dependencies:
        # Prefer a lightweight requirements file when provided.  Fallback to the
        # repository's default requirements.txt, even though it contains training
        # dependencies as well.
        if requirements_path is not None:
            requirements_file = repo_dir / requirements_path
        else:
            candidate = repo_dir / "requirements_eval.txt"
            requirements_file = candidate if candidate.exists() else repo_dir / "requirements.txt"

        if requirements_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True,
            )
        else:
            raise FileNotFoundError(
                f"Could not locate requirements file for LatentSync at {requirements_file}."
            )

    return repo_dir


def _ensure_repo_on_path(repo_dir: Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _resolve_checkpoint_filename(relative_path: Path) -> str:
    """Map a checkpoint path to the file stored in the Hugging Face repo."""

    if relative_path == LEGACY_SYNCNET_MODEL_PATH:
        return DEFAULT_INITIAL_MODEL.name
    parts = relative_path.parts
    if relative_path.is_absolute():
        parts = parts[1:]
    return "/".join(parts) if parts else relative_path.name


def _ensure_checkpoint(
    file_path: Path,
    repo_root: Path,
    repo_id: str = DEFAULT_CHECKPOINT_REPO,
) -> Path:
    """Download a checkpoint file via :mod:`huggingface_hub` when missing."""

    if file_path.exists():
        return file_path

    relative_path = file_path
    if not file_path.is_absolute():
        file_path = repo_root / file_path
        relative_path = file_path.relative_to(repo_root)
    else:
        with contextlib.suppress(ValueError):
            relative_path = file_path.relative_to(repo_root)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise LatentSyncDependencyError(
            "huggingface_hub is required to download LatentSync checkpoints."
        ) from exc

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=_resolve_checkpoint_filename(relative_path),
        local_dir=str(repo_root),
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path != file_path and not file_path.exists():
        shutil.copy2(str(downloaded_path), str(file_path))

    return file_path


def _patch_check_model_and_download(repo_root: Path, repo_id: str) -> None:
    """Replace LatentSync's CLI-based downloader with our Python implementation."""

    util_module = importlib.import_module("latentsync.utils.util")

    def _patched(ckpt_path: str, huggingface_model_id: str = repo_id) -> None:
        path_obj = Path(ckpt_path)
        if not path_obj.is_absolute():
            path_obj = repo_root / path_obj
        _ensure_checkpoint(path_obj, repo_root, huggingface_model_id)

    util_module.check_model_and_download = _patched  # type: ignore[attr-defined]


def evaluate(
    data_list: List[VideoData],
    device: str = "cuda",
    batch_size: int = 16,
    sampling: int = 16,
    model_args: Dict[str, Any] | None = None,
) -> List[VideoData]:
    """Run LatentSync Stable SyncNet evaluation on the provided videos."""

    del batch_size, sampling  # Unused but kept for a consistent signature.

    model_args = model_args or {}

    repo_dir_arg = model_args.get("repo_dir")
    repo_dir = Path(repo_dir_arg) if repo_dir_arg else None
    repo_dir = _prepare_latentsync_repo(
        repo_dir=repo_dir,
        repo_url=model_args.get("repo_url", LATENTSYNC_REPO_URL),
        install_dependencies=model_args.get("install_dependencies", False),
        force_clone=model_args.get("force_clone", False),
        requirements_path=model_args.get("requirements_path"),
    )

    _ensure_repo_on_path(repo_dir)

    # Patching must happen before importing modules that depend on the helper.
    _patch_check_model_and_download(repo_dir, model_args.get("checkpoint_repo_id", DEFAULT_CHECKPOINT_REPO))

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL
        from einops import rearrange
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise LatentSyncDependencyError(
            "Stable SyncNet evaluation requires torch, accelerate, diffusers and einops."
        ) from exc

    if device.startswith("cuda"):
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    config_path = Path(model_args.get("config_path", "configs/syncnet/syncnet_16_latent.yaml"))
    if not config_path.is_absolute():
        config_path = repo_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"LatentSync config not found at {config_path}")

    base_config = OmegaConf.load(str(config_path))

    checkpoint_repo_id = model_args.get("checkpoint_repo_id", DEFAULT_CHECKPOINT_REPO)
    ckpt_path_arg = model_args.get("inference_ckpt_path") or base_config.ckpt.get("inference_ckpt_path")
    if not ckpt_path_arg:
        ckpt_path_arg = DEFAULT_INITIAL_MODEL

    ckpt_path = Path(ckpt_path_arg)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_dir / ckpt_path

    _ensure_checkpoint(ckpt_path, repo_dir, checkpoint_repo_id)

    model_dtype = torch.float16 if torch_device.type == "cuda" else torch.float32

    syncnet_module = importlib.import_module("latentsync.models.stable_syncnet")

    syncnet = syncnet_module.StableSyncNet(OmegaConf.to_container(base_config.model)).to(torch_device)

    try:
        checkpoint = torch.load(str(ckpt_path), map_location=torch_device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location=torch_device)

    state_dict = checkpoint.get("state_dict", checkpoint)
    syncnet.load_state_dict(state_dict)
    syncnet.to(dtype=model_dtype)
    syncnet.requires_grad_(False)
    syncnet.eval()

    latent_space = bool(model_args.get("latent_space", base_config.data.get("latent_space", True)))

    vae = None
    if latent_space:
        vae_kwargs = {"subfolder": "vae", "torch_dtype": model_dtype}
        if torch_device.type == "cuda":
            vae_kwargs["revision"] = "fp16"
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-inpainting", **vae_kwargs)
        vae.requires_grad_(False)
        vae.to(torch_device)

    similarity_threshold = float(model_args.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    syncnet_batch_size = int(model_args.get("syncnet_batch_size", DEFAULT_SYNCNET_BATCH_SIZE))
    syncnet_num_batches = int(model_args.get("syncnet_num_batches", DEFAULT_SYNCNET_NUM_BATCHES))
    syncnet_num_workers = int(model_args.get("syncnet_num_workers", 0))
    seed = int(model_args.get("seed", base_config.run.get("seed", 42)))
    results_key = model_args.get("results_key", DEFAULT_RESULTS_KEY)

    if syncnet_batch_size <= 0:
        raise ValueError("syncnet_batch_size must be a positive integer")
    if syncnet_num_batches <= 0:
        raise ValueError("syncnet_num_batches must be a positive integer")

    set_seed(seed)

    dataset_module = importlib.import_module("latentsync.data.syncnet_dataset")

    config_overrides = {
        "num_frames": int(model_args.get("num_frames", base_config.data.get("num_frames", 16))),
        "resolution": int(model_args.get("resolution", base_config.data.get("resolution", 256))),
        "audio_sample_rate": int(
            model_args.get("audio_sample_rate", base_config.data.get("audio_sample_rate", 16000))
        ),
        "video_fps": int(model_args.get("video_fps", base_config.data.get("video_fps", 25))),
        "lower_half": bool(model_args.get("lower_half", base_config.data.get("lower_half", False))),
    }

    with tempfile.TemporaryDirectory(prefix="latentsync_eval_") as workspace_dir:
        workspace = Path(workspace_dir)

        for data in data_list:
            video_path = Path(data.video_path).resolve()

            video_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

            video_config.data.latent_space = latent_space
            video_config.data.batch_size = syncnet_batch_size
            video_config.data.num_workers = syncnet_num_workers
            video_config.data.num_val_samples = syncnet_batch_size * syncnet_num_batches
            video_config.run.seed = seed

            for key, value in config_overrides.items():
                setattr(video_config.data, key, value)

            audio_cache_dir = workspace / f"mel_cache_{video_path.stem}"
            audio_cache_dir.mkdir(parents=True, exist_ok=True)
            video_config.data.audio_mel_cache_dir = str(audio_cache_dir)

            filelist_path = workspace / f"video_list_{video_path.stem}.txt"
            with filelist_path.open("w", encoding="utf-8") as handle:
                for _ in range(video_config.data.num_val_samples):
                    handle.write(str(video_path) + "\n")

            video_config.data.val_fileslist = str(filelist_path)
            video_config.data.val_data_dir = ""

            dataset = dataset_module.SyncNetDataset(
                video_config.data.val_data_dir,
                video_config.data.val_fileslist,
                video_config,
            )

            dataset.worker_init_fn(0)

            dataloader = DataLoader(
                dataset,
                batch_size=syncnet_batch_size,
                shuffle=False,
                num_workers=syncnet_num_workers,
                drop_last=False,
                worker_init_fn=dataset.worker_init_fn,
            )

            num_val_batches = video_config.data.num_val_samples // syncnet_batch_size
            num_val_batches = max(1, num_val_batches)

            num_correct_preds = 0
            num_total_preds = 0
            similarities: List[float] = []
            positive_sims: List[float] = []
            negative_sims: List[float] = []

            batches_consumed = 0
            dataloader_iter: Iterator = iter(dataloader)

            while batches_consumed < num_val_batches:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                frames = batch["frames"].to(device=torch_device, dtype=model_dtype)
                audio_samples = batch["audio_samples"].to(device=torch_device, dtype=model_dtype)
                labels = batch["y"].to(device=torch_device, dtype=model_dtype).squeeze(1)

                if latent_space:
                    frames = rearrange(frames, "b f c h w -> (b f) c h w")
                    with torch.no_grad():
                        frames = vae.encode(frames).latent_dist.sample() * 0.18215
                    frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=video_config.data.num_frames)
                else:
                    frames = rearrange(frames, "b f c h w -> b (f c) h w")

                if video_config.data.lower_half:
                    height = frames.shape[2]
                    frames = frames[:, :, height // 2 :, :]

                with torch.no_grad():
                    vision_embeds, audio_embeds = syncnet(frames, audio_samples)

                sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
                preds_bool = sims > similarity_threshold
                labels_bool = labels >= 0.5

                matches = (preds_bool == labels_bool).sum().item()
                batch_size_actual = len(sims)

                num_correct_preds += matches
                num_total_preds += batch_size_actual

                sims_cpu = sims.detach().cpu().tolist()
                labels_cpu = labels.detach().cpu().tolist()
                similarities.extend(sims_cpu)

                for sim_value, label_value in zip(sims_cpu, labels_cpu):
                    if label_value >= 0.5:
                        positive_sims.append(sim_value)
                    else:
                        negative_sims.append(sim_value)

                batches_consumed += 1

            accuracy = float(num_correct_preds) / float(num_total_preds) if num_total_preds else 0.0
            mean_similarity = fmean(similarities) if similarities else 0.0
            positive_mean = fmean(positive_sims) if positive_sims else 0.0
            negative_mean = fmean(negative_sims) if negative_sims else 0.0

            result = {
                "sync_accuracy": accuracy,
                "num_samples": num_total_preds,
                "mean_similarity": mean_similarity,
                "positive_similarity": positive_mean,
                "negative_similarity": negative_mean,
                "similarity_threshold": similarity_threshold,
            }

            data.register_result(results_key, result)

    return data_list
