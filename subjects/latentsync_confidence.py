"""LatentSync audio-visual synchrony evaluation subject.

This module wraps the lip-sync evaluation utilities that ship with the
LatentSync project (https://github.com/bytedance/LatentSync).  The original
repository exposes a command line entry-point ``eval/eval_sync_conf.py``.  The
logic below mirrors that behaviour while integrating it with the
:mod:`Video-Eval` orchestration framework.

Key features
============
* The LatentSync repository is cloned on demand (``third_party/LatentSync`` by
  default) and optionally its Python dependencies can be installed.
* Checkpoints required by SyncNet and S3FD are fetched automatically from the
  official Hugging Face repository.  This avoids depending on the ``huggingface
  cli`` wrapper used upstream.
* Each :class:`~video.VideoData` instance is evaluated individually and the
  resulting confidence/offset statistics are registered under the
  ``"latentsync_confidence"`` subject key by default.

The intent is to keep the orchestration lightweight while delegating the
heavyweight face detection, cropping and SyncNet inference to the upstream
codebase.
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
from typing import Any, Dict, List, Tuple

from video import VideoData

LATENTSYNC_REPO_URL = "https://github.com/bytedance/LatentSync.git"
DEFAULT_REPO_SUBDIR = Path("third_party") / "LatentSync"
DEFAULT_CHECKPOINT_REPO = "ByteDance/LatentSync-1.5"
DEFAULT_INITIAL_MODEL = Path("checkpoints") / "auxiliary" / "syncnet_v2.model"
DEFAULT_RESULTS_KEY = "latentsync_confidence"
DEFAULT_SYNCNET_BATCH_SIZE = 20
DEFAULT_SYNCNET_VSHIFT = 15
DEFAULT_MIN_TRACK = 50


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
        filename="/".join(relative_path.parts),
        local_dir=str(repo_root),
        local_dir_use_symlinks=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path != file_path:
        shutil.move(str(downloaded_path), str(file_path))

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


def _run_syncnet_evaluation(
    syncnet,
    syncnet_detector,
    video_path: Path,
    workspace: Path,
    *,
    min_track: int,
    scale_videos: bool,
    syncnet_batch_size: int,
    syncnet_vshift: int,
) -> Tuple[int, float, List[Dict[str, float]]]:
    """Execute the SyncNet confidence evaluation for a single video."""

    detect_root = Path(syncnet_detector.detect_results_dir)
    if detect_root.exists():
        shutil.rmtree(detect_root)
    detect_root.mkdir(parents=True, exist_ok=True)

    syncnet_detector(video_path=str(video_path), min_track=min_track, scale=scale_videos)

    crop_dir = detect_root / "crop"
    crop_videos = sorted(crop_dir.glob("*.mp4"))
    if not crop_videos:
        raise RuntimeError(f"Face not detected in {video_path}")

    per_crop_results: List[Dict[str, float]] = []
    offsets: List[int] = []
    confidences: List[float] = []

    temp_root = workspace / "syncnet_temp"

    for crop_video in crop_videos:
        crop_temp = temp_root / crop_video.stem
        av_offset, _, conf = syncnet.evaluate(
            video_path=str(crop_video),
            temp_dir=str(crop_temp),
            batch_size=syncnet_batch_size,
            vshift=syncnet_vshift,
        )

        av_offset_int = int(av_offset)
        conf_float = float(conf)
        offsets.append(av_offset_int)
        confidences.append(conf_float)
        per_crop_results.append(
            {
                "crop_video": str(crop_video),
                "confidence": conf_float,
                "av_offset": av_offset_int,
            }
        )

    average_offset = int(round(fmean(offsets)))
    average_confidence = float(fmean(confidences))

    return average_offset, average_confidence, per_crop_results


def evaluate(
    data_list: List[VideoData],
    device: str = "cuda",
    batch_size: int = 16,
    sampling: int = 16,
    model_args: Dict[str, Any] | None = None,
) -> List[VideoData]:
    """Run LatentSync SyncNet confidence evaluation on the provided videos."""

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

    syncnet_module = importlib.import_module("eval.syncnet")
    detector_module = importlib.import_module("eval.syncnet_detect")
    util_module = importlib.import_module("latentsync.utils.util")

    util_module.check_ffmpeg_installed()

    initial_model = model_args.get("initial_model", DEFAULT_INITIAL_MODEL)
    initial_model_path = Path(initial_model)
    if not initial_model_path.is_absolute():
        initial_model_path = repo_dir / initial_model_path

    _ensure_checkpoint(initial_model_path, repo_dir, model_args.get("checkpoint_repo_id", DEFAULT_CHECKPOINT_REPO))

    syncnet = syncnet_module.SyncNetEval(device=device)
    syncnet.loadParameters(str(initial_model_path))

    min_track = int(model_args.get("min_track", DEFAULT_MIN_TRACK))
    scale_videos = bool(model_args.get("scale_videos", False))
    syncnet_batch_size = int(model_args.get("syncnet_batch_size", DEFAULT_SYNCNET_BATCH_SIZE))
    syncnet_vshift = int(model_args.get("syncnet_vshift", DEFAULT_SYNCNET_VSHIFT))
    results_key = model_args.get("results_key", DEFAULT_RESULTS_KEY)

    with tempfile.TemporaryDirectory(prefix="latentsync_eval_") as workspace_dir:
        workspace = Path(workspace_dir)
        detect_results_dir = workspace / "detect_results"
        syncnet_detector = detector_module.SyncNetDetector(
            device=device,
            detect_results_dir=str(detect_results_dir),
        )
        for data in data_list:
            video_path = Path(data.video_path)
            try:
                avg_offset, avg_conf, crop_details = _run_syncnet_evaluation(
                    syncnet,
                    syncnet_detector,
                    video_path,
                    workspace,
                    min_track=min_track,
                    scale_videos=scale_videos,
                    syncnet_batch_size=syncnet_batch_size,
                    syncnet_vshift=syncnet_vshift,
                )
                result = {
                    "confidence": avg_conf,
                    "av_offset": avg_offset,
                    "crop_details": crop_details,
                }
            except Exception as exc:  # pragma: no cover - safeguard for runtime failures
                result = {"error": str(exc)}
            finally:
                shutil.rmtree(syncnet_detector.detect_results_dir, ignore_errors=True)

            data.register_result(results_key, result)

    return data_list
