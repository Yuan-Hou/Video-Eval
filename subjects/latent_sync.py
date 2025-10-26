"""Audio-visual synchronisation evaluation using LatentSync."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Tuple

from video import VideoData

LATENTSYNC_REPO_URL = "https://github.com/bytedance/LatentSync.git"
DEFAULT_REPO_SUBDIR = Path("third_party") / "LatentSync"
DEFAULT_HF_REPO_ID = "ByteDance/LatentSync-1.5"
REQUIRED_CHECKPOINTS = (
    Path("checkpoints/auxiliary/syncnet_v2.model"),
    Path("checkpoints/auxiliary/sfd_face.pth"),
)
SUBJECT_NAME = "latent_sync"


class LatentSyncDependencyError(RuntimeError):
    """Raised when a mandatory dependency for LatentSync is missing."""


def _prepare_latentsync_repo(
    repo_dir: Path | None = None,
    repo_url: str = LATENTSYNC_REPO_URL,
    force_clone: bool = False,
) -> Path:
    """Clone the LatentSync repository if it is not already available."""

    repo_dir = (repo_dir or (Path(__file__).resolve().parent.parent / DEFAULT_REPO_SUBDIR)).resolve()
    if force_clone and repo_dir.exists():
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    return repo_dir


def _ensure_checkpoints(repo_dir: Path, repo_id: str = DEFAULT_HF_REPO_ID) -> None:
    """Download the SyncNet checkpoints required by the evaluation script."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - import guarded by environment
        raise LatentSyncDependencyError(
            "huggingface_hub is required to download LatentSync checkpoints. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    for relative_path in REQUIRED_CHECKPOINTS:
        destination = repo_dir / relative_path
        if destination.exists():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        parts = relative_path.parts
        if len(parts) < 2:
            # The checkpoint paths provided by LatentSync always contain at least
            # two components (e.g. "checkpoints/auxiliary/file").  Bail out if
            # the format changes unexpectedly.
            raise ValueError(f"Unexpected checkpoint path format: {relative_path}")

        # LatentSync stores the checkpoints inside a top-level "checkpoints"
        # directory.  The Hugging Face repository mirrors this structure but the
        # download API expects a path relative to that root directory.
        local_dir = repo_dir / parts[0]
        filename = str(Path(*parts[1:]))
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )


def _import_latentsync_modules(repo_dir: Path):
    """Import LatentSync evaluation helpers from the cloned repository."""

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    try:
        from eval.syncnet import SyncNetEval  # type: ignore[attr-defined]
        from eval.syncnet_detect import SyncNetDetector  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - depends on external repo
        raise LatentSyncDependencyError(
            "Failed to import LatentSync evaluation modules. Ensure that all "
            "dependencies listed in the LatentSync repository are installed."
        ) from exc

    return SyncNetEval, SyncNetDetector


def _run_syncnet(
    syncnet,
    detector,
    video_path: str,
    temp_root: Path,
    min_track: int,
) -> Tuple[int, float, int]:
    """Execute the SyncNet pipeline for a single video file."""

    detect_dir = temp_root / "detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    detector.detect_results_dir = str(detect_dir)
    detector(video_path=video_path, min_track=min_track)

    crop_dir = detect_dir / "crop"
    crop_videos = sorted(crop_dir.glob("*.mp4"))
    if not crop_videos:
        raise RuntimeError(f"Face not detected in {video_path}")

    offsets: List[int] = []
    confidences: List[float] = []

    eval_temp = temp_root / "syncnet_temp"
    for crop_video in crop_videos:
        av_offset, _, conf = syncnet.evaluate(video_path=str(crop_video), temp_dir=str(eval_temp))
        offsets.append(int(av_offset))
        confidences.append(float(conf))

    shutil.rmtree(detect_dir, ignore_errors=True)
    return int(fmean(offsets)), float(fmean(confidences)), len(crop_videos)


def evaluate(
    data_list: Iterable[VideoData],
    device: str = "cuda",
    batch_size: int | None = None,
    model_args: Dict | None = None,
    sampling: int | None = None,
) -> List[VideoData]:
    """Evaluate lip-sync quality for each :class:`VideoData` using LatentSync."""

    del batch_size, sampling  # These parameters are not used by this subject.
    model_args = model_args or {}

    repo_dir_arg = model_args.get("repo_dir")
    repo_dir = _prepare_latentsync_repo(
        repo_dir=Path(repo_dir_arg) if repo_dir_arg else None,
        force_clone=bool(model_args.get("force_clone", False)),
    )

    _ensure_checkpoints(repo_dir, repo_id=model_args.get("huggingface_repo_id", DEFAULT_HF_REPO_ID))
    SyncNetEval, SyncNetDetector = _import_latentsync_modules(repo_dir)

    syncnet = SyncNetEval(device=device)
    checkpoint = model_args.get(
        "syncnet_checkpoint",
        repo_dir / REQUIRED_CHECKPOINTS[0],
    )
    syncnet.loadParameters(str(checkpoint))

    detector = SyncNetDetector(device=device, detect_results_dir=str(repo_dir / "detect_results"))

    min_track = int(model_args.get("min_track", 50))
    subject_name = model_args.get("subject_name", SUBJECT_NAME)

    for video_data in data_list:
        with tempfile.TemporaryDirectory(prefix="latentsync_eval_") as tmpdir:
            temp_root = Path(tmpdir)
            try:
                av_offset, confidence, crops = _run_syncnet(
                    syncnet,
                    detector,
                    video_data.video_path,
                    temp_root=temp_root,
                    min_track=min_track,
                )
                result = {
                    "confidence": confidence,
                    "av_offset": av_offset,
                    "num_crops": crops,
                }
            except Exception as exc:  # pragma: no cover - depends on runtime data
                result = {"error": str(exc)}

        video_data.register_result(subject_name, result)

    return list(data_list)
