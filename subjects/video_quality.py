"""FineVQ video quality evaluation subject.

This module integrates the third-party FineVQ project to score the
perceived quality of generated videos.  The evaluation procedure is:

1. Automatically clone (and optionally install) the FineVQ repository.
2. Prepare a temporary workspace that exposes the evaluated videos in the
   structure expected by FineVQ.
3. Invoke the official inference entrypoint provided by FineVQ via
   ``torch.distributed.run`` so that their distributed initialisation logic
   remains untouched.
4. Collect the generated CSV results and register them on each
   :class:`~video.VideoData` instance under the ``"video_quality"`` key.

The goal is to keep the orchestration logic lightweight while delegating the
actual quality prediction to the upstream project.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from video import VideoData


FINEVQ_REPO_URL = "https://github.com/IntMeGroup/FineVQ.git"
DEFAULT_REPO_SUBDIR = Path("third_party") / "FineVQ"


def _prepare_finevq_repo(
    repo_dir: Path | None = None,
    repo_url: str = FINEVQ_REPO_URL,
    install_dependencies: bool = True,
    force_clone: bool = False,
) -> Path:
    """Clone the FineVQ repository and (optionally) install dependencies.

    Parameters
    ----------
    repo_dir:
        Target directory where the repository should live.  When ``None`` the
        default ``third_party/FineVQ`` directory inside the current project is
        used.
    repo_url:
        Git URL of the FineVQ repository.
    install_dependencies:
        When ``True`` the repository requirements will be installed via ``pip``
        once per checkout (a marker file is written to avoid redundant
        installs).
    force_clone:
        Remove the existing checkout and clone it again when ``True``.  This is
        helpful if the repository becomes corrupted or needs to be refreshed.
    """

    repo_dir = repo_dir or (Path(__file__).resolve().parent.parent / DEFAULT_REPO_SUBDIR)
    repo_dir = repo_dir.resolve()
    if force_clone and repo_dir.exists():
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    if install_dependencies:
        marker = repo_dir / ".video_eval_installed"
        if not marker.exists():
            requirements = repo_dir / "requirements.txt"
            if requirements.exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                    check=True,
                )
            marker.write_text("installed\n", encoding="utf-8")

    return repo_dir


def _ensure_link(source: Path, destination: Path) -> None:
    """Create a symbolic link and fall back to copying when unavailable."""

    try:
        if destination.exists():
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        os.symlink(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _build_video_workspace(
    videos: Iterable[Tuple[str, Path]],
    workspace: Path,
) -> Tuple[Path, Path]:
    """Expose the target videos in a temporary directory.

    Parameters
    ----------
    videos:
        Iterable of ``(alias, path)`` tuples.  The alias is the filename that
        FineVQ should see while the path points to the actual video file.
    workspace:
        Temporary directory allocated for the evaluation run.

    Returns
    -------
    A tuple ``(videos_dir, names_txt)`` where ``videos_dir`` contains the
    symlinks/copies and ``names_txt`` is the text file consumed by FineVQ.
    """

    videos_dir = workspace / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    names_file = workspace / "video_names.txt"

    with names_file.open("w", encoding="utf-8") as handle:
        for alias, original in videos:
            destination = videos_dir / alias
            _ensure_link(original, destination)
            handle.write(f"{alias}\n")

    return videos_dir, names_file


def _write_meta_file(root: Path, names_txt: Path, destination: Path, count: int) -> Path:
    """Write the FineVQ meta configuration file."""

    meta = {
        "video_eval": {
            "root": str(root),
            "video_name_txt": str(names_txt),
            "data_augment": False,
            "repeat_time": 1,
            "length": count,
        }
    }
    destination.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def _invoke_finevq(
    repo_dir: Path,
    meta_path: Path,
    output_dir: Path,
    output_csv: Path,
    metrics_path: Path,
    per_device_batch_size: int,
    nproc_per_node: int,
    use_bf16: bool,
    extra_env: Dict[str, str] | None,
    extra_args: List[str] | None,
) -> None:
    """Run the FineVQ inference script via ``torch.distributed.run``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _find_free_port() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return str(sock.getsockname()[1])

    master_port = os.environ.get("MASTER_PORT") or _find_free_port()
    script_path = repo_dir / "internvl" / "train" / "inference.py"

    args = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        master_port,
        str(script_path),
        "--model_name_or_path",
        "IntMeGroup/FineVQ_score",
        "--conv_style",
        "internlm2-chat",
        "--output_dir",
        str(output_dir),
        "--meta_path",
        str(meta_path),
        "--overwrite_output_dir",
        "True",
        "--force_image_size",
        "448",
        "--max_dynamic_patch",
        "6",
        "--down_sample_ratio",
        "0.5",
        "--drop_path_rate",
        "0.1",
        "--freeze_llm",
        "True",
        "--freeze_mlp",
        "True",
        "--freeze_backbone",
        "True",
        "--vision_select_layer",
        "-1",
        "--dataloader_num_workers",
        "4",
        "--bf16",
        "True" if use_bf16 else "False",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        str(per_device_batch_size),
        "--gradient_accumulation_steps",
        "1",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "no",
        "--learning_rate",
        "0.0",
        "--weight_decay",
        "0.0",
        "--warmup_ratio",
        "0.0",
        "--logging_steps",
        "1",
        "--max_seq_length",
        "4096",
        "--do_train",
        "False",
        "--grad_checkpoint",
        "False",
        "--group_by_length",
        "False",
        "--dynamic_image_size",
        "True",
        "--use_thumbnail",
        "True",
        "--ps_version",
        "v2",
        "--output_file",
        str(output_csv),
        "--metrics_file",
        str(metrics_path),
    ]

    if extra_args:
        args.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("LAUNCHER", "pytorch")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    repo_python_path = str(repo_dir)
    env["PYTHONPATH"] = (
        f"{repo_python_path}{os.pathsep}{env['PYTHONPATH']}"
        if "PYTHONPATH" in env and env["PYTHONPATH"]
        else repo_python_path
    )
    if extra_env:
        env.update(extra_env)

    subprocess.run(args, check=True, cwd=repo_dir, env=env)


def _collect_scores(csv_path: Path) -> Dict[str, float]:
    """Load the FineVQ CSV output into a dictionary."""

    if not csv_path.exists():
        raise FileNotFoundError(f"FineVQ output CSV not found: {csv_path}")

    results: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("video_name", "").strip()
            score_str = row.get("pred_score")
            if not name:
                continue
            try:
                score = float(score_str)
            except (TypeError, ValueError):
                continue
            results[name] = score
    return results


def evaluate(
    data_list: List[VideoData],
    device: str = "cuda",
    batch_size: int = 1,
    sampling: int = 0,
    model_args: Dict | None = None,
) -> List[VideoData]:
    """Run FineVQ video quality scoring on the provided videos."""

    if not data_list:
        return data_list

    model_args = model_args or {}
    repo_dir = _prepare_finevq_repo(
        repo_dir=Path(model_args.get("repo_dir", "")) if model_args.get("repo_dir") else None,
        repo_url=model_args.get("repo_url", FINEVQ_REPO_URL),
        install_dependencies=model_args.get("install_dependencies", True),
        force_clone=model_args.get("force_clone", False),
    )

    use_bf16 = bool(model_args.get("use_bf16", False))
    nproc_per_node = int(model_args.get("nproc_per_node", 1))
    extra_env = dict(model_args.get("env") or {})
    device = device or "cuda"
    if device.startswith("cuda"):
        if ":" in device:
            extra_env.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])
    elif device == "cpu":
        extra_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    extra_args = model_args.get("extra_args") or []

    with tempfile.TemporaryDirectory(prefix="finevq_eval_") as tmp_dir:
        workspace = Path(tmp_dir)
        alias_pairs: List[Tuple[str, Path]] = []
        for index, video in enumerate(data_list):
            video_path = Path(video.video_path)
            alias = f"{index:04d}_{video_path.name}"
            alias_pairs.append((alias, video_path))

        videos_dir, names_txt = _build_video_workspace(alias_pairs, workspace)
        meta_path = _write_meta_file(
            videos_dir, names_txt, workspace / "meta.json", len(alias_pairs)
        )

        output_dir = workspace / "outputs"
        output_csv = workspace / "finevq_results.csv"
        metrics_file = workspace / "finevq_metrics.txt"

        _invoke_finevq(
            repo_dir=repo_dir,
            meta_path=meta_path,
            output_dir=output_dir,
            output_csv=output_csv,
            metrics_path=metrics_file,
            per_device_batch_size=max(1, int(model_args.get("per_device_batch_size", batch_size))),
            nproc_per_node=nproc_per_node,
            use_bf16=use_bf16,
            extra_env=extra_env,
            extra_args=extra_args,
        )

        score_map = _collect_scores(output_csv)

    for (alias, _), video in zip(alias_pairs, data_list):
        score = score_map.get(alias)
        results = {
            "video_quality_score": score,
            "details": {
                "alias": alias,
                "source_video": video.video_path,
            },
        }
        video.register_result("video_quality", results)

    return data_list

