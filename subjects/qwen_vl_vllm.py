"""Qwen3-VL evaluation subject powered by vLLM.

This subject loads a multi-modal Large Language Model through vLLM and
queries it with the generated video together with a textual prompt.  The
prompt can be produced from a user supplied template which receives the
per-video text file contents and a few metadata fields.  The model
response is then stored on each :class:`~video.VideoData` instance under
the ``"qwen_vl_vllm"`` key.

Example ``--model_args`` value::

    '{
        "model_name": "Qwen/Qwen3-VL-2B-Thinking",
        "prompt_template_path": "./templates/video_eval.txt",
        "system_prompt": "You are a video critic.",
        "sampling_options": {"max_tokens": 512, "temperature": 0.0}
    }'

The template receives the following fields::

    {video_text}         -> Contents of ``VideoData.text_path`` (empty when missing).
    {video_filename}     -> Basename of the video without extension.
    {video_path}         -> Absolute path to the video file.
    {audio_path}         -> Absolute path to the reference audio or empty string.
    {text_path}          -> Absolute path to the associated text file or empty string.

Additional placeholders can be supported by providing default values via
``template_variables`` in ``model_args``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from tqdm import tqdm

from video import VideoData

try:  # pragma: no cover - optional dependency
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - raise a helpful error when missing
    raise ImportError(
        "The 'vllm' package is required for the 'qwen_vl_vllm' subject. "
        "Please install it with `pip install vllm` (optionally including the "
        "appropriate extras for your hardware)."
    ) from exc


DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"
DEFAULT_PLACEHOLDER = "{video_text}"


class _SafeDict(dict):
    """Dictionary that preserves unknown placeholders during ``format_map``."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return "{" + key + "}"


@dataclass
class TemplateRenderer:
    """Render prompts using ``str.format`` with sensible defaults."""

    template: str
    base_context: Mapping[str, Any]

    def render(self, extra_context: Mapping[str, Any]) -> str:
        context = _SafeDict(self.base_context)
        context.update(extra_context)
        return self.template.format_map(context)


def _load_template(model_args: Mapping[str, Any]) -> TemplateRenderer:
    template_text: Optional[str] = model_args.get("prompt_template")
    template_path = model_args.get("prompt_template_path")
    if template_text and template_path:
        raise ValueError(
            "Specify only one of 'prompt_template' or 'prompt_template_path'."
        )
    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(
                f"Prompt template file '{template_file}' does not exist."
            )
        template_text = template_file.read_text(encoding="utf-8")

    if template_text is None:
        template_text = DEFAULT_PLACEHOLDER

    base_variables = model_args.get("template_variables", {})
    if not isinstance(base_variables, Mapping):
        raise TypeError("'template_variables' must be a mapping of placeholder defaults.")

    return TemplateRenderer(template=template_text, base_context=base_variables)


def _prepare_messages(
    video_path: Path,
    prompt: str,
    system_prompt: Optional[str],
    include_video: bool
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            }
        )

    user_content: List[Dict[str, Any]] = []
    if include_video:
        user_content.append(
            {
                "type": "video_url",
                "video_url": {"url": f"file://{video_path}"},
            }
        )

    if prompt:
        user_content.append({"type": "text", "text": prompt})

    if not user_content:
        raise ValueError("At least one of video or prompt content must be provided.")

    messages.append({"role": "user", "content": user_content})
    return messages


def _build_sampling_params(options: Mapping[str, Any]) -> SamplingParams:
    if options and not isinstance(options, Mapping):
        raise TypeError("'sampling_options' must be a mapping of SamplingParams arguments.")

    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": 512,
        "temperature": 0.0,
    }
    if options:
        sampling_kwargs.update(options)

    return SamplingParams(**sampling_kwargs)


def _collect_llm_kwargs(model_args: Mapping[str, Any]) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = dict(model_args.get("engine_options", {}))
    if not isinstance(llm_kwargs, MutableMapping):
        raise TypeError("'engine_options' must be a mapping of keyword arguments.")

    for key in ("tensor_parallel_size", "dtype", "gpu_memory_utilization", "max_model_len", "quantization", "revision", "enforce_eager"):
        if key in model_args and key not in llm_kwargs:
            llm_kwargs[key] = model_args[key]

    if "trust_remote_code" not in llm_kwargs:
        llm_kwargs["trust_remote_code"] = model_args.get("trust_remote_code", True)

    return llm_kwargs


def _read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def evaluate(
    data_list: Iterable[VideoData],
    device: str = "cuda",
    batch_size: int = 1,
    sampling: int = 16,
    model_args: Optional[Dict[str, Any]] = None,
) -> List[VideoData]:
    """Run the Qwen3-VL model via vLLM on each video entry."""

    del device, batch_size, sampling  # Parameters kept for API compatibility.

    model_args = model_args or {}
    model_name = model_args.get("model_name", DEFAULT_MODEL_NAME)
    include_video = model_args.get("include_video", True)
    system_prompt = model_args.get("system_prompt")
    keep_full_messages = model_args.get("store_messages", False)

    template = _load_template(model_args)
    sampling_params = _build_sampling_params(model_args.get("sampling_options", {}))
    llm_kwargs = _collect_llm_kwargs(model_args)

    llm = LLM(model=model_name, allowed_local_media_path="*", **llm_kwargs)

    results: List[VideoData] = []
    for data in tqdm(data_list, desc="Evaluating Qwen3-VL via vLLM"):
        video_path = Path(data.video_path).resolve()
        text_content = _read_text_file(data.text_path)
        prompt = template.render(
            {
                "video_text": text_content,
                "video_filename": data.video_filename,
                "video_path": str(video_path),
                "audio_path": str(Path(data.audio_path).resolve()) if data.audio_path else "",
                "text_path": str(Path(data.text_path).resolve()) if data.text_path else "",
            }
        ).strip()

        messages = _prepare_messages(video_path, prompt, system_prompt, include_video)

        raw_output = llm.chat(messages=messages, sampling_params=sampling_params)
        responses = [
            {
                "text": output.outputs[0].text if output.outputs else "",
                "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                "usage": {
                    "prompt_tokens": getattr(output, "prompt_token_ids", None),
                    "completion_tokens": getattr(output.outputs[0], "token_ids", None)
                    if output.outputs
                    else None,
                },
            }
            for output in raw_output
        ]

        if not responses:
            raise RuntimeError("vLLM returned no outputs for the given request.")

        primary_response = responses[0]["text"]
        record: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "response": primary_response,
        }

        if keep_full_messages:
            record["messages"] = messages
            record["raw_outputs"] = responses

        data.register_result("qwen_vl_vllm", record)
        results.append(data)

    return results
