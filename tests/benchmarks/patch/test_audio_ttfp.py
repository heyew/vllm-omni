# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Focused audio-TTFP event validity tests without optional mock plugins."""

import base64
import io
import json
import wave

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.patch import patch as benchmark_patch

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class _MockResponse:
    def __init__(self, chunks: list[bytes]):
        self.status = 200
        self.reason = "OK"
        self._chunks = chunks
        self.content = self

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class _MockSession:
    def __init__(self, response: _MockResponse):
        self._response = response

    def post(self, *args, **kwargs):
        return self._response


def _sse_chunk(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


def _request_input() -> RequestFuncInput:
    return RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )


def _wav_b64(pcm_bytes: bytes, *, sample_rate: int = 24_000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm_bytes)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.asyncio
async def test_audio_ttfp_ignores_zero_frame_wav_header(monkeypatch):
    """A WAV container without PCM frames is metadata, not first audio."""
    chunks = [
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(b"")}}],
                "modality": "audio",
            }
        ),
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(b"\x00\x00" * 4)}}],
                "modality": "audio",
            }
        ),
        b"data: [DONE]\n\n",
    ]
    timestamps = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(benchmark_patch.time, "perf_counter", timestamps.__next__)

    output = await benchmark_patch.async_request_openai_chat_omni_completions(
        _request_input(),
        _MockSession(_MockResponse(chunks)),
    )

    assert output.success is True
    assert output.audio_ttfp == pytest.approx(2.0)
    assert output.audio_frames == 4


@pytest.mark.asyncio
async def test_zero_frame_wav_after_pcm_does_not_extend_audio_generation(monkeypatch):
    """A trailing header-only event must not inflate RTF after real audio."""
    one_second_pcm = b"\x00\x00" * 24_000
    chunks = [
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(one_second_pcm)}}],
                "modality": "audio",
            }
        ),
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(b"")}}],
                "modality": "audio",
            }
        ),
        b"data: [DONE]\n\n",
    ]
    timestamps = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(benchmark_patch.time, "perf_counter", timestamps.__next__)

    output = await benchmark_patch.async_request_openai_chat_omni_completions(
        _request_input(),
        _MockSession(_MockResponse(chunks)),
    )

    assert output.audio_ttfp == pytest.approx(1.0)
    assert output.audio_frames == 24_000
    assert output.audio_duration == pytest.approx(1.0)
    assert output.audio_rtf == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_inconsistent_wav_chunk_does_not_extend_audio_generation(monkeypatch):
    """A dropped WAV chunk cannot affect the last-valid-audio timestamp."""
    chunks = [
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(b"\x00\x00" * 24_000)}}],
                "modality": "audio",
            }
        ),
        _sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "content": _wav_b64(
                                b"\x00\x00" * 16_000,
                                sample_rate=16_000,
                            )
                        }
                    }
                ],
                "modality": "audio",
            }
        ),
        b"data: [DONE]\n\n",
    ]
    timestamps = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(benchmark_patch.time, "perf_counter", timestamps.__next__)

    output = await benchmark_patch.async_request_openai_chat_omni_completions(
        _request_input(),
        _MockSession(_MockResponse(chunks)),
    )

    assert output.audio_ttfp == pytest.approx(1.0)
    assert output.audio_frames == 24_000
    assert output.audio_duration == pytest.approx(1.0)
    assert output.audio_rtf == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_only_zero_frame_wav_never_produces_audio_metrics(monkeypatch):
    """A response containing only a WAV header has no usable audio."""
    chunks = [
        _sse_chunk(
            {
                "choices": [{"delta": {"content": _wav_b64(b"")}}],
                "modality": "audio",
            }
        ),
        b"data: [DONE]\n\n",
    ]
    timestamps = iter([100.0, 101.0])
    monkeypatch.setattr(benchmark_patch.time, "perf_counter", timestamps.__next__)

    output = await benchmark_patch.async_request_openai_chat_omni_completions(
        _request_input(),
        _MockSession(_MockResponse(chunks)),
    )

    assert output.audio_ttfp == 0.0
    assert output.audio_frames == 0
    assert output.audio_duration == 0.0
    assert output.audio_rtf == 0.0
