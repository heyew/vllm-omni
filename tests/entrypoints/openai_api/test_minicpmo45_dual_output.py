from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _serving_chat_with_stages(*stage_args):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = SimpleNamespace(
        stage_configs=[SimpleNamespace(engine_args=engine_args) for engine_args in stage_args]
    )
    instance.use_harmony = False
    return instance


def _minicpmo_serving_chat():
    return _serving_chat_with_stages(
        {
            "model_arch": "MiniCPMO45OmniForConditionalGeneration",
            "model_stage": "llm",
        },
        {
            "model_arch": "MiniCPMO45OmniForConditionalGeneration",
            "model_stage": "tts",
        },
        {
            "model_arch": "MiniCPMO45Code2Wav",
            "model_stage": "code2wav",
        },
    )


def _simple_request(**overrides):
    values = {
        "modalities": ["text", "audio"],
        "tools": None,
        "echo": False,
        "logprobs": False,
        "return_token_ids": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_minicpmo45_audio_stream_keeps_thinker_and_talker_delta():
    instance = _minicpmo_serving_chat()
    params = [SamplingParams(max_tokens=8) for _ in range(3)]
    for param in params:
        param.output_kind = RequestOutputKind.FINAL_ONLY

    result = instance._fix_minicpmo45_audio_stream_output_kinds(params, ["text", "audio"])

    assert result[0].output_kind == RequestOutputKind.DELTA
    assert result[1].output_kind == RequestOutputKind.DELTA
    assert result[2].output_kind == RequestOutputKind.FINAL_ONLY


def test_minicpmo45_text_only_request_does_not_override_output_kinds():
    instance = _minicpmo_serving_chat()
    params = [SamplingParams(max_tokens=8)]
    params[0].output_kind = RequestOutputKind.FINAL_ONLY

    result = instance._fix_minicpmo45_audio_stream_output_kinds(params, ["text"])

    assert result[0].output_kind == RequestOutputKind.FINAL_ONLY


def test_first_and_final_text_delta_coalescer_preserves_complete_text():
    instance = _serving_chat_with_stages()
    emitted = []
    first_emitted = False
    pending = ""

    for delta, finished in [("昇", False), ("腾", False), ("AI", False), ("芯片", True)]:
        current, first_emitted, pending = instance._coalesce_first_and_final_text_delta(
            delta,
            finished=finished,
            first_emitted=first_emitted,
            pending_text=pending,
        )
        if current is not None:
            emitted.append(current)

    assert emitted == ["昇", "腾AI芯片"]
    assert "".join(emitted) == "昇腾AI芯片"
    assert pending == ""


def test_first_and_final_text_delta_coalescer_flushes_on_empty_terminal_delta():
    instance = _serving_chat_with_stages()
    first, first_emitted, pending = instance._coalesce_first_and_final_text_delta(
        "first", finished=False, first_emitted=False, pending_text=""
    )
    middle, first_emitted, pending = instance._coalesce_first_and_final_text_delta(
        " middle", finished=False, first_emitted=first_emitted, pending_text=pending
    )
    final, first_emitted, pending = instance._coalesce_first_and_final_text_delta(
        "", finished=True, first_emitted=first_emitted, pending_text=pending
    )

    assert first == "first"
    assert middle is None
    assert final == " middle"
    assert pending == ""


def test_simple_minicpmo_audio_request_enables_coalescing():
    instance = _minicpmo_serving_chat()

    assert instance._should_coalesce_minicpmo45_text_deltas(_simple_request(), None)


@pytest.mark.parametrize(
    ("request_overrides", "reasoning_parser", "use_harmony"),
    [
        ({"modalities": ["text"]}, None, False),
        ({"tools": [object()]}, None, False),
        ({"echo": True}, None, False),
        ({"logprobs": True}, None, False),
        ({"return_token_ids": True}, None, False),
        ({}, object(), False),
        ({}, None, True),
    ],
)
def test_advanced_or_non_audio_requests_bypass_coalescing(
    request_overrides,
    reasoning_parser,
    use_harmony,
):
    instance = _minicpmo_serving_chat()
    instance.use_harmony = use_harmony

    assert not instance._should_coalesce_minicpmo45_text_deltas(
        _simple_request(**request_overrides),
        reasoning_parser,
    )


def test_non_minicpmo_pipeline_bypasses_coalescing():
    instance = _serving_chat_with_stages(
        {"model_arch": "OtherOmniModel", "model_stage": "llm"}
    )

    assert not instance._should_coalesce_minicpmo45_text_deltas(_simple_request(), None)
