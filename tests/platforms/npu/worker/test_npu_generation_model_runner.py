# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused output-routing tests for the Ascend generation runner."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.worker.npu_ar_model_runner import ExecuteModelState
from vllm_omni.platforms.npu.worker.npu_generation_model_runner import (
    NPUGenerationModelRunner,
)
from vllm_omni.utils.mm_outputs import OMIT_OMNI_OUTPUT_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self):
        self.req_ids = ["setup", "audio"]
        self.req_id_to_index = {
            req_id: index for index, req_id in enumerate(self.req_ids)
        }
        self.num_reqs = len(self.req_ids)
        self.vocab_size = 10


def _make_runner():
    runner = object.__new__(NPUGenerationModelRunner)
    runner.execute_model_state = ExecuteModelState(
        scheduler_output=None,
        logits=torch.empty(0),
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=torch.empty(0),
        sample_hidden_states=torch.empty(0),
        aux_hidden_states=None,
        attn_metadata=None,
        positions=torch.empty(0, dtype=torch.int64),
        ec_connector_output=None,
        cudagraph_stats=None,
        batch_desc=None,
        multimodal_outputs={
            "model_outputs": [torch.empty(0), torch.ones(8)],
            "sr": [torch.tensor(24000), torch.tensor(24000)],
            OMIT_OMNI_OUTPUT_KEY: [torch.tensor(True), torch.tensor(False)],
        },
    )
    runner.kv_connector_output = None
    runner.input_batch = _DummyInputBatch()
    runner.requests = {}
    runner._async_chunk = True
    runner.use_async_scheduling = False
    runner.supports_mm_inputs = False
    runner.speculative_config = None
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=False)
    )
    runner.ascend_config = SimpleNamespace(
        profiling_chunk_config=SimpleNamespace(enabled=False)
    )
    runner.dynamic_eplb = False
    runner.debugger = None
    runner.get_omni_connector_output = lambda: None
    return runner


def test_sample_tokens_omits_setup_only_but_keeps_sibling_audio():
    """A setup row must disappear without suppressing its audio sibling."""
    output = NPUGenerationModelRunner.sample_tokens(_make_runner(), grammar_output=None)

    assert output.inter_stage_outputs is None
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[0] is None
    assert output.multimodal_outputs[1] is not None
    assert output.multimodal_outputs[1]["model_outputs"].numel() == 8
    assert output.multimodal_outputs[1]["sr"].item() == 24000
    assert OMIT_OMNI_OUTPUT_KEY not in output.multimodal_outputs[1]
