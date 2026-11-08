"""대화 모델 불러오기 옵션: GPU가 있으면 GPU에 올리고 int8로 줄인다. GPU가 없으면 CPU에 올린다."""

import torch

from app.services.companion.llm_client import llm_load_kwargs


def test_gpu_with_int8_loads_on_gpu_in_8bit():
    kw = llm_load_kwargs(torch.bfloat16, cuda_available=True, quant="int8")
    assert kw["device_map"] == {"": 0}
    assert kw["quantization_config"].load_in_8bit is True
    assert kw["local_files_only"] is True


def test_gpu_without_quant_loads_on_gpu_full_precision():
    kw = llm_load_kwargs(torch.bfloat16, cuda_available=True, quant="none")
    assert kw["device_map"] == {"": 0}
    assert "quantization_config" not in kw


def test_no_gpu_loads_on_cpu_without_device_map():
    kw = llm_load_kwargs(torch.bfloat16, cuda_available=False, quant="int8")
    assert "device_map" not in kw
    assert "quantization_config" not in kw
