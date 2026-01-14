# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the VllmBackendWithCache class.
"""

import pytest
import torch

from vllm.compilation.backends import VllmBackendWithCache
from vllm.compilation.caching import VllmSerializableFunction
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.utils.torch_utils import is_torch_equal_or_newer


def make_vllm_config() -> VllmConfig:
    """Create a test VllmConfig."""
    return VllmConfig(
        compilation_config=CompilationConfig(
            level=CompilationMode.VLLM_COMPILE,
        )
    )


@pytest.mark.skipif(
    not is_torch_equal_or_newer("2.10.0.dev"), reason="requires torch 2.10"
)
class TestVllmBackendWithCache:
    """Test the VllmBackendWithCache class."""

    def test_init_empty_submodules(self):
        """Test initialization of VllmBackendWithCache with no submodules."""
        inductor_compiled_artifacts = (
            VllmSerializableFunction.InductorCompiledArtifacts()
        )
        vllm_config = make_vllm_config()

        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=inductor_compiled_artifacts,
            vllm_config=vllm_config,
            prefix="test",
            submod_names=[],
        )

        assert backend.inductor_compiled_artifacts == inductor_compiled_artifacts
        assert backend.vllm_config == vllm_config
        assert backend.prefix == "test"
        assert backend.submod_names == []
        assert backend.compiled_callables == {}

    def test_build_dispatch_callable_empty_cache(self):
        """Test building dispatch callable with an empty cache and no submodules."""
        inductor_compiled_artifacts = (
            VllmSerializableFunction.InductorCompiledArtifacts()
        )
        vllm_config = make_vllm_config()

        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=inductor_compiled_artifacts,
            vllm_config=vllm_config,
            prefix="test",
            submod_names=[],
        )

        assert backend.compiled_callables == {}

    def test_call_empty_submodules(self):
        """Test creating split_gm with empty submod_names."""
        backend = VllmBackendWithCache(
            inductor_compiled_artifacts=VllmSerializableFunction.InductorCompiledArtifacts(),
            vllm_config=make_vllm_config(),
            prefix="test",
            submod_names=[],
            sym_shape_indices_map={},
            returns_tuple_map={},
        )

        import torch.fx as fx

        gm = fx.GraphModule(torch.nn.Module(), fx.Graph())

        assert backend.create_split_gm_from_cache(gm) is gm
