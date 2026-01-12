# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import inspect
import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any, Literal
from unittest.mock import patch

import torch
from torch.utils import _pytree as pytree

import vllm.envs as envs
from vllm.compilation.compiler_interface import get_inductor_factors
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.utils import hash_factors
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash

try:
    from torch._dynamo.aot_compile import SerializableCallable
except ImportError:
    SerializableCallable = object

assert isinstance(SerializableCallable, type)

logger = init_logger(__name__)


class VllmSerializableFunction(SerializableCallable):  # type: ignore[misc]
    """
    A wrapper around a compiled function by vllm. It will forward the tensor
    inputs to the compiled function and return the result.
    It also implements a serialization interface to support PyTorch's precompile
    with custom backend, so that we can save and load the compiled function on
    disk. There's no need to wrap around the compiled function if we don't want
    to serialize them in particular cases.
    Right now serialization for the custom backend is done via
    serializing the Dynamo fx graph plus example inputs.
    """

    class InductorCompiledArtifacts:
        def __init__(self):
            # dict from submodule name to byte hash
            self.submodule_bytes = {}
            # dict from byte hash to bytes
            self.submodule_bytes_store = {}
            # dict from byte hash to loaded module
            self.loaded_submodule_store = {}

        def insert(self, submod_name: str, shape: str, entry: bytes):
            hasher = hashlib.sha256()
            hasher.update(entry)
            hex_digest = hasher.hexdigest()
            self.submodule_bytes[f"{submod_name}_{shape}"] = hex_digest
            if hex_digest not in self.submodule_bytes_store:
                self.submodule_bytes_store[hex_digest] = entry
                logger.debug(
                    "Inserting new inductor artifact for submod %s with shape %s "
                    "(%s bytes) at hash %s",
                    submod_name,
                    shape,
                    len(entry),
                    hex_digest,
                )
            else:
                logger.debug(
                    "Reusing existing inductor cache artifact for submod %s "
                    "with shape %s (%s bytes) at hash %s",
                    submod_name,
                    shape,
                    len(entry),
                    hex_digest,
                )

        def get(self, submod_name: str, shape: str) -> bytes:
            logger.debug(
                "Getting inductor artifact for submod %s with shape %s",
                submod_name,
                shape,
            )
            return self.submodule_bytes_store[
                self.submodule_bytes[f"{submod_name}_{shape}"]
            ]

        def get_loaded(self, submod_name: str, shape: str):
            logger.debug(
                "Getting inductor artifact for submod %s with shape %s",
                submod_name,
                shape,
            )
            return self.loaded_submodule_store[
                self.submodule_bytes[f"{submod_name}_{shape}"]
            ]

        def size_bytes(self) -> int:
            return sum(len(entry) for entry in self.submodule_bytes_store.values())

        def num_artifacts(self) -> int:
            return len(self.submodule_bytes_store)

        def num_entries(self) -> int:
            return len(self.submodule_bytes)

        def load_all(self) -> None:
            import concurrent.futures

            # check if already loaded
            if len(self.loaded_submodule_store) == len(self.submodule_bytes_store):
                return

            from torch._inductor.standalone_compile import AOTCompiledArtifact

            def _load_entry(entry_bytes) -> AOTCompiledArtifact:
                # Unpickle the bundled cache entry first
                entry = pickle.loads(entry_bytes)
                return AOTCompiledArtifact.deserialize(entry)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                entries = list(self.submodule_bytes_store.values())
                loaded_entries = list(executor.map(_load_entry, entries))

            for i, k in enumerate(self.submodule_bytes_store.keys()):
                self.loaded_submodule_store[k] = loaded_entries[i]

            logger.debug("loaded all %s submodules", self.num_artifacts())

        def __getstate__(self):
            return {
                "submodule_bytes": self.submodule_bytes,
                "submodule_bytes_store": self.submodule_bytes_store,
            }

        def __setstate__(self, state):
            self.submodule_bytes = state["submodule_bytes"]
            self.submodule_bytes_store = state["submodule_bytes_store"]
            self.loaded_submodule_store = {}

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        prefix: str,
        optimized_call: Callable[..., Any],
        is_encoder: bool = False,
        submod_names: list[str] | None = None,
        collect_inductor_compiled_artifacts: Callable | None = None,
        # reference to VllmBackend for accessing split_gm and sym_tensor_indices
        # not serialized - recreated during deserialization
        vllm_backend: Any | None = None,
        # stored separately for serialization since vllm_backend isn't serialized
        sym_tensor_indices: list[int] | None = None,
    ) -> None:
        assert isinstance(graph_module, torch.fx.GraphModule)
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.prefix = prefix
        self.optimized_call = optimized_call
        self.is_encoder = is_encoder
        self.shape_env = None
        self.submod_names = submod_names or []
        self.collect_inductor_compiled_artifacts = collect_inductor_compiled_artifacts
        self.vllm_backend = vllm_backend
        self.sym_tensor_indices = sym_tensor_indices
        sym_input = next(
            (i for i in self.example_inputs if isinstance(i, torch.SymInt)), None
        )
        if sym_input is not None:
            self.shape_env = sym_input.node.shape_env

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimized_call(*args, **kwargs)

    @classmethod
    def serialize_compile_artifacts(
        cls, compiled_fn: "VllmSerializableFunction"
    ) -> bytes:
        import sympy
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler, Options

        state = compiled_fn.__dict__.copy()
        state.pop("optimized_call")
        state.pop("shape_env")
        state.pop("vllm_backend", None)
        for node in state["graph_module"].graph.nodes:
            node.meta.pop("source_fn_stack", None)
            node.meta.pop("nn_module_stack", None)
        for name, submod in state["graph_module"].named_children():
            if hasattr(submod, "graph"):
                for node in submod.graph.nodes:
                    node.meta.pop("source_fn_stack", None)
                    node.meta.pop("nn_module_stack", None)

        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(
            self: GraphPickler, obj: Any
        ) -> tuple[Callable[..., Any], tuple[Any, ...]] | Any:
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        if state.get("sym_tensor_indices"):
            # put tensor inputs on meta device since their data
            # isn't needed, yet we need the meta for make_copy_and_call
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )
        else:
            # mask off all tensor inputs since they are large and not needed.
            state["example_inputs"] = pytree.tree_map_only(
                torch.Tensor,
                lambda inp: torch.empty_like(inp, device="meta"),
                state["example_inputs"],
            )
        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["graph_module"] = GraphPickler.dumps(
                state["graph_module"], Options(ops_filter=None)
            )
            state["example_inputs"] = GraphPickler.dumps(state["example_inputs"])

        if compiled_fn.collect_inductor_compiled_artifacts:
            (
                inductor_compiled_artifacts,
                sym_shape_indices_map,
                returns_tuple_map,
            ) = compiled_fn.collect_inductor_compiled_artifacts(
                compiled_fn.submod_names
            )
            state["inductor_compiled_artifacts"] = inductor_compiled_artifacts
            state["sym_shape_indices_map"] = sym_shape_indices_map
            state["returns_tuple_map"] = returns_tuple_map
            state.pop("collect_inductor_compiled_artifacts")
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes) -> "VllmSerializableFunction":
        from torch._guards import TracingContext, tracing
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        state = pickle.loads(data)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        state["graph_module"] = GraphPickler.loads(state["graph_module"], fake_mode)
        state["graph_module"].recompile()
        state["example_inputs"] = GraphPickler.loads(state["example_inputs"], fake_mode)

        # Pop inductor artifact keys from state unconditionally since
        # VllmSerializableFunction.__init__ doesn't accept them
        inductor_compiled_artifacts = state.pop("inductor_compiled_artifacts", None)
        sym_shape_indices_map = state.pop("sym_shape_indices_map", {})
        returns_tuple_map = state.pop("returns_tuple_map", {})

        # Check if we should use VllmBackendWithCache
        if envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS:
            from vllm.compilation.backends import VllmBackendWithCache

            submod_names = state.get("submod_names")

            has_artifacts = (
                inductor_compiled_artifacts is not None
                and hasattr(inductor_compiled_artifacts, "num_artifacts")
                and inductor_compiled_artifacts.num_artifacts() > 0
            )

            num_artifacts = (
                inductor_compiled_artifacts.num_artifacts()
                if inductor_compiled_artifacts
                else 0
            )
            num_submods = len(submod_names) if submod_names else 0

            logger.info(
                "VllmBackendWithCache check: has_artifacts=%s, "
                "num_artifacts=%d, num_submods=%d",
                has_artifacts,
                num_artifacts,
                num_submods,
            )

            if not has_artifacts or not submod_names:
                # Cache doesn't exist yet or is incomplete
                # Fall back to standard compilation path which will populate it
                logger.warning(
                    "VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS is set "
                    "but inductor cache is empty (artifacts=%d) or submod_names "
                    "is missing (len=%d). Falling back to standard compilation "
                    "path to populate cache.",
                    num_artifacts,
                    num_submods,
                )
                # Continue to fallback path below instead of raising
            else:
                # Cache exists, use VllmBackendWithCache
                logger.info(
                    "Loading from VllmBackendWithCache with %d artifacts "
                    "and %d submodules",
                    num_artifacts,
                    num_submods,
                )

                vllm_backend_with_cache = VllmBackendWithCache(
                    inductor_compiled_artifacts=inductor_compiled_artifacts,
                    vllm_config=get_current_vllm_config(),
                    prefix=state["prefix"],
                    submod_names=submod_names,
                    sym_shape_indices_map=sym_shape_indices_map,
                    returns_tuple_map=returns_tuple_map,
                    is_encoder=state.get("is_encoder", False),
                )

                # Get the split_gm from the deserialized state and populate it
                # with piecewise backends from cache
                split_gm = state["graph_module"]

                # Populate split_gm with piecewise backends from cache
                # This replaces the submodules with cached piecewise backends
                populated_split_gm = vllm_backend_with_cache.create_split_gm_from_cache(
                    split_gm
                )

                # Check if we need to create copy_and_call wrapper
                # for cudagraph_copy_inputs
                sym_tensor_indices = state.get("sym_tensor_indices")
                vllm_config = get_current_vllm_config()
                cudagraph_copy_inputs = (
                    vllm_config.compilation_config.cudagraph_copy_inputs
                    if vllm_config.compilation_config
                    else False
                )

                if cudagraph_copy_inputs and sym_tensor_indices:
                    # Import here to avoid circular dependency
                    from vllm.compilation.backends import make_copy_and_call

                    example_inputs = state.get("example_inputs", [])
                    input_buffers = [
                        torch.empty_like(
                            example_inputs[idx], device=vllm_config.device_config.device
                        )
                        for idx in sym_tensor_indices
                    ]

                    optimized_call_with_cache = make_copy_and_call(
                        sym_tensor_indices, input_buffers, populated_split_gm
                    )
                else:
                    optimized_call_with_cache = populated_split_gm

                fn: VllmSerializableFunction = cls(
                    **state,
                    optimized_call=optimized_call_with_cache,
                    vllm_backend=vllm_backend_with_cache,
                )
                logger.info("Successfully created VllmBackendWithCache function")
                return fn

        # Fall back to standard VllmBackend
        from vllm.compilation.backends import VllmBackend

        is_encoder = state.get("is_encoder", False)
        vllm_backend: VllmBackend = VllmBackend(
            get_current_vllm_config(), state["prefix"], is_encoder
        )

        def optimized_call(*example_inputs: Any) -> Any:
            """
            On the first run of the optimized call, we rerun the compiler
            backend which should result in a cache hit. After the backend
            call returns, we just do a one-time replacement of the optimized
            call with the compiled function, so that subsequent calls are on
            the AOT compiled path.
            """
            compile_inputs = [
                inp if inp is not None else example_inputs[i]
                for i, inp in enumerate(fn.example_inputs)
            ]
            with tracing(TracingContext(fake_mode)):
                fn.optimized_call = vllm_backend(
                    state["graph_module"], compile_inputs
                ).optimized_call
            return fn.optimized_call(*example_inputs)

        fn = cls(**state, optimized_call=optimized_call)
        return fn

    @property
    def co_name(self) -> Literal["VllmSerializableFunction"]:
        """
        Used for depyf debugging.
        """
        return "VllmSerializableFunction"


def compilation_config_hash_factors(vllm_config: VllmConfig) -> list[str]:
    factors = []
    # 0. factors come from the env, for example, The values of
    # VLLM_PP_LAYER_PARTITION will affect the computation graph.
    env_hash = hash_factors(envs.compile_factors())
    factors.append(env_hash)

    # 1. factors come from the vllm_config (it mainly summarizes how the
    #    model is created)
    config_hash = vllm_config.compute_hash()
    factors.append(config_hash)

    # 2. inductor factors if applicable
    if envs.VLLM_USE_BACKEND_WITH_INDUCTOR_COMPILED_ARTIFACTS:
        factors.extend(get_inductor_factors())

    return factors


def _compute_code_hash_with_content(file_contents: dict[str, str]) -> str:
    items = list(sorted(file_contents.items(), key=lambda x: x[0]))
    hash_content = []
    for filepath, content in items:
        hash_content.append(filepath)
        if filepath == "<string>":
            # This means the function was dynamically generated, with
            # e.g. exec(). We can't actually check these.
            continue
        hash_content.append(content)
    return safe_hash(
        "\n".join(hash_content).encode(), usedforsecurity=False
    ).hexdigest()


def _compute_code_hash(files: set[str]) -> str:
    logger.debug(
        "Traced files (to be considered for compilation cache):\n%s", "\n".join(files)
    )
    file_contents = {}
    for filepath in files:
        # Skip files that don't exist (e.g., <string>, <frozen modules>, etc.)
        if not os.path.isfile(filepath):
            file_contents[filepath] = ""
        else:
            with open(filepath) as f:
                file_contents[filepath] = f.read()
    return _compute_code_hash_with_content(file_contents)
