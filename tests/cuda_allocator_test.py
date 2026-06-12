import importlib
import sys
from contextlib import nullcontext
from types import ModuleType
import pytest

class TestCudaAllocator():
    def fresh_cuda_module(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        monkeypatch.delitem(sys.modules, "cbclib_v2.cuda", raising=False)
        return importlib.import_module("cbclib_v2.cuda")

    def remove_jax_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in list(sys.modules):
            if name == "jax" or name.startswith("jax."):
                monkeypatch.delitem(sys.modules, name, raising=False)

    def jax_init_warning(self, cuda: ModuleType):
        if cuda.get_allocator_config()["jax_initialized"]:
            return pytest.warns(RuntimeWarning, match="after JAX backend initialization")
        return nullcontext()

    @pytest.fixture
    def cuda(self, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
        return self.fresh_cuda_module(monkeypatch)

    def test_import_skips_jax(self, monkeypatch: pytest.MonkeyPatch):
        self.remove_jax_modules(monkeypatch)
        cuda = self.fresh_cuda_module(monkeypatch)

        assert all(name != "jax" and not name.startswith("jax.") for name in sys.modules)
        assert cuda.get_allocator_config()["jax_initialized"] is False

    def test_jax_allocator_env(self, cuda: ModuleType, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TF_GPU_ALLOCATOR", raising=False)

        with self.jax_init_warning(cuda):
            cuda.set_jax_allocator("cuda_malloc_async")
        assert cuda.os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
        assert cuda.get_allocator_config()["jax_allocator"] == "cuda_malloc_async"

        with self.jax_init_warning(cuda):
            cuda.set_jax_allocator("default")
        assert "TF_GPU_ALLOCATOR" not in cuda.os.environ
        assert cuda.get_allocator_config()["jax_allocator"] == "default"

    def test_jax_allocator_preserves_env(self, cuda: ModuleType,
                                         monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TF_GPU_ALLOCATOR", "platform")

        with self.jax_init_warning(cuda):
            cuda.set_jax_allocator("cuda_malloc_async")
        assert cuda.os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"

        with self.jax_init_warning(cuda):
            cuda.set_jax_allocator("default")
        assert cuda.os.environ["TF_GPU_ALLOCATOR"] == "platform"

    def test_jax_limit_rejects_async(self, cuda: ModuleType):
        with self.jax_init_warning(cuda):
            cuda.set_jax_allocator("cuda_malloc_async")

        with pytest.raises(RuntimeError, match="JAX memory limits"):
            cuda.set_jax_limit(0.5)

    def test_jax_limit_fraction(self, cuda: ModuleType, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)

        with self.jax_init_warning(cuda):
            cuda.set_jax_limit("40%")
        assert cuda.os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.4"
        assert cuda.get_allocator_config()["jax_limit"] == 0.4

        with self.jax_init_warning(cuda):
            cuda.set_jax_limit(None)
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in cuda.os.environ
        assert cuda.get_allocator_config()["jax_limit"] is None

    def test_strict_after_jax_init(self, cuda: ModuleType, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cuda, "_is_jax_backend_initialized", lambda: True)

        with pytest.raises(RuntimeError, match="after JAX backend initialization"):
            cuda.set_jax_allocator("cuda_malloc_async", strict=True)

    def test_strict_before_jax_init(self, cuda: ModuleType, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "jax", ModuleType("jax"))
        monkeypatch.setattr(cuda, "_is_jax_backend_initialized", lambda: False)

        cuda.set_jax_allocator("cuda_malloc_async", strict=True)

    def test_jax_init_false_without_jax(self, cuda: ModuleType,
                                        monkeypatch: pytest.MonkeyPatch):
        self.remove_jax_modules(monkeypatch)

        assert cuda._is_jax_backend_initialized() is False

    def test_jax_init_false_before_backend(self, cuda: ModuleType,
                                           monkeypatch: pytest.MonkeyPatch):
        xla_bridge = ModuleType("jax._src.xla_bridge")
        xla_bridge.backends_are_initialized = lambda: False

        monkeypatch.setitem(sys.modules, "jax", ModuleType("jax"))
        monkeypatch.setitem(sys.modules, "jax._src", ModuleType("jax._src"))
        monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xla_bridge)

        assert cuda._is_jax_backend_initialized() is False

    def test_jax_init_true_after_backend(self, cuda: ModuleType,
                                         monkeypatch: pytest.MonkeyPatch):
        xla_bridge = ModuleType("jax._src.xla_bridge")
        xla_bridge.backends_are_initialized = lambda: True

        monkeypatch.setitem(sys.modules, "jax", ModuleType("jax"))
        monkeypatch.setitem(sys.modules, "jax._src", ModuleType("jax._src"))
        monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", xla_bridge)

        assert cuda._is_jax_backend_initialized() is True

    def test_cupy_limit_rejects_async(self, cuda: ModuleType):
        cuda._cupy_allocator = "cuda_malloc_async"

        with pytest.raises(RuntimeError, match="CuPy memory limits"):
            cuda.set_cupy_limit("8GB")

    def test_parse_cupy_limits(self, cuda: ModuleType):
        assert cuda._parse_limit("8GB", allow_bytes=True) == (8_000_000_000, None)
        assert cuda._parse_limit("512MiB", allow_bytes=True) == (512 * 1024 ** 2, None)
        assert cuda._parse_limit("25%", allow_bytes=True) == (None, 0.25)
