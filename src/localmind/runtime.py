from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    torch_available: bool
    cuda_available: bool
    cuda_name: str | None
    onnx_available: bool
    onnx_providers: tuple[str, ...]

    @property
    def recommended_reranker_runtime(self) -> str:
        if self.cuda_available:
            return "cuda"
        if self.onnx_available:
            return "onnx"
        return "cpu"

    @property
    def default_reranker_enabled(self) -> bool:
        return self.cuda_available


@dataclass(frozen=True, slots=True)
class ResolvedRerankerRuntime:
    requested: str
    name: str
    backend: str
    device: str
    provider: str | None
    available: bool
    message: str


@lru_cache(maxsize=1)
def detect_runtime_capabilities() -> RuntimeCapabilities:
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    cuda_name = None
    if torch_available:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_name = str(torch.cuda.get_device_name(0))
        except Exception:
            cuda_available = False

    onnx_available = (
        importlib.util.find_spec("onnxruntime") is not None
        and importlib.util.find_spec("optimum") is not None
    )
    providers: tuple[str, ...] = ()
    if onnx_available:
        try:
            import onnxruntime

            providers = tuple(str(item) for item in onnxruntime.get_available_providers())
        except Exception:
            onnx_available = False
    return RuntimeCapabilities(
        torch_available=torch_available,
        cuda_available=cuda_available,
        cuda_name=cuda_name,
        onnx_available=onnx_available,
        onnx_providers=providers,
    )


def resolve_reranker_runtime(
    requested: str = "auto",
    capabilities: RuntimeCapabilities | None = None,
) -> ResolvedRerankerRuntime:
    caps = capabilities or detect_runtime_capabilities()
    normalized = requested.lower().strip()
    if normalized not in {"auto", "cuda", "onnx", "cpu"}:
        normalized = "auto"
    selected = caps.recommended_reranker_runtime if normalized == "auto" else normalized

    if selected == "cuda":
        available = caps.cuda_available
        return ResolvedRerankerRuntime(
            requested=normalized,
            name="cuda",
            backend="torch",
            device="cuda",
            provider=None,
            available=available,
            message=(f"CUDA / {caps.cuda_name}" if available else "当前 PyTorch 未检测到 CUDA"),
        )
    if selected == "onnx":
        available = caps.onnx_available
        provider = None
        if "CUDAExecutionProvider" in caps.onnx_providers:
            provider = "CUDAExecutionProvider"
        elif "CPUExecutionProvider" in caps.onnx_providers:
            provider = "CPUExecutionProvider"
        return ResolvedRerankerRuntime(
            requested=normalized,
            name="onnx",
            backend="onnx",
            device="cuda" if provider == "CUDAExecutionProvider" else "cpu",
            provider=provider,
            available=available,
            message=(
                f"ONNX / {provider or '默认 Provider'}"
                if available
                else "未安装 ONNX Runtime，可安装项目的 onnx 可选依赖"
            ),
        )
    return ResolvedRerankerRuntime(
        requested=normalized,
        name="cpu",
        backend="torch",
        device="cpu",
        provider=None,
        available=caps.torch_available,
        message="PyTorch CPU（精排默认关闭）" if caps.torch_available else "未安装 PyTorch",
    )
