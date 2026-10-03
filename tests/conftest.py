"""Test setup.

Off Apple Silicon there is no ``mlx``/``mlx_lm``. Several modules import them at the top
level, so we install inert stand-ins to let the model-free logic (CHI maths, compaction,
agent helpers) be tested everywhere. Tests that need real tensors skip on ``HAS_REAL_MLX``.
"""

import sys
import types

try:  # pragma: no cover - depends on platform
    import mlx.core  # noqa: F401
    import mlx_lm  # noqa: F401

    HAS_REAL_MLX = True
except ImportError:
    HAS_REAL_MLX = False

    def _module(name: str, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__stub__ = True
        sys.modules[name] = mod
        return mod

    class _Module:  # stands in for mlx.nn.Module
        pass

    core = _module("mlx.core", array=object)
    nn = _module("mlx.nn", Module=_Module)
    mlx = _module("mlx", core=core, nn=nn)
    _module("mlx_lm", load=None, generate=None, stream_generate=None)
