"""
Model loading and basic generation.

Thin wrapper around mlx_lm for loading models and generating text.
Use this as the single entry point for inference in the package.
"""

from mlx_lm import load, generate

__all__ = ["load", "generate"]
