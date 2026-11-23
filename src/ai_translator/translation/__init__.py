"""Ollama-backed translation candidate generation."""

from .config import ModelConfig, TranslationConfig
from .service import TranslationService

__all__ = ["ModelConfig", "TranslationConfig", "TranslationService"]

