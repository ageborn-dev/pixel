"""PIXEL: Pattern-Indexed eXtreme Efficient LLM compression library."""

from pixel.core.config import PIXELConfig, CompressionConfig
from pixel.core.patterns import PatternDictionary, PatternType
from pixel.core.synthesis import WeightSynthesizer
from pixel.core.compression import PIXELCompressor

__version__ = "0.1.0"
__all__ = [
    "PIXELConfig",
    "CompressionConfig",
    "PatternDictionary",
    "PatternType",
    "WeightSynthesizer",
    "PIXELCompressor",
]
