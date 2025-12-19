# PIXEL: Pattern-Indexed eXtreme Efficient LLM Compression

> ⚠️ **Experimental Research Project** - This is a research exploration into pattern-based LLM compression. For production use, consider established methods like [GPTQ](https://github.com/IST-DASLab/gptq) or [AWQ](https://github.com/mit-han-lab/llm-awq).

## Overview

PIXEL explores novel approaches to LLM weight compression through pattern discovery, SVD decomposition, and quantization. This project documents our research journey and findings.

## Key Findings

### What We Learned

| Approach | Compression | Perplexity Increase | Verdict |
|----------|-------------|---------------------|---------|
| SVD-only | ~4x | +2,500% | ❌ Unusable |
| Pattern matching | ~1x | +77% | ❌ No compression |
| Calibrated SVD | ~5x | +250,000% | ❌ Worse |
| **GPTQ-style quantization** | **7.87x** | **+38%** | ✅ Best result |

### Key Insight

> Pure SVD/pattern compression destroys model quality. Error compensation (like GPTQ uses) is essential for maintaining quality during compression.

## Installation

```bash
git clone https://github.com/ageborn-dev/pixel.git
cd pixel
pip install -e .
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- transformers (for benchmarks)

## Quick Start

### GPTQ-Style Compression (Best Results)

```python
from pixel.core import GPTQStyleCompressor
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
compressor = GPTQStyleCompressor(bits=4)

# Compress all layers
compressed = compressor.compress_model_sequential(model)

# Apply compressed weights
model = compressor.apply_compressed_weights(model, compressed)
```

### Pattern Analysis (Experimental)

```python
from pixel import PIXELCompressor, PIXELConfig

config = PIXELConfig()
compressor = PIXELCompressor(config)

# Analyze weight patterns
result = compressor.compress(model_weights)
print(f"Patterns discovered: {len(compressor.pattern_dict)}")
```

## Benchmarks

Run the benchmarks to reproduce our findings:

```bash
# GPTQ-style compression (best results)
python benchmarks/gptq_style_benchmark.py

# Original pattern-based approach
python benchmarks/llm_benchmark.py

# Perplexity comparison
python benchmarks/perplexity_benchmark.py
```

## Project Structure

```
pixel/
├── core/
│   ├── config.py         # Configuration classes
│   ├── patterns.py       # Pattern dictionary
│   ├── synthesis.py      # Weight synthesis
│   ├── compression.py    # Main compressor
│   ├── adaptive.py       # Adaptive compression with presets
│   ├── calibrated.py     # Calibration-based compression
│   └── gptq_style.py     # GPTQ-style with error compensation
├── nn/
│   ├── layers.py         # PIXEL layers
│   └── attention.py      # Pattern attention
├── experimental/
│   └── svd_hybrid.py     # SVD + pattern hybrid
└── utils/
    ├── math_ops.py       # Math utilities
    ├── memory.py         # Memory tracking
    └── io.py             # I/O utilities
```

## Compression Methods

### 1. GPTQStyleCompressor (Recommended)
- 4-bit quantization with error compensation
- Column-wise quantization
- Best quality/compression trade-off
- **7.87x compression, 38% perplexity increase**

### 2. AdaptiveCompressor
- SVD-based with per-layer tuning
- Three presets: `high_quality`, `balanced`, `max_compression`
- Good compression ratio but high perplexity increase

### 3. PIXELCompressor (Original)
- Pattern-based weight representation
- Interesting for analysis, not practical for compression
- Pattern discovery and dictionary building

## Research Directions

This project explored several ideas:

1. **Pattern-based compression** - Store weights as patterns + scales
2. **SVD decomposition** - Low-rank approximation of weight matrices
3. **Calibration-based importance** - Use activation data to identify critical weights
4. **Error compensation** - GPTQ's key innovation that makes quantization work

### Why Pure SVD/Patterns Don't Work

- Even 99% SVD energy retention causes significant perplexity increase
- LLM weights are sensitive - small errors compound catastrophically
- Without error compensation, quality degrades rapidly

### What Works

- **Quantization + error compensation** (GPTQ/AWQ approach)
- Adjusting subsequent weights to compensate for quantization error
- Using calibration data to guide compression decisions

## Contributing

This is a research project open to exploration. Ideas welcome:

1. Better pattern discovery algorithms
2. Combining patterns with quantization
3. Per-layer adaptive compression
4. Fine-tuning after compression

## License

MIT License

## Acknowledgments

- Inspired by GPTQ, AWQ, and other LLM compression research
- Built with PyTorch
