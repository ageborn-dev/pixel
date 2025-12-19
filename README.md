# PIXEL: Pattern-Indexed eXtreme Efficient LLM

A novel approach to neural network weight compression using pattern-based storage and dynamic weight synthesis.

## Overview

PIXEL compresses neural network weights by storing common patterns instead of full weight matrices. During inference, weights are reconstructed on-demand from pattern combinations, dramatically reducing model size while maintaining accuracy.

### Key Features

- **Pattern-based compression** - Store patterns + scales instead of full matrices
- **SVD+Pattern hybrid** - Combine low-rank SVD with pattern matching for better compression
- **Drop-in replacement** - Convert `nn.Linear` layers to compressed `PIXELLinear`
- **PyTorch native** - Built on PyTorch with GPU acceleration support

## Installation

```bash
git clone https://github.com/ageborn-dev/pixel.git
cd pixel
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Compression

```python
import torch
from pixel import PIXELConfig, PIXELCompressor

compressor = PIXELCompressor()

weights = {
    "layer1.weight": torch.randn(256, 256),
    "layer2.weight": torch.randn(256, 256),
}

result = compressor.compress(weights)
print(f"Compression: {result.compression_ratio:.2f}x")
print(f"Error: {result.reconstruction_error:.4f}")

reconstructed = compressor.decompress(result.pattern_refs)
```

### Replace Linear Layers

```python
import torch.nn as nn
from pixel import PatternDictionary
from pixel.nn.layers import replace_linear_layers

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
)

pattern_dict = PatternDictionary(max_patterns=100)
pattern_dict.generate_base_patterns(128)
pattern_dict.generate_base_patterns(256)

errors = replace_linear_layers(model, pattern_dict)
```

### SVD + Pattern Hybrid

```python
from pixel.experimental import SVDHybridCompressor

compressor = SVDHybridCompressor(
    svd_energy_threshold=0.85,
    pattern_error_threshold=0.05,
)

weight = torch.randn(512, 512)
result = compressor.compress(weight)

print(f"SVD rank: {result['svd_rank']}")
print(f"Patterns: {result['num_patterns']}")
print(f"Ratio: {result['compression_ratio']:.2f}x")
```

## Architecture

```
Weight Matrix
     │
     ▼
┌─────────────────┐
│  SVD Decompose  │  (optional, for global structure)
│   W ≈ U·Σ·V^T   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pattern Match   │  Find similar patterns in dictionary
│  residual → Σ   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scale Optimize  │  Least-squares for optimal scales
│   argmin ||W -  │
│   Σ(Pi·si)||²   │
└────────┬────────┘
         │
         ▼
  Compressed Form:
  (U, Σ, V) + [(p₁,s₁), (p₂,s₂), ...] + residual
```

## API Reference

### PIXELConfig

```python
from pixel import PIXELConfig

config = PIXELConfig(
    matrix_size=256,
    hidden_size=768,
    num_layers=12,
)

config = PIXELConfig.for_small_model()
config = PIXELConfig.for_large_model()
```

### PIXELCompressor

```python
from pixel import PIXELCompressor

compressor = PIXELCompressor(config)
result = compressor.compress(weight_dict)
weights = compressor.decompress(pattern_refs)
compressor.save("model.pt")
compressor = PIXELCompressor.load("model.pt")
```

### PatternDictionary

```python
from pixel import PatternDictionary

pattern_dict = PatternDictionary(max_patterns=1024, cache_size=256)
pattern_dict.generate_base_patterns(size=64)
pid = pattern_dict.add(pattern_tensor)
pattern = pattern_dict.get(pid)
matches = pattern_dict.find_best_matches(target, top_k=5)
```

## Benchmarks

Run benchmarks:

```bash
python benchmarks/memory_benchmark.py
python benchmarks/accuracy_benchmark.py
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
pixel/
├── pixel/
│   ├── core/           # Core compression logic
│   │   ├── config.py
│   │   ├── patterns.py
│   │   ├── synthesis.py
│   │   └── compression.py
│   ├── nn/             # Neural network layers
│   │   ├── layers.py
│   │   └── attention.py
│   ├── utils/          # Utilities
│   └── experimental/   # Experimental features
├── tests/
├── examples/
├── benchmarks/
└── docs/
```

## How It Works

1. **Pattern Discovery**: Identify recurring patterns in weight matrices
2. **Pattern Dictionary**: Store unique patterns with usage tracking
3. **Weight Decomposition**: Express weights as pattern combinations
4. **Dynamic Synthesis**: Reconstruct weights on-demand during inference

### Memory Targets

| Component | Target Size |
|-----------|-------------|
| Pattern Dictionary | ~100MB |
| Synthesis Network | ~50MB |
| Runtime Cache | ~200MB |
| **Total** | **~350MB** |

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use PIXEL in your research, please cite:

```bibtex
@software{pixel2025,
  title={PIXEL: Pattern-Indexed eXtreme Efficient LLM},
  year={2025},
  url={https://github.com/ageborn-dev/pixel.git}
}
```
