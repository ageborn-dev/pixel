"""Perplexity benchmark - tests actual model quality after compression."""
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from pixel.core.adaptive import AdaptiveCompressor


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    model.eval()
    model.to(device)
    
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss)
    return perplexity.item()


def apply_compressed_weights(model, compressed_weights: dict, compressor: AdaptiveCompressor):
    decompressed = compressor.decompress_model(compressed_weights)
    
    state_dict = model.state_dict()
    for name, weight in decompressed.items():
        if name in state_dict:
            state_dict[name] = weight.to(state_dict[name].dtype)
    
    model.load_state_dict(state_dict)
    return model


def run_perplexity_benchmark():
    print("=" * 60)
    print("PIXEL Perplexity Benchmark")
    print("=" * 60)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog. This is a simple test sentence to evaluate the language model's ability to predict common English phrases and patterns.",
        "In machine learning, neural networks are computational systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.",
        "Python is a high-level programming language known for its simplicity and readability. It is widely used in data science, web development, and artificial intelligence.",
    ]
    
    print("\nLoading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    print("\nCalculating original perplexity...")
    original_perplexities = []
    for i, text in enumerate(test_texts):
        ppl = calculate_perplexity(model, tokenizer, text)
        original_perplexities.append(ppl)
        print(f"  Text {i+1}: {ppl:.2f}")
    
    avg_original = sum(original_perplexities) / len(original_perplexities)
    print(f"  Average: {avg_original:.2f}")
    
    weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            weights[name] = param.data.clone()
    
    presets = [
        ("high_quality", AdaptiveCompressor.high_quality()),
        ("balanced", AdaptiveCompressor.balanced()),
        ("max_compression", AdaptiveCompressor.max_compression()),
    ]
    
    results = []
    
    for preset_name, compressor in presets:
        print(f"\n{'=' * 60}")
        print(f"Testing preset: {preset_name}")
        print("=" * 60)
        
        model_copy = GPT2LMHeadModel.from_pretrained("gpt2")
        
        print("Compressing weights...")
        compressed = compressor.compress_model(weights, verbose=False)
        
        total_original = sum(w.numel() * 4 for w in weights.values())
        total_compressed = sum(r["size_bytes"] for r in compressed.values())
        ratio = total_original / total_compressed
        
        print(f"  Compression ratio: {ratio:.2f}x")
        
        print("Applying compressed weights...")
        model_copy = apply_compressed_weights(model_copy, compressed, compressor)
        
        print("Calculating compressed perplexity...")
        compressed_perplexities = []
        for i, text in enumerate(test_texts):
            ppl = calculate_perplexity(model_copy, tokenizer, text)
            compressed_perplexities.append(ppl)
            print(f"  Text {i+1}: {ppl:.2f}")
        
        avg_compressed = sum(compressed_perplexities) / len(compressed_perplexities)
        ppl_increase = (avg_compressed - avg_original) / avg_original * 100
        
        print(f"  Average: {avg_compressed:.2f}")
        print(f"  Perplexity increase: {ppl_increase:+.2f}%")
        
        results.append({
            "preset": preset_name,
            "compression_ratio": ratio,
            "original_ppl": avg_original,
            "compressed_ppl": avg_compressed,
            "ppl_increase": ppl_increase,
        })
        
        del model_copy
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Preset':<20} {'Compression':<15} {'Orig PPL':<12} {'Comp PPL':<12} {'Increase':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['preset']:<20} {r['compression_ratio']:<15.2f}x {r['original_ppl']:<12.2f} {r['compressed_ppl']:<12.2f} {r['ppl_increase']:+.2f}%")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
- Perplexity measures how "surprised" the model is by text
- Lower is better (original GPT-2 typically ~20-40)
- <5% increase = excellent quality
- <20% increase = acceptable for most uses
- >50% increase = significant quality loss
""")


if __name__ == "__main__":
    run_perplexity_benchmark()
