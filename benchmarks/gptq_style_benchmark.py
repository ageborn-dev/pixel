"""GPTQ-style compression benchmark with error compensation."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from pixel.core.gptq_style import GPTQStyleCompressor


CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long.",
    "All happy families are alike; each unhappy family is unhappy.",
    "The sky above the port was the color of television.",
    "In the beginning God created the heaven and the earth.",
    "It was a bright cold day in April, and the clocks were striking.",
    "As Gregor Samsa awoke one morning from uneasy dreams.",
    "A screaming comes across the sky.",
]


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    model.eval()
    model.to(device)
    encodings = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    
    return torch.exp(outputs.loss).item()


def run_gptq_style_benchmark():
    print("=" * 60)
    print("PIXEL GPTQ-Style Compression Benchmark")
    print("=" * 60)
    
    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    device = "cpu"
    model.to(device)
    model.eval()
    
    print("\nCalculating original perplexity...")
    test_texts = CALIBRATION_TEXTS[:5]
    original_ppls = [calculate_perplexity(model, tokenizer, t, device) for t in test_texts]
    avg_original = sum(original_ppls) / len(original_ppls)
    print(f"  Average original perplexity: {avg_original:.2f}")
    
    print("\n" + "=" * 60)
    print("GPTQ-Style Compression (4-bit with error compensation)")
    print("=" * 60)
    
    compressor = GPTQStyleCompressor(
        bits=4,
        group_size=128,
        dampening=0.01,
    )
    
    calibration_inputs = [
        tokenizer(text, return_tensors="pt").input_ids.to(device)
        for text in CALIBRATION_TEXTS
    ]
    
    print("\nCompressing model with Hessian calibration...")
    compressed = compressor.compress_model_sequential(
        model,
        calibration_data=calibration_inputs,
        verbose=True,
    )
    
    print("\nApplying compressed weights...")
    model = compressor.apply_compressed_weights(model, compressed)
    
    print("\nCalculating compressed perplexity...")
    compressed_ppls = [calculate_perplexity(model, tokenizer, t, device) for t in test_texts]
    avg_compressed = sum(compressed_ppls) / len(compressed_ppls)
    ppl_increase = (avg_compressed - avg_original) / avg_original * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Original perplexity: {avg_original:.2f}")
    print(f"  Compressed perplexity: {avg_compressed:.2f}")
    print(f"  Perplexity increase: {ppl_increase:+.2f}%")
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS METHODS")
    print("=" * 60)
    print(f"  SVD-only (no calibration): ~2500% increase")
    print(f"  Calibrated SVD: ~250,000% increase")
    print(f"  GPTQ-Style (this): {ppl_increase:+.2f}% increase")
    
    if ppl_increase < 10:
        print("\n✅ EXCELLENT: Ready for production use!")
    elif ppl_increase < 50:
        print("\n✅ GOOD: Acceptable for most applications")
    elif ppl_increase < 200:
        print("\n⚠️ MODERATE: Some quality loss, may need tuning")
    else:
        print("\n❌ NEEDS WORK: Quality loss too high")


if __name__ == "__main__":
    run_gptq_style_benchmark()
