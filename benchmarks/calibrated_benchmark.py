"""Calibrated compression benchmark with GPT-2."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from pixel.core.calibrated import CalibratedCompressor


CALIBRATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, never mind how long.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "In the beginning God created the heaven and the earth.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Happy families are all alike; every unhappy family is unhappy in its own way.",
    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed.",
    "Once upon a time and a very good time it was there was a moocow coming down along the road.",
    "A screaming comes across the sky.",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendía.",
    "The sun shone, having no alternative, on the nothing new.",
    "I am an invisible man.",
    "Mother died today. Or maybe yesterday, I don't know.",
    "Someone must have slandered Josef K., for one morning, without having done anything truly wrong.",
    "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.",
    "Whether I shall turn out to be the hero of my own life.",
]


def calculate_perplexity(model, tokenizer, text: str, device: str = "cpu") -> float:
    model.eval()
    encodings = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(encodings.input_ids, labels=encodings.input_ids)
    
    return torch.exp(outputs.loss).item()


def apply_compressed_weights(model, compressed: dict, compressor: CalibratedCompressor):
    decompressed = compressor.decompress_model(compressed)
    
    state_dict = model.state_dict()
    for name, weight in decompressed.items():
        if name in state_dict:
            state_dict[name] = weight.to(state_dict[name].dtype)
    
    model.load_state_dict(state_dict)
    return model


def run_calibrated_benchmark():
    print("=" * 60)
    print("PIXEL Calibrated Compression Benchmark")
    print("=" * 60)
    
    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
    
    print("\nCalculating original perplexity...")
    test_texts = CALIBRATION_TEXTS[:5]
    original_ppls = [calculate_perplexity(model, tokenizer, t) for t in test_texts]
    avg_original = sum(original_ppls) / len(original_ppls)
    print(f"  Average original perplexity: {avg_original:.2f}")
    
    print("\n" + "=" * 60)
    print("Testing Calibrated Compressor")
    print("=" * 60)
    
    compressor = CalibratedCompressor(
        protection_ratio=0.05,
        base_svd_energy=0.90,
        protected_svd_energy=0.98,
        quantize=True,
    )
    
    print(f"\nSettings:")
    print(f"  Protection ratio: {compressor.protection_ratio:.1%}")
    print(f"  Base SVD energy: {compressor.base_svd_energy:.0%}")
    print(f"  Protected SVD energy: {compressor.protected_svd_energy:.0%}")
    
    calibration_inputs = [
        tokenizer(text, return_tensors="pt").input_ids
        for text in CALIBRATION_TEXTS
    ]
    
    compressor.calibrate(model, calibration_inputs, tokenizer=None)
    
    weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            weights[name] = param.data.clone()
    
    print("\nCompressing weights...")
    compressed = compressor.compress_model(weights, verbose=True)
    
    total_original = sum(w.numel() * 4 for w in weights.values())
    total_compressed = sum(r["size_bytes"] for r in compressed.values())
    
    print("\nApplying compressed weights...")
    model_copy = GPT2LMHeadModel.from_pretrained("gpt2")
    model_copy = apply_compressed_weights(model_copy, compressed, compressor)
    
    print("\nCalculating compressed perplexity...")
    compressed_ppls = [calculate_perplexity(model_copy, tokenizer, t) for t in test_texts]
    avg_compressed = sum(compressed_ppls) / len(compressed_ppls)
    ppl_increase = (avg_compressed - avg_original) / avg_original * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Original size: {total_original / 1024 / 1024:.2f} MB")
    print(f"  Compressed size: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {total_original / total_compressed:.2f}x")
    print(f"  Original perplexity: {avg_original:.2f}")
    print(f"  Compressed perplexity: {avg_compressed:.2f}")
    print(f"  Perplexity increase: {ppl_increase:+.2f}%")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Previous (no calibration): ~2500% perplexity increase")
    print(f"  With calibration: {ppl_increase:+.2f}% perplexity increase")
    
    if ppl_increase < 20:
        print("\n✅ SUCCESS: Calibration significantly improved quality!")
    elif ppl_increase < 100:
        print("\n⚠️ IMPROVED: Better than before, but still needs work")
    else:
        print("\n❌ NEEDS MORE WORK: Still too much quality loss")


if __name__ == "__main__":
    run_calibrated_benchmark()
