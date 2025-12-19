"""Example: Compress a PyTorch model's weights."""
import torch
import torch.nn as nn

from pixel import PIXELConfig, PatternDictionary
from pixel.nn.layers import replace_linear_layers, PIXELLinear


class SampleMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("PIXEL Model Compression Example")
    print("=" * 50)

    model = SampleMLP()
    original_params = count_parameters(model)
    print(f"Original model: {original_params:,} parameters")

    x = torch.randn(8, 128)
    original_output = model(x)

    pattern_dict = PatternDictionary(max_patterns=200)
    for size in [64, 128, 256]:
        pattern_dict.generate_base_patterns(size)
    print(f"Generated {len(pattern_dict)} base patterns")

    print("\nConverting layers...")
    errors = replace_linear_layers(model, pattern_dict, max_patterns=16)

    for name, error in errors.items():
        print(f"  {name}: error = {error:.4f}")

    pixel_output = model(x)

    output_diff = torch.linalg.norm(original_output - pixel_output)
    output_norm = torch.linalg.norm(original_output)
    relative_error = output_diff / output_norm

    print(f"\nOutput comparison:")
    print(f"  - Relative difference: {relative_error:.4f}")

    print(f"\nModel structure after conversion:")
    for name, module in model.named_modules():
        if isinstance(module, PIXELLinear):
            print(f"  {name}: PIXELLinear({module.in_features}, {module.out_features})")
            print(f"    - Pattern refs: {len(module.pattern_refs)}")
            print(f"    - Has residual: {module.residual is not None}")


if __name__ == "__main__":
    main()
