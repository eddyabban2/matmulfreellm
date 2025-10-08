"""
Benchmark script to compare optimized vs unoptimized fixed-point implementation
"""

import torch
import time
from hgrn_8bit_fixed import HGRNFixed8bit, generate_test_weights

def benchmark_ternary_matmul(batch_size, seq_len, hidden_size, iterations=100):
    """Benchmark the ternary_matmul operation"""

    hgrn = HGRNFixed8bit()

    # Generate test data
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    x_fixed = hgrn.to_fixed(x).cuda()

    w_ternary, w_scale = generate_test_weights(hidden_size, hidden_size)
    w_ternary = w_ternary.cuda()
    w_scale = w_scale.cuda()

    # Warm-up
    for _ in range(10):
        _ = hgrn.ternary_matmul(x_fixed, w_ternary, w_scale)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        output = hgrn.ternary_matmul(x_fixed, w_ternary, w_scale)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations * 1000  # ms
    return avg_time

def benchmark_full_forward(batch_size, seq_len, hidden_size, iterations=20):
    """Benchmark full HGRN forward pass"""

    hgrn = HGRNFixed8bit()

    # Generate test data
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    x = x.cuda()

    w_i, w_scale_i = generate_test_weights(hidden_size, hidden_size)
    w_f, w_scale_f = generate_test_weights(hidden_size, hidden_size)
    w_g, w_scale_g = generate_test_weights(hidden_size, hidden_size)
    w_o, w_scale_o = generate_test_weights(hidden_size, hidden_size)

    # Move to GPU
    w_i, w_f, w_g, w_o = w_i.cuda(), w_f.cuda(), w_g.cuda(), w_o.cuda()
    w_scale_i = w_scale_i.cuda()
    w_scale_f = w_scale_f.cuda()
    w_scale_g = w_scale_g.cuda()
    w_scale_o = w_scale_o.cuda()

    # Warm-up
    for _ in range(5):
        _ = hgrn.forward(x, w_i, w_f, w_g, w_o,
                        w_scale_i, w_scale_f, w_scale_g, w_scale_o)

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        output = hgrn.forward(x, w_i, w_f, w_g, w_o,
                             w_scale_i, w_scale_f, w_scale_g, w_scale_o)
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations * 1000  # ms
    return avg_time

if __name__ == "__main__":
    print("="*80)
    print("FIXED-POINT HGRN PERFORMANCE BENCHMARK")
    print("="*80)

    configs = [
        (1, 16, 512),   # Small
        (1, 32, 1024),  # Medium
        (1, 64, 2048),  # Large
    ]

    print("\nTernary Matrix Multiplication Performance:")
    print(f"{'Config':<20} {'Time (ms)':<15} {'Throughput (GOPS)':<20}")
    print("-" * 80)

    for batch, seq_len, hidden in configs:
        avg_time = benchmark_ternary_matmul(batch, seq_len, hidden, iterations=100)
        # Calculate throughput: 2 * batch * seq_len * hidden * hidden operations
        ops = 2 * batch * seq_len * hidden * hidden
        throughput = ops / (avg_time * 1e6)  # GOPS

        config_str = f"[{batch}, {seq_len}, {hidden}]"
        print(f"{config_str:<20} {avg_time:<15.4f} {throughput:<20.2f}")

    print("\n" + "="*80)
    print("Full HGRN Forward Pass Performance:")
    print(f"{'Config':<20} {'Time (ms)':<15} {'Tokens/sec':<20}")
    print("-" * 80)

    for batch, seq_len, hidden in configs:
        avg_time = benchmark_full_forward(batch, seq_len, hidden, iterations=20)
        tokens_per_sec = (batch * seq_len * 1000) / avg_time

        config_str = f"[{batch}, {seq_len}, {hidden}]"
        print(f"{config_str:<20} {avg_time:<15.4f} {tokens_per_sec:<20.2f}")

    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY:")
    print("✓ Using cuBLAS-accelerated matmul for ternary operations")
    print("✓ Cached sigmoid lookup tables")
    print("✓ Pre-allocated output buffers")
    print("✓ Vectorized operations throughout")
    print("="*80)
