# #!/usr/bin/env python3
# # Memory benchmark script for different ToMe merge functions
# # This script evaluates memory usage and speed without requiring a validation dataset

# import os
# import sys
# import time
# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.cuda import max_memory_allocated, reset_peak_memory_stats
# import timm
# from tqdm import tqdm

# # Import from merge.py without modifying the original code
# sys.path.append('.')  # Add current directory to path
# from tome import merge

# # Available merge functions to benchmark
# MERGE_FUNCTIONS = {
#     'bipartite_soft_matching': merge.bipartite_soft_matching,
#     'grouped_bipartite_soft_matching': merge.grouped_bipartite_soft_matching,
#     'kmeans_bipartite_soft_matching': merge.kmeans_bipartite_soft_matching,
#     'kth_bipartite_soft_matching': merge.kth_bipartite_soft_matching,
#     'random_bipartite_soft_matching': merge.random_bipartite_soft_matching,
# }

# class BenchmarkResults:
#     """Class to store benchmark results"""
#     def __init__(self):
#         self.memory_usage = {}
#         self.inference_times = {}
#         self.throughputs = {}
    
#     def add_result(self, function_name, r_value, memory, time, throughput):
#         key = f"{function_name}_r{r_value}"
#         self.memory_usage[key] = memory
#         self.inference_times[key] = time
#         self.throughputs[key] = throughput

#     def print_summary(self):
#         print("\n===== BENCHMARK SUMMARY =====")
#         print(f"{'Function':30} {'r':5} {'Memory (MB)':15} {'Time (ms)':15} {'Throughput (im/s)':20}")
#         print("-" * 90)
        
#         for key in sorted(self.memory_usage.keys()):
#             func_name, r_value = key.rsplit('_r', 1)
#             print(f"{func_name:30} {r_value:5} {self.memory_usage[key]:15.2f} "
#                   f"{self.inference_times[key]*1000:15.2f} {self.throughputs[key]:20.2f}")
    
#     def plot_results(self, output_dir='./benchmark_results'):
#         """Plot the benchmark results"""
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Group by function name
#         functions = {}
#         r_values = set()
#         for key in self.memory_usage.keys():
#             func_name, r_value = key.rsplit('_r', 1)
#             r_value = int(r_value)
#             r_values.add(r_value)
            
#             if func_name not in functions:
#                 functions[func_name] = []
#             functions[func_name].append(r_value)
        
#         r_values = sorted(list(r_values))
        
#         # Plot memory usage
#         plt.figure(figsize=(12, 8))
#         for func_name, func_r_values in functions.items():
#             # Sort r values for proper line plotting
#             func_r_values = sorted(func_r_values)
#             y_values = [self.memory_usage[f"{func_name}_r{r}"] for r in func_r_values]
#             plt.plot(func_r_values, y_values, marker='o', label=func_name)
        
#         plt.xlabel('r value (tokens to merge)')
#         plt.ylabel('Memory Usage (MB)')
#         plt.title('Memory Usage vs. Token Reduction (r)')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
        
#         # Plot throughput
#         plt.figure(figsize=(12, 8))
#         for func_name, func_r_values in functions.items():
#             # Sort r values for proper line plotting
#             func_r_values = sorted(func_r_values)
#             y_values = [self.throughputs[f"{func_name}_r{r}"] for r in func_r_values]
#             plt.plot(func_r_values, y_values, marker='o', label=func_name)
        
#         plt.xlabel('r value (tokens to merge)')
#         plt.ylabel('Throughput (images/s)')
#         plt.title('Throughput vs. Token Reduction (r)')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, 'throughput.png'))
        
#         # Plot memory-throughput trade-off for each r value
#         for r in r_values:
#             plt.figure(figsize=(10, 8))
#             x_values = []
#             y_values = []
#             labels = []
            
#             for func_name in functions:
#                 if r in functions[func_name]:
#                     x_values.append(self.memory_usage[f"{func_name}_r{r}"])
#                     y_values.append(self.throughputs[f"{func_name}_r{r}"])
#                     labels.append(func_name)
            
#             plt.scatter(x_values, y_values, s=100)
            
#             # Add labels to points
#             for i, label in enumerate(labels):
#                 plt.annotate(label, (x_values[i], y_values[i]), 
#                            xytext=(10, 5), textcoords='offset points')
            
#             plt.xlabel('Memory Usage (MB)')
#             plt.ylabel('Throughput (images/s)')
#             plt.title(f'Memory-Throughput Trade-off for r={r}')
#             plt.grid(True)
#             plt.savefig(os.path.join(output_dir, f'tradeoff_r{r}.png'))
#             plt.close()

# def get_model(model_name="vit_base_patch16_224", pretrained=True):
#     """Create and return the model to benchmark"""
#     model = timm.create_model(model_name, pretrained=pretrained)
#     return model

# def patch_model_with_tome(model, merge_function, r_value):
#     """Patch a model with ToMe using the specified merge function"""
#     if not hasattr(model, 'blocks'):
#         raise ValueError("Model does not have 'blocks' attribute. This script is designed for ViT models.")
    
#     # Save the original attention functions
#     original_attentions = []
#     for block in model.blocks:
#         if hasattr(block, 'attn'):
#             original_attentions.append(block.attn.forward)
    
#     # Apply merge function to each block
#     for i, block in enumerate(model.blocks):
#         if hasattr(block, 'attn'):
#             # Monkey patch the attention forward function
#             orig_forward = block.attn.forward
            
#             def make_new_forward(block_idx, original_forward, merge_fn):
#                 def new_forward(x, *args, **kwargs):
#                     # Get the class token flag
#                     class_token = hasattr(model, 'cls_token')
                    
#                     # Apply merge before attention
#                     B, N, C = x.shape
#                     merge_op, unmerge_op = merge_fn(x, r_value, class_token=class_token)
#                     x = merge_op(x)
                    
#                     # Apply original attention
#                     x = original_forward(x, *args, **kwargs)
                    
#                     # Apply unmerge after attention
#                     x = unmerge_op(x)
#                     return x
                
#                 return new_forward
            
#             block.attn.forward = make_new_forward(i, orig_forward, merge_function)
    
#     return original_attentions

# def restore_original_attention(model, original_attentions):
#     """Restore the original attention functions"""
#     for i, block in enumerate(model.blocks):
#         if hasattr(block, 'attn') and i < len(original_attentions):
#             block.attn.forward = original_attentions[i]

# def benchmark_memory(model, batch_size, input_size, device, warmup=5, runs=20):
#     """Benchmark memory usage of the model using random data"""
#     model.eval()
#     model.to(device)
    
#     # Create random input tensors
#     random_inputs = torch.randn(batch_size, 3, input_size[1], input_size[2], device=device)
    
#     # Warmup
#     for _ in range(warmup):
#         with torch.no_grad():
#             _ = model(random_inputs)
    
#     # Reset peak memory stats
#     reset_peak_memory_stats(device)
    
#     # Benchmark
#     torch.cuda.synchronize()
#     start_time = time.time()
    
#     for _ in range(runs):
#         with torch.no_grad():
#             _ = model(random_inputs)
    
#     torch.cuda.synchronize()
#     end_time = time.time()
    
#     # Get peak memory usage in MB
#     memory_usage = max_memory_allocated(device) / (1024 * 1024)
    
#     # Calculate throughput
#     total_time = end_time - start_time
#     total_images = batch_size * runs
#     throughput = total_images / total_time
#     inference_time = total_time / runs  # Average time per batch
    
#     return memory_usage, inference_time, throughput

# def main():
#     parser = argparse.ArgumentParser(description='Benchmark ToMe merge functions memory usage')
#     parser.add_argument('--model', type=str, default='vit_base_patch16_224',
#                         help='Model name from timm (default: vit_base_patch16_224)')
#     parser.add_argument('--batch-size', type=int, default=32,
#                         help='Batch size for benchmark (default: 32)')
#     parser.add_argument('--device', type=str, default='cuda:0',
#                         help='Device to use for evaluation (default: cuda:0)')
#     parser.add_argument('--r-values', type=int, nargs='+', default=[8, 16, 24, 32],
#                         help='List of r values to benchmark (default: [8, 16, 24, 32])')
#     parser.add_argument('--functions', type=str, nargs='+', 
#                         default=list(MERGE_FUNCTIONS.keys()),
#                         help=f'List of merge functions to benchmark (default: all)')
#     parser.add_argument('--output-dir', type=str, default='./benchmark_results',
#                         help='Directory to save benchmark results (default: ./benchmark_results)')
#     parser.add_argument('--runs', type=int, default=20,
#                         help='Number of runs for benchmarking (default: 20)')
#     parser.add_argument('--warmup', type=int, default=5,
#                         help='Number of warmup iterations (default: 5)')
#     parser.add_argument('--pretrained', action='store_true',
#                         help='Use pretrained model weights (slower to load but more realistic)')
    
#     args = parser.parse_args()
    
#     # Check if CUDA is available
#     if not torch.cuda.is_available() and 'cuda' in args.device:
#         print("CUDA is not available, falling back to CPU")
#         args.device = 'cpu'
    
#     # Adjust batch size for CPU
#     if args.device == 'cpu':
#         args.batch_size = min(args.batch_size, 8)
#         args.runs = min(args.runs, 5)
#         print(f"Adjusted batch size to {args.batch_size} and runs to {args.runs} for CPU testing")
    
#     # Initialize results object
#     results = BenchmarkResults()
    
#     # Get baseline model performance
#     print(f"Loading base model: {args.model}...")
#     model = get_model(args.model, pretrained=args.pretrained)
#     input_size = model.default_cfg["input_size"]
    
#     print(f"Model: {args.model}")
#     print(f"Input size: {input_size}")
#     print(f"Device: {args.device}")
#     print(f"Batch size: {args.batch_size}")
    
#     print("\nMeasuring baseline memory usage...")
#     baseline_memory, baseline_time, baseline_throughput = benchmark_memory(
#         model, args.batch_size, input_size, args.device, 
#         warmup=args.warmup, runs=args.runs
#     )
#     print(f"Baseline memory usage: {baseline_memory:.2f} MB")
#     print(f"Baseline inference time: {baseline_time*1000:.2f} ms per batch")
#     print(f"Baseline throughput: {baseline_throughput:.2f} images/s")
    
#     # Add baseline to results
#     results.add_result("baseline", 0, baseline_memory, baseline_time, baseline_throughput)
    
#     # Create a dictionary to store the function runtimes
#     function_runtimes = {}
    
#     # Benchmark each merge function with different r values
#     for func_name in args.functions:
#         if func_name not in MERGE_FUNCTIONS:
#             print(f"Warning: {func_name} is not a valid merge function. Skipping.")
#             continue
        
#         merge_function = MERGE_FUNCTIONS[func_name]
#         function_runtimes[func_name] = {}
        
#         for r in args.r_values:
#             print(f"\nBenchmarking {func_name} with r={r}...")
            
#             # Create a new model instance to avoid interference
#             model = get_model(args.model, pretrained=args.pretrained)
            
#             # Measure the function's runtime separately
#             # Create a dummy input tensor similar to what would be passed to the merge function
#             dummy_batch_size = 1
#             dummy_seq_len = 197  # 196 patches + 1 class token for 224x224 ViT
#             dummy_dim = model.embed_dim
#             dummy_input = torch.randn(dummy_batch_size, dummy_seq_len, dummy_dim, device=args.device)
            
#             # Measure time to run the merge function
#             torch.cuda.synchronize()
#             func_start = time.time()
#             for _ in range(100):  # Run multiple times for more accurate measurement
#                 with torch.no_grad():
#                     class_token = True  # Most ViT models have a class token
#                     _, _ = merge_function(dummy_input, r, class_token=class_token)
#             torch.cuda.synchronize()
#             func_end = time.time()
#             func_time = (func_end - func_start) / 100
#             function_runtimes[func_name][r] = func_time
            
#             # Patch the model with the current merge function
#             original_attentions = patch_model_with_tome(model, merge_function, r)
            
#             # Measure memory usage
#             memory_usage, inference_time, throughput = benchmark_memory(
#                 model, args.batch_size, input_size, args.device,
#                 warmup=args.warmup, runs=args.runs
#             )
#             print(f"Memory usage: {memory_usage:.2f} MB ({memory_usage/baseline_memory:.2f}x baseline)")
#             print(f"Inference time: {inference_time*1000:.2f} ms per batch ({baseline_time/inference_time:.2f}x faster)")
#             print(f"Throughput: {throughput:.2f} images/s ({throughput/baseline_throughput:.2f}x baseline)")
            
#             # Add result
#             results.add_result(func_name, r, memory_usage, inference_time, throughput)
            
#             # Restore original attention functions
#             restore_original_attention(model, original_attentions)
            
#             # Clean up
#             del model
#             torch.cuda.empty_cache()
    
#     # Print and plot results
#     results.print_summary()
#     results.plot_results(args.output_dir)
#     print(f"Plots saved to {args.output_dir}")
    
#     # Print function runtimes
#     print("\n===== MERGE FUNCTION RUNTIMES =====")
#     print(f"{'Function':30} {'r':5} {'Runtime (ms)':15}")
#     print("-" * 55)
    
#     for func_name in function_runtimes:
#         for r in sorted(function_runtimes[func_name].keys()):
#             runtime = function_runtimes[func_name][r] * 1000  # Convert to ms
#             print(f"{func_name:30} {r:5} {runtime:15.4f}")

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
# Memory benchmark script for different ToMe merge functions
# This script evaluates memory usage and speed without requiring a validation dataset

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
import timm
from tqdm import tqdm

# Import from merge.py without modifying the original code
sys.path.append('.')  # Add current directory to path
from tome import merge

# Available merge functions to benchmark
MERGE_FUNCTIONS = {
    'bipartite_soft_matching': merge.bipartite_soft_matching,
    'grouped_bipartite_soft_matching': merge.grouped_bipartite_soft_matching,
    'iga_bipartite_soft_matching': merge.iga_bipartite_soft_matching,  # Add the IGA function here
}

# Remove kmeans function
if 'kmeans_bipartite_soft_matching' in MERGE_FUNCTIONS:
    del MERGE_FUNCTIONS['kmeans_bipartite_soft_matching']

class BenchmarkResults:
    """Class to store benchmark results"""
    def __init__(self):
        self.memory_usage = {}
        self.inference_times = {}
        self.throughputs = {}
    
    def add_result(self, function_name, r_value, memory, time, throughput):
        key = f"{function_name}_r{r_value}"
        self.memory_usage[key] = memory
        self.inference_times[key] = time
        self.throughputs[key] = throughput

    def print_summary(self):
        print("\n===== BENCHMARK SUMMARY =====")
        print(f"{'Function':30} {'r':5} {'Memory (MB)':15} {'Time (ms)':15} {'Throughput (im/s)':20}")
        print("-" * 90)
        
        for key in sorted(self.memory_usage.keys()):
            func_name, r_value = key.rsplit('_r', 1)
            print(f"{func_name:30} {r_value:5} {self.memory_usage[key]:15.2f} "
                  f"{self.inference_times[key]*1000:15.2f} {self.throughputs[key]:20.2f}")
    
    def plot_results(self, output_dir='./benchmark_results'):
        """Plot the benchmark results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by function name
        functions = {}
        r_values = set()
        for key in self.memory_usage.keys():
            func_name, r_value = key.rsplit('_r', 1)
            r_value = int(r_value)
            r_values.add(r_value)
            
            if func_name not in functions:
                functions[func_name] = []
            functions[func_name].append(r_value)
        
        r_values = sorted(list(r_values))
        
        # Plot memory usage
        plt.figure(figsize=(12, 8))
        for func_name, func_r_values in functions.items():
            # Sort r values for proper line plotting
            func_r_values = sorted(func_r_values)
            y_values = [self.memory_usage[f"{func_name}_r{r}"] for r in func_r_values]
            plt.plot(func_r_values, y_values, marker='o', label=func_name)
        
        plt.xlabel('r value (tokens to merge)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs. Token Reduction (r)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
        
        # Plot throughput
        plt.figure(figsize=(12, 8))
        for func_name, func_r_values in functions.items():
            # Sort r values for proper line plotting
            func_r_values = sorted(func_r_values)
            y_values = [self.throughputs[f"{func_name}_r{r}"] for r in func_r_values]
            plt.plot(func_r_values, y_values, marker='o', label=func_name)
        
        plt.xlabel('r value (tokens to merge)')
        plt.ylabel('Throughput (images/s)')
        plt.title('Throughput vs. Token Reduction (r)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'throughput.png'))
        
        # Plot memory-throughput trade-off for each r value
        for r in r_values:
            plt.figure(figsize=(10, 8))
            x_values = []
            y_values = []
            labels = []
            
            for func_name in functions:
                if r in functions[func_name]:
                    x_values.append(self.memory_usage[f"{func_name}_r{r}"])
                    y_values.append(self.throughputs[f"{func_name}_r{r}"])
                    labels.append(func_name)
            
            plt.scatter(x_values, y_values, s=100)
            
            # Add labels to points
            for i, label in enumerate(labels):
                plt.annotate(label, (x_values[i], y_values[i]), 
                           xytext=(10, 5), textcoords='offset points')
            
            plt.xlabel('Memory Usage (MB)')
            plt.ylabel('Throughput (images/s)')
            plt.title(f'Memory-Throughput Trade-off for r={r}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'tradeoff_r{r}.png'))
            plt.close()

def get_model(model_name="vit_base_patch16_224", pretrained=True):
    """Create and return the model to benchmark"""
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

def patch_model_with_tome(model, merge_function, r_value):
    """Patch a model with ToMe using the specified merge function"""
    if not hasattr(model, 'blocks'):
        raise ValueError("Model does not have 'blocks' attribute. This script is designed for ViT models.")
    
    # Save the original attention functions
    original_attentions = []
    for block in model.blocks:
        if hasattr(block, 'attn'):
            original_attentions.append(block.attn.forward)
    
    # Apply merge function to each block
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn'):
            # Monkey patch the attention forward function
            orig_forward = block.attn.forward
            
            def make_new_forward(block_idx, original_forward, merge_fn):
                def new_forward(x, *args, **kwargs):
                    # Get the class token flag
                    class_token = hasattr(model, 'cls_token')
                    
                    # Apply merge before attention
                    B, N, C = x.shape
                    merge_op, unmerge_op = merge_fn(x, r_value, class_token=class_token)
                    x = merge_op(x)
                    
                    # Apply original attention
                    x = original_forward(x, *args, **kwargs)
                    
                    # Apply unmerge after attention
                    x = unmerge_op(x)
                    return x
                
                return new_forward
            
            block.attn.forward = make_new_forward(i, orig_forward, merge_function)
    
    return original_attentions

def restore_original_attention(model, original_attentions):
    """Restore the original attention functions"""
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'attn') and i < len(original_attentions):
            block.attn.forward = original_attentions[i]

def benchmark_memory(model, batch_size, input_size, device, warmup=5, runs=20):
    """Benchmark memory usage of the model using random data"""
    model.eval()
    model.to(device)
    
    # Create random input tensors
    random_inputs = torch.randn(batch_size, 3, input_size[1], input_size[2], device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(random_inputs)
    
    # Reset peak memory stats
    reset_peak_memory_stats(device)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(runs):
        with torch.no_grad():
            _ = model(random_inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Get peak memory usage in MB
    memory_usage = max_memory_allocated(device) / (1024 * 1024)
    
    # Calculate throughput
    total_time = end_time - start_time
    total_images = batch_size * runs
    throughput = total_images / total_time
    inference_time = total_time / runs  # Average time per batch
    
    return memory_usage, inference_time, throughput

def main():
    parser = argparse.ArgumentParser(description='Benchmark ToMe merge functions memory usage')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                        help='Model name from timm (default: vit_base_patch16_224)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for benchmark (default: 32)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for evaluation (default: cuda:0)')
    parser.add_argument('--r-values', type=int, nargs='+', default=[8, 16, 24, 32],
                        help='List of r values to benchmark (default: [8, 16, 24, 32])')
    parser.add_argument('--functions', type=str, nargs='+', 
                        default=list(MERGE_FUNCTIONS.keys()),
                        help=f'List of merge functions to benchmark (default: all)')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                        help='Directory to save benchmark results (default: ./benchmark_results)')
    parser.add_argument('--runs', type=int, default=20,
                        help='Number of runs for benchmarking (default: 20)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model weights (slower to load but more realistic)')
    
    # Add IGA specific parameters
    parser.add_argument('--iga-iterations', type=int, default=2,
                        help='Number of iterations for IGA algorithm (default: 2)')
    parser.add_argument('--iga-memory-efficient', action='store_true',
                        help='Use memory-efficient mode for IGA algorithm')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    # Adjust batch size for CPU
    if args.device == 'cpu':
        args.batch_size = min(args.batch_size, 8)
        args.runs = min(args.runs, 5)
        print(f"Adjusted batch size to {args.batch_size} and runs to {args.runs} for CPU testing")
    
    # Initialize results object
    results = BenchmarkResults()
    
    # Get baseline model performance
    print(f"Loading base model: {args.model}...")
    model = get_model(args.model, pretrained=args.pretrained)
    input_size = model.default_cfg["input_size"]
    
    print(f"Model: {args.model}")
    print(f"Input size: {input_size}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    print("\nMeasuring baseline memory usage...")
    baseline_memory, baseline_time, baseline_throughput = benchmark_memory(
        model, args.batch_size, input_size, args.device, 
        warmup=args.warmup, runs=args.runs
    )
    print(f"Baseline memory usage: {baseline_memory:.2f} MB")
    print(f"Baseline inference time: {baseline_time*1000:.2f} ms per batch")
    print(f"Baseline throughput: {baseline_throughput:.2f} images/s")
    
    # Add baseline to results
    results.add_result("baseline", 0, baseline_memory, baseline_time, baseline_throughput)
    
    # Create a dictionary to store the function runtimes
    function_runtimes = {}
    
    # Benchmark each merge function with different r values
    for func_name in args.functions:
        if func_name not in MERGE_FUNCTIONS:
            print(f"Warning: {func_name} is not a valid merge function. Skipping.")
            continue
        
        merge_function = MERGE_FUNCTIONS[func_name]
        function_runtimes[func_name] = {}
        
        # Set IGA specific parameters if testing IGA
        if func_name == 'iga_bipartite_soft_matching':
            # Create a wrapper function to pass additional IGA parameters
            orig_merge_function = merge_function
            merge_function = lambda metric, r, **kwargs: orig_merge_function(
                metric, r, 
                max_iterations=args.iga_iterations,
                memory_efficient=args.iga_memory_efficient,
                **kwargs
            )
        
        for r in args.r_values:
            print(f"\nBenchmarking {func_name} with r={r}...")
            
            # Create a new model instance to avoid interference
            model = get_model(args.model, pretrained=args.pretrained)
            
            # Measure the function's runtime separately
            # Create a dummy input tensor similar to what would be passed to the merge function
            dummy_batch_size = 1
            dummy_seq_len = 197  # 196 patches + 1 class token for 224x224 ViT
            dummy_dim = model.embed_dim
            dummy_input = torch.randn(dummy_batch_size, dummy_seq_len, dummy_dim, device=args.device)
            
            # Measure time to run the merge function
            torch.cuda.synchronize()
            func_start = time.time()
            for _ in range(100):  # Run multiple times for more accurate measurement
                with torch.no_grad():
                    class_token = True  # Most ViT models have a class token
                    _, _ = merge_function(dummy_input, r, class_token=class_token)
            torch.cuda.synchronize()
            func_end = time.time()
            func_time = (func_end - func_start) / 100
            function_runtimes[func_name][r] = func_time
            
            # Patch the model with the current merge function
            original_attentions = patch_model_with_tome(model, merge_function, r)
            
            # Measure memory usage
            memory_usage, inference_time, throughput = benchmark_memory(
                model, args.batch_size, input_size, args.device,
                warmup=args.warmup, runs=args.runs
            )
            print(f"Memory usage: {memory_usage:.2f} MB ({memory_usage/baseline_memory:.2f}x baseline)")
            print(f"Inference time: {inference_time*1000:.2f} ms per batch ({baseline_time/inference_time:.2f}x faster)")
            print(f"Throughput: {throughput:.2f} images/s ({throughput/baseline_throughput:.2f}x baseline)")
            
            # Add result
            results.add_result(func_name, r, memory_usage, inference_time, throughput)
            
            # Restore original attention functions
            restore_original_attention(model, original_attentions)
            
            # Clean up
            del model
            torch.cuda.empty_cache()
    
    # Print and plot results
    results.print_summary()
    results.plot_results(args.output_dir)
    print(f"Plots saved to {args.output_dir}")
    
    # Print function runtimes
    print("\n===== MERGE FUNCTION RUNTIMES =====")
    print(f"{'Function':30} {'r':5} {'Runtime (ms)':15}")
    print("-" * 55)
    
    for func_name in function_runtimes:
        for r in sorted(function_runtimes[func_name].keys()):
            runtime = function_runtimes[func_name][r] * 1000  # Convert to ms
            print(f"{func_name:30} {r:5} {runtime:15.4f}")

if __name__ == '__main__':
    main()