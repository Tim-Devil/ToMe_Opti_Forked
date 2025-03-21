# import timm
# import tome
# import torch
# import gc

# def measure_memory_usage(func, *args, **kwargs):
#     """测量函数执行过程中的峰值GPU显存使用量"""
#     # 清空缓存
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 记录初始显存使用量
#     torch.cuda.reset_peak_memory_stats()
#     initial_memory = torch.cuda.memory_allocated()
    
#     # 执行函数
#     result = func(*args, **kwargs)
    
#     # 获取峰值显存使用量
#     peak_memory = torch.cuda.max_memory_allocated()
    
#     # 计算显存增长量（MB）
#     memory_used = (peak_memory - initial_memory) / (1024 ** 2)
    
#     return result, memory_used

# # Use any ViT model here (see timm.models.vision_transformer)
# model_name = "vit_base_patch16_224"

# # Load a pretrained model
# model = timm.create_model(model_name, pretrained=True)

# # Set this to be whatever device you want to benchmark on
# # If you don't have a GPU, you can use "cpu" but you probably want to set the # runs to be lower
# device = "cuda:0"
# runs = 50
# batch_size = 256  # Lower this if you don't have that much memory
# input_size = model.default_cfg["input_size"]

# # 创建一个包装函数来运行benchmark
# def run_benchmark(model):
#     return tome.utils.benchmark(
#         model,
#         device=device,
#         verbose=True,
#         runs=runs,
#         batch_size=batch_size,
#         input_size=input_size
#     )

# print(f"Model: {model_name}")
# print(f"Batch size: {batch_size}")
# print(f"Input size: {input_size}")
# print("-" * 50)

# # Baseline benchmark with memory measurement
# print("\nBaseline (Original Model):")
# baseline_throughput, baseline_memory = measure_memory_usage(run_benchmark, model)
# print(f"Memory usage: {baseline_memory:.2f} MB")

# # Applying ToMe
# # Simply patch the model after initialization to enable ToMe.
# tome.patch.timm(model)

# # ToMe with r=16
# print("\nToMe with r=16:")
# model.r = 16
# tome_throughput, tome_memory = measure_memory_usage(run_benchmark, model)
# print(f"Memory usage: {tome_memory:.2f} MB")
# print(f"Memory reduction: {baseline_memory / tome_memory:.2f}x")
# print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

# # ToMe with r=16 and a decreasing schedule
# print("\nToMe with r=(16, -1.0) (decreasing schedule):")
# model.r = (16, -1.0)
# tome_decr_throughput, tome_decr_memory = measure_memory_usage(run_benchmark, model)
# print(f"Memory usage: {tome_decr_memory:.2f} MB")
# print(f"Memory reduction: {baseline_memory / tome_decr_memory:.2f}x")
# print(f"Throughput improvement: {tome_decr_throughput / baseline_throughput:.2f}x")

# print("-" * 50)

# # 测试其它的r值以观察显存和速度的变化
# print("\nTest other r value:")
# r_values = [4, 8, 12, 20, 24]

# results = []
# for r in r_values:
#     print(f"\nToMe with r={r}:")
#     model.r = r
#     throughput, memory = measure_memory_usage(run_benchmark, model)
#     speedup = throughput / baseline_throughput
#     mem_reduction = baseline_memory / memory
    
#     results.append((r, throughput, memory, speedup, mem_reduction))
    
#     print(f"Memory usage: {memory:.2f} MB")
#     print(f"Memory reduction: {mem_reduction:.2f}x")
#     print(f"Throughput improvement: {speedup:.2f}x")

# # 打印结果表格
# print("\nResult:")
# print(f"{'r Value':^5} | {'Throughput':^8} | {'Memory(MB)':^10} | {'Accelerate ratio':^8} | {'Saved Memory':^8}")
# print("-" * 50)
# for r, throughput, memory, speedup, mem_reduction in results:
#     print(f"{r:^5} | {throughput:^8.2f} | {memory:^10.2f} | {speedup:^8.2f}x | {mem_reduction:^8.2f}x")



# import timm
# import tome
# import torch
# import gc
# import numpy as np
# from tqdm import tqdm

# def measure_representation_changes(model, model_with_tome, num_samples=100, r_value=16):
#     """Measure how ToMe affects model representations using random inputs"""
#     device = next(model.parameters()).device
#     input_size = model.default_cfg["input_size"]
#     batch_size = 1  # Process one image at a time to measure detailed changes
    
#     # Set up ToMe with specific r value
#     if isinstance(model_with_tome.r, tuple):
#         r_desc = f"{model_with_tome.r[0]},{model_with_tome.r[1]}"
#     else:
#         r_desc = str(model_with_tome.r)
        
#     print(f"Testing with r={r_desc}")
    
#     # Metrics to track
#     output_diffs = []
#     output_rank_changes = []
    
#     model.eval()
#     model_with_tome.eval()
    
#     with torch.no_grad():
#         for i in tqdm(range(num_samples)):
#             # Create random input with proper normalization
#             random_input = torch.randn(batch_size, *input_size).to(device)
            
#             # Get original model outputs
#             original_output = model(random_input)
            
#             # Get ToMe model outputs
#             tome_output = model_with_tome(random_input)
            
#             # Calculate differences in outputs
#             # L2 distance between output logits, normalized by feature dimension
#             output_diff = torch.norm(original_output - tome_output) / np.sqrt(original_output.shape[-1])
#             output_diffs.append(output_diff.item())
            
#             # How much do the top-k predictions change?
#             k = 5
#             orig_topk = torch.topk(original_output, k, dim=-1).indices.squeeze()
#             tome_topk = torch.topk(tome_output, k, dim=-1).indices.squeeze()
            
#             # Count how many of the top-k predictions changed
#             changes = k - len(set(orig_topk.cpu().numpy()).intersection(set(tome_topk.cpu().numpy())))
#             output_rank_changes.append(changes)
            
#     # Compute summary statistics    
#     results = {
#         "output_diff_mean": np.mean(output_diffs),
#         "output_diff_std": np.std(output_diffs),
#         "top5_changes_mean": np.mean(output_rank_changes),
#         "top5_changes_pct": np.mean(output_rank_changes) / k * 100
#     }
    
#     return results

# def measure_memory_usage(func, *args, **kwargs):
#     """测量函数执行过程中的峰值GPU显存使用量"""
#     # 清空缓存
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # 记录初始显存使用量
#     torch.cuda.reset_peak_memory_stats()
#     initial_memory = torch.cuda.memory_allocated()
    
#     # 执行函数
#     result = func(*args, **kwargs)
    
#     # 获取峰值显存使用量
#     peak_memory = torch.cuda.max_memory_allocated()
    
#     # 计算显存增长量（MB）
#     memory_used = (peak_memory - initial_memory) / (1024 ** 2)
    
#     return result, memory_used

# # Use any ViT model here
# model_name = "vit_base_patch16_224"

# # Load a pretrained model
# model = timm.create_model(model_name, pretrained=True)

# # Set device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# # Parameters for benchmarking
# runs = 20  # Reduced for quicker testing
# batch_size = 128  # Adjust based on your GPU memory
# input_size = model.default_cfg["input_size"]

# # 创建一个包装函数来运行benchmark
# def run_benchmark(model):
#     return tome.utils.benchmark(
#         model,
#         device=device,
#         verbose=False,  # Set to False to reduce output
#         runs=runs,
#         batch_size=batch_size,
#         input_size=input_size
#     )

# print(f"Model: {model_name}")
# print(f"Batch size: {batch_size}")
# print(f"Input size: {input_size}")
# print("-" * 50)

# # Baseline benchmark with memory measurement
# print("\nBaseline (Original Model):")
# baseline_throughput, baseline_memory = measure_memory_usage(run_benchmark, model)
# print(f"Memory usage: {baseline_memory:.2f} MB")

# # Create a ToMe model for each r value to test
# r_values = [4, 8, 12, 16, 20, 24]
# results = []

# for r in r_values:
#     print(f"\nToMe with r={r}:")
    
#     # Create a fresh model instance for each test
#     tome_model = timm.create_model(model_name, pretrained=True)
#     tome_model = tome_model.to(device)
#     tome.patch.timm(tome_model)
#     tome_model.r = r
    
#     # Measure memory and throughput
#     throughput, memory = measure_memory_usage(run_benchmark, tome_model)
#     speedup = throughput / baseline_throughput
#     mem_reduction = baseline_memory / memory
    
#     print(f"Memory usage: {memory:.2f} MB")
#     print(f"Memory reduction: {mem_reduction:.2f}x")
#     print(f"Throughput improvement: {speedup:.2f}x")
    
#     # Measure representation changes as a proxy for accuracy
#     accuracy_proxy = measure_representation_changes(model, tome_model, num_samples=100, r_value=r)
    
#     # Store results
#     results.append((r, throughput, memory, speedup, mem_reduction, accuracy_proxy))
    
#     print(f"Output diff: {accuracy_proxy['output_diff_mean']:.4f} ± {accuracy_proxy['output_diff_std']:.4f}")
#     print(f"Top-5 prediction changes: {accuracy_proxy['top5_changes_mean']:.2f} ({accuracy_proxy['top5_changes_pct']:.1f}%)")

# # Also test decreasing schedule
# print("\nToMe with r=(16, -1.0) (decreasing schedule):")
# decr_model = timm.create_model(model_name, pretrained=True)
# decr_model = decr_model.to(device)
# tome.patch.timm(decr_model)
# decr_model.r = (16, -1.0)

# decr_throughput, decr_memory = measure_memory_usage(run_benchmark, decr_model)
# decr_speedup = decr_throughput / baseline_throughput
# decr_mem_reduction = baseline_memory / decr_memory

# decr_accuracy_proxy = measure_representation_changes(model, decr_model, num_samples=100, r_value=(16, -1.0))

# print(f"Memory usage: {decr_memory:.2f} MB")
# print(f"Memory reduction: {decr_mem_reduction:.2f}x")
# print(f"Throughput improvement: {decr_speedup:.2f}x")
# print(f"Output diff: {decr_accuracy_proxy['output_diff_mean']:.4f} ± {decr_accuracy_proxy['output_diff_std']:.4f}")
# print(f"Top-5 prediction changes: {decr_accuracy_proxy['top5_changes_mean']:.2f} ({decr_accuracy_proxy['top5_changes_pct']:.1f}%)")

# # Print results table
# print("\nResult Table:")
# print(f"{'r Value':^8} | {'Throughput':^10} | {'Memory(MB)':^10} | {'Speed Up':^8} | {'Mem Saved':^9} | {'Output Diff':^10} | {'Top-5 Δ%':^8}")
# print("-" * 80)
# print(f"{'Original':^8} | {baseline_throughput:^10.2f} | {baseline_memory:^10.2f} | {1.00:^8.2f}x | {1.00:^9.2f}x | {0.00:^10.2f} | {0.0:^8.1f}")

# for r, throughput, memory, speedup, mem_reduction, acc_proxy in results:
#     print(f"{r:^8} | {throughput:^10.2f} | {memory:^10.2f} | {speedup:^8.2f}x | {mem_reduction:^9.2f}x | {acc_proxy['output_diff_mean']:^10.4f} | {acc_proxy['top5_changes_pct']:^8.1f}")

# print(f"{'16, -1.0':^8} | {decr_throughput:^10.2f} | {decr_memory:^10.2f} | {decr_speedup:^8.2f}x | {decr_mem_reduction:^9.2f}x | {decr_accuracy_proxy['output_diff_mean']:^10.4f} | {decr_accuracy_proxy['top5_changes_pct']:^8.1f}")


import timm
import tome
import torch
import gc
import numpy as np
from tqdm import tqdm
import importlib
import sys
import time
from functools import partial

# Add path to import merge module
sys.path.append('.')  # Assumes merge.py is in the current directory

# Import merge functions for testing
# We'll dynamically replace functions later
from tome.merge import (
    bipartite_soft_matching,
    grouped_bipartite_soft_matching,
    kmeans_bipartite_soft_matching
)

def measure_representation_changes(model, model_with_tome, num_samples=50):
    """Measure how ToMe affects model representations using random inputs"""
    device = next(model.parameters()).device
    input_size = model.default_cfg["input_size"]
    batch_size = 1  # Process one image at a time to measure detailed changes
    
    # Set up ToMe description
    if hasattr(model_with_tome, 'r'):
        if isinstance(model_with_tome.r, tuple):
            r_desc = f"{model_with_tome.r[0]},{model_with_tome.r[1]}"
        else:
            r_desc = str(model_with_tome.r)
        print(f"Testing with r={r_desc}")
    
    # Metrics to track
    output_diffs = []
    output_rank_changes = []
    
    model.eval()
    model_with_tome.eval()
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Measuring representation changes"):
            # Create random input with proper normalization
            random_input = torch.randn(batch_size, *input_size).to(device)
            
            # Get original model outputs
            original_output = model(random_input)
            
            # Get ToMe model outputs
            tome_output = model_with_tome(random_input)
            
            # Calculate differences in outputs
            # L2 distance between output logits, normalized by feature dimension
            output_diff = torch.norm(original_output - tome_output) / np.sqrt(original_output.shape[-1])
            output_diffs.append(output_diff.item())
            
            # How much do the top-k predictions change?
            k = 5
            orig_topk = torch.topk(original_output, k, dim=-1).indices.squeeze()
            tome_topk = torch.topk(tome_output, k, dim=-1).indices.squeeze()
            
            # Count how many of the top-k predictions changed
            changes = k - len(set(orig_topk.cpu().numpy()).intersection(set(tome_topk.cpu().numpy())))
            output_rank_changes.append(changes)
            
    # Compute summary statistics    
    results = {
        "output_diff_mean": np.mean(output_diffs),
        "output_diff_std": np.std(output_diffs),
        "top5_changes_mean": np.mean(output_rank_changes),
        "top5_changes_pct": np.mean(output_rank_changes) / k * 100
    }
    
    return results

def measure_memory_usage(func, *args, **kwargs):
    """Measure peak GPU memory usage during function execution"""
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Record initial memory usage
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Execute function
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Calculate memory growth (MB)
    memory_used = (peak_memory - initial_memory) / (1024 ** 2)
    
    return result, memory_used, execution_time

def patch_model_with_merge_function(model, merge_function_name):
    """
    Patch the model with a specific merge function from the merge module
    """
    # First apply standard ToMe patching
    tome.patch.timm(model)
    
    # Now override the bipartite_soft_matching function with our desired function
    if merge_function_name == "bipartite_soft_matching":
        # This is already the default, no need to replace
        pass
    elif merge_function_name == "grouped_bipartite_soft_matching":
        # Replace the default matching function with our grouped version
        from tome import merge
        importlib.reload(merge)  # Ensure we have the latest version
        tome.bipartite_soft_matching = merge.grouped_bipartite_soft_matching
    elif merge_function_name == "kmeans_bipartite_soft_matching":
        # Replace with K-means version
        from tome import merge
        importlib.reload(merge)
        tome.bipartite_soft_matching = merge.kmeans_bipartite_soft_matching
    else:
        raise ValueError(f"Unknown merge function: {merge_function_name}")
    
    return model

def run_benchmark(model, runs=20, batch_size=128):
    """Run performance benchmark on the model"""
    input_size = model.default_cfg["input_size"]
    device = next(model.parameters()).device
    
    return tome.utils.benchmark(
        model,
        device=device,
        verbose=False,
        runs=runs,
        batch_size=batch_size,
        input_size=input_size
    )

def main():
    # Model and device setup
    model_name = "vit_base_patch16_224"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Parameters for benchmarking
    runs = 20  # Reduced for quicker testing
    batch_size = 64  # Adjust based on your GPU memory
    r_value = 16  # Token reduction parameter
    
    # Load the base model for comparison
    print(f"Loading model: {model_name}")
    base_model = timm.create_model(model_name, pretrained=True)
    base_model = base_model.to(device)
    
    # Get baseline performance
    print("\nBaseline (Original Model):")
    baseline_throughput, baseline_memory, baseline_time = measure_memory_usage(
        partial(run_benchmark, base_model, runs=runs, batch_size=batch_size)
    )
    print(f"Memory usage: {baseline_memory:.2f} MB")
    print(f"Execution time: {baseline_time:.2f} seconds")
    
    # Merge functions to test
    merge_functions = [
        "bipartite_soft_matching",
        "grouped_bipartite_soft_matching",
        "kmeans_bipartite_soft_matching"
    ]
    
    # Results collection
    results = []
    
    # Test each merge function
    for merge_func in merge_functions:
        print(f"\nTesting {merge_func}:")
        
        # Create a fresh model instance
        tome_model = timm.create_model(model_name, pretrained=True)
        tome_model = tome_model.to(device)
        
        # Patch with the specific merge function
        tome_model = patch_model_with_merge_function(tome_model, merge_func)
        tome_model.r = r_value
        
        # Measure performance
        throughput, memory, exec_time = measure_memory_usage(
            partial(run_benchmark, tome_model, runs=runs, batch_size=batch_size)
        )
        
        speedup = throughput / baseline_throughput
        mem_reduction = baseline_memory / memory
        
        print(f"Memory usage: {memory:.2f} MB")
        print(f"Memory reduction: {mem_reduction:.2f}x")
        print(f"Throughput improvement: {speedup:.2f}x")
        print(f"Execution time: {exec_time:.2f} seconds")
        
        # Measure representation changes (accuracy proxy)
        accuracy_proxy = measure_representation_changes(base_model, tome_model, num_samples=50)
        
        print(f"Output diff: {accuracy_proxy['output_diff_mean']:.4f} ± {accuracy_proxy['output_diff_std']:.4f}")
        print(f"Top-5 prediction changes: {accuracy_proxy['top5_changes_mean']:.2f} ({accuracy_proxy['top5_changes_pct']:.1f}%)")
        
        # Store results
        results.append((
            merge_func, 
            throughput, 
            memory, 
            speedup, 
            mem_reduction, 
            exec_time,
            accuracy_proxy
        ))
    
    # Print results table
    print("\nResults Summary:")
    print(f"{'Merge Function':^25} | {'Throughput':^10} | {'Memory(MB)':^10} | {'Speed Up':^8} | {'Mem Saved':^9} | {'Time(s)':^8} | {'Output Diff':^10} | {'Top-5 Δ%':^8}")
    print("-" * 110)
    
    print(f"{'Original':^25} | {baseline_throughput:^10.2f} | {baseline_memory:^10.2f} | {1.00:^8.2f}x | {1.00:^9.2f}x | {baseline_time:^8.2f} | {0.00:^10.2f} | {0.0:^8.1f}")
    
    for func, throughput, memory, speedup, mem_reduction, exec_time, acc_proxy in results:
        print(f"{func:^25} | {throughput:^10.2f} | {memory:^10.2f} | {speedup:^8.2f}x | {mem_reduction:^9.2f}x | {exec_time:^8.2f} | {acc_proxy['output_diff_mean']:^10.4f} | {acc_proxy['top5_changes_pct']:^8.1f}")

if __name__ == "__main__":
    main()