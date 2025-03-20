# import timm
# import tome

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

# # Baseline benchmark
# baseline_throughput = tome.utils.benchmark(
#     model,
#     device=device,
#     verbose=True,
#     runs=runs,
#     batch_size=batch_size,
#     input_size=input_size
# )

# # Applying ToMe
# # Simply patch the model after initialization to enable ToMe.

# # Apply ToMe
# tome.patch.timm(model)

# # ToMe with r=16
# model.r = 16
# tome_throughput = tome.utils.benchmark(
#     model,
#     device=device,
#     verbose=True,
#     runs=runs,
#     batch_size=batch_size,
#     input_size=input_size
# )
# print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

# # ToMe with r=16 and a decreasing schedule
# model.r = (16, -1.0)
# tome_decr_throughput = tome.utils.benchmark(
#     model,
#     device=device,
#     verbose=True,
#     runs=runs,
#     batch_size=batch_size,
#     input_size=input_size
# )
# print(f"Throughput improvement: {tome_decr_throughput / baseline_throughput:.2f}x")


import timm
import tome
import torch
import gc

def measure_memory_usage(func, *args, **kwargs):
    """测量函数执行过程中的峰值GPU显存使用量"""
    # 清空缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录初始显存使用量
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # 执行函数
    result = func(*args, **kwargs)
    
    # 获取峰值显存使用量
    peak_memory = torch.cuda.max_memory_allocated()
    
    # 计算显存增长量（MB）
    memory_used = (peak_memory - initial_memory) / (1024 ** 2)
    
    return result, memory_used

# Use any ViT model here (see timm.models.vision_transformer)
model_name = "vit_base_patch16_224"

# Load a pretrained model
model = timm.create_model(model_name, pretrained=True)

# Set this to be whatever device you want to benchmark on
# If you don't have a GPU, you can use "cpu" but you probably want to set the # runs to be lower
device = "cuda:0"
runs = 50
batch_size = 256  # Lower this if you don't have that much memory
input_size = model.default_cfg["input_size"]

# 创建一个包装函数来运行benchmark
def run_benchmark(model):
    return tome.utils.benchmark(
        model,
        device=device,
        verbose=True,
        runs=runs,
        batch_size=batch_size,
        input_size=input_size
    )

print(f"Model: {model_name}")
print(f"Batch size: {batch_size}")
print(f"Input size: {input_size}")
print("-" * 50)

# Baseline benchmark with memory measurement
print("\nBaseline (Original Model):")
baseline_throughput, baseline_memory = measure_memory_usage(run_benchmark, model)
print(f"Memory usage: {baseline_memory:.2f} MB")

# Applying ToMe
# Simply patch the model after initialization to enable ToMe.
tome.patch.timm(model)

# ToMe with r=16
print("\nToMe with r=16:")
model.r = 16
tome_throughput, tome_memory = measure_memory_usage(run_benchmark, model)
print(f"Memory usage: {tome_memory:.2f} MB")
print(f"Memory reduction: {baseline_memory / tome_memory:.2f}x")
print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")

# ToMe with r=16 and a decreasing schedule
print("\nToMe with r=(16, -1.0) (decreasing schedule):")
model.r = (16, -1.0)
tome_decr_throughput, tome_decr_memory = measure_memory_usage(run_benchmark, model)
print(f"Memory usage: {tome_decr_memory:.2f} MB")
print(f"Memory reduction: {baseline_memory / tome_decr_memory:.2f}x")
print(f"Throughput improvement: {tome_decr_throughput / baseline_throughput:.2f}x")

# 测试不同的r值以观察显存和速度的变化
print("\n测试不同的r值:")
r_values = [4, 8, 12, 20, 24]

results = []
for r in r_values:
    print(f"\nToMe with r={r}:")
    model.r = r
    throughput, memory = measure_memory_usage(run_benchmark, model)
    speedup = throughput / baseline_throughput
    mem_reduction = baseline_memory / memory
    
    results.append((r, throughput, memory, speedup, mem_reduction))
    
    print(f"Memory usage: {memory:.2f} MB")
    print(f"Memory reduction: {mem_reduction:.2f}x")
    print(f"Throughput improvement: {speedup:.2f}x")

# 打印结果表格
print("\n结果汇总:")
print(f"{'r值':^5} | {'吞吐量':^8} | {'显存(MB)':^10} | {'加速比':^8} | {'显存减少':^8}")
print("-" * 50)
for r, throughput, memory, speedup, mem_reduction in results:
    print(f"{r:^5} | {throughput:^8.2f} | {memory:^10.2f} | {speedup:^8.2f}x | {mem_reduction:^8.2f}x")