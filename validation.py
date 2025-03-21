# import torch
# from torch.cuda import memory_allocated
# from tome.merge import bipartite_soft_matching, grouped_bipartite_soft_matching


# def memory_usage(func, metric, r, class_token, distill_token):
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()

#     merge, _ = func(metric, r, class_token, distill_token)

#     x = torch.randn_like(metric, device="cuda")
#     merged_x = merge(x)

#     torch.cuda.synchronize()

#     memory_used = memory_allocated() / (1024 ** 2)  # Convert to MB

#     del merged_x
#     torch.cuda.empty_cache()

#     return memory_used


# def compare_memory():
#     metric = torch.randn(16, 512, 768, device="cuda")  # [batch, tokens, channels]
#     r = 128
#     class_token = True
#     distill_token = False

#     mem_grouped = memory_usage(grouped_bipartite_soft_matching, metric, r, class_token, distill_token)
#     mem_bipartite = memory_usage(bipartite_soft_matching, metric, r, class_token, distill_token)

#     print(f"Grouped Bipartite Soft Matching memory usage: {mem_grouped:.2f} MB")
#     print(f"Bipartite Soft Matching memory usage: {mem_bipartite:.2f} MB")


# if __name__ == '__main__':
#     compare_memory()

# import torch
# import timm
# import tome
# from tome.merge import bipartite_soft_matching, grouped_bipartite_soft_matching

# def measure_gpu_memory(model, merge_func, device, batch_size, input_size, r):
#     torch.cuda.empty_cache()

#     # 显式设置merge_func
#     model._tome_info["merge_func"] = merge_func
#     model.r = r

#     dummy_input = torch.randn(batch_size, *input_size, device=device)

#     with torch.no_grad():
#         torch.cuda.reset_peak_memory_stats(device)
#         model(dummy_input)
#         torch.cuda.synchronize(device)
#         peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

#     return peak_memory

# def compare_memory_with_different_r(r_values, batch_size=32, device="cuda:0"):
#     model_name = "vit_large_patch16_224"
#     input_size = (3, 224, 224)

#     for r in r_values:
#         # grouped matching
#         model_grouped = timm.create_model(model_name, pretrained=False).to(device)
#         tome.patch.timm(model_grouped)

#         mem_grouped = measure_gpu_memory(
#             model_grouped,
#             lambda metric, r, class_token=False, distill_token=False:
#                 grouped_bipartite_soft_matching(metric, r, class_token, distill_token, chunk_size=2),
#             device, batch_size, input_size, r
#         )

#         del model_grouped
#         torch.cuda.empty_cache()

#         # bipartite matching
#         model_bipartite = timm.create_model(model_name, pretrained=False).to(device)
#         tome.patch.timm(model_bipartite)

#         mem_bipartite = measure_gpu_memory(
#             model_bipartite,
#             bipartite_soft_matching,
#             device, batch_size, input_size, r
#         )

#         del model_bipartite
#         torch.cuda.empty_cache()

#         print(f"r={r} | Grouped Matching: {mem_grouped:.2f} MB | Bipartite Matching: {mem_bipartite:.2f} MB")

# if __name__ == '__main__':
#     r_values = [8, 16, 32, 64, 128]
#     compare_memory_with_different_r(r_values, batch_size=256, device="cuda:0")



import torch
from torch.cuda import memory_allocated
from tome.merge import bipartite_soft_matching, grouped_bipartite_soft_matching

def memory_usage(func, metric, r, class_token, distill_token):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    merge, _ = func(metric, r, class_token, distill_token)
    x = torch.randn_like(metric, device="cuda")
    merged_x = merge(x)

    torch.cuda.synchronize()
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)

    del merged_x, x, merge
    torch.cuda.empty_cache()
    return memory_used

def compare_memory_different_r(r_values, batch_size, tokens, channels, chunk_size=2):
    metric = torch.randn(batch_size, tokens, channels, device="cuda")
    class_token = True
    distill_token = False

    for r in r_values:
        mem_grouped = memory_usage(
            lambda m, r, c, d: grouped_bipartite_soft_matching(m, r, c, d, chunk_size=chunk_size),
            metric, r, class_token, distill_token
        )
        mem_bipartite = memory_usage(
            bipartite_soft_matching, metric, r, class_token, distill_token
        )

        print(f"r={r} | Grouped Matching (chunk={chunk_size}): {mem_grouped:.2f} MB | Bipartite Matching: {mem_bipartite:.2f} MB")

if __name__ == '__main__':
    r_values = [128, 256, 512, 768, 1024]
    chunk = [2, 4, 8, 16, 32, 64, 128, 256]
    compare_memory_different_r(r_values, batch_size=16, tokens=2048, channels=768, chunk_size=2)
