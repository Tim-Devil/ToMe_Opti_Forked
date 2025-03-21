# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    # Return the size of metric's second dimension(token num.)
    r = min(r, (t - protected) // 2)
    # In reality we should reduce how much token.
    # Should not exceed half of 't'.

    if r <= 0:
        return do_nothing, do_nothing
    
    # Just do nothing. Protection method.

    with torch.no_grad():
        # create context and clear it automatically.
        # Use it to disable grad computation temporarily.

        metric = metric / metric.norm(dim=-1, keepdim=True)
        # normalize it by L2.

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        # divide it to odd and even index.
        scores = a @ b.transpose(-1, -2)
        # compute similarity score between a and b.

        # do masking operation.
        if class_token:
            scores[..., 0, :] = -math.inf
        # class token: (0, s)
        if distill_token:
            scores[..., :, 0] = -math.inf
        # distill token: (s, 0)

        node_max, node_idx = scores.max(dim=-1)
        # max: return `max num` and `max index`.
        # dim=-1: find max num in a row
        # about dim: https://zhuanlan.zhihu.com/p/525276061
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        # argsort: sort and return index. 
        # the highest similarity should be merged first.

        unm_idx = edge_idx[..., r:, :]  
        # store Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  
        # store Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # use None to shape it. Keep shape alignment.

        # Put class token at first to protect it.
        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        # odd and even element.

        n, t1, c = src.shape
        # n: size of batch(row)
        # t1: token length(col)
        # c: 

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    # Used to restore origin position.
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

# ------------------------------------------------------------------------------------------

'''
ADD:
1. grouped_bipartite_soft_matching:
    Use chunk technique to make wlb between speed and memory. 
    You can achieve it by adjusting `chunk_size`.
    (A general method right?)

2. K_means_bipartite_soft_matching:
    Rewrite the whole judging standard.
    Use k-means to compute the nearest element.
    (All in all we just need to know what we need to merge.)
    OK so we add group technique. After all they can coexist.
    Simply using K_means may also save Memory. Maybe.

    Update: OK I must confess that it is useless.
    Because we still need hadamard times.
    ......Waste my time. Skip this.

3. IGA_bipartite_soft_matching:
    Use Iterative greedy algorithm.
    ? Do we need to merge and compute a new score???

'''

# ------------------------------------------------------------------------------------------
# Grouped:

# def grouped_bipartite_soft_matching(
#     metric: torch.Tensor,   # input matrix
#     r: int, # expected num of removed token
#     class_token: bool = False,
#     distill_token: bool = False,
#     chunk_size: int = 64,  
#     # size of chunk. Used to make wlb between speed and memory.
# ) -> Tuple[Callable, Callable]:
    
#     # Nothing special.
#     protected = 0
#     if class_token:
#         protected += 1
#     if distill_token:
#         protected += 1

#     t = metric.shape[1]
#     r = min(r, (t - protected) // 2)

#     if r <= 0:
#         return do_nothing, do_nothing

#     # Compute similarity score matrix(with chunk technique)
#     with torch.no_grad():
#         # Normalize
#         metric = metric / metric.norm(dim=-1, keepdim=True)
        
#         # odd and even divide.
#         a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
#         # Follow my habit. First determine data struct.
#         batch_size = metric.shape[0]
#         num_a_tokens = a.shape[1]
        
#         # Prepare a new matrix to include our result.
#         best_scores = torch.full((batch_size, num_a_tokens), -float('inf'), device=metric.device)
#         best_indices = torch.zeros((batch_size, num_a_tokens), dtype=torch.long, device=metric.device)
        
#         # Process in chunks to save memory.
#         num_b_tokens = b.shape[1]
#         for chunk_start in range(0, num_b_tokens, chunk_size):
#             # Maybe not exact division
#             chunk_end = min(chunk_start + chunk_size, num_b_tokens)
#             # divide chunk from b matrix
#             b_chunk = b[..., chunk_start:chunk_end, :]
            
#             # Compute partial similarity scores
#             # (b chunk with whole a)
#             chunk_scores = a @ b_chunk.transpose(-1, -2)  # [batch, num_a, chunk_size]
            
#             # Apply masking for special tokens within this chunk
#             # Be advised that we only apply on the first chunk of each row
#             if class_token and chunk_start <= 0 < chunk_end:
#                 chunk_scores[..., 0, :] = -math.inf
#             if distill_token and chunk_start <= 0 < chunk_end:
#                 chunk_scores[..., :, 0] = -math.inf
            
#             # Update best scores and indices
#             chunk_max_scores, chunk_max_indices = chunk_scores.max(dim=-1)
            
#             # Adjust indices to account for chunking
#             chunk_max_indices = chunk_max_indices + chunk_start
            
#             # Update the global best if better matches are found in this chunk
#             better_matches = chunk_max_scores > best_scores
#             best_scores[better_matches] = chunk_max_scores[better_matches]
#             best_indices[better_matches] = chunk_max_indices[better_matches]
        
#         # Sort to find the top-r token pairs to merge
#         sorted_indices = best_scores.argsort(dim=-1, descending=True)
        
#         # Get indices of tokens to merge and to keep unchanged
#         src_idx = sorted_indices[..., :r, None]  # Tokens to merge
#         unm_idx = sorted_indices[..., r:, None]  # Tokens to keep unchanged
        
#         # Get the destination indices (where to merge into)
#         dst_idx = best_indices[..., None].gather(dim=-2, index=src_idx)
        
#         # If we have a class token, ensure it's at the beginning
#         if class_token:
#             unm_idx = unm_idx.sort(dim=1)[0]

#     def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
#         src, dst = x[..., ::2, :], x[..., 1::2, :]
        
#         n, t1, c = src.shape
        
#         # Gather tokens that remain unchanged
#         unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
#         # Gather source tokens that will be merged
#         src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
#         # Merge tokens into their destinations
#         dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
#         # Combine the results
#         if distill_token:
#             return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
#         else:
#             return torch.cat([unm, dst], dim=1)

#     def unmerge(x: torch.Tensor) -> torch.Tensor:
#         unm_len = unm_idx.shape[1]
#         unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
#         n, _, c = unm.shape

#         src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

#         out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

#         out[..., 1::2, :] = dst

#         out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
#         out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

#         return out

#     return merge, unmerge

def get_optimal_chunk_size(
    batch_size: int,
    num_tokens: int,
    feature_dim: int,
    available_memory_mb: int = 1024,  # 默认1GB缓冲
    dtype_bytes: int = 4              # 默认float32
) -> int:
    """
    根据可用GPU内存和模型维度确定最佳分块大小。
    
    参数:
        batch_size: 处理的批量大小
        num_tokens: 每个序列中的token数量
        feature_dim: token特征的维度
        available_memory_mb: 估计可用的GPU内存(MB)
        dtype_bytes: 每个元素的字节数(float32为4,float16为2)
        
    返回:
        最佳分块大小
    """
    # 将可用内存转换为字节
    available_memory = available_memory_mb * 1024 * 1024
    
    # 计算计算中不同张量的内存
    # 我们需要考虑:
    # 1. a和b张量
    # 2. 相似度矩阵(最内存密集)
    # 3. 计算的额外缓冲区
    
    # token特征内存(a和b)
    token_features_memory = 2 * batch_size * num_tokens * feature_dim * dtype_bytes
    
    # 可用于相似度矩阵的内存
    similarity_matrix_memory = available_memory - token_features_memory
    
    # 根据可用内存计算最大分块大小
    # 相似度矩阵形状: [batch_size, num_tokens, chunk_size]
    max_chunk_size = similarity_matrix_memory // (batch_size * num_tokens * dtype_bytes)
    
    # 应用合理的界限并确保至少为1
    max_chunk_size = max(1, min(max_chunk_size, num_tokens))
    
    # 舍入到2的幂以获得更好的内存对齐
    chunk_size = 2 ** int(math.log2(max_chunk_size))
    
    return chunk_size

def grouped_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    chunk_size: int = None,  # 设置为None以自动确定
    available_memory_mb: int = 1024  # 要使用的内存缓冲区
) -> Tuple[Callable, Callable]:
    # 提取形状
    batch_size = metric.shape[0]
    num_tokens = metric.shape[1] // 2  # 奇偶分割后的一半token
    feature_dim = metric.shape[2]
    
    # 如果未指定,确定分块大小
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(
            batch_size=batch_size,
            num_tokens=num_tokens,
            feature_dim=feature_dim,
            available_memory_mb=available_memory_mb,
            dtype_bytes=metric.element_size()  # 根据张量dtype获取实际大小
        )
    
    # Nothing special.
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    # Compute similarity score matrix(with chunk technique)
    with torch.no_grad():
        # Normalize
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # odd and even divide.
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # Follow my habit. First determine data struct.
        batch_size = metric.shape[0]
        num_a_tokens = a.shape[1]
        
        # Prepare a new matrix to include our result.
        best_scores = torch.full((batch_size, num_a_tokens), -float('inf'), device=metric.device)
        best_indices = torch.zeros((batch_size, num_a_tokens), dtype=torch.long, device=metric.device)
        
        # Process in chunks to save memory.
        num_b_tokens = b.shape[1]
        for chunk_start in range(0, num_b_tokens, chunk_size):
            # Maybe not exact division
            chunk_end = min(chunk_start + chunk_size, num_b_tokens)
            # divide chunk from b matrix
            b_chunk = b[..., chunk_start:chunk_end, :]
            
            # Compute partial similarity scores
            # (b chunk with whole a)
            chunk_scores = a @ b_chunk.transpose(-1, -2)  # [batch, num_a, chunk_size]
            
            # Apply masking for special tokens within this chunk
            # Be advised that we only apply on the first chunk of each row
            if class_token and chunk_start <= 0 < chunk_end:
                chunk_scores[..., 0, :] = -math.inf
            if distill_token and chunk_start <= 0 < chunk_end:
                chunk_scores[..., :, 0] = -math.inf
            
            # Update best scores and indices
            chunk_max_scores, chunk_max_indices = chunk_scores.max(dim=-1)
            
            # Adjust indices to account for chunking
            chunk_max_indices = chunk_max_indices + chunk_start
            
            # Update the global best if better matches are found in this chunk
            better_matches = chunk_max_scores > best_scores
            best_scores[better_matches] = chunk_max_scores[better_matches]
            best_indices[better_matches] = chunk_max_indices[better_matches]
        
        # Sort to find the top-r token pairs to merge
        sorted_indices = best_scores.argsort(dim=-1, descending=True)
        
        # Get indices of tokens to merge and to keep unchanged
        src_idx = sorted_indices[..., :r, None]  # Tokens to merge
        unm_idx = sorted_indices[..., r:, None]  # Tokens to keep unchanged
        
        # Get the destination indices (where to merge into)
        dst_idx = best_indices[..., None].gather(dim=-2, index=src_idx)
        
        # If we have a class token, ensure it's at the beginning
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        
        n, t1, c = src.shape
        
        # Gather tokens that remain unchanged
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
        # Gather source tokens that will be merged
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # Merge tokens into their destinations
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        # Combine the results
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst

        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

# ------------------------------------------------------------------------------------------
# K-means:

def kmeans_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    max_iters: int = 3,
) -> Tuple[Callable, Callable]:
    """
    使用K-means聚类应用ToMe算法进行令牌合并。
    
    输入大小为 [batch, tokens, channels]。
    r 表示要移除的令牌数（最多为50%的令牌）。
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # 限制移除的令牌数量不超过50%
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    # 执行K-means聚类以确定要合并的令牌
    with torch.no_grad():
        # 规范化特征向量
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # 分离偶数和奇数位置的令牌
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # 获取形状信息
        batch_size, num_a, feat_dim = a.shape
        
        # 创建与原始bipartite_soft_matching相同模式的存储结构
        # 这样可以确保形状一致性
        scores = a @ b.transpose(-1, -2)  # 计算相似度矩阵
        
        # 应用特殊令牌的掩码
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
            
        # 使用K-means聚类来找到合并对
        # 我们将使用一个简化的K-means实现:
        # 1. 随机初始化r个聚类中心
        # 2. 分配每个令牌到最近的中心
        # 3. 更新中心为分配给它的所有令牌的平均值
        # 4. 重复直到收敛或达到最大迭代次数
        
        # 从b中初始化聚类中心
        random_indices = torch.randperm(b.shape[1], device=metric.device)[:r]
        centroids = b[:, random_indices, :]
        
        # 执行K-means迭代
        for _ in range(max_iters):
            # 计算每个令牌到每个中心的相似度
            similarities = torch.bmm(a, centroids.transpose(1, 2))
            
            # 应用掩码以保护特殊令牌
            if class_token:
                similarities[:, 0, :] = -math.inf
                
            # 为每个令牌找到最近的中心
            _, assignments = similarities.max(dim=2)
            
            # 更新中心
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(batch_size, r, 1, device=metric.device)
            
            for i in range(r):
                mask = (assignments == i).unsqueeze(-1)
                new_centroids[:, i:i+1, :] = torch.sum(a * mask, dim=1, keepdim=True)
                counts[:, i:i+1, :] = mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            new_centroids = new_centroids / counts
            
            # 检查收敛性
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
                
            centroids = new_centroids
            
        # 现在我们有了聚类分配，从这里我们可以创建相似于bipartite_soft_matching的结构
        # 以确保兼容性和一致性
        
        # 对于每个批次和每个聚类，找到一个要合并的令牌
        node_max = torch.zeros(batch_size, num_a, device=metric.device)
        node_idx = torch.zeros(batch_size, num_a, dtype=torch.long, device=metric.device)
        
        # 为每个批次单独填充
        for batch_idx in range(batch_size):
            for i in range(num_a):
                cluster = assignments[batch_idx, i].item()
                # 模拟相似度分数，使具有相同聚类ID的令牌相互吸引
                node_max[batch_idx, i] = 1.0 if i < r else 0.0
                node_idx[batch_idx, i] = cluster if cluster < b.shape[1] else 0
        
        # 按相似度排序，选择顶部的r个令牌进行合并
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        # 划分合并和未合并的令牌索引
        src_idx = edge_idx[..., :r, :]  # 要合并的令牌
        unm_idx = edge_idx[..., r:, :]  # 保持不变的令牌
        
        # 获取目标索引
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        # 确保类令牌始终在开头
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
            
    # 完全遵循bipartite_soft_matching的合并和解合并函数
    # 这将确保兼容性和一致的维度
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        
        n, t1, c = src.shape
        
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

# ------------------------------------------------------------------------------------------

def iga_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    max_iterations: int = 2,
    memory_efficient: bool = True
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with iterative greedy algorithm for token merging.
    
    This implementation aims to minimize peak memory usage by:
    1. Processing tokens in smaller batches
    2. Using an iterative approach to avoid large matrix multiplications
    3. Releasing intermediate tensors as soon as possible
    
    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    
    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
     - max_iterations: Number of greedy iterations to perform
     - memory_efficient: Whether to use the most memory-efficient approach
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        # Normalize features for better similarity comparison
        if memory_efficient:
            # Normalize in-place to save memory
            norm = metric.norm(dim=-1, keepdim=True)
            metric.div_(norm)
            del norm  # Explicitly release the tensor
        else:
            metric = metric / metric.norm(dim=-1, keepdim=True)
            
        # Split tokens into odd and even indices
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        batch_size, num_a_tokens, feat_dim = a.shape
        
        # Initialize arrays to track merging decisions
        # Use the same dtype as the input tensor to avoid type conversion issues
        node_scores = torch.full((batch_size, num_a_tokens), -float('inf'), 
                               device=metric.device, dtype=metric.dtype)
        node_indices = torch.zeros((batch_size, num_a_tokens), 
                                 device=metric.device, dtype=torch.long)
        
        # Process in smaller chunks to reduce memory footprint
        chunk_size = min(64, num_a_tokens)
        
        # Iterative greedy algorithm
        for iteration in range(max_iterations):
            # In each iteration, we consider already-selected token pairs
            # and try to find better pairs
            
            # Process each chunk of the first token set
            for chunk_start in range(0, num_a_tokens, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_a_tokens)
                a_chunk = a[:, chunk_start:chunk_end, :]
                
                # Compute similarities with second token set
                # This is the memory-intensive operation we're optimizing
                similarities = torch.bmm(a_chunk, b.transpose(-1, -2))
                
                # Apply masking for special tokens
                if class_token and chunk_start == 0:
                    similarities[:, 0, :] = -float('inf')
                if distill_token:
                    similarities[:, :, 0] = -float('inf')
                
                # Find best matches for this chunk
                chunk_max_scores, chunk_max_indices = similarities.max(dim=-1)
                
                # Update global scores if better matches found
                for i in range(chunk_end - chunk_start):
                    pos = chunk_start + i
                    better_mask = chunk_max_scores[:, i] > node_scores[:, pos]
                    node_scores[better_mask, pos] = chunk_max_scores[better_mask, i]
                    node_indices[better_mask, pos] = chunk_max_indices[better_mask, i]
                
                # Free memory explicitly
                del similarities, chunk_max_scores, chunk_max_indices
                
            # After first iteration, consider swapping pairs for better matches
            if iteration > 0 and iteration < max_iterations - 1:
                # Find potential swap candidates
                # This is a simplified approach to avoid quadratic memory usage
                sorted_scores, sorted_indices = node_scores.sort(dim=-1, descending=True)
                
                # Consider top-k pairs for potential swaps
                top_k = min(r * 2, num_a_tokens)
                for i in range(top_k - 1):
                    for j in range(i + 1, top_k):
                        idx_i = sorted_indices[:, i]
                        idx_j = sorted_indices[:, j]
                        
                        # Check if swapping target indices improves score
                        for batch_idx in range(batch_size):
                            orig_score = (node_scores[batch_idx, idx_i[batch_idx]] + 
                                         node_scores[batch_idx, idx_j[batch_idx]])
                            
                            # Calculate potential new scores
                            i_token = a[batch_idx, idx_i[batch_idx]].unsqueeze(0)
                            j_target = b[batch_idx, node_indices[batch_idx, idx_j[batch_idx]]].unsqueeze(0)
                            j_token = a[batch_idx, idx_j[batch_idx]].unsqueeze(0)
                            i_target = b[batch_idx, node_indices[batch_idx, idx_i[batch_idx]]].unsqueeze(0)
                            
                            new_i_score = (i_token @ j_target.transpose(-1, -2)).item()
                            new_j_score = (j_token @ i_target.transpose(-1, -2)).item()
                            new_score = new_i_score + new_j_score
                            
                            # Swap if better
                            if new_score > orig_score:
                                temp = node_indices[batch_idx, idx_i[batch_idx]].item()
                                node_indices[batch_idx, idx_i[batch_idx]] = node_indices[batch_idx, idx_j[batch_idx]]
                                node_indices[batch_idx, idx_j[batch_idx]] = temp
                                
                                node_scores[batch_idx, idx_i[batch_idx]] = new_i_score
                                node_scores[batch_idx, idx_j[batch_idx]] = new_j_score
        
        # Create the final token merging plan
        edge_idx = node_scores.argsort(dim=-1, descending=True)[..., None]
        
        # Divide into merged and unmerged tokens
        src_idx = edge_idx[..., :r, :]  # Tokens to merge
        unm_idx = edge_idx[..., r:, :]  # Tokens to keep unchanged
        
        # Get destination indices
        dst_idx = node_indices[..., None].gather(dim=-2, index=src_idx)
        
        # Ensure class token is at the beginning if present
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
    
    # The merge and unmerge functions are kept similar to the original
    # implementation for consistency
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        
        n, t1, c = src.shape
        
        # Process tokens in smaller groups to reduce peak memory
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # Use scatter_reduce which is more memory efficient for merging
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        # Combine results
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        
        # Retrieve source tokens
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        
        # Create output tensor directly with the right size to avoid
        # intermediate copies
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        
        # Place tokens in correct positions
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        
        return out

    return merge, unmerge

# ------------------------------------------------------------------------------------------

def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


# add weight to origin x matrix.
def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


# display merge process by matrix.
def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source
