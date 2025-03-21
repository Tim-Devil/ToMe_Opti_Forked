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
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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

# def get_optimal_chunk_size(
#     batch_size: int,
#     num_tokens: int,
#     feature_dim: int,
#     available_memory_mb: int = 1024,
#     dtype_bytes: int = 4             
# ) -> int:
#     """
#     根据可用GPU内存和模型维度确定最佳分块大小。
    
#     参数:
#         batch_size: 处理的批量大小
#         num_tokens: 每个序列中的token数量
#         feature_dim: token特征的维度
#         available_memory_mb: 估计可用的GPU内存(MB)
#         dtype_bytes: 每个元素的字节数(float32为4,float16为2)
        
#     返回:
#         最佳分块大小
#     """
#     # 将可用内存转换为字节
#     available_memory = available_memory_mb * 1024 * 1024
    
#     # 计算计算中不同张量的内存
#     # 我们需要考虑:
#     # 1. a和b张量
#     # 2. 相似度矩阵(最内存密集)
#     # 3. 计算的额外缓冲区
    
#     # token特征内存(a和b)
#     token_features_memory = 2 * batch_size * num_tokens * feature_dim * dtype_bytes
    
#     # 可用于相似度矩阵的内存
#     similarity_matrix_memory = available_memory - token_features_memory
    
#     # 根据可用内存计算最大分块大小
#     # 相似度矩阵形状: [batch_size, num_tokens, chunk_size]
#     max_chunk_size = similarity_matrix_memory // (batch_size * num_tokens * dtype_bytes)
    
#     # 应用合理的界限并确保至少为1
#     max_chunk_size = max(1, min(max_chunk_size, num_tokens))
    
#     # 舍入到2的幂以获得更好的内存对齐
#     chunk_size = 2 ** int(math.log2(max_chunk_size))
    
#     return chunk_size

import torch
from typing import Callable, Tuple
import math


def do_nothing(x, mode=None):
    return x


def grouped_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    chunk_size: int = 64
) -> Tuple[Callable, Callable]:
    protected = int(class_token) + int(distill_token)
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric.div_(metric.norm(dim=-1, keepdim=True) + 1e-6)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        batch_size, num_a_tokens, feat_dim = a.shape

        best_scores = torch.full((batch_size, num_a_tokens), -float('inf'), device=metric.device)
        best_indices = torch.zeros((batch_size, num_a_tokens), dtype=torch.long, device=metric.device)

        num_b_tokens = b.shape[1]

        for start in range(0, num_b_tokens, chunk_size):
            end = min(start + chunk_size, num_b_tokens)
            chunk_scores = torch.bmm(a, b[:, start:end, :].transpose(-1, -2))

            if class_token and start == 0:
                chunk_scores[:, 0, :] = -float('inf')
            if distill_token and start == 0:
                chunk_scores[:, :, 0] = -float('inf')

            max_scores, max_indices = chunk_scores.max(dim=-1)
            max_indices += start

            better = max_scores > best_scores
            best_scores = torch.where(better, max_scores, best_scores)
            best_indices = torch.where(better, max_indices, best_indices)

            del chunk_scores, max_scores, max_indices, better

        sorted_indices = best_scores.argsort(dim=-1, descending=True)
        src_idx = sorted_indices[:, :r, None]
        unm_idx = sorted_indices[:, r:, None]
        dst_idx = best_indices.gather(dim=-1, index=src_idx.squeeze(-1)).unsqueeze(-1)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, _, c = src.shape

        unm = src.gather(dim=1, index=unm_idx.expand(n, num_a_tokens - r, c))
        src = src.gather(dim=1, index=src_idx.expand(n, r, c))
        dst.scatter_reduce_(dim=1, index=dst_idx.expand(n, r, c), src=src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[:, :unm_len, :], x[:, unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=1, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[:, 1::2, :] = dst
        out.scatter_(dim=1, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=1, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


# ------------------------------------------------------------------------------------------
# K-means:(In fact it is useless.)

def kmeans_bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    max_iters: int = 3,
) -> Tuple[Callable, Callable]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        batch_size, num_a, _ = a.shape
        
        scores = a @ b.transpose(-1, -2)
        # Still need to `@` to determine k-means so it is useless.

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        
        # 从b中初始化聚类中心
        random_indices = torch.randperm(b.shape[1], device=metric.device)[:r]
        centroids = b[:, random_indices, :]
        
        # 
        for _ in range(max_iters):
            similarities = torch.bmm(a, centroids.transpose(1, 2))
            
            if class_token:
                similarities[:, 0, :] = -math.inf
                
            _, assignments = similarities.max(dim=2)
            
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(batch_size, r, 1, device=metric.device)
            
            for i in range(r):
                mask = (assignments == i).unsqueeze(-1)
                new_centroids[:, i:i+1, :] = torch.sum(a * mask, dim=1, keepdim=True)
                counts[:, i:i+1, :] = mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            new_centroids = new_centroids / counts
            
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
                
            centroids = new_centroids
            
        node_max = torch.zeros(batch_size, num_a, device=metric.device)
        node_idx = torch.zeros(batch_size, num_a, dtype=torch.long, device=metric.device)
        
        for batch_idx in range(batch_size):
            for i in range(num_a):
                cluster = assignments[batch_idx, i].item()
                
                node_max[batch_idx, i] = 1.0 if i < r else 0.0
                node_idx[batch_idx, i] = cluster if cluster < b.shape[1] else 0
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        

        src_idx = edge_idx[..., :r, :]
        unm_idx = edge_idx[..., r:, :]
        
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
            
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
    max_iterations: int = 1,
    memory_efficient: bool = True
) -> Tuple[Callable, Callable]:

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing
    
    with torch.no_grad():
        if memory_efficient:
            norm = metric.norm(dim=-1, keepdim=True)
            metric.div_(norm)
            del norm
        else:
            metric = metric / metric.norm(dim=-1, keepdim=True)
            
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        batch_size, num_a_tokens, feat_dim = a.shape
        
        similarities = torch.bmm(a, b.transpose(-1, -2))
        
        if class_token:
            similarities[:, 0, :] = -float('inf')
        if distill_token:
            similarities[:, :, 0] = -float('inf')
        
        node_scores, node_indices = similarities.max(dim=-1)
        
        if memory_efficient:
            del similarities
        
        if max_iterations > 1:
            sorted_scores, sorted_indices = node_scores.sort(dim=-1, descending=True)
            top_indices = sorted_indices[:, :min(r*2, num_a_tokens)]
            
            top_k = top_indices.shape[1]
            for iter_idx in range(1, max_iterations):
                improved = False
                
                for i in range(top_k-1):
                    for j in range(i+1, top_k):
                        for batch_idx in range(batch_size):
                            idx_i = top_indices[batch_idx, i].item()
                            idx_j = top_indices[batch_idx, j].item()
                            
                            curr_i_score = node_scores[batch_idx, idx_i]
                            curr_j_score = node_scores[batch_idx, idx_j]
                            curr_score = curr_i_score + curr_j_score
                            
                            target_i = node_indices[batch_idx, idx_i].item()
                            target_j = node_indices[batch_idx, idx_j].item()
                            
                            new_i_score = torch.dot(a[batch_idx, idx_i], b[batch_idx, target_j])
                            new_j_score = torch.dot(a[batch_idx, idx_j], b[batch_idx, target_i])
                            new_score = new_i_score + new_j_score
                            
                            if new_score > curr_score:
                                improved = True
                                node_indices[batch_idx, idx_i] = target_j
                                node_indices[batch_idx, idx_j] = target_i
                                node_scores[batch_idx, idx_i] = new_i_score
                                node_scores[batch_idx, idx_j] = new_j_score
                
                if not improved:
                    break
        
        edge_idx = node_scores.argsort(dim=-1, descending=True)[..., None]
        
        src_idx = edge_idx[..., :r, :]
        unm_idx = edge_idx[..., r:, :]
        
        dst_idx = node_indices[..., None].gather(dim=-2, index=src_idx)
        
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
    
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
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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
