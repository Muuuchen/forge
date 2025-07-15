# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/ea/ceauxet7cfownjqmnczgemi3f3pdc7elerak52xmsx4mam5l3xhe.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#    => mm_default
# Graph fragment:
#   %mm_default : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg2_1, %permute), kwargs = {})
triton_tem_fused_addmm_0 = async_compile.triton('triton_tem_fused_addmm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=2,
    num_warps=2,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_0', 'backend_hash': '2699C509628CC18320D88A848659EBAA6D04FE717DCFB43170EA35D4B5C723E2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2']},
)
@triton.jit
def triton_tem_fused_addmm_0(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 32
    N = 5
    K = 10
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 10
    stride_ak = 1
    stride_bk = 1
    stride_bn = 10

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and M >= BLOCK_M:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and N >= BLOCK_N:
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 10*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 5*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 10*idx_n, xindex.shape)).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 5*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tn/ctnztsfgoerytaomvarsqpjlq27waw5qsmaa6kj522cwi6nc3qqe.py
# Topologically Sorted Source Nodes: [add, x_1, x_2], Original ATen: [aten.addmm, aten.relu, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   add => add_tensor
#   x_1 => relu
#   x_2 => mean, pow_1
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg1_1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
triton_poi_fused_addmm_mean_pow_relu_1 = async_compile.triton('triton_poi_fused_addmm_mean_pow_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_mean_pow_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '2699C509628CC18320D88A848659EBAA6D04FE717DCFB43170EA35D4B5C723E2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_mean_pow_relu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (2))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (3))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (4))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = tmp5 * tmp5
    tmp10 = tmp7 + tmp9
    tmp11 = triton_helpers.maximum(tmp4, tmp10)
    tmp12 = tmp11 * tmp11
    tmp13 = tmp6 + tmp12
    tmp17 = tmp14 + tmp16
    tmp18 = triton_helpers.maximum(tmp4, tmp17)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp13 + tmp19
    tmp24 = tmp21 + tmp23
    tmp25 = triton_helpers.maximum(tmp4, tmp24)
    tmp26 = tmp25 * tmp25
    tmp27 = tmp20 + tmp26
    tmp31 = tmp28 + tmp30
    tmp32 = triton_helpers.maximum(tmp4, tmp31)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp27 + tmp33
    tmp35 = 5.0
    tmp36 = (tmp34 / tmp35)
    tl.store(out_ptr0 + (x0), tmp36, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/f4/cf4cxbrvhxgko76ahielhpomcnzb5s7zv6b3vin4pm4paesf7iob.py
# Topologically Sorted Source Nodes: [add, x_1, x_2], Original ATen: [aten.addmm, aten.relu, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_tensor
#   x_1 => relu
#   x_2 => add, mul, mul_1, rsqrt
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg1_1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
triton_poi_fused_add_addmm_mul_relu_rsqrt_2 = async_compile.triton('triton_poi_fused_add_addmm_mul_relu_rsqrt_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_relu_rsqrt_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '2699C509628CC18320D88A848659EBAA6D04FE717DCFB43170EA35D4B5C723E2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_relu_rsqrt_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 5)
    x1 = xindex // 5
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = 1.1920928955078125e-07
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ap/capvkbbvq5oc6hckeglqcofdqngxkq5y3stoepyduk25gn5y5ezk.py
# Topologically Sorted Source Nodes: [add, x_1, x_2, x_3], Original ATen: [aten.addmm, aten.relu, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_tensor
#   x_1 => relu
#   x_2 => add, mul, mul_1, rsqrt
#   x_3 => addmm_1
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg1_1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %addmm_1 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg5_1, %mul_1, %permute_1), kwargs = {})
triton_tem_fused_add_addmm_mul_relu_rsqrt_3 = async_compile.triton('triton_tem_fused_add_addmm_mul_relu_rsqrt_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=2,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=114, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_mul_relu_rsqrt_3', 'backend_hash': '2699C509628CC18320D88A848659EBAA6D04FE717DCFB43170EA35D4B5C723E2', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2']},
)
@triton.jit
def triton_tem_fused_add_addmm_mul_relu_rsqrt_3(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 32
    N = 2
    K = 5
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 5
    stride_ak = 1
    stride_bk = 1
    stride_bn = 5

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and M >= BLOCK_M:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and N >= BLOCK_N:
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 5*idx_m
        a = tl.load(A + (xindex), mask=a_mask, other=0.0)

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 2*idx_m
        b = tl.load(B + ((tl.broadcast_to(idx_m + 5*idx_n, xindex.shape)).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 2*idx_m
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (5, 10), (10, 1))
    assert_size_stride(arg1_1, (5, ), (1, ))
    assert_size_stride(arg2_1, (32, 10), (10, 1))
    assert_size_stride(arg3_1, (5, ), (1, ))
    assert_size_stride(arg4_1, (2, 5), (5, 1))
    assert_size_stride(arg5_1, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 5), (5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_addmm_0.run(arg2_1, arg0_1, buf0, 1, 1, 1, stream=stream0)
        del arg0_1
        del arg2_1
        buf1 = empty_strided_cuda((32, 1), (1, 32), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_1, x_2], Original ATen: [aten.addmm, aten.relu, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_mean_pow_relu_1.run(buf0, arg1_1, buf1, 32, stream=stream0)
        buf2 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [add, x_1, x_2], Original ATen: [aten.addmm, aten.relu, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_relu_rsqrt_2.run(buf2, arg1_1, buf1, arg3_1, 160, stream=stream0)
        del arg1_1
        del arg3_1
        del buf1
        buf3 = empty_strided_cuda((32, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_1, x_2, x_3], Original ATen: [aten.addmm, aten.relu, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_tem_fused_add_addmm_mul_relu_rsqrt_3.run(arg5_1, buf2, arg4_1, buf3, 1, 1, 1, stream=stream0)
        del arg4_1
        del arg5_1
        del buf2
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((5, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
