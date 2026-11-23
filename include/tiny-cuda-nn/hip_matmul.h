/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   hip_matmul.h
 *  @brief  AMD ROCm implementation of the fc_multiply / fc_multiply_split_k GEMM helpers that
 *          cutlass_matmul.h provides on NVIDIA GPUs. Selected by cutlass_matmul.h when the
 *          build defines TCNN_USE_HIP.
 *
 *  Semantics are identical to the CUTLASS versions:
 *    fc_multiply(A, B, C, D, act, transfer=false, sum_source)  D = act(A*B [+ C])
 *    fc_multiply(A, B, C, D, act, transfer=true)               D = act'(A*B) given forward values C
 *    fc_multiply_split_k(A, B, C, D, slices, beta)             D = A*B + beta*C
 *  The GEMM runs on hipBLASLt (fp16/fp32 inputs, fp32 accumulation), with hipBLAS as the
 *  fallback when hipBLASLt has no solution for a problem; the activation epilogue is a separate
 *  elementwise kernel. Split-K slicing is a CUTLASS tiling detail; hipBLASLt picks its own
 *  decomposition (including split-K), so `split_k_slices` is accepted and ignored.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace tcnn {

#define HIPBLAS_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                     \
		hipblasStatus_t _result = x;                                                                                         \
		if (_result != HIPBLAS_STATUS_SUCCESS)                                                                               \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + hipblasStatusToString(_result)); \
	} while(0)

// The CUTLASS build selects threadblock/warp tile shapes through these configuration tags.
// hipBLASLt/hipBLAS choose their own kernels, so on ROCm they are empty tags that keep call sites unchanged.
struct LayerConfigHip {};
using FullLayerK = LayerConfigHip;
using LastLayerK = LayerConfigHip;
using FullLayer = LayerConfigHip;
using LastLayer = LayerConfigHip;

// One hipBLAS handle per (host thread, device). Handles are intentionally never destroyed:
// destroying them from static destructors races with HIP runtime teardown at process exit.
inline hipblasHandle_t& hipblas_handle_slot() {
	static thread_local std::unordered_map<int, hipblasHandle_t> s_handles;
	return s_handles[cuda_device()];
}

inline hipblasHandle_t hipblas_handle(cudaStream_t stream) {
	hipblasHandle_t& handle = hipblas_handle_slot();
	if (!handle) {
		// Creating a handle synchronizes with the legacy stream, which is not allowed while a stream is
		// being captured into a graph. CudaGraph therefore creates the handle before it starts capturing.
		cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
		CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &capture_status));
		if (capture_status != cudaStreamCaptureStatusNone) {
			throw std::runtime_error{"hipBLAS must be initialized (hipblas_initialize()) before graph capture begins."};
		}
		HIPBLAS_CHECK_THROW(hipblasCreate(&handle));
	}
	HIPBLAS_CHECK_THROW(hipblasSetStream(handle, stream));
	return handle;
}

template <typename T> constexpr hipDataType hipblas_data_type();
template <> constexpr hipDataType hipblas_data_type<float>() { return HIP_R_32F; }
template <> constexpr hipDataType hipblas_data_type<__half>() { return HIP_R_16F; }

// hipBLASLt handles, one per (host thread, device, stream), never destroyed for the same reason as above.
// hipBLASLt keeps per-handle device state for its split-K kernels, so split-K GEMMs that run concurrently
// on different streams must not share a handle -- and the MLPs compute the weight gradients of different
// layers concurrently on auxiliary streams. Creating a handle is not capturable: a stream that first needs
// a handle while it is being captured into a graph gets none (its GEMMs fall back to hipBLAS) and is
// remembered, and hipblas_initialize(), which CudaGraph calls before capturing, creates its handle.
struct HipblasLtHandles {
	std::unordered_map<cudaStream_t, hipblasLtHandle_t> by_stream;
	std::vector<cudaStream_t> deferred;
};

inline HipblasLtHandles& hipblaslt_handles() {
	static thread_local std::unordered_map<int, HipblasLtHandles> s_handles;
	return s_handles[cuda_device()];
}

inline void hipblaslt_create_deferred_handles() {
	HipblasLtHandles& handles = hipblaslt_handles();
	for (cudaStream_t stream : handles.deferred) {
		if (!handles.by_stream.count(stream)) {
			hipblasLtHandle_t handle;
			HIPBLAS_CHECK_THROW(hipblasLtCreate(&handle));
			handles.by_stream[stream] = handle;
		}
	}
	handles.deferred.clear();
}

// Returns nullptr when the stream has no handle yet and is being captured.
inline hipblasLtHandle_t hipblaslt_handle(cudaStream_t stream) {
	HipblasLtHandles& handles = hipblaslt_handles();
	auto it = handles.by_stream.find(stream);
	if (it != handles.by_stream.end()) {
		return it->second;
	}

	cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
	CUDA_CHECK_THROW(cudaStreamIsCapturing(stream, &capture_status));
	if (capture_status != cudaStreamCaptureStatusNone) {
		if (std::find(handles.deferred.begin(), handles.deferred.end(), stream) == handles.deferred.end()) {
			handles.deferred.push_back(stream);
		}
		return nullptr;
	}

	hipblasLtHandle_t handle;
	HIPBLAS_CHECK_THROW(hipblasLtCreate(&handle));
	handles.by_stream[stream] = handle;
	return handle;
}

// Column-major GEMM problem D(m x n) = op_a(A) * op_b(B) on one handle, keyed for the per-thread plan cache.
struct HipblasLtGemmKey {
	hipblasLtHandle_t handle;
	hipblasOperation_t op_a, op_b;
	int m, n, k;
	int lda, ldb, ldd;
	hipDataType type;
	hipblasLtEpilogue_t epilogue;

	bool operator==(const HipblasLtGemmKey& other) const {
		return handle == other.handle && op_a == other.op_a && op_b == other.op_b && m == other.m && n == other.n && k == other.k &&
			lda == other.lda && ldb == other.ldb && ldd == other.ldd && type == other.type && epilogue == other.epilogue;
	}
};

struct HipblasLtGemmKeyHash {
	size_t operator()(const HipblasLtGemmKey& key) const {
		size_t h = 0;
		for (int64_t v : {(int64_t)(intptr_t)key.handle, (int64_t)key.op_a, (int64_t)key.op_b, (int64_t)key.m, (int64_t)key.n, (int64_t)key.k,
			(int64_t)key.lda, (int64_t)key.ldb, (int64_t)key.ldd, (int64_t)key.type, (int64_t)key.epilogue}) {
			h ^= std::hash<int64_t>{}(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
		}
		return h;
	}
};

// Matmul descriptor, operand layouts, and the algorithm that hipBLASLt's heuristic ranks first for one
// problem. Built on first use and kept for the lifetime of the thread, like the library handles.
// `usable == false` records that hipBLASLt has no solution, so the problem goes to hipBLAS.
struct HipblasLtGemmPlan {
	bool usable = false;
	hipblasLtMatmulDesc_t desc = nullptr;
	hipblasLtMatrixLayout_t layout_a = nullptr;
	hipblasLtMatrixLayout_t layout_b = nullptr;
	hipblasLtMatrixLayout_t layout_d = nullptr;
	hipblasLtMatmulAlgo_t algo = {};
	size_t workspace_size = 0;
};

// Upper bound on the scratch memory a selected algorithm may use (split-K partial products).
static constexpr uint64_t HIPBLASLT_MAX_WORKSPACE_BYTES = 32ull << 20;

inline const HipblasLtGemmPlan& hipblaslt_gemm_plan(const HipblasLtGemmKey& key) {
	static thread_local std::unordered_map<HipblasLtGemmKey, HipblasLtGemmPlan, HipblasLtGemmKeyHash> s_plans;
	auto it = s_plans.find(key);
	if (it != s_plans.end()) {
		return it->second;
	}

	HipblasLtGemmPlan plan;
	const hipblasOperation_t op_a = key.op_a, op_b = key.op_b;
	// Matrix layouts describe the operands as stored, i.e. before op_a/op_b are applied.
	bool ok = hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
		hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)) == HIPBLAS_STATUS_SUCCESS &&
		hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)) == HIPBLAS_STATUS_SUCCESS &&
		(key.epilogue == HIPBLASLT_EPILOGUE_DEFAULT ||
			hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &key.epilogue, sizeof(key.epilogue)) == HIPBLAS_STATUS_SUCCESS) &&
		hipblasLtMatrixLayoutCreate(&plan.layout_a, key.type, op_a == HIPBLAS_OP_N ? key.m : key.k, op_a == HIPBLAS_OP_N ? key.k : key.m, key.lda) == HIPBLAS_STATUS_SUCCESS &&
		hipblasLtMatrixLayoutCreate(&plan.layout_b, key.type, op_b == HIPBLAS_OP_N ? key.k : key.n, op_b == HIPBLAS_OP_N ? key.n : key.k, key.ldb) == HIPBLAS_STATUS_SUCCESS &&
		hipblasLtMatrixLayoutCreate(&plan.layout_d, key.type, key.m, key.n, key.ldd) == HIPBLAS_STATUS_SUCCESS;

	if (ok) {
		// The heuristic query can leave std::cout in fixed-point format with precision 2 (observed on
		// gfx942 with hipBLASLt 1.4), which would change how the caller prints numbers. Restore it.
		const std::ios_base::fmtflags cout_flags = std::cout.flags();
		const std::streamsize cout_precision = std::cout.precision();
		ScopeGuard restore_cout{[&]() {
			std::cout.flags(cout_flags);
			std::cout.precision(cout_precision);
		}};

		hipblasLtMatmulPreference_t preference = nullptr;
		if (hipblasLtMatmulPreferenceCreate(&preference) == HIPBLAS_STATUS_SUCCESS) {
			uint64_t max_workspace = HIPBLASLT_MAX_WORKSPACE_BYTES;
			hipblasLtMatmulHeuristicResult_t result = {};
			int n_results = 0;
			if (hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)) == HIPBLAS_STATUS_SUCCESS &&
				hipblasLtMatmulAlgoGetHeuristic(key.handle, plan.desc, plan.layout_a, plan.layout_b, plan.layout_d, plan.layout_d, preference, 1, &result, &n_results) == HIPBLAS_STATUS_SUCCESS &&
				n_results > 0 && result.state == HIPBLAS_STATUS_SUCCESS) {
				plan.algo = result.algo;
				plan.workspace_size = result.workspaceSize;
				plan.usable = true;
			}
			hipblasLtMatmulPreferenceDestroy(preference);
		}
	}

	return s_plans.emplace(key, plan).first->second;
}

// D (m x n, column-major, leading dimension ldd) = epilogue(alpha * op_a(A) * op_b(B) + beta * D) through
// hipBLASLt. Returns false, without touching D, when hipBLASLt has no handle, no solution, or rejects the
// operands, so that the caller can fall back to hipBLAS.
template <typename T>
bool hipblaslt_gemm(cudaStream_t stream, hipblasOperation_t op_a, hipblasOperation_t op_b, int m, int n, int k, const T* A, int lda, const T* B, int ldb, T* D, int ldd, float alpha, float beta, hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT) {
	if (k == 0) {
		return false;
	}

	hipblasLtHandle_t handle = hipblaslt_handle(stream);
	if (!handle) {
		return false;
	}

	const HipblasLtGemmPlan& plan = hipblaslt_gemm_plan({handle, op_a, op_b, m, n, k, lda, ldb, ldd, hipblas_data_type<T>(), epilogue});
	if (!plan.usable) {
		return false;
	}

	GPUMemoryArena::Allocation workspace;
	if (plan.workspace_size > 0) {
		workspace = allocate_workspace(stream, plan.workspace_size);
	}

	return hipblasLtMatmul(
		handle, plan.desc,
		&alpha, A, plan.layout_a, B, plan.layout_b,
		&beta, D, plan.layout_d, D, plan.layout_d,
		&plan.algo, plan.workspace_size > 0 ? workspace.data() : nullptr, plan.workspace_size, stream
	) == HIPBLAS_STATUS_SUCCESS;
}

enum class FcEpilogueMode {
	Copy,          // out = source
	Activation,    // out = act(in)
	AddActivation, // out = act(in + source)
	Transfer,      // out = act'(in) given forward activations in `source`
};

template <typename T>
__global__ void fc_epilogue_kernel(
	const uint32_t n_elements,
	const uint32_t m,
	const FcEpilogueMode mode,
	const Activation act,
	MatrixView<const T> in,
	MatrixView<const T> source,
	MatrixView<T> out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t row = i % m;
	const uint32_t col = i / m;

	switch (mode) {
		case FcEpilogueMode::Copy: out(row, col) = source(row, col); break;
		case FcEpilogueMode::Activation: out(row, col) = (T)activation<float>(act, (float)in(row, col)); break;
		case FcEpilogueMode::AddActivation: out(row, col) = (T)activation<float>(act, (float)in(row, col) + (float)source(row, col)); break;
		case FcEpilogueMode::Transfer: out(row, col) = (T)activation_backward<float>(act, (float)in(row, col), (float)source(row, col)); break;
	}
}

// Fast path of fc_epilogue_kernel for views that share one contiguous layout: the matrices are processed
// as flat arrays, N_VEC elements (16 bytes) per thread, without per-element index arithmetic, and with the
// epilogue mode and activation resolved at compile time.
template <typename T, uint32_t N_VEC, FcEpilogueMode MODE, Activation ACT>
__global__ void fc_epilogue_flat_kernel(const uint32_t n_vectors, const T* in, const T* source, T* out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_vectors) return;

	using vec_t = tvec<T, N_VEC, sizeof(T) * N_VEC>;
	vec_t result;
	if constexpr (MODE == FcEpilogueMode::Copy) {
		result = ((const vec_t*)source)[i];
	} else {
		const vec_t a = ((const vec_t*)in)[i];
		tvec<float, N_VEC> x;
		TCNN_PRAGMA_UNROLL
		for (uint32_t k = 0; k < N_VEC; ++k) {
			x[k] = (float)a[k];
		}

		tvec<float, N_VEC> y;
		if constexpr (MODE == FcEpilogueMode::Activation) {
			y = vec_activation<float>(ACT, x);
		} else {
			const vec_t b = ((const vec_t*)source)[i];
			tvec<float, N_VEC> z;
			TCNN_PRAGMA_UNROLL
			for (uint32_t k = 0; k < N_VEC; ++k) {
				z[k] = (float)b[k];
			}

			if constexpr (MODE == FcEpilogueMode::AddActivation) {
				y = vec_activation<float>(ACT, x + z);
			} else {
				y = vec_activation_backward<float>(ACT, x, z);
			}
		}

		TCNN_PRAGMA_UNROLL
		for (uint32_t k = 0; k < N_VEC; ++k) {
			result[k] = (T)y[k];
		}
	}

	((vec_t*)out)[i] = result;
}

template <typename T, FcEpilogueMode MODE, Activation ACT>
void fc_epilogue_flat(cudaStream_t stream, uint32_t n_elements, const T* in, const T* source, T* out) {
	// Larger workgroups than N_THREADS_LINEAR: launching many small workgroups is rate-limited on AMD GPUs.
	static constexpr uint32_t N_THREADS = 512;
	static constexpr uint32_t N_VEC = 16 / sizeof(T);
	const uint32_t n_vectors = n_elements / N_VEC;
	fc_epilogue_flat_kernel<T, N_VEC, MODE, ACT><<<n_blocks_linear(n_vectors, N_THREADS), N_THREADS, 0, stream>>>(n_vectors, in, source, out);
}

template <typename T, FcEpilogueMode MODE>
void fc_epilogue_flat(cudaStream_t stream, uint32_t n_elements, Activation act, const T* in, const T* source, T* out) {
	switch (act) {
		case Activation::ReLU: fc_epilogue_flat<T, MODE, Activation::ReLU>(stream, n_elements, in, source, out); break;
		case Activation::LeakyReLU: fc_epilogue_flat<T, MODE, Activation::LeakyReLU>(stream, n_elements, in, source, out); break;
		case Activation::SiLU: fc_epilogue_flat<T, MODE, Activation::SiLU>(stream, n_elements, in, source, out); break;
		case Activation::Exponential: fc_epilogue_flat<T, MODE, Activation::Exponential>(stream, n_elements, in, source, out); break;
		case Activation::Sine: fc_epilogue_flat<T, MODE, Activation::Sine>(stream, n_elements, in, source, out); break;
		case Activation::Sigmoid: fc_epilogue_flat<T, MODE, Activation::Sigmoid>(stream, n_elements, in, source, out); break;
		case Activation::Squareplus: fc_epilogue_flat<T, MODE, Activation::Squareplus>(stream, n_elements, in, source, out); break;
		case Activation::Softplus: fc_epilogue_flat<T, MODE, Activation::Softplus>(stream, n_elements, in, source, out); break;
		case Activation::Tanh: fc_epilogue_flat<T, MODE, Activation::Tanh>(stream, n_elements, in, source, out); break;
		case Activation::None: fc_epilogue_flat<T, MODE, Activation::None>(stream, n_elements, in, source, out); break;
		default: throw std::runtime_error{"fc_epilogue: unsupported activation."};
	}
}

template <typename T>
void fc_epilogue(cudaStream_t stream, uint32_t m, uint32_t n, FcEpilogueMode mode, Activation act, MatrixView<const T> in, MatrixView<const T> source, MatrixView<T> out) {
	CHECK_THROW((uint64_t)m * n < (1ull << 32));
	const uint32_t n_elements = m * n;
	if (n_elements == 0) {
		return;
	}

	// Element (row, col) has the same flat offset in all three views when they share strides that
	// describe a contiguous column- or row-major m x n matrix.
	const bool uses_in = mode != FcEpilogueMode::Copy;
	const bool uses_source = mode != FcEpilogueMode::Activation;
	const bool contiguous = (out.stride_i == 1 && out.stride_j == m) || (out.stride_j == 1 && out.stride_i == n);
	const bool same_layout = (!uses_in || (in.stride_i == out.stride_i && in.stride_j == out.stride_j)) &&
		(!uses_source || (source.stride_i == out.stride_i && source.stride_j == out.stride_j));
	const auto aligned = [](const void* ptr) { return ((uintptr_t)ptr % 16) == 0; };
	if (contiguous && same_layout && n_elements % (16 / sizeof(T)) == 0 && aligned(out.data) && (!uses_in || aligned(in.data)) && (!uses_source || aligned(source.data))) {
		switch (mode) {
			case FcEpilogueMode::Copy: fc_epilogue_flat<T, FcEpilogueMode::Copy, Activation::None>(stream, n_elements, in.data, source.data, out.data); break;
			case FcEpilogueMode::Activation: fc_epilogue_flat<T, FcEpilogueMode::Activation>(stream, n_elements, act, in.data, source.data, out.data); break;
			case FcEpilogueMode::AddActivation: fc_epilogue_flat<T, FcEpilogueMode::AddActivation>(stream, n_elements, act, in.data, source.data, out.data); break;
			case FcEpilogueMode::Transfer: fc_epilogue_flat<T, FcEpilogueMode::Transfer>(stream, n_elements, act, in.data, source.data, out.data); break;
		}
		return;
	}

	fc_epilogue_kernel<T><<<n_blocks_linear(n_elements), N_THREADS_LINEAR, 0, stream>>>(n_elements, m, mode, act, in, source, out);
}

template <typename T>
MatrixView<T> matrix_view(T* data, MatrixLayout layout, uint32_t stride) {
	return {data, layout == CM ? 1u : stride, layout == CM ? stride : 1u};
}

// D (M x N, layout LayoutD, leading dimension ldd) = epilogue(alpha * A * B + beta * D) through hipBLASLt.
// Returns false, without touching D, when hipBLASLt cannot run the problem.
template <typename T, MatrixLayout LayoutA, MatrixLayout LayoutB, MatrixLayout LayoutD>
bool hipblaslt_fc_gemm(cudaStream_t stream, const GPUMatrix<T, LayoutA>& A, const GPUMatrix<T, LayoutB>& B, T* D, uint32_t ldd, float alpha, float beta, hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT) {
	if (LayoutD == CM) {
		return hipblaslt_gemm<T>(
			stream,
			LayoutA == CM ? HIPBLAS_OP_N : HIPBLAS_OP_T, LayoutB == CM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			A.m(), B.n(), A.n(), A.data(), (int)A.stride(), B.data(), (int)B.stride(), D, (int)ldd, alpha, beta, epilogue
		);
	} else {
		// Row-major D: compute D^T = B^T * A^T in column-major (elementwise epilogues are unaffected).
		return hipblaslt_gemm<T>(
			stream,
			LayoutB == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T, LayoutA == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			B.n(), A.m(), A.n(), B.data(), (int)B.stride(), A.data(), (int)A.stride(), D, (int)ldd, alpha, beta, epilogue
		);
	}
}

// D (m x n, layout LayoutD, leading dimension ldd) = alpha * A * B + beta * D, column-major BLAS mapping.
template <typename T, MatrixLayout LayoutA, MatrixLayout LayoutB, MatrixLayout LayoutD>
void hipblas_fc_gemm(cudaStream_t stream, const GPUMatrix<T, LayoutA>& A, const GPUMatrix<T, LayoutB>& B, T* D, uint32_t ldd, float alpha, float beta) {
	const int M = A.m();
	const int K = A.n();
	const int N = B.n();

	if (M == 0 || N == 0) {
		return;
	}

	// hipBLASLt first: through hipBLAS (rocBLAS) the tall-K weight-gradient products (M, N <= 128,
	// K = batch size) can resolve to a single-workgroup kernel that is orders of magnitude slower
	// than the split-K kernels hipBLASLt selects for the same problem.
	if (hipblaslt_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D, ldd, alpha, beta)) {
		return;
	}

	hipblasHandle_t handle = hipblas_handle(stream);
	constexpr hipDataType type = hipblas_data_type<T>();

	if (LayoutD == CM) {
		// D = op(A) * op(B), where a row-major matrix is its column-major transpose.
		HIPBLAS_CHECK_THROW(hipblasGemmEx(
			handle,
			LayoutA == CM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			LayoutB == CM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			M, N, K,
			&alpha,
			A.data(), type, (int)A.stride(),
			B.data(), type, (int)B.stride(),
			&beta,
			D, type, (int)ldd,
			HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT
		));
	} else {
		// Row-major D: compute D^T = B^T * A^T in column-major.
		HIPBLAS_CHECK_THROW(hipblasGemmEx(
			handle,
			LayoutB == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			LayoutA == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T,
			N, M, K,
			&alpha,
			B.data(), type, (int)B.stride(),
			A.data(), type, (int)A.stride(),
			&beta,
			D, type, (int)ldd,
			HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT
		));
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrix<TypeC, LayoutC>& C,
	const GPUMatrix<TypeD, LayoutD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(std::is_same<TypeA, TypeD>::value, "fc_multiply on ROCm requires equal input and output types");
	static_assert(LayoutC == LayoutD, "Layout of matrix C and D must be equal");

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const uint32_t M = A.m();
	const uint32_t N = B.n();

	if (C.m() != M || C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	using T = TypeD;
	const MatrixView<const T> c_view = matrix_view((const T*)C.data(), LayoutC, C.stride());
	const MatrixView<T> d_view = matrix_view((T*)D.data(), LayoutD, D.stride());
	const bool c_aliases_d = (const void*)C.data() == (const void*)D.data();

	if (transfer) {
		if (c_aliases_d) {
			// The forward activations would be overwritten by the GEMM: stage the product in scratch memory.
			auto tmp = allocate_workspace(stream, sizeof(T) * M * N);
			T* tmp_data = (T*)tmp.data();
			const uint32_t ld = LayoutD == CM ? M : N;
			hipblas_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, tmp_data, ld, 1.0f, 0.0f);
			fc_epilogue<T>(stream, M, N, FcEpilogueMode::Transfer, act, matrix_view((const T*)tmp_data, LayoutD, ld), c_view, d_view);
		} else {
			hipblas_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D.data(), D.stride(), 1.0f, 0.0f);
			if (act != Activation::None) { // the transfer of Activation::None is the identity
				fc_epilogue<T>(stream, M, N, FcEpilogueMode::Transfer, act, d_view, c_view, d_view);
			}
		}
		return;
	}

	if (sum_source && c_aliases_d) {
		hipblas_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D.data(), D.stride(), 1.0f, 1.0f);
		sum_source = false;
	} else {
		// ReLU is applied by the GEMM epilogue when hipBLASLt provides one, saving a pass over D.
		if (!sum_source && act == Activation::ReLU && M > 0 && N > 0 &&
			hipblaslt_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D.data(), D.stride(), 1.0f, 0.0f, HIPBLASLT_EPILOGUE_RELU)) {
			return;
		}
		hipblas_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D.data(), D.stride(), 1.0f, 0.0f);
	}

	if (sum_source) {
		fc_epilogue<T>(stream, M, N, FcEpilogueMode::AddActivation, act, d_view, c_view, d_view);
	} else if (act != Activation::None) {
		fc_epilogue<T>(stream, M, N, FcEpilogueMode::Activation, act, d_view, c_view, d_view);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply<config>(stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None,
	bool transfer = false,
	bool sum_source = false
) {
	if (B.layout() == CM) {
		fc_multiply<config>(stream, A, B.cm(), C, D, act, transfer, sum_source);
	} else {
		fc_multiply<config>(stream, A, B.rm(), C, D, act, transfer, sum_source);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeD>
void fc_multiply(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeD>& D,
	Activation act = Activation::None
) {
	fc_multiply<config>(stream, A, B, D, D, act);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, MatrixLayout LayoutC, typename TypeD, MatrixLayout LayoutD>
void fc_multiply_split_k(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrix<TypeC, LayoutC>& C,
	const GPUMatrix<TypeD, LayoutD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	static_assert(std::is_same<TypeA, TypeB>::value, "Type of matrix A and B must be equal");
	static_assert(std::is_same<TypeC, TypeD>::value, "Type of matrix C and D must be equal");
	static_assert(std::is_same<TypeA, TypeD>::value, "fc_multiply_split_k on ROCm requires equal input and output types");
	static_assert(LayoutC == LayoutD, "Layout of matrix C and D must be equal");

	if (A.n() != B.m()) {
		throw std::runtime_error("Matrices A and B can not be multiplied together");
	}

	const uint32_t M = A.m();
	const uint32_t N = B.n();

	if (C.m() != M || C.n() != N) {
		throw std::runtime_error{fmt::format("Matrix C has incorrect size {}x{} != {}x{}", C.m(), C.n(), M, N)};
	}

	if (D.m() != M || D.n() != N) {
		throw std::runtime_error{fmt::format("Matrix D has incorrect size {}x{} != {}x{}", D.m(), D.n(), M, N)};
	}

	using T = TypeD;
	if (beta != 0.0f && (const void*)C.data() != (const void*)D.data()) {
		fc_epilogue<T>(
			stream, M, N, FcEpilogueMode::Copy, Activation::None,
			matrix_view((const T*)D.data(), LayoutD, D.stride()),
			matrix_view((const T*)C.data(), LayoutC, C.stride()),
			matrix_view((T*)D.data(), LayoutD, D.stride())
		);
	}

	hipblas_fc_gemm<T, LayoutA, LayoutB, LayoutD>(stream, A, B, D.data(), D.stride(), 1.0f, beta);
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, MatrixLayout LayoutB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrix<TypeB, LayoutB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (C.layout() != D.layout()) {
		throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
	}

	if (D.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
	}
}

template <typename config, typename TypeA, MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	cudaStream_t stream,
	const GPUMatrix<TypeA, LayoutA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (B.layout() == CM) {
		fc_multiply_split_k<config>(stream, A, B.cm(), C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A, B.rm(), C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeC, typename TypeD>
void fc_multiply_split_k(
	cudaStream_t stream,
	const GPUMatrixDynamic<TypeA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeC>& C,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices = 1,
	float beta = 0.0f
) {
	if (A.layout() == CM) {
		fc_multiply_split_k<config>(stream, A.cm(), B, C, D, split_k_slices, beta);
	} else {
		fc_multiply_split_k<config>(stream, A.rm(), B, C, D, split_k_slices, beta);
	}
}

template <typename config, typename TypeA, typename TypeB, typename TypeD>
void fc_multiply_split_k(
	cudaStream_t stream,
	const GPUMatrixDynamic<TypeA>& A,
	const GPUMatrixDynamic<TypeB>& B,
	const GPUMatrixDynamic<TypeD>& D,
	uint32_t split_k_slices,
	float beta
) {
	fc_multiply_split_k<config>(stream, A, B, D, D, split_k_slices, beta);
}

}
