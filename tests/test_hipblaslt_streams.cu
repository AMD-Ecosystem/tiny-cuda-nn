/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   test_hipblaslt_streams.cu
 *  @brief  ROCm only: weight-gradient GEMMs that run concurrently on auxiliary streams must give the same
 *          result as the same GEMMs run one after the other.
 *
 *  CutlassMLP and FullyFusedMLP compute the weight gradients of different layers concurrently, on the
 *  auxiliary streams of a SyncedMultiStream. On ROCm these tall-K products (M, N <= 128, K = batch size)
 *  go through hipBLASLt, which selects split-K kernels for them. Those kernels keep per-HANDLE device
 *  state, so concurrent split-K GEMMs that share one hipBLASLt handle return wrong results: on MI300X
 *  (gfx942) 475 of 500 concurrent runs of the 64x64x2^18 product below differed from the serial result
 *  with one shared handle, and none with one handle per stream (hip_matmul.h, hipblaslt_handle()). In
 *  training this showed up as NaN losses within a few steps. With hipblaslt_handle() changed to return
 *  one handle for all streams, this test fails on MI325X (gfx942): 78 to 92 of the 100 concurrent runs
 *  of the 64x64 product differ (six runs).
 *
 *  The race needs the split-K solutions hipBLASLt picks on gfx942. On MI350X/MI355X (gfx950) it picks
 *  different solutions for these shapes, which did not race even with a shared handle, so this test is
 *  a meaningful regression test on gfx942 and a plain consistency check elsewhere.
 */

#include "test_common.h"

#include <tiny-cuda-nn/cutlass_matmul.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstring>

using namespace tcnn;

namespace {

using T = __half;

// K = batch size of the weight-gradient products.
constexpr uint32_t K = 1u << 18;

// D (m x n, row-major, as the MLP weight gradients) = A (m x K, column-major) * B (K x n), as issued by
// CutlassMLP::backward: A = the layer's output gradient, B = the transposed layer input.
struct Problem {
	Problem(uint32_t m, uint32_t n, MatrixLayout layout_b) : m{m}, n{n}, layout_b{layout_b}, a{m, K}, b{K, n, layout_b}, d{m, n} {}

	uint32_t m, n;
	MatrixLayout layout_b;
	GPUMatrix<T, CM> a;
	GPUMatrixDynamic<T> b;
	GPUMatrix<T, RM> d;
	std::vector<T> serial;
};

void run(cudaStream_t stream, Problem& p) {
	// Poison D so that a GEMM that does not run (or writes nothing) cannot reproduce the reference.
	CUDA_CHECK_THROW(cudaMemsetAsync(p.d.data(), 0xFF, p.d.n_bytes(), stream));
	if (p.layout_b == RM) {
		fc_multiply_split_k<FullLayerK>(stream, p.a, p.b.rm(), p.d, p.d, 1, 0.0f);
	} else {
		fc_multiply_split_k<FullLayerK>(stream, p.a, p.b.cm(), p.d, p.d, 1, 0.0f);
	}
}

// The macro tile and split-K (GSU) parameters of a hipBLASLt kernel name.
std::string solution_summary(const std::string& kernel_name) {
	std::string result;
	size_t begin = 0;
	while (begin <= kernel_name.size()) {
		size_t end = kernel_name.find('_', begin);
		if (end == std::string::npos) {
			end = kernel_name.size();
		}
		const std::string token = kernel_name.substr(begin, end - begin);
		if (token.rfind("MT", 0) == 0 || token.rfind("GSU", 0) == 0 || token.rfind("ISA", 0) == 0) {
			result += (result.empty() ? "" : " ") + token;
		}
		begin = end + 1;
	}
	return result.empty() ? kernel_name : result;
}

std::vector<T> to_host(const GPUMatrix<T, RM>& d) {
	std::vector<T> result(d.n_elements());
	CUDA_CHECK_THROW(cudaMemcpy(result.data(), d.data(), d.n_bytes(), cudaMemcpyDeviceToHost));
	return result;
}

} // namespace

TEST_CASE("Concurrent weight-gradient GEMMs on auxiliary streams match a serial run", "[gemm][hipblaslt]") {
	tcnn_test_setup();

	const uint32_t n_reps = 100;

	// The three products of the hash-grid image sample (input layer, hidden layer, output layer), which
	// hipBLASLt runs as split-K kernels on gfx942: column-major 32x64 (op T,T), 64x64 (op N,T) and 64x16
	// (op N,T) in BLAS terms, since a row-major D is computed as D^T.
	std::vector<Problem> problems;
	problems.reserve(3);
	problems.emplace_back(64, 32, CM);
	problems.emplace_back(64, 64, RM);
	problems.emplace_back(16, 64, RM);

	pcg32 rng{1337};
	for (auto& p : problems) {
		p.a.initialize_uniform(rng, -1.0f, 1.0f);
		p.b.initialize_uniform(rng, -0.25f, 0.25f);
	}

	cudaStream_t main_stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&main_stream));

	{
		// One auxiliary stream per problem, as CutlassMLP uses one per layer.
		SyncedMultiStream streams{main_stream, problems.size() + 1};

		// Serial reference: every problem on its own stream (and thus with its own hipBLASLt handle and
		// plan), one at a time.
		for (size_t j = 0; j < problems.size(); ++j) {
			cudaStream_t stream = streams.get(j + 1);
			run(stream, problems[j]);
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			problems[j].serial = to_host(problems[j].d);
		}

		// The test is only meaningful if the products actually ran on hipBLASLt (not the hipBLAS
		// fallback); report which solution was selected.
		for (size_t j = 0; j < problems.size(); ++j) {
			const Problem& p = problems[j];
			cudaStream_t stream = streams.get(j + 1);
			hipblasLtHandle_t handle = hipblaslt_handle(stream);
			REQUIRE(handle != nullptr);
			const hipblasOperation_t op_a = p.layout_b == RM ? HIPBLAS_OP_N : HIPBLAS_OP_T;
			const HipblasLtGemmPlan& plan = hipblaslt_gemm_plan(
				{handle, op_a, HIPBLAS_OP_T, (int)p.n, (int)p.m, (int)K, (int)p.b.stride(), (int)p.a.stride(), (int)p.d.stride(), HIP_R_16F, HIPBLASLT_EPILOGUE_DEFAULT}
			);
			CHECK(plan.usable);
			if (plan.usable) {
				hipblasLtMatmulAlgo_t algo = plan.algo;
				fmt::print("problem {}: D {}x{} K={} B {} workspace {} solution {}\n", j, p.m, p.n, K, p.layout_b == RM ? "RM" : "CM", plan.workspace_size, solution_summary(hipblaslt_ext::getKernelNameFromAlgo(handle, algo)));
			}
		}

		// Concurrent runs: all problems in flight on their streams at the same time.
		std::vector<uint32_t> n_mismatches(problems.size(), 0);
		for (uint32_t rep = 0; rep < n_reps; ++rep) {
			for (size_t j = 0; j < problems.size(); ++j) {
				run(streams.get(j + 1), problems[j]);
			}
			CUDA_CHECK_THROW(cudaDeviceSynchronize());
			for (size_t j = 0; j < problems.size(); ++j) {
				const std::vector<T> result = to_host(problems[j].d);
				if (std::memcmp(result.data(), problems[j].serial.data(), result.size() * sizeof(T)) != 0) {
					++n_mismatches[j];
				}
			}
		}

		for (size_t j = 0; j < problems.size(); ++j) {
			fmt::print("problem {}: concurrent runs differing from the serial run: {} of {}\n", j, n_mismatches[j], n_reps);
			INFO("problem " << j << " (" << problems[j].m << "x" << problems[j].n << "x" << K << ")");
			CHECK(n_mismatches[j] == 0);
		}
	}

	CUDA_CHECK_THROW(cudaStreamDestroy(main_stream));
}
