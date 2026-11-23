/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   test_grid_gradients.cu
 *  @brief  ROCm only: the parameter gradients of GridEncoding::backward against the reference scatter
 *          kernel_grid_backward.
 *
 *  On ROCm the grid parameter gradient is computed by two kernels (grid.h, backward_impl): levels whose
 *  gradient fits in shared memory are accumulated per workgroup of 4096 samples in fp32 shared memory and
 *  then added to the global gradient (kernel_grid_backward_shared), and the other levels use direct atomics
 *  (kernel_grid_backward_hip); the two run concurrently on two streams. The CUDA build uses
 *  kernel_grid_backward for all levels, which is still compiled in the ROCm build and serves as the
 *  reference here: same inputs, same vertex selection, but one global atomic per sample and vertex.
 *
 *  Covered: hash, dense and tiled grids; configurations with both shared-memory and direct levels, with
 *  shared-memory levels only and with direct levels only; 1, 2, 4 and 8 features per level; linear,
 *  nearest and smoothstep interpolation; stochastic interpolation; a per-sample max level
 *  (max_level_gpu) and a global one; overwrite and accumulate gradient modes. The batch (3 x 4096 + 256)
 *  spans several shared-memory workgroups plus a partial one.
 *
 *  Both sides add fp32 values with float atomics in a non-deterministic order, and the shared-memory path
 *  sums per-workgroup partials first, so the results differ by rounding only. The error of each element
 *  is measured relative to |reference| + RMS of the reference over its level (the RMS keeps elements that
 *  nearly cancel on an absolute scale); see GRADIENT_TOLERANCE for the measured values.
 */

#include "test_common.h"

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/multi_level_interface.h>

#include <algorithm>
#include <cmath>

using namespace tcnn;

namespace {

// Largest relative error measured over all cases below, two runs each on MI325X (gfx942) and MI355X (gfx950):
// 1.9e-6. Two runs of the reference kernel differ from each other by about as much (up to 1.8e-6), so this is the
// rounding of fp32 sums taken in a different order. The tolerance leaves about 50x headroom and is far
// below real defects: dropping a single sample per 4096-sample workgroup in the shared-memory kernel gives
// a relative error of at least 0.19 in every case that has shared-memory levels.
constexpr double GRADIENT_TOLERANCE = 1e-4;

constexpr uint32_t BATCH_SIZE = 3 * 4096 + 256;

struct Case {
	std::string label;
	json config;
	bool per_sample_max_level;
	float max_level;
	GradientMode mode;
};

template <uint32_t N_POS_DIMS, uint32_t N_FEATURES_PER_LEVEL>
void check_grid_gradients(const Case& c) {
	INFO(c.label);

	std::unique_ptr<MultiLevelEncoding<float>> grid{dynamic_cast<MultiLevelEncoding<float>*>(create_encoding<float>(N_POS_DIMS, c.config))};
	REQUIRE(grid);
	REQUIRE(grid->n_pos_dims() == N_POS_DIMS);
	REQUIRE(grid->n_features_per_level() == N_FEATURES_PER_LEVEL);

	const uint32_t n_levels = grid->n_levels();
	const uint32_t n_features = n_levels * N_FEATURES_PER_LEVEL;
	const size_t n_params = grid->n_params();
	const ParamsOffsetTable offset_table = grid->params_offset_table();

	// Levels that backward_impl accumulates in shared memory (same rule as grid.h).
	uint32_t n_shared_levels = 0;
	while (n_shared_levels < n_levels) {
		const uint32_t level_params = offset_table.data[n_shared_levels + 1] - offset_table.data[n_shared_levels];
		if (level_params > 4096 || level_params * N_FEATURES_PER_LEVEL * sizeof(float) > 32 * 1024) {
			break;
		}
		++n_shared_levels;
	}

	pcg32 rng{42 + (uint64_t)n_params};

	GPUMemory<float> params(n_params);
	std::vector<float> params_host(n_params);
	for (auto& v : params_host) {
		v = rng.next_float() * 2.0f - 1.0f;
	}
	params.copy_from_host(params_host);

	// Gradient initial values (only used in accumulate mode).
	std::vector<float> initial_gradient(n_params);
	for (auto& v : initial_gradient) {
		v = rng.next_float() * 2.0f - 1.0f;
	}
	GPUMemory<float> gradients(n_params);
	gradients.copy_from_host(initial_gradient);
	grid->set_params(params.data(), params.data(), gradients.data());

	GPUMatrix<float> input{N_POS_DIMS, BATCH_SIZE};
	input.initialize_uniform(rng, 0.0f, 1.0f);

	const uint32_t padded_width = grid->padded_output_width();
	GPUMatrix<float, RM> output{padded_width, BATCH_SIZE};
	GPUMatrix<float, RM> dL_doutput{padded_width, BATCH_SIZE};
	dL_doutput.initialize_uniform(rng, -1.0f, 1.0f);

	GPUMemory<float> max_level_gpu;
	if (c.per_sample_max_level) {
		std::vector<float> max_level(BATCH_SIZE);
		for (auto& v : max_level) {
			v = rng.next_float();
		}
		max_level_gpu.resize(BATCH_SIZE);
		max_level_gpu.copy_from_host(max_level);
		grid->set_max_level_gpu(max_level_gpu.data());
	} else {
		grid->set_max_level(c.max_level);
	}

	// Gradient under test.
	{
		auto ctx = grid->forward(input, &output);
		grid->backward(*ctx, input, output, dL_doutput, nullptr, false, c.mode);
	}
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	std::vector<float> actual(n_params);
	gradients.copy_to_host(actual);

	// Reference: kernel_grid_backward over all levels, on the same (row-major = SoA) output gradient.
	static constexpr uint32_t N_THREADS = 256;
	static constexpr uint32_t N_FEATURES_PER_THREAD = std::min(2u, N_FEATURES_PER_LEVEL);
	const bool stochastic = c.config.value("stochastic_interpolation", false);
	const InterpolationType interpolation = string_to_interpolation_type(c.config.value("interpolation", "Linear"));
	const GridType grid_type = string_to_grid_type(c.config.value("type", "Hash"));

	GPUMemory<float> reference_gpu(n_params);
	std::vector<float> reference_runs[2];
	for (auto& reference : reference_runs) {
		reference_gpu.copy_from_host(c.mode == GradientMode::Accumulate ? initial_gradient : std::vector<float>(n_params, 0.0f));
		const dim3 blocks = {div_round_up(BATCH_SIZE * N_FEATURES_PER_LEVEL / N_FEATURES_PER_THREAD, N_THREADS), n_levels, 1};
		kernel_grid_backward<float, float, N_POS_DIMS, N_FEATURES_PER_LEVEL, N_FEATURES_PER_THREAD, HashType::CoherentPrime><<<blocks, N_THREADS>>>(
			BATCH_SIZE,
			n_features,
			offset_table,
			c.config.value("base_resolution", 16u),
			std::log2(c.config.value("per_level_scale", 2.0f)),
			c.per_sample_max_level ? 1000.0f : c.max_level,
			c.per_sample_max_level ? max_level_gpu.data() : nullptr,
			stochastic,
			interpolation,
			grid_type,
			reference_gpu.data(),
			input.view(),
			dL_doutput.data()
		);
		CUDA_CHECK_THROW(cudaGetLastError());
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
		reference.resize(n_params);
		reference_gpu.copy_to_host(reference);
	}
	const std::vector<float>& reference = reference_runs[0];

	// Per-level relative error with the level's RMS as floor.
	double max_rel = 0.0, max_rel_shared = 0.0, max_rel_reference = 0.0;
	size_t n_nonzero = 0, n_nonfinite = 0, worst = 0;
	for (uint32_t level = 0; level < n_levels; ++level) {
		const size_t begin = (size_t)offset_table.data[level] * N_FEATURES_PER_LEVEL;
		const size_t end = (size_t)offset_table.data[level + 1] * N_FEATURES_PER_LEVEL;
		double sum_sq = 0.0;
		for (size_t i = begin; i < end; ++i) {
			sum_sq += (double)reference[i] * reference[i];
		}
		const double rms = std::sqrt(sum_sq / (double)(end - begin));
		for (size_t i = begin; i < end; ++i) {
			const double a = actual[i], b = reference[i];
			if (!std::isfinite(a)) {
				++n_nonfinite;
				continue;
			}
			n_nonzero += b != 0.0;
			const double denom = std::abs(b) + rms;
			const double rel = denom > 0.0 ? std::abs(a - b) / denom : std::abs(a - b);
			const double rel_reference = denom > 0.0 ? std::abs((double)reference_runs[1][i] - b) / denom : std::abs((double)reference_runs[1][i] - b);
			if (rel > max_rel) {
				max_rel = rel;
				worst = i;
			}
			if (level < n_shared_levels) {
				max_rel_shared = std::max(max_rel_shared, rel);
			}
			max_rel_reference = std::max(max_rel_reference, rel_reference);
		}
	}

	fmt::print(
		"{} | levels {} (shared memory {}, direct {}) | params {} nonzero {} | max rel {:.2e} (shared-memory levels {:.2e}) | reference vs itself {:.2e}\n",
		c.label, n_levels, n_shared_levels, n_levels - n_shared_levels, n_params, n_nonzero, max_rel, max_rel_shared, max_rel_reference
	);
	INFO("max relative error " << max_rel << " at parameter " << worst << ": " << actual[worst] << " vs " << reference[worst]);
	CHECK(n_nonfinite == 0);
	CHECK(n_nonzero > 0);
	CHECK(max_rel < GRADIENT_TOLERANCE);
}

json grid_config(const char* type, uint32_t n_levels, uint32_t n_features_per_level, uint32_t base_resolution, uint32_t log2_hashmap_size, const char* interpolation, bool stochastic) {
	return {
		{"otype", "Grid"},
		{"type", type},
		{"n_levels", n_levels},
		{"n_features_per_level", n_features_per_level},
		{"base_resolution", base_resolution},
		{"per_level_scale", 2.0f},
		{"log2_hashmap_size", log2_hashmap_size},
		{"interpolation", interpolation},
		{"stochastic_interpolation", stochastic},
	};
}

// Interpolation / max-level / gradient-mode variants run for every grid layout.
std::vector<Case> variants(const std::string& name, const char* type, uint32_t n_levels, uint32_t n_features_per_level, uint32_t base_resolution, uint32_t log2_hashmap_size) {
	auto make = [&](const char* variant, const char* interpolation, bool stochastic, bool per_sample_max_level, float max_level, GradientMode mode) {
		return Case{
			fmt::format("{} {}", name, variant),
			grid_config(type, n_levels, n_features_per_level, base_resolution, log2_hashmap_size, interpolation, stochastic),
			per_sample_max_level,
			max_level,
			mode,
		};
	};
	return {
		make("linear", "Linear", false, false, 1000.0f, GradientMode::Overwrite),
		make("nearest", "Nearest", false, false, 1000.0f, GradientMode::Overwrite),
		make("smoothstep", "Smoothstep", false, false, 1000.0f, GradientMode::Overwrite),
		make("linear stochastic", "Linear", true, false, 1000.0f, GradientMode::Overwrite),
		make("linear per-sample max level", "Linear", false, true, 0.0f, GradientMode::Overwrite),
		make("smoothstep stochastic per-sample max level", "Smoothstep", true, true, 0.0f, GradientMode::Overwrite),
		make("linear max level 0.6", "Linear", false, false, 0.6f, GradientMode::Overwrite),
		make("linear accumulate", "Linear", false, false, 1000.0f, GradientMode::Accumulate),
	};
}

} // namespace

TEST_CASE("Grid parameter gradients match kernel_grid_backward", "[encoding][grid]") {
	tcnn_test_setup();

	// 2D hash grid, 2 features: levels 16^2, 32^2, 64^2 in shared memory, 5 hashed levels direct.
	for (const auto& c : variants("hash 2D F2", "Hash", 8, 2, 16, 14)) {
		check_grid_gradients<2, 2>(c);
	}
	// 3D hash grid, 2 features: levels 8^3, 16^3 in shared memory, 4 hashed levels direct.
	for (const auto& c : variants("hash 3D F2", "Hash", 6, 2, 8, 15)) {
		check_grid_gradients<3, 2>(c);
	}
	// 3D dense grid, 2 features: levels 8^3, 16^3 in shared memory, 32^3 and 64^3 direct.
	for (const auto& c : variants("dense 3D F2", "Dense", 4, 2, 8, 19)) {
		check_grid_gradients<3, 2>(c);
	}
	// 2D tiled grid with 16^2 tiles: every level in shared memory (single stream).
	for (const auto& c : variants("tiled 2D F2", "Tiled", 4, 2, 16, 19)) {
		check_grid_gradients<2, 2>(c);
	}
	// 3D tiled grid with 32^3 tiles: every level direct.
	for (const auto& c : variants("tiled 3D F2", "Tiled", 3, 2, 32, 19)) {
		check_grid_gradients<3, 2>(c);
	}
	// 2D hash grid, 1 feature (scalar atomics): 3 levels in shared memory, 3 direct.
	for (const auto& c : variants("hash 2D F1", "Hash", 6, 1, 16, 14)) {
		check_grid_gradients<2, 1>(c);
	}
	// 2D hash grid, 4 features: 64^2 x 4 floats exceeds the shared-memory budget, so 2 levels in shared memory.
	for (const auto& c : variants("hash 2D F4", "Hash", 4, 4, 16, 14)) {
		check_grid_gradients<2, 4>(c);
	}
	// 2D hash grid, 8 features: 3 levels in shared memory, 1 direct.
	for (const auto& c : variants("hash 2D F8", "Hash", 4, 8, 8, 14)) {
		check_grid_gradients<2, 8>(c);
	}
}
