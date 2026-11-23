/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   test_fused_mlp_parity.cu
 *  @brief  Cross-implementation checks of the MLP networks that do not rely on runtime compilation:
 *          FullyFusedMLP against CutlassMLP with identical parameters (forward, inference, input
 *          gradient and parameter gradient), and CutlassMLP against a CPU fp32 reference.
 *
 *  test_networks compares FullyFusedMLP and CutlassMLP only indirectly, through the JIT-fused
 *  versions of both. On builds without JIT support those checks do not run, so this test compares
 *  the two implementations directly.
 */

#include "test_common.h"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/network.h>

#include <algorithm>
#include <cmath>
#if defined(TCNN_USE_HIP)
#include <set>
#endif

using namespace tcnn;

namespace {

struct ErrorStats {
	double mean_rel = 0.0;   // mean |a - b| / mean |b|
	double p99_rel = 0.0;    // 99th percentile of |a - b| / (|b| + rms(b))
	double max_rel = 0.0;    // maximum of |a - b| / (|b| + rms(b))
	size_t n_nonfinite = 0;  // elements where either value is not finite
};

template <typename T, typename U>
ErrorStats compare(const std::vector<T>& actual, const std::vector<U>& expected) {
	REQUIRE(actual.size() == expected.size());
	REQUIRE(!expected.empty());

	ErrorStats stats;
	double sum_abs_ref = 0.0, sum_sq_ref = 0.0, sum_abs_err = 0.0;
	for (size_t i = 0; i < expected.size(); ++i) {
		const double a = (double)(float)actual[i];
		const double b = (double)(float)expected[i];
		if (!std::isfinite(a) || !std::isfinite(b)) {
			++stats.n_nonfinite;
			continue;
		}
		sum_abs_ref += std::abs(b);
		sum_sq_ref += b * b;
		sum_abs_err += std::abs(a - b);
	}

	const double n = (double)expected.size();
	const double mean_abs_ref = sum_abs_ref / n;
	const double rms_ref = std::sqrt(sum_sq_ref / n);
	stats.mean_rel = mean_abs_ref > 0.0 ? (sum_abs_err / n) / mean_abs_ref : sum_abs_err / n;

	// Per-element error relative to the element's own magnitude, with the RMS as a floor so that
	// near-zero elements are judged on an absolute scale.
	std::vector<double> rel;
	rel.reserve(expected.size());
	for (size_t i = 0; i < expected.size(); ++i) {
		const double a = (double)(float)actual[i];
		const double b = (double)(float)expected[i];
		if (std::isfinite(a) && std::isfinite(b)) {
			const double denom = std::abs(b) + rms_ref;
			rel.push_back(denom > 0.0 ? std::abs(a - b) / denom : std::abs(a - b));
		}
	}
	if (!rel.empty()) {
		const size_t k = std::min(rel.size() - 1, (size_t)std::floor(0.99 * (double)rel.size()));
		std::nth_element(rel.begin(), rel.begin() + k, rel.end());
		stats.p99_rel = rel[k];
		stats.max_rel = *std::max_element(rel.begin() + k, rel.end());
	}
	return stats;
}

// A systematic error (wrong lane mapping, wrong rows copied, wrong epilogue) produces O(1) errors in
// most elements. Rounding to fp16 gives mean errors of order 1e-3. Gradients through ReLU and
// LeakyReLU additionally have isolated O(1) outliers wherever rounding moves a pre-activation across
// the kink, and parameter gradients accumulate those over the batch. The unmodified CUDA
// implementation on H100 shows, over this test's configurations (worst case):
//   outputs:                         mean 1.6e-3, p99 3.3e-3
//   gradients, smooth activations:   mean 9.5e-3, p99 1.2e-2
//   gradients, ReLU / LeakyReLU:     mean 3.4e-2, p99 1.2e-1
// The thresholds below leave headroom over those values while staying far below the O(1) errors
// of a systematic defect.
//
// On HIP both implementations accumulate in fp32 (rocWMMA and hipBLASLt), so they agree much more
// closely and the bounds can be tighter. There the maximum per-element error is checked as well:
// a defect that corrupts only a few elements (one element per matrix fragment, one row per block)
// barely moves the mean or the 99th percentile but shows up as an O(1) maximum. The worst case
// measured on MI300-series (gfx942) and MI350X (gfx950), over this test's configurations
// including the comparison against the CPU reference:
//   outputs:                         mean 1.1e-3, p99 1.8e-3, max 1.2e-2
//   gradients, smooth activations:   mean 4.0e-3, p99 5.8e-3, max 2.5e-2
//   gradients, ReLU / LeakyReLU:     mean 3.7e-4, p99 6.1e-4, max 1.3e-3
// The HIP bounds leave at least 2x headroom over those values.
//
// A ReLU / LeakyReLU gradient can additionally contain one kink crossing: a pre-activation within
// rounding of zero that lands on different sides of the kink in the two implementations. It flips one
// activation derivative, which changes the dL_dinput column of that one sample and the dL_dparams row
// of that one neuron (measured on gfx942 with the hipBLASLt GEMMs: max 2.2e-2 in dL_dinput, 8.8e-3 in
// dL_dparams, width=128, LeakyReLU, n_output_dims=32, column-major input). The HIP check therefore
// allows elements above `max` only if they all fall in ONE sample column of dL_dinput and ONE neuron
// row of dL_dparams, and keeps every element below `ceiling`. A defect that corrupts one element per
// fragment or one row per block touches many columns / rows and still fails. When a crossing occurs,
// the test recomputes the pre-activation of that sample and neuron on the CPU and checks that it is
// within rounding of zero.
struct Thresholds {
	double mean;
	double p99;
	double max; // not checked when infinite
#if defined(TCNN_USE_HIP)
	double ceiling; // hard per-element bound; elements in (max, ceiling) are allowed in one sample / neuron only
#endif
};
#if defined(TCNN_USE_HIP)
constexpr Thresholds OUTPUT_THRESHOLDS = {5e-3, 1e-2, 5e-2, 5e-2};
constexpr Thresholds SMOOTH_GRADIENT_THRESHOLDS = {1.5e-2, 2.5e-2, 1e-1, 1e-1};
constexpr Thresholds KINK_GRADIENT_THRESHOLDS = {1.5e-3, 2.5e-3, 5e-3, 5e-2};

// A kink crossing puts the pre-activation within rounding of zero: |z| is a small multiple of the fp16
// rounding of the layer's inputs, well below this fraction of the layer's RMS pre-activation.
constexpr double KINK_CROSSING_MAX_REL_PREACTIVATION = 1e-2;
#else
constexpr Thresholds OUTPUT_THRESHOLDS = {1e-2, 5e-2, INFINITY};
constexpr Thresholds SMOOTH_GRADIENT_THRESHOLDS = {2e-2, 5e-2, INFINITY};
constexpr Thresholds KINK_GRADIENT_THRESHOLDS = {1e-1, 3e-1, INFINITY};
#endif

Thresholds gradient_thresholds(const std::string& activation) {
	return activation == "ReLU" || activation == "LeakyReLU" ? KINK_GRADIENT_THRESHOLDS : SMOOTH_GRADIENT_THRESHOLDS;
}

template <typename T, typename U>
void check_match(const std::string& label, const char* what, const std::vector<T>& actual, const std::vector<U>& expected, const Thresholds& thresholds) {
	const ErrorStats stats = compare(actual, expected);
	fmt::print("{} | {:<16} | mean {:.2e} | p99 {:.2e} | max {:.2e} | non-finite {}\n", label, what, stats.mean_rel, stats.p99_rel, stats.max_rel, stats.n_nonfinite);
	INFO(what << ": mean relative error " << stats.mean_rel << ", p99 relative error " << stats.p99_rel << ", max relative error " << stats.max_rel << ", non-finite " << stats.n_nonfinite);
	CHECK(stats.n_nonfinite == 0);
	CHECK(stats.mean_rel < thresholds.mean);
	CHECK(stats.p99_rel < thresholds.p99);
	if (std::isfinite(thresholds.max)) {
		CHECK(stats.max_rel < thresholds.max);
	}
}

template <typename T>
std::vector<T> device_to_host(const T* data, size_t n) {
	std::vector<T> result(n);
	CUDA_CHECK_THROW(cudaMemcpy(result.data(), data, n * sizeof(T), cudaMemcpyDeviceToHost));
	return result;
}

std::string layout_name(MatrixLayout layout) {
	return layout == RM ? "RM" : "CM";
}
#if defined(TCNN_USE_HIP)

using LayerSizes = std::vector<std::pair<uint32_t, uint32_t>>; // (rows, cols) of each row-major weight matrix

// Elements whose relative error (as in compare()) is at least `threshold`, and the distinct groups they
// fall in: the sample (column) of dL_dinput or the neuron (row) of dL_dparams, given by `group_of`.
struct Outliers {
	size_t count = 0;
	std::vector<size_t> groups; // sorted, distinct
};

template <typename T, typename U, typename GroupOf>
Outliers find_outliers(const std::vector<T>& actual, const std::vector<U>& expected, double threshold, GroupOf group_of) {
	double sum_sq_ref = 0.0;
	for (size_t i = 0; i < expected.size(); ++i) {
		const double a = (double)(float)actual[i];
		const double b = (double)(float)expected[i];
		if (std::isfinite(a) && std::isfinite(b)) {
			sum_sq_ref += b * b;
		}
	}
	const double rms_ref = std::sqrt(sum_sq_ref / (double)expected.size());

	Outliers result;
	std::set<size_t> groups;
	for (size_t i = 0; i < expected.size(); ++i) {
		const double a = (double)(float)actual[i];
		const double b = (double)(float)expected[i];
		if (!std::isfinite(a) || !std::isfinite(b)) {
			continue;
		}
		const double denom = std::abs(b) + rms_ref;
		const double rel = denom > 0.0 ? std::abs(a - b) / denom : std::abs(a - b);
		if (rel >= threshold) {
			++result.count;
			groups.insert(group_of(i));
		}
	}
	result.groups.assign(groups.begin(), groups.end());
	return result;
}

std::string groups_string(const std::vector<size_t>& groups) {
	std::string result;
	for (size_t i = 0; i < groups.size() && i < 8; ++i) {
		result += fmt::format("{}{}", i == 0 ? "" : ",", groups[i]);
	}
	return groups.size() > 8 ? result + ",..." : result;
}

// Like check_match, but elements above thresholds.max are allowed if they all fall in one group (one
// kink crossing, see above) and stay below thresholds.ceiling.
template <typename T, typename U, typename GroupOf>
Outliers check_gradient_match(const std::string& label, const char* what, const char* group_kind, const std::vector<T>& actual, const std::vector<U>& expected, const Thresholds& thresholds, GroupOf group_of) {
	const ErrorStats stats = compare(actual, expected);
	const Outliers outliers = find_outliers(actual, expected, thresholds.max, group_of);
	fmt::print(
		"{} | {:<16} | mean {:.2e} | p99 {:.2e} | max {:.2e} | non-finite {} | over {:.0e}: {} in {} {} [{}]\n",
		label, what, stats.mean_rel, stats.p99_rel, stats.max_rel, stats.n_nonfinite,
		thresholds.max, outliers.count, outliers.groups.size(), group_kind, groups_string(outliers.groups)
	);
	INFO(what << ": mean relative error " << stats.mean_rel << ", p99 relative error " << stats.p99_rel << ", max relative error " << stats.max_rel
		<< ", non-finite " << stats.n_nonfinite << ", " << outliers.count << " elements over " << thresholds.max << " in " << outliers.groups.size() << " " << group_kind);
	CHECK(stats.n_nonfinite == 0);
	CHECK(stats.mean_rel < thresholds.mean);
	CHECK(stats.p99_rel < thresholds.p99);
	CHECK(outliers.groups.size() <= 1);
	CHECK(stats.max_rel < thresholds.ceiling);
	return outliers;
}

// Sample (column) of an element of a [n_dims x batch_size] matrix.
auto sample_of(uint32_t n_dims, uint32_t batch_size, MatrixLayout layout) {
	return [=](size_t i) -> size_t { return layout == CM ? i / n_dims : i % batch_size; };
}

// Neuron row, counted over all layers, of an element of a parameter vector made of row-major matrices.
auto neuron_row_of(const LayerSizes& layers) {
	return [layers](size_t i) -> size_t {
		size_t offset = 0, first_row = 0;
		for (const auto& [rows, cols] : layers) {
			if (i < offset + (size_t)rows * cols) {
				return first_row + (i - offset) / cols;
			}
			offset += (size_t)rows * cols;
			first_row += rows;
		}
		return first_row;
	};
}

// CPU fp32 pre-activations z_l = W_l * a_l of all layers for all samples. a_0 is the input and
// a_{l+1} = act(z_l), rounded to T as the networks store it. Result: z[l][sample * rows_l + row].
template <typename T>
std::vector<std::vector<float>> cpu_preactivations(const std::vector<T>& params, const LayerSizes& layers, const std::vector<T>& input, uint32_t batch_size, MatrixLayout input_layout, Activation activation, Activation output_activation) {
	const uint32_t n_input_dims = layers.front().second;
	std::vector<float> a((size_t)n_input_dims * batch_size);
	for (uint32_t b = 0; b < batch_size; ++b) {
		for (uint32_t d = 0; d < n_input_dims; ++d) {
			a[(size_t)b * n_input_dims + d] = (float)input[input_layout == CM ? (size_t)b * n_input_dims + d : (size_t)d * batch_size + b];
		}
	}

	std::vector<std::vector<float>> z(layers.size());
	size_t offset = 0;
	for (size_t l = 0; l < layers.size(); ++l) {
		const auto [rows, cols] = layers[l];
		const Activation act = l + 1 == layers.size() ? output_activation : activation;
		z[l].resize((size_t)rows * batch_size);
		std::vector<float> next((size_t)rows * batch_size);
		for (uint32_t b = 0; b < batch_size; ++b) {
			for (uint32_t r = 0; r < rows; ++r) {
				float sum = 0.0f;
				for (uint32_t c = 0; c < cols; ++c) {
					sum += (float)params[offset + (size_t)r * cols + c] * a[(size_t)b * cols + c];
				}
				z[l][(size_t)b * rows + r] = sum;
				next[(size_t)b * rows + r] = (float)(T)tcnn::activation<float>(act, sum);
			}
		}
		a = std::move(next);
		offset += (size_t)rows * cols;
	}
	return z;
}

// For a gradient comparison with elements above the max (one sample column of dL_dinput and/or one neuron
// row of dL_dparams), prints the pre-activation of that sample at that neuron. If both are known, checks
// that it is within rounding of zero, i.e. that the outliers are a kink crossing.
template <typename T>
void check_kink_crossing(
	const std::string& label,
	const Outliers& dinput_outliers,
	const Outliers& dparams_outliers,
	const LayerSizes& layers,
	const std::vector<T>& params,
	const std::vector<T>& input,
	uint32_t batch_size,
	MatrixLayout input_layout,
	Activation activation,
	Activation output_activation
) {
	if (dinput_outliers.groups.size() != 1 && dparams_outliers.groups.size() != 1) {
		return;
	}

	const auto z = cpu_preactivations(params, layers, input, batch_size, input_layout, activation, output_activation);
	std::vector<double> rms(layers.size());
	for (size_t l = 0; l < layers.size(); ++l) {
		double sum_sq = 0.0;
		for (float v : z[l]) {
			sum_sq += (double)v * v;
		}
		rms[l] = std::sqrt(sum_sq / (double)z[l].size());
	}
	auto rel_z = [&](size_t l, size_t sample, size_t row) {
		return std::abs((double)z[l][sample * layers[l].first + row]) / rms[l];
	};

	if (dparams_outliers.groups.size() == 1) {
		// Neuron (layer, row) of the dL_dparams outliers, and the sample closest to its kink.
		size_t l = 0, row = dparams_outliers.groups.front();
		while (l + 1 < layers.size() && row >= layers[l].first) {
			row -= layers[l].first;
			++l;
		}
		size_t closest = 0;
		for (size_t b = 1; b < batch_size; ++b) {
			if (rel_z(l, b, row) < rel_z(l, closest, row)) {
				closest = b;
			}
		}

		if (dinput_outliers.groups.size() == 1) {
			const size_t sample = dinput_outliers.groups.front();
			const float value = z[l][sample * layers[l].first + row];
			fmt::print(
				"{} | kink crossing    | sample {} layer {} neuron {}: pre-activation {:.3e} = {:.2e} x RMS {:.3e} (sample closest to the kink of this neuron: {}, {:.2e} x RMS)\n",
				label, sample, l, row, value, rel_z(l, sample, row), rms[l], closest, rel_z(l, closest, row)
			);
			INFO("kink crossing: sample " << sample << " layer " << l << " neuron " << row << " pre-activation " << value << " RMS " << rms[l]);
			CHECK(rel_z(l, sample, row) < KINK_CROSSING_MAX_REL_PREACTIVATION);
		} else {
			fmt::print(
				"{} | kink crossing    | layer {} neuron {}: sample closest to the kink {}, pre-activation {:.3e} = {:.2e} x RMS {:.3e}\n",
				label, l, row, closest, z[l][closest * layers[l].first + row], rel_z(l, closest, row), rms[l]
			);
		}
	} else {
		// Only the sample is known: report the neuron closest to the kink for it.
		const size_t sample = dinput_outliers.groups.front();
		size_t best_l = 0, best_row = 0;
		for (size_t l = 0; l < layers.size(); ++l) {
			for (size_t r = 0; r < layers[l].first; ++r) {
				if (rel_z(l, sample, r) < rel_z(best_l, sample, best_row)) {
					best_l = l;
					best_row = r;
				}
			}
		}
		fmt::print(
			"{} | kink crossing    | sample {}: neuron closest to the kink layer {} neuron {}, pre-activation {:.3e} = {:.2e} x RMS {:.3e}\n",
			label, sample, best_l, best_row, z[best_l][sample * layers[best_l].first + best_row], rel_z(best_l, sample, best_row), rms[best_l]
		);
	}
}
#endif

} // namespace

TEST_CASE("FullyFusedMLP matches CutlassMLP", "[network][parity]") {
	using T = network_precision_t;

	tcnn_test_setup();

	if (MIN_GPU_ARCH <= 70) {
		SUCCEED("FullyFusedMLP is not available for this GPU target.");
		return;
	}

	const std::vector<uint32_t> widths = {16, 32, 64, 128};
	const std::vector<std::string> activations = {"None", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softplus", "Squareplus", "Exponential"};
	const std::vector<uint32_t> output_widths = {16, 32};
	const std::vector<MatrixLayout> input_layouts = {CM, RM};

	const uint32_t n_input_dims = 32;
	const uint32_t n_hidden_layers = 2;
	const uint32_t batch_size = 4 * BATCH_SIZE_GRANULARITY;

	uint32_t n_configs = 0;

	for (uint32_t width : widths) {
		for (const auto& activation : activations) {
			// The output activation exercises the output-layer epilogue (forward) and transfer (backward).
			// Exponential is not used as output activation on top of Exponential hidden layers: the
			// result overflows fp16 in both implementations.
			const bool vary_output_activation = activation != "None" && activation != "Exponential";
			for (uint32_t output_variant = 0; output_variant < (vary_output_activation ? 2u : 1u); ++output_variant) {
				const std::string output_activation = output_variant == 0 ? std::string{"None"} : activation;
				for (uint32_t n_out : output_widths) {
					for (MatrixLayout input_layout : input_layouts) {
						const std::string label = fmt::format("width={} activation={} output_activation={} n_output_dims={} input_layout={}",
							width, activation, output_activation, n_out, layout_name(input_layout));
						INFO(label);

						json config = {
							{"otype", "CutlassMLP"},
							{"n_input_dims", n_input_dims},
							{"n_output_dims", n_out},
							{"activation", activation},
							{"output_activation", output_activation},
							{"n_neurons", width},
							{"n_hidden_layers", n_hidden_layers},
						};

						std::shared_ptr<Network<T>> cutlass{create_network<T>(config)};
						config["otype"] = "FullyFusedMLP";
						std::shared_ptr<Network<T>> fused{create_network<T>(config)};
						REQUIRE(cutlass->n_params() == fused->n_params());
						REQUIRE(cutlass->padded_output_width() == fused->padded_output_width());

						std::shared_ptr<Optimizer<T>> optimizer{create_optimizer<T>(json::object())};
						std::shared_ptr<Loss<T>> loss{create_loss<T>(json::object())};
						auto cutlass_trainer = std::make_shared<Trainer<T, T, T>>(cutlass, optimizer, loss);
						auto fused_trainer = std::make_shared<Trainer<T, T, T>>(fused, optimizer, loss);

						// Identical parameters (same layout: input, hidden, output weight matrices, row-major).
						fused_trainer->set_params_full_precision(cutlass_trainer->params_full_precision(), cutlass->n_params(), true);

						pcg32 rng{1337 + n_configs};
						const uint32_t n_out_padded = cutlass->padded_output_width();
						GPUMatrixDynamic<T> input{n_input_dims, batch_size, input_layout};
						input.initialize_uniform(rng, -1.0f, 1.0f);
						GPUMatrix<T> output_gradient{n_out_padded, batch_size};
						output_gradient.initialize_uniform(rng, -1.0f, 1.0f);

						GPUMatrix<T> cutlass_output{n_out_padded, batch_size};
						GPUMatrix<T> fused_output{n_out_padded, batch_size};
						GPUMatrixDynamic<T> cutlass_dinput{n_input_dims, batch_size, input_layout};
						GPUMatrixDynamic<T> fused_dinput{n_input_dims, batch_size, input_layout};

						// Inference path (separate fused kernel instantiation).
						cutlass->inference_mixed_precision(input, cutlass_output);
						fused->inference_mixed_precision(input, fused_output);
						CUDA_CHECK_THROW(cudaDeviceSynchronize());
						check_match(label, "inference output", fused_output.to_cpu_vector(), cutlass_output.to_cpu_vector(), OUTPUT_THRESHOLDS);

						// Training path: forward + backward with parameter and input gradients.
						{
							auto ctx = cutlass->forward(input, &cutlass_output, false, true);
							cutlass->backward(*ctx, input, cutlass_output, output_gradient, &cutlass_dinput, false, GradientMode::Overwrite);
						}
						{
							auto ctx = fused->forward(input, &fused_output, false, true);
							fused->backward(*ctx, input, fused_output, output_gradient, &fused_dinput, false, GradientMode::Overwrite);
						}
						CUDA_CHECK_THROW(cudaDeviceSynchronize());

						check_match(label, "forward output", fused_output.to_cpu_vector(), cutlass_output.to_cpu_vector(), OUTPUT_THRESHOLDS);
#if defined(TCNN_USE_HIP)
						{
							const LayerSizes layers = cutlass->layer_sizes();
							const Outliers dinput_outliers = check_gradient_match(
								label, "dL_dinput", "samples", fused_dinput.to_cpu_vector(), cutlass_dinput.to_cpu_vector(), gradient_thresholds(activation),
								sample_of(n_input_dims, batch_size, input_layout)
							);
							const Outliers dparams_outliers = check_gradient_match(
								label,
								"dL_dparams",
								"neurons",
								device_to_host(fused_trainer->param_gradients(), fused->n_params()),
								device_to_host(cutlass_trainer->param_gradients(), cutlass->n_params()),
								gradient_thresholds(activation),
								neuron_row_of(layers)
							);
							check_kink_crossing(
								label, dinput_outliers, dparams_outliers, layers, device_to_host(cutlass_trainer->params(), cutlass->n_params()), input.to_cpu_vector(),
								batch_size, input_layout, string_to_activation(activation), string_to_activation(output_activation)
							);
						}
#else
						check_match(label, "dL_dinput", fused_dinput.to_cpu_vector(), cutlass_dinput.to_cpu_vector(), gradient_thresholds(activation));
						check_match(
							label,
							"dL_dparams",
							device_to_host(fused_trainer->param_gradients(), fused->n_params()),
							device_to_host(cutlass_trainer->param_gradients(), cutlass->n_params()),
							gradient_thresholds(activation)
						);
#endif

						++n_configs;
					}
				}
			}
		}
	}

	// 4 widths x (2 + 6 x 2 activation pairs) x 2 output widths x 2 input layouts
	CHECK(n_configs == 4 * 14 * 2 * 2);
}

TEST_CASE("CutlassMLP matches a CPU reference", "[network][parity]") {
	using T = network_precision_t;

	tcnn_test_setup();

	const uint32_t n_input_dims = 32;
	const uint32_t width = 64;
	const uint32_t n_out = 32;
	const uint32_t n_hidden_layers = 2;
	const uint32_t batch_size = BATCH_SIZE_GRANULARITY;

	const auto [activation_name, output_activation_name] = GENERATE(table<std::string, std::string>({
		{"ReLU", "None"},
		{"Tanh", "Sigmoid"},
	}));
	CAPTURE(activation_name, output_activation_name);

	json config = {
		{"otype", "CutlassMLP"},
		{"n_input_dims", n_input_dims},
		{"n_output_dims", n_out},
		{"activation", activation_name},
		{"output_activation", output_activation_name},
		{"n_neurons", width},
		{"n_hidden_layers", n_hidden_layers},
	};

	std::shared_ptr<Network<T>> network{create_network<T>(config)};
	REQUIRE(network->padded_output_width() == n_out);
	std::shared_ptr<Optimizer<T>> optimizer{create_optimizer<T>(json::object())};
	std::shared_ptr<Loss<T>> loss{create_loss<T>(json::object())};
	auto trainer = std::make_shared<Trainer<T, T, T>>(network, optimizer, loss);

	const Activation activation = string_to_activation(activation_name);
	const Activation output_activation = string_to_activation(output_activation_name);

	pcg32 rng{4242};
	GPUMatrix<T> input{n_input_dims, batch_size};
	input.initialize_uniform(rng, -1.0f, 1.0f);
	GPUMatrix<T> output_gradient{n_out, batch_size};
	output_gradient.initialize_uniform(rng, -1.0f, 1.0f);
	GPUMatrix<T> output{n_out, batch_size};
	GPUMatrix<T> dinput{n_input_dims, batch_size};

	{
		auto ctx = network->forward(input, &output, false, true);
		network->backward(*ctx, input, output, output_gradient, &dinput, false, GradientMode::Overwrite);
	}
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	// CPU fp32 reference. Matrices are column-major (feature index fastest); weights are row-major
	// [n_out x n_in] matrices stored back to back: input layer, hidden layers, output layer.
	const auto params = device_to_host(trainer->params(), network->n_params());
	const auto x = input.to_cpu_vector();
	const auto dy = output_gradient.to_cpu_vector();

	std::vector<std::pair<uint32_t, uint32_t>> layers; // (n_out, n_in)
	layers.emplace_back(width, n_input_dims);
	for (uint32_t i = 1; i < n_hidden_layers; ++i) {
		layers.emplace_back(width, width);
	}
	layers.emplace_back(n_out, width);

	std::vector<size_t> offsets;
	size_t offset = 0;
	for (const auto& [rows, cols] : layers) {
		offsets.push_back(offset);
		offset += (size_t)rows * cols;
	}
	REQUIRE(offset == network->n_params());

	auto weight = [&](size_t layer, uint32_t row, uint32_t col) {
		return (float)params[offsets[layer] + (size_t)row * layers[layer].second + col];
	};

	// Forward: acts[0] is the input, acts[l + 1] the output of layer l after its activation.
	std::vector<std::vector<float>> acts(layers.size() + 1);
	acts[0].resize((size_t)n_input_dims * batch_size);
	for (size_t i = 0; i < acts[0].size(); ++i) {
		acts[0][i] = (float)x[i];
	}

	for (size_t l = 0; l < layers.size(); ++l) {
		const auto [rows, cols] = layers[l];
		const Activation act = l + 1 == layers.size() ? output_activation : activation;
		acts[l + 1].resize((size_t)rows * batch_size);
		for (uint32_t b = 0; b < batch_size; ++b) {
			for (uint32_t r = 0; r < rows; ++r) {
				float sum = 0.0f;
				for (uint32_t c = 0; c < cols; ++c) {
					sum += weight(l, r, c) * acts[l][(size_t)b * cols + c];
				}
				acts[l + 1][(size_t)b * rows + r] = tcnn::activation<float>(act, sum);
			}
		}
	}

	// Backward
	std::vector<float> dparams(network->n_params(), 0.0f);
	std::vector<float> grad((size_t)n_out * batch_size);
	for (size_t i = 0; i < grad.size(); ++i) {
		grad[i] = (float)dy[i];
	}

	std::vector<float> dx;
	for (size_t l = layers.size(); l-- > 0;) {
		const auto [rows, cols] = layers[l];
		const Activation act = l + 1 == layers.size() ? output_activation : activation;

		// Gradient w.r.t. the pre-activation of layer l
		std::vector<float> delta((size_t)rows * batch_size);
		for (size_t i = 0; i < delta.size(); ++i) {
			delta[i] = tcnn::activation_backward<float>(act, grad[i], acts[l + 1][i]);
		}

		for (uint32_t r = 0; r < rows; ++r) {
			for (uint32_t c = 0; c < cols; ++c) {
				float sum = 0.0f;
				for (uint32_t b = 0; b < batch_size; ++b) {
					sum += delta[(size_t)b * rows + r] * acts[l][(size_t)b * cols + c];
				}
				dparams[offsets[l] + (size_t)r * cols + c] = sum;
			}
		}

		std::vector<float> prev((size_t)cols * batch_size, 0.0f);
		for (uint32_t b = 0; b < batch_size; ++b) {
			for (uint32_t c = 0; c < cols; ++c) {
				float sum = 0.0f;
				for (uint32_t r = 0; r < rows; ++r) {
					sum += weight(l, r, c) * delta[(size_t)b * rows + r];
				}
				prev[(size_t)b * cols + c] = sum;
			}
		}
		grad = std::move(prev);
	}
	dx = std::move(grad);

	const std::string label = fmt::format("CutlassMLP vs CPU activation={} output_activation={}", activation_name, output_activation_name);
	check_match(label, "forward output", output.to_cpu_vector(), acts.back(), OUTPUT_THRESHOLDS);
#if defined(TCNN_USE_HIP)
	const LayerSizes layer_sizes = network->layer_sizes();
	const Outliers dinput_outliers = check_gradient_match(label, "dL_dinput", "samples", dinput.to_cpu_vector(), dx, gradient_thresholds(activation_name), sample_of(n_input_dims, batch_size, CM));
	const Outliers dparams_outliers = check_gradient_match(
		label, "dL_dparams", "neurons", device_to_host(trainer->param_gradients(), network->n_params()), dparams, gradient_thresholds(activation_name), neuron_row_of(layer_sizes)
	);
	check_kink_crossing(label, dinput_outliers, dparams_outliers, layer_sizes, params, x, batch_size, CM, activation, output_activation);
#else
	check_match(label, "dL_dinput", dinput.to_cpu_vector(), dx, gradient_thresholds(activation_name));
	check_match(label, "dL_dparams", device_to_host(trainer->param_gradients(), network->n_params()), dparams, gradient_thresholds(activation_name));
#endif
}
