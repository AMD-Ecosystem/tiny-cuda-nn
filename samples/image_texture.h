/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   image_texture.h
 *  @brief  Image lookup used by the image-learning sample and benchmark.
 *
 *  On NVIDIA GPUs this is the CUDA texture object (bilinear filtering, clamped addressing,
 *  normalized coordinates) that the sources create. HIP's image/texture API is unavailable on the
 *  AMD Instinct targets of the HIP build (__HIP_NO_IMAGE_SUPPORT), so the HIP build emulates the
 *  same lookup in software on the RGBA float image buffer. CUDA's linear filtering uses 8-bit
 *  fixed-point interpolation weights while the emulation uses fp32 weights, so sample outputs
 *  differ slightly between the two.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#if defined(TCNN_USE_HIP)

struct ImageTexture {
	const float4* data;
	int width;
	int height;
};

inline ImageTexture create_image_texture(float* rgba_image, int width, int height) {
	return {(const float4*)rgba_image, width, height};
}

__device__ inline float4 sample_image_texture(const ImageTexture& tex, float u, float v) {
	// Same convention as CUDA's linear texture filtering: texel centers sit at (i + 0.5) / size.
	const float x = u * tex.width - 0.5f;
	const float y = v * tex.height - 0.5f;
	const float x_floor = floorf(x);
	const float y_floor = floorf(y);
	const float wx = x - x_floor;
	const float wy = y - y_floor;

	auto texel = [&](int xi, int yi) {
		xi = min(max(xi, 0), tex.width - 1);
		yi = min(max(yi, 0), tex.height - 1);
		return tex.data[xi + yi * tex.width];
	};

	const int x0 = (int)x_floor;
	const int y0 = (int)y_floor;
	const float4 c00 = texel(x0, y0), c10 = texel(x0 + 1, y0), c01 = texel(x0, y0 + 1), c11 = texel(x0 + 1, y0 + 1);

	auto lerp4 = [](const float4& a, const float4& b, float t) {
		return float4{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
	};

	return lerp4(lerp4(c00, c10, wx), lerp4(c01, c11, wx), wy);
}

#else

using ImageTexture = cudaTextureObject_t;

__device__ inline float4 sample_image_texture(ImageTexture tex, float u, float v) {
	return tex2D<float4>(tex, u, v);
}

#endif
