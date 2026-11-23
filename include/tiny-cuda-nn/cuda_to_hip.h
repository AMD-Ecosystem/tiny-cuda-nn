/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file   cuda_to_hip.h
 *  @brief  Portability layer that lets tiny-cuda-nn's CUDA-spelled sources compile for AMD GPUs
 *          with HIP/ROCm when the build defines TCNN_USE_HIP.
 *
 *  Without TCNN_USE_HIP (the default NVIDIA build) this header includes NOTHING and only defines a
 *  handful of portability macros that expand to the original CUDA spellings, so CUDA compilation
 *  is unchanged. It must stay include-free on that path: common.h reaches it from the runtime
 *  (NVRTC) JIT preamble, where only a few stub headers exist and <cuda_runtime.h> cannot be found.
 *
 *  Macros provided on both platforms:
 *    TCNN_GPU_COMPILER       defined when compiling with a GPU compiler (nvcc / clang-CUDA / hip-clang)
 *    TCNN_DEVICE_CODE        defined only in the device compilation pass
 *    TCNN_DEVICE_ARCH        feature level used by `>= N` guards in device code. On NVIDIA it is
 *                            __CUDA_ARCH__ (device pass only). On AMD it is 800 in BOTH passes,
 *                            because hip-clang semantically analyzes device code in the host pass
 *                            too; all AMD Instinct targets provide the features tiny-cuda-nn gates
 *                            on sm_60..sm_80 (native half2 arithmetic, half2/float atomics, __hmax).
 *    TCNN_FULL_WARP_MASK     full-warp/wavefront mask for the *_sync shuffle intrinsics
 *                            (32-bit on NVIDIA, 64-bit on AMD as required by ROCm 7+).
 */

#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#  define TCNN_GPU_COMPILER 1
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#  define TCNN_DEVICE_CODE 1
#endif

#if !defined(TCNN_USE_HIP)

// No includes here (see the file comment): the NVRTC JIT preamble reaches this header.

#if defined(__CUDA_ARCH__)
#  define TCNN_DEVICE_ARCH __CUDA_ARCH__
#endif

#define TCNN_FULL_WARP_MASK 0xffffffffu

#else // TCNN_USE_HIP

#if !defined(__HIP_PLATFORM_AMD__)
#  define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>
#if defined(TCNN_GPU_COMPILER)
#  include <hip/hip_fp16.h>
#endif

#include <cstdint>

#if defined(TCNN_GPU_COMPILER)
#  define TCNN_DEVICE_ARCH 800
#endif

#define TCNN_FULL_WARP_MASK 0xffffffffffffffffull

#if defined(TCNN_GPU_COMPILER)
#include <cmath>
// CUDA's math headers also declare a host-side rsqrtf (evaluated as 1/sqrt in double precision);
// HIP only provides the device version.
__host__ inline float rsqrtf(float x) { return (float)(1.0 / std::sqrt((double)x)); }
#endif

// ---------------------------------------------------------------------------------------------
// CUDA runtime API -> HIP runtime API (only the symbols tiny-cuda-nn uses).
// Sources that were already hipified (e.g. by PyTorch's build-time hipify) spell the HIP names
// directly and are unaffected by these aliases.
// ---------------------------------------------------------------------------------------------
#define cudaStream_t                                 hipStream_t
#define cudaEvent_t                                  hipEvent_t
#define cudaError_t                                  hipError_t
#define cudaDeviceProp                               hipDeviceProp_t
#define cudaGraph_t                                  hipGraph_t
#define cudaGraphExec_t                              hipGraphExec_t
#define cudaGraphNode_t                              hipGraphNode_t
#define cudaGraphExecUpdateResult                    hipGraphExecUpdateResult
#define cudaStreamCaptureStatus                      hipStreamCaptureStatus
#define cudaTextureObject_t                          hipTextureObject_t
#define cudaResourceDesc                             hipResourceDesc
#define cudaTextureDesc                              hipTextureDesc

#define cudaSuccess                                  hipSuccess
#define cudaErrorStreamCaptureImplicit               hipErrorStreamCaptureImplicit
#define cudaMemcpyHostToDevice                       hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost                       hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice                     hipMemcpyDeviceToDevice
#define cudaMemcpyDefault                            hipMemcpyDefault
#define cudaFuncAttributeMaxDynamicSharedMemorySize  hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaStreamLegacy                             hipStreamLegacy
#define cudaStreamCaptureStatusNone                  hipStreamCaptureStatusNone
#define cudaStreamCaptureModeRelaxed                 hipStreamCaptureModeRelaxed
#define cudaGraphExecUpdateSuccess                   hipGraphExecUpdateSuccess
#define cudaAddressModeClamp                         hipAddressModeClamp
#define cudaFilterModeLinear                         hipFilterModeLinear
#define cudaChannelFormatKindFloat                   hipChannelFormatKindFloat
#define cudaResourceTypePitch2D                      hipResourceTypePitch2D
#define cudaReadModeElementType                      hipReadModeElementType

#define cudaCreateChannelDesc                        hipCreateChannelDesc
#define cudaCreateTextureObject                      hipCreateTextureObject
#define cudaDestroyTextureObject                     hipDestroyTextureObject
#define cudaDeviceSynchronize                        hipDeviceSynchronize
#define cudaEventCreate                              hipEventCreate
#define cudaEventDestroy                             hipEventDestroy
#define cudaEventRecord                              hipEventRecord
#define cudaEventSynchronize                         hipEventSynchronize
#define cudaEventElapsedTime                         hipEventElapsedTime
#define cudaFree                                     hipFree
#define cudaFuncSetAttribute                         hipFuncSetAttribute
#define cudaGetDevice                                hipGetDevice
#define cudaGetDeviceCount                           hipGetDeviceCount
#define cudaGetDeviceProperties                      hipGetDeviceProperties
#define cudaGetErrorString                           hipGetErrorString
#define cudaGetLastError                             hipGetLastError
#define cudaGraphDestroy                             hipGraphDestroy
#define cudaGraphExecDestroy                         hipGraphExecDestroy
#define cudaGraphExecUpdate                          hipGraphExecUpdate
#define cudaGraphInstantiate                         hipGraphInstantiate
#define cudaGraphLaunch                              hipGraphLaunch
#define cudaMalloc                                   hipMalloc
#define cudaMallocManaged                            hipMallocManaged
#define cudaMemGetInfo                               hipMemGetInfo
#define cudaMemcpy                                   hipMemcpy
#define cudaMemcpyAsync                              hipMemcpyAsync
#define cudaMemset                                   hipMemset
#define cudaMemsetAsync                              hipMemsetAsync
#define cudaRuntimeGetVersion                        hipRuntimeGetVersion
#define cudaSetDevice                                hipSetDevice
#define cudaStreamBeginCapture                       hipStreamBeginCapture
#define cudaStreamCreate                             hipStreamCreate
#define cudaStreamDestroy                            hipStreamDestroy
#define cudaStreamEndCapture                         hipStreamEndCapture
#define cudaStreamIsCapturing                        hipStreamIsCapturing
#define cudaStreamSynchronize                        hipStreamSynchronize
#define cudaStreamWaitEvent                          hipStreamWaitEvent

// CUDA's runtime offers a templated cudaFuncSetAttribute(kernel, ...) overload; HIP only offers the
// `const void*` form. Provide the templated overload so kernels can be passed directly.
template <typename T>
inline hipError_t hipFuncSetAttribute(T* func, hipFuncAttribute attr, int value) {
	return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

// ---------------------------------------------------------------------------------------------
// CUDA driver API (virtual memory management) -> HIP VMM API.
// CUdeviceptr is an integer on CUDA while HIP's VMM functions take void*, so the address-range
// functions are thin inline wrappers instead of plain aliases (tiny-cuda-nn does integer
// arithmetic on the reserved base address).
// ---------------------------------------------------------------------------------------------
using CUresult = hipError_t;
using CUdeviceptr = uintptr_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;

#define CUDA_SUCCESS                                              hipSuccess
#define CUDA_ERROR_NOT_SUPPORTED                                  hipErrorNotSupported
#define CU_MEM_ALLOCATION_TYPE_PINNED                             hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE                               hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE                        hipMemAccessFlagsProtReadWrite
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM                          hipMemAllocationGranularityMinimum
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED  hipDeviceAttributeVirtualMemoryManagementSupported

// tiny-cuda-nn only uses cuGetErrorName for diagnostics and ignores its status.
inline void cuGetErrorName(CUresult error, const char** str) { (void)hipDrvGetErrorName(error, str); }
inline CUresult cuDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int device) { return hipDeviceGetAttribute(value, attr, device); }
inline CUresult cuMemGetAllocationGranularity(size_t* granularity, const CUmemAllocationProp* prop, hipMemAllocationGranularity_flags option) {
	return hipMemGetAllocationGranularity(granularity, prop, option);
}
inline CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, size_t size, const CUmemAllocationProp* prop, unsigned long long flags) {
	return hipMemCreate(handle, size, prop, flags);
}
inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) { return hipMemRelease(handle); }
inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
	void* result = nullptr;
	CUresult status = hipMemAddressReserve(&result, size, alignment, (void*)addr, flags);
	*ptr = (CUdeviceptr)result;
	return status;
}
inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) { return hipMemAddressFree((void*)ptr, size); }
inline CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
	return hipMemMap((void*)ptr, size, offset, handle, flags);
}
inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) { return hipMemUnmap((void*)ptr, size); }
inline CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count) {
	return hipMemSetAccess((void*)ptr, size, desc, count);
}

// Driver-API module types referenced by the runtime-compilation (RTC) interface in rtc_kernel.h.
// RTC itself is not supported on HIP (TCNN_RTC is never defined), but the interface must compile.
using CUmodule = hipModule_t;
using CUfunction = hipFunction_t;
using CUfunction_attribute = hipFunction_attribute;
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
inline CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int shmem, hipStream_t stream, void** params, void** extra) {
	return hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shmem, stream, params, extra);
}

#endif // TCNN_USE_HIP
