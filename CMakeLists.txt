# Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

option(TCNN_USE_HIP "Build tiny-cuda-nn for AMD GPUs with HIP/ROCm instead of CUDA?" OFF)

if (TCNN_USE_HIP)
	cmake_minimum_required(VERSION 3.21) # HIP language support
	if (NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT DEFINED ENV{HIPARCHS})
		if (GPU_TARGETS)
			# Honor the ROCm-wide GPU_TARGETS variable (also set by TheRock and other ROCm super-builds).
			set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}" CACHE STRING "AMD GPU architectures to build for")
		else()
			# AMD Instinct MI300 series (gfx942) and MI350 series (gfx950)
			set(CMAKE_HIP_ARCHITECTURES "gfx942;gfx950" CACHE STRING "AMD GPU architectures to build for")
		endif()
	endif()
	set(TCNN_LANGUAGES CXX HIP)
else()
	set(TCNN_LANGUAGES CXX CUDA)
endif()

project(
	tiny-cuda-nn
	VERSION 2.0
	DESCRIPTION "Lightning fast & tiny C++/CUDA neural network framework"
	LANGUAGES ${TCNN_LANGUAGES}
)

option(TCNN_ALLOW_CUBLAS_CUSOLVER "Allows tiny-cuda-nn to use cuBLAS and cuSolver. Only required for the Shampoo optimizer." OFF)
option(TCNN_BUILD_BENCHMARK "Build tiny-cuda-nn example benchmark?" ON)
option(TCNN_BUILD_EXAMPLES "Build tiny-cuda-nn example applications?" ON)
option(TCNN_BUILD_NO_FWD_BWD "Build without offline compiled forward and backward kernels?" OFF)
option(TCNN_BUILD_TESTS "Build tiny-cuda-nn's tests?" OFF)
option(TCNN_BUILD_WITH_RTC "Build support for runtime compilation of fully fused kernels?" ON)
option(TCNN_BUILD_USE_FAST_MATH "Build tiny-cuda-nn with '--use_fast_math' option?" ON)

set(TCNN_EXTERNAL_FMT "" CACHE STRING "If non-empty, the `fmt` target is supplied externally with the given name.")

set(TCNN_CUDA_ARCHITECTURES "" CACHE STRING "Build tiny-cuda-nn for a specific GPU architecture.")
option(TCNN_LINK_CUDA "Link tiny-cuda-nn to CUDA libraries?" ON)
option(TCNN_INSTALL "Generate install rules and a CMake package config for the tiny-cuda-nn library?" OFF)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cutlass/CMakeLists.txt")
	message(FATAL_ERROR
		"Some tiny-cuda-nn dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    set(CMAKE_CUDA_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# HIP compiler setup (AMD GPUs)
###############################################################################

if (TCNN_USE_HIP)
	get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)

	set(CMAKE_HIP_STANDARD 17)
	set(CMAKE_HIP_STANDARD_REQUIRED ON)
	set(CMAKE_HIP_EXTENSIONS OFF)

	if (TCNN_BUILD_WITH_RTC)
		message(STATUS "Runtime compilation (JIT fusion) of fully fused kernels is not supported on HIP. Disabling TCNN_BUILD_WITH_RTC.")
		set(TCNN_BUILD_WITH_RTC OFF CACHE BOOL "Build support for runtime compilation of fully fused kernels?" FORCE)
	endif()
	if (TCNN_ALLOW_CUBLAS_CUSOLVER)
		message(WARNING "TCNN_ALLOW_CUBLAS_CUSOLVER (Shampoo optimizer) is not supported on HIP and is ignored.")
	endif()

	find_package(hip REQUIRED)
	find_package(hipblas REQUIRED)
	find_package(hipblaslt REQUIRED)
	# rocWMMA is header-only. Use the include directory exported by its package config rather than
	# the roc::rocwmma target, whose link interface also pulls in hip::device and OpenMP.
	find_package(rocwmma CONFIG REQUIRED)
	set(TCNN_ROCWMMA_INCLUDE_DIR "${rocwmma_INCLUDE_DIR}")

	# AMD Instinct GPUs provide every feature that tiny-cuda-nn gates on NVIDIA compute capability
	# (fp16 arithmetic and atomics, matrix cores for the fully fused MLP), so the HIP build uses the
	# feature level of compute capability 8.0.
	set(MIN_GPU_ARCH 80)

	option(TCNN_HALF_PRECISION "Enable half precision (FP16) arithmetic" ON)
	if (TCNN_HALF_PRECISION)
		list(APPEND TCNN_DEFINITIONS -DTCNN_HALF_PRECISION=1)
	else()
		list(APPEND TCNN_DEFINITIONS -DTCNN_HALF_PRECISION=0)
	endif()
	message(STATUS "TCNN_HALF_PRECISION: ${TCNN_HALF_PRECISION}")
	message(STATUS "Targeting HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

	list(APPEND TCNN_DEFINITIONS -DTCNN_USE_HIP)
	list(APPEND TCNN_INCLUDES "include" "${TCNN_ROCWMMA_INCLUDE_DIR}")
	list(APPEND TCNN_LIBRARIES hip::host roc::hipblas roc::hipblaslt)

	if (TCNN_BUILD_USE_FAST_MATH)
		# Closest equivalent of nvcc's --use_fast_math. Deliberately not -ffast-math, whose
		# finite-math-only assumption would break tiny-cuda-nn's isfinite() guards.
		list(APPEND TCNN_HIP_FLAGS -fgpu-flush-denormals-to-zero -fgpu-approx-transcendentals)
	endif()
	# Use hardware floating-point atomics (relaxed IEEE semantics), as nvcc does by default. This is
	# applied to tiny-cuda-nn's own sources only (PRIVATE below): it changes the semantics of
	# atomicAdd(float*) on fine-grained memory, so it must not leak into consumers' code. The
	# gradient-accumulation helpers in vec.h call unsafeAtomicAdd explicitly, so header code that
	# consumers instantiate gets the hardware atomics without the flag.
	set(TCNN_HIP_PRIVATE_FLAGS -munsafe-fp-atomics)
	# `#pragma unroll` on loops with runtime trip counts is a hint that clang reports when it cannot honor it.
	list(APPEND TCNN_HIP_FLAGS -Wno-pass-failed)

	if (TCNN_HAS_PARENT)
		set(TCNN_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES} PARENT_SCOPE)
		set(TCNN_HALF_PRECISION ${TCNN_HALF_PRECISION} PARENT_SCOPE)
		set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
	endif()
else()

###############################################################################
# CUDA compiler setup
###############################################################################

# Figure out CUDA version
if(CMAKE_CUDA_COMPILER_LOADED)
	if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
		set(CUDA_VERSION "${CMAKE_MATCH_1}")
	endif()
endif()

# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(TCNN_AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
	if (NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		if (CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
			set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cu")
		else()
			set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cpp")
		endif()

		file(WRITE ${file} ""
			"#include <cuda_runtime.h>\n"
			"#include <cstdio>\n"
			"int main() {\n"
			"	int count = 0;\n"
			"	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
			"	if (count == 0) return -1;\n"
			"	for (int device = 0; device < count; ++device) {\n"
			"		cudaDeviceProp prop;\n"
			"		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
			"			std::printf(\"%d%d\", prop.major, prop.minor);\n"
			"			if (device < count - 1) std::printf(\";\");\n"
			"		}\n"
			"	}\n"
			"	return 0;\n"
			"}\n"
		)

		try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
		if (run_result EQUAL 0)
			# If the user has multiple GPUs with the same compute capability installed, list that capability only once.
			list(REMOVE_DUPLICATES compute_capabilities)
			set(TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
		endif()
	endif()

	if (NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
		message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
		set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
	else()
		set(${OUT_VARIABLE} ${TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
	endif()
endfunction()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)
if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained CUDA architectures from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
elseif (TCNN_CUDA_ARCHITECTURES)
	message(STATUS "Obtained CUDA architectures from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
else()
	message(STATUS "Obtained CUDA architectures automatically from installed GPUs")
	TCNN_AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
endif()

# If the CUDA version does not support the chosen architecture, target
# the latest supported one instead.
if (CUDA_VERSION VERSION_LESS 11.0)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 75)
elseif (CUDA_VERSION VERSION_LESS 11.1)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 80)
elseif (CUDA_VERSION VERSION_LESS 11.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 86)
elseif (CUDA_VERSION VERSION_LESS 12.8)
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)
else()
	set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 120)
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 75)
elseif (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 50)
else()
	set(EARLIEST_SUPPORTED_CUDA_ARCHITECTURE 20)
endif()

foreach (CUDA_CC IN LISTS CMAKE_CUDA_ARCHITECTURES)
	if (CUDA_CC GREATER ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(WARNING "CUDA version ${CUDA_VERSION} is too low for detected architecture ${CUDA_CC}. Targeting the highest supported architecture ${LATEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()

	if (CUDA_CC LESS ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		message(ERROR "CUDA version ${CUDA_VERSION} no longer supports detected architecture ${CUDA_CC}. Targeting the lowest supported architecture ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE} instead.")
		list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${CUDA_CC})
		if (NOT CMAKE_CUDA_ARCHITECTURES)
			list(APPEND CMAKE_CUDA_ARCHITECTURES ${EARLIEST_SUPPORTED_CUDA_ARCHITECTURE})
		endif()
	endif()
endforeach(CUDA_CC)

if (NOT CMAKE_CUDA_ARCHITECTURES)
	list(APPEND CMAKE_CUDA_ARCHITECTURES ${LATEST_SUPPORTED_CUDA_ARCHITECTURE})
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

if (MIN_GPU_ARCH EQUAL 61 OR MIN_GPU_ARCH LESS_EQUAL 52)
    set(TCNN_HALF_PRECISION_DEFAULT OFF)
else()
    set(TCNN_HALF_PRECISION_DEFAULT ON)
endif()

option(TCNN_HALF_PRECISION "Enable half precision (FP16) arithmetic" ${TCNN_HALF_PRECISION_DEFAULT})

if (TCNN_HALF_PRECISION)
    list(APPEND TCNN_DEFINITIONS -DTCNN_HALF_PRECISION=1)
    message(STATUS "TCNN_HALF_PRECISION: ON")
else()
    list(APPEND TCNN_DEFINITIONS -DTCNN_HALF_PRECISION=0)
    message(STATUS "TCNN_HALF_PRECISION: OFF")
endif()

message(STATUS "Targeting CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (TCNN_HAS_PARENT)
	set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(TCNN_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
	set(TCNN_HALF_PRECISION ${TCNN_HALF_PRECISION} PARENT_SCOPE)
endif()

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND TCNN_INCLUDES "include")

if (TCNN_HAS_PARENT)
	set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
endif()

endif() # TCNN_USE_HIP

# Only compile the shampoo optimizer if
# a recent enough cuBLAS version is available.
if (TCNN_ALLOW_CUBLAS_CUSOLVER AND NOT TCNN_USE_HIP AND CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	set(TCNN_BUILD_WITH_SHAMPOO ON)
else()
	set(TCNN_BUILD_WITH_SHAMPOO OFF)
endif()

if (TCNN_BUILD_WITH_SHAMPOO)
	list(APPEND TCNN_DEFINITIONS -DTCNN_SHAMPOO)
endif()

if (TCNN_BUILD_WITH_RTC)
	list(APPEND TCNN_DEFINITIONS -DTCNN_RTC)
endif()

if (TCNN_BUILD_USE_FAST_MATH)
	list(APPEND TCNN_DEFINITIONS -DTCNN_RTC_USE_FAST_MATH)
endif()

if (TCNN_BUILD_NO_FWD_BWD)
	list(APPEND TCNN_DEFINITIONS -DTCNN_NO_FWD_BWD)
endif()

if (TCNN_LINK_CUDA AND NOT TCNN_USE_HIP)
	list(APPEND TCNN_LIBRARIES cuda)
	if (TCNN_BUILD_WITH_SHAMPOO)
		list(APPEND TCNN_LIBRARIES cublas)
	endif()
	if (TCNN_BUILD_WITH_RTC)
		list(APPEND TCNN_LIBRARIES nvrtc)
	endif()
endif()

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/Zc:preprocessor")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
if (TCNN_BUILD_USE_FAST_MATH)
	list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")


###############################################################################
# Dependencies
###############################################################################

if (NOT MSVC)
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()

set(BUILD_SHARED_LIBS OFF)

if (TCNN_EXTERNAL_FMT)
	if (TCNN_EXTERNAL_FMT MATCHES "::" AND NOT TARGET "${TCNN_EXTERNAL_FMT}")
		# An imported fmt target (e.g. fmt::fmt) that the parent project has not created yet: use the installed fmt package.
		find_package(fmt CONFIG REQUIRED)
	endif()
	list(APPEND TCNN_LIBRARIES "${TCNN_EXTERNAL_FMT}")
	list(APPEND TCNN_DEFINITIONS -DFMT_UNICODE=0)
else()
	set(FMT_UNICODE OFF CACHE BOOL " " FORCE)
	add_subdirectory("dependencies/fmt")
	list(APPEND TCNN_LIBRARIES fmt)
	list(APPEND TCNN_INCLUDES "dependencies/fmt/include")
endif()

###############################################################################
# tiny-cuda-nn library, samples, and benchmarks
###############################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

set(TCNN_SOURCES
	src/common_host.cu
	src/cpp_api.cu
	src/cutlass_mlp.cu
	src/encoding.cu
	src/loss.cu
	src/network.cu
	src/object.cu
	src/optimizer.cu
	src/reduce_sum.cu
	src/rtc_kernel.cu
)

if (MIN_GPU_ARCH GREATER 70)
	list(APPEND TCNN_SOURCES src/fully_fused_mlp.cu)
endif()

list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})

if (TCNN_USE_HIP)
	set_source_files_properties(${TCNN_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

###############################################################################
# Linker / library
###############################################################################

include("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cmrc/CMakeRC.cmake")
cmrc_add_resource_library(tiny-cuda-nn-resources NAMESPACE tcnn)
list(APPEND TCNN_DEFINITIONS -DTCNN_CMRC)
list(APPEND TCNN_LIBRARIES tiny-cuda-nn-resources)

if (TCNN_BUILD_WITH_RTC)
	# Fetch CUDA headers and folders that will be required by the runtime compiler
	# and include those headers with the compiled binary of tcnn.
	foreach (CUDA_INCLUDE_CANDIDATE IN LISTS CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
		if (EXISTS "${CUDA_INCLUDE_CANDIDATE}/cuda_fp16.h")
			set(CUDA_INCLUDE "${CUDA_INCLUDE_CANDIDATE}")
			break()
		endif()
	endforeach(CUDA_INCLUDE_CANDIDATE)

	if (NOT CUDA_INCLUDE)
		# If the CUDA include dir couldn't be found via CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES,
		# try a relative path w.r.t. the CUDA compiler binary as a last-ditch effort.
		get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
		get_filename_component(CUDA_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)
		set(CUDA_INCLUDE "${CUDA_DIR}/include")
	endif()

	file(GLOB CUDA_HEADERS "${CUDA_INCLUDE}/cuda_fp16*" "${CUDA_INCLUDE}/vector*")
	if (NOT CUDA_HEADERS)
		message(WARNING "FP16 headers could not be found. JIT compilation will likely fail.")
	endif()

	file(GLOB_RECURSE TCNN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/tiny-cuda-nn/*")
	file(GLOB PCG32_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/pcg32/*")

	cmrc_add_resources(tiny-cuda-nn-resources WHENCE "${CUDA_INCLUDE}" ${CUDA_HEADERS})
	cmrc_add_resources(tiny-cuda-nn-resources WHENCE "${CMAKE_CURRENT_SOURCE_DIR}/include" ${TCNN_HEADERS})
	cmrc_add_resources(tiny-cuda-nn-resources WHENCE "${CMAKE_CURRENT_SOURCE_DIR}/dependencies" ${PCG32_HEADERS})
endif()

list(APPEND TCNN_INCLUDES
	"include"
	"dependencies"
	"dependencies/cutlass/include"
	"dependencies/cutlass/tools/util/include"
)

add_library(tiny-cuda-nn STATIC ${TCNN_SOURCES})
target_compile_definitions(tiny-cuda-nn PUBLIC ${TCNN_DEFINITIONS})
target_compile_options(tiny-cuda-nn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
if (TCNN_USE_HIP)
	target_compile_options(tiny-cuda-nn PUBLIC $<$<COMPILE_LANGUAGE:HIP>:${TCNN_HIP_FLAGS}>)
	target_compile_options(tiny-cuda-nn PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${TCNN_HIP_PRIVATE_FLAGS}>)
endif()
if (TCNN_INSTALL)
	include(GNUInstallDirs)
	# Source-tree include directories are build-only; installed consumers use the installed headers.
	foreach (TCNN_INCLUDE IN LISTS TCNN_INCLUDES)
		get_filename_component(TCNN_INCLUDE "${TCNN_INCLUDE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
		target_include_directories(tiny-cuda-nn PUBLIC $<BUILD_INTERFACE:${TCNN_INCLUDE}>)
	endforeach()
	target_include_directories(tiny-cuda-nn PUBLIC
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tiny-cuda-nn/dependencies>
	)
else()
	target_include_directories(tiny-cuda-nn PUBLIC ${TCNN_INCLUDES})
endif()
target_link_libraries(tiny-cuda-nn PUBLIC ${TCNN_LIBRARIES})

if (TCNN_BUILD_EXAMPLES)
	add_subdirectory("samples")
endif()

if (TCNN_BUILD_BENCHMARK)
	add_subdirectory("benchmarks/image")
	add_subdirectory("benchmarks/mlp")
endif()

if (TCNN_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
	list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
endif()

###############################################################################
# Installation
###############################################################################

if (TCNN_INSTALL)
	include(CMakePackageConfigHelpers)

	install(TARGETS tiny-cuda-nn tiny-cuda-nn-resources cmrc-base
		EXPORT tiny-cuda-nn-targets
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
	install(DIRECTORY include/tiny-cuda-nn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
	# Header-only dependencies included by the public headers, kept out of the top-level include directory.
	install(DIRECTORY dependencies/json dependencies/pcg32 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tiny-cuda-nn/dependencies)
	if (NOT TCNN_EXTERNAL_FMT)
		install(TARGETS fmt EXPORT tiny-cuda-nn-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
		install(DIRECTORY dependencies/fmt/include/fmt DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tiny-cuda-nn/dependencies)
	endif()

	set(TCNN_CONFIG_DEPENDENCIES "")
	if (TCNN_USE_HIP)
		string(APPEND TCNN_CONFIG_DEPENDENCIES "find_dependency(hip)\nfind_dependency(hipblas)\nfind_dependency(hipblaslt)\n")
	endif()
	if (TCNN_EXTERNAL_FMT MATCHES "^fmt::")
		string(APPEND TCNN_CONFIG_DEPENDENCIES "find_dependency(fmt)\n")
	endif()
	set(TCNN_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/tiny-cuda-nn)
	file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config.cmake.in"
		"@PACKAGE_INIT@\n"
		"include(CMakeFindDependencyMacro)\n"
		"${TCNN_CONFIG_DEPENDENCIES}"
		"include(\"\${CMAKE_CURRENT_LIST_DIR}/tiny-cuda-nn-targets.cmake\")\n"
		"check_required_components(tiny-cuda-nn)\n"
	)
	configure_package_config_file("${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config.cmake.in"
		"${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config.cmake"
		INSTALL_DESTINATION ${TCNN_CMAKE_DIR}
	)
	write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config-version.cmake"
		COMPATIBILITY SameMajorVersion
	)
	install(EXPORT tiny-cuda-nn-targets NAMESPACE tiny-cuda-nn:: DESTINATION ${TCNN_CMAKE_DIR})
	install(FILES
		"${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-config-version.cmake"
		DESTINATION ${TCNN_CMAKE_DIR}
	)
endif()
