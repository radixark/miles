/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Host surface for the generated Cake launchers (``*_binding.cu``): a read-only tensor view,
// argument checks that raise Python exceptions, the current CUDA stream, the launchers' common
// helpers and the stage registry behind the package's one extension module, implemented on
// ATen / pybind11 so the launchers build as a torch CUDA extension.
#pragma once

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/dlpack.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cake_host {

// Accessor names the generated launchers use, over an ATen tensor.
class TensorView {
 public:
  TensorView(at::Tensor tensor) : tensor_(std::move(tensor)) {}  // NOLINT(google-explicit-constructor)

  DLDevice device() const {
    DLDevice dev;
    dev.device_type = tensor_.is_cuda() ? kDLCUDA : kDLCPU;
    dev.device_id = tensor_.is_cuda() ? static_cast<int32_t>(tensor_.get_device()) : 0;
    return dev;
  }
  DLDataType dtype() const { return at::getDLDataType(tensor_); }
  int ndim() const { return static_cast<int>(tensor_.dim()); }
  int64_t size(int axis) const { return tensor_.size(axis); }
  int64_t stride(int axis) const { return tensor_.stride(axis); }
  int64_t numel() const { return tensor_.numel(); }
  bool IsContiguous() const { return tensor_.is_contiguous(); }
  void* data_ptr() const { return tensor_.data_ptr(); }

 private:
  at::Tensor tensor_;
};

template <typename T>
using Optional = std::optional<T>;

// Collects a failed check's message and raises it as the named Python exception kind
// (``ValueError`` / ``TypeError`` / ``RuntimeError``) when the statement ends.
class CheckFailure {
 public:
  CheckFailure(const char* kind, const char* file, int line) : kind_(kind), file_(file), line_(line) {}
  CheckFailure(const CheckFailure&) = delete;
  CheckFailure& operator=(const CheckFailure&) = delete;

  template <typename T>
  CheckFailure& operator<<(const T& value) {
    message_ << value;
    return *this;
  }

  ~CheckFailure() noexcept(false) {
    const c10::SourceLocation where{"cake_host", file_, static_cast<uint32_t>(line_)};
    const std::string kind(kind_);
    if (kind == "ValueError") throw c10::ValueError(where, message_.str());
    if (kind == "TypeError") throw c10::TypeError(where, message_.str());
    throw c10::Error(where, message_.str());
  }

 private:
  const char* kind_;
  const char* file_;
  int line_;
  std::ostringstream message_;
};

// The stream torch has made current on ``device_id``; the launchers enqueue on it.
inline void* CurrentStream(int /*device_type*/, int device_id) {
  return static_cast<void*>(c10::cuda::getCurrentCUDAStream(device_id).stream());
}

// ``Run(TensorView..., int64_t..., double...)`` is exposed with ``at::Tensor`` in place of
// ``TensorView``; scalars pass through unchanged.
template <typename T>
struct PyArg {
  using type = T;
};
template <>
struct PyArg<TensorView> {
  using type = at::Tensor;
};

template <typename R, typename... Args>
auto Wrap(R (*fn)(Args...)) {
  return [fn](typename PyArg<Args>::type... args) -> R { return fn(Args(std::move(args))...); };
}

// One package builds into one extension module: every launcher translation unit registers its
// stage here at load time and ``CAKE_HOST_MODULE()`` (``cake_module.cu``) exposes them all.
using StageRegistrar = void (*)(pybind11::module_&);
struct StageEntry {
  const char* name;
  StageRegistrar registrar;
};
inline std::vector<StageEntry>& Stages() {
  static std::vector<StageEntry> stages;
  return stages;
}
struct StageRegistration {
  StageRegistration(const char* name, StageRegistrar registrar) { Stages().push_back({name, registrar}); }
};

}  // namespace cake_host

#define CAKE_HOST_CHECK(cond, Kind) \
  if (cond) {                       \
  } else                            \
    ::cake_host::CheckFailure(#Kind, __FILE__, __LINE__)

// The launchers' common helpers (generated once by the compiler, shipped once here).
namespace cake_host {


class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    CAKE_HOST_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      CAKE_HOST_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

inline int64_t CakeDeviceMultiprocessorCount(int device_id) {
  constexpr int kMaxCachedCudaDevices = 64;
  CAKE_HOST_CHECK(device_id >= 0 && device_id < kMaxCachedCudaDevices, RuntimeError)
      << "physical-SM-count cache does not cover cuda:" << device_id;
  static std::atomic<int> count_by_device[kMaxCachedCudaDevices]{};
  int cached = count_by_device[device_id].load(std::memory_order_acquire);
  if (cached > 0) return cached;

  int count = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device_id);
  CAKE_HOST_CHECK(error == cudaSuccess && count > 0, RuntimeError)
      << "querying multiProcessorCount failed for cuda:" << device_id
      << ": cudaError=" << static_cast<int>(error) << ", count=" << count;
  // Concurrent first launches may repeat the immutable device query, but all
  // publication is atomic and every later hot-path lookup is one acquire load.
  count_by_device[device_id].store(count, std::memory_order_release);
  return count;
}

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  CAKE_HOST_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  CAKE_HOST_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckCurrentCudaDevice(
    const TensorView& reference,
    const char* reference_name) {
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  CAKE_HOST_CHECK(error == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed while validating " << reference_name
      << ": cudaError=" << static_cast<int>(error);
  CAKE_HOST_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match " << reference_name
      << ": current=cuda:" << current_device
      << ", tensor=cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  CAKE_HOST_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  CAKE_HOST_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  CAKE_HOST_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  CAKE_HOST_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      CAKE_HOST_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

}  // namespace cake_host

#define CAKE_HOST_STAGE(name, fn)                                            \
  static const ::cake_host::StageRegistration cake_host_stage_##name{        \
      #name, [](pybind11::module_& m) { m.def(#name, ::cake_host::Wrap(fn)); }}

#define CAKE_HOST_MODULE()                                                   \
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {                                 \
    for (const auto& stage : ::cake_host::Stages()) stage.registrar(m);      \
  }
