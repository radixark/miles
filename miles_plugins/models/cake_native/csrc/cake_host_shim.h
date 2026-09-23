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
// argument checks that raise Python exceptions, the current CUDA stream and the module
// export, implemented on ATen / pybind11 so the launchers build as a torch CUDA extension.
#pragma once

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/dlpack.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

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

}  // namespace cake_host

#define CAKE_HOST_CHECK(cond, Kind) \
  if (cond) {                       \
  } else                            \
    ::cake_host::CheckFailure(#Kind, __FILE__, __LINE__)

#define CAKE_HOST_EXPORT(name, fn) \
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(#name, ::cake_host::Wrap(fn)); }
