/*
 * Minimal CUDA kernel that performs an illegal memory access (IMA),
 * causing the NVIDIA driver to log XID 13 (Graphics Engine Exception)
 * to the kernel ring buffer (/dev/kmsg / dmesg).
 *
 * Used by FaultInjectorActor.trigger_gpu_xid() for E2E testing of the
 * KmsgCollector -> metric -> detector pipeline.
 *
 * Build:  nvcc -o /tmp/trigger_xid trigger_xid.cu
 * Run:    /tmp/trigger_xid          (exits non-zero after the IMA)
 */

#include <cstdio>
#include <cstdlib>

__global__ void illegal_access_kernel() {
    *(volatile int*)0xDEADBEEF = 42;
}

int main() {
    illegal_access_kernel<<<1, 1>>>();
    cudaError_t sync_err = cudaDeviceSynchronize();
    cudaError_t last_err = cudaGetLastError();

    if (sync_err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(sync_err));
    }
    if (last_err != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError: %s\n",
                cudaGetErrorString(last_err));
    }

    cudaDeviceReset();
    return (sync_err != cudaSuccess || last_err != cudaSuccess) ? 1 : 0;
}
