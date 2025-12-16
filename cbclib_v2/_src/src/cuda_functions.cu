#include <cuda_runtime.h>
#include "geometry.hpp"

namespace cbclib::cuda {

void handle_cuda_error(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		throw std::runtime_error(cudaGetErrorString(error));
	}
}

// template <typename T>
// __global__ void add(int size, T * a, T * b)
// {
// 	int index = blockIdx.x * blockDim.x + threadIdx.x;
// 	int stride = blockDim.x * gridDim.x;
// 	for (int i = index; i < size; i += stride) b[i] = a[i] + b[i];
// }

// template <typename T>
// void run_kernel(int size, T * a, T * b)
// {
// 	int block_size = 256;
// 	int num_blocks = (size + block_size - 1) / block_size;

// 	add<T><<<num_blocks, block_size>>>(size, a, b);

// 	handle_cuda_error(cudaGetLastError());
// }

// template <typename T>
// py::array_t<T> sum(py::array_t<T, py::array::c_style | py::array::forcecast> a, py::array_t<T, py::array::c_style | py::array::forcecast> b)
// {
// 	py::buffer_info buf_a = a.request();
// 	py::buffer_info buf_b = b.request();

// 	if (buf_a.size != buf_b.size)
// 	{
// 		throw std::runtime_error("Input arrays must have the same size");
// 	}

// 	T * a_ptr, * b_ptr;
// 	handle_cuda_error(cudaMalloc(&a_ptr, buf_a.size * sizeof(T)));
// 	handle_cuda_error(cudaMalloc(&b_ptr, buf_b.size * sizeof(T)));

// 	handle_cuda_error(cudaMemcpy(a_ptr, a.data(), buf_a.size * sizeof(T), cudaMemcpyHostToDevice));
// 	handle_cuda_error(cudaMemcpy(b_ptr, b.data(), buf_b.size * sizeof(T), cudaMemcpyHostToDevice));

// 	run_kernel<T>(buf_a.size, a_ptr, b_ptr);

// 	py::array_t<T> out {buf_a};

// 	handle_cuda_error(cudaMemcpy(out.mutable_data(), b_ptr, buf_b.size * sizeof(T), cudaMemcpyDeviceToHost));

// 	handle_cuda_error(cudaFree(a_ptr));
// 	handle_cuda_error(cudaFree(b_ptr));

// 	return out;
// }

} // namespace cbclib

PYBIND11_MODULE(cuda_library, m)
{
	using namespace cbclib;

	try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

	// m.def("sum", cuda::sum<float>, py::arg("a"), py::arg("b"));
	// m.def("sum", cuda::sum<double>, py::arg("a"), py::arg("b"));
}
