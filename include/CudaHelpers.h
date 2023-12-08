#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <iostream>


inline void syncAndCheckErrors()
{
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		std::cout << "kernel launch failed with error \"" << cudaGetErrorString(cudaerr) << "\"."
			<< std::endl;
}

#define errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
		if (abort)
			exit(code);
	}
}

#endif // CUDA_HELPERS_H
