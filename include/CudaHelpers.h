#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <iostream>
#include <format>


inline void syncAndCheckErrors()
{
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
		std::cout << std::format("kernel launch failed with error \"{}\".",cudaGetErrorString(cudaerr))
			<< std::endl;
}

#endif // CUDA_HELPERS_H
