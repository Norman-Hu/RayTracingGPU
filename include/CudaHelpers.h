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

#endif // CUDA_HELPERS_H
