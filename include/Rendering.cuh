#ifndef RENDERING_H
#define RENDERING_H

#include <Vec3.cuh>
#include <Matrix.cuh>
#include <Scene.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState);

// random state
__global__ void setupRandomState(curandState_t * state, uint64_t seed);
__device__ float randUniform(curandState_t * state);

// random helpers
__device__ Vec3 randomInSphere(curandState_t * state);

// debug
__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface);
__global__ void renderStraight(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState);

#endif // RENDERING_H
