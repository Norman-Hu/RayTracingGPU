#ifndef RENDERING_H
#define RENDERING_H

#include <Vec3.cuh>
#include <Matrix.cuh>
#include <Scene.cuh>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>

__host__ __device__ float distributionGGX(const Vec3 & N, const Vec3 & H, float roughness);
__host__ __device__ float geometrySchlickGGX(float NdotV, float roughness);
__host__ __device__ float geometrySmith(const Vec3 & N, const Vec3 & V, const Vec3 & L, float roughness);
__host__ __device__ Vec3 fresnelSchlick(float cosTheta, const Vec3 & F0);
__host__ __device__ Vec3 sampleUniformHemisphere(float u, float v);
__host__ __device__ Vec3 sampleHemisphereAroundNormal(float u, float v, const Vec3 & normal);
__host__ __device__ Vec3 sampleUniformSphere(float u, float v);
__device__ float schlick(float cosine, float ref_idx);

__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState, bool sampleLights, int sampleCount);
__global__ void renderSimple(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState, bool sampleLights, int sampleCount);

// random state
__global__ void setupRandomState(curandState_t * state, uint64_t seed);
__device__ float randUniform(curandState_t * state);

// random helpers
__device__ Vec3 randomInSphere(curandState_t * state);

// debug
__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface);
__global__ void renderStraight(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface);

#endif // RENDERING_H
