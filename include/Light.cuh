#ifndef LIGHT_H
#define LIGHT_H


#include <Vec3.cuh>
#include <curand.h>
#include <curand_kernel.h>


class LightSamples
{
public:
	static constexpr unsigned int MAX_SAMPLES = 16;
	Vec3 samples[MAX_SAMPLES];
	unsigned int size;
};

class Light
{
public:
	__device__ virtual ~Light() {};
	__device__ virtual LightSamples getSamples(curandState_t * state) = 0;

	Vec3 color;
};

class PointLight : public Light
{
public:
	__device__ LightSamples getSamples(curandState_t * state) override;
	Vec3 p;
};

class AreaLight : public Light
{
public:
	__device__ LightSamples getSamples(curandState_t * state) override;
	Vec3 p;
	Vec3 n;
	Vec3 right;
	Vec3 up;
};

#endif // LIGHT_H
