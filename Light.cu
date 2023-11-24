#include <Light.cuh>
#include <Rendering.cuh>

__device__ LightSamples PointLight::getSamples(curandState_t * state)
{
	LightSamples samples;
	samples.samples[0] = p;
	samples.size = 1;
	return samples;
}

__device__ LightSamples AreaLight::getSamples(curandState_t * state)
{
	constexpr unsigned int SAMPLE_COUNT = 8;
	LightSamples samples;
	for (int i=0; i<SAMPLE_COUNT; ++i)
	{
		float u = randUniform(state);
		float v = randUniform(state);
		samples.samples[i] = p+u*right+v*up;
	}
	samples.size = SAMPLE_COUNT;
	return samples;
}