#include <Light.cuh>
#include <Rendering.cuh>

__device__ LightSamples PointLight::getSamples(curandState_t * state)
{
	LightSamples samples;
	samples.samples[0] = p;
	samples.size = 1;
	return samples;
}

PointLight * createPointLight(const Vec3 & p, const Vec3 & color)
{
	PointLight * res;
	PointLight ** ptr_d_light;
	cudaMalloc(&ptr_d_light, sizeof(PointLight *));
	d_createPointLight<<<1, 1>>>(ptr_d_light, p, color);
	cudaMemcpy(&res, ptr_d_light, sizeof(PointLight *), cudaMemcpyDeviceToHost);
	cudaFree(ptr_d_light);
	return res;
}

__global__ void d_createPointLight(PointLight ** ptr_d_light, Vec3 p, Vec3 color)
{
	PointLight * light = new PointLight();
	light->p = p;
	light->color = color;
	*ptr_d_light = light;
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