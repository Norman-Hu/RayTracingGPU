#include <Rendering.cuh>
#include <surface_indirect_functions.h>
#include <cstdio>


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface, curandState_t * randState)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<w && y<h)
	{
		float ndc_x = (2.f*(float)x / (float)w)-1.f;
		float ndc_y = 1.f - (2.f*(float)y / (float)h);
		Vec4 vec{ndc_x, ndc_y, camNear, 1.f};
		vec = rayTransform*vec;
		Vec3 dir = Vec3{vec[0], vec[1], vec[2]};
		dir.normalize();

		Ray ray{camPos, dir.normalized()};

		uchar4 val;
		Vec3 colorVal = Vec3(0.0f, 0.0f, 0.0f);
		int lightsCount = scene->lightCount;

		Vec3 color(0.0f, 0.0f, 0.0f);
		constexpr int maxBounces = 10;

		for (int i=0; i<maxBounces; ++i)
		{
			Hit hitInfo;
			bool hit = scene->hit(ray, 0.001f, 50.0f, hitInfo);
			if (hit)
			{
				BlinnPhongMaterial & mat = scene->materials[hitInfo.materialId];
				if (mat.mirror > 0.f)
					ray = {hitInfo.p, Vec3::reflect(ray.direction, hitInfo.normal)};
				else
				if (mat.refraction)
				{
					Vec3 incident = ray.direction;
					Vec3 normal;
					float ratio;
					if (Vec3::dot(ray.direction, hitInfo.normal) > 0.0f)
					{
						normal = -1.f * hitInfo.normal;
						ratio = mat.refractiveIndex;
					}
					else
					{
						ratio = 1.f / mat.refractiveIndex;
						normal = hitInfo.normal;
					}
					ray = {hitInfo.p, Vec3::refract(incident, normal, ratio).normalized()};
				}
				else
				{
					int lightHits = 0;
					for (int l=0; l<lightsCount; l++)
					{
						Light * light = scene->lights[l];
						LightSamples samples = light->getSamples(randState);

						int sampleHits = 0;

						Vec3 colorForLight(0.0f, 0.0f, 0.0f);
						for (int lightSample=0; lightSample<samples.size; ++lightSample)
						{
							Vec3 lightPos = samples.samples[lightSample];

							Vec3 lightDir = (lightPos - hitInfo.p);
							float lightDistance = lightDir.length();
							lightDir.normalize();

							// check if obstructed
							Hit _unused;
							if (scene->hit({hitInfo.p, lightDir}, 0.001f, lightDistance, _unused))
								continue;

							++sampleHits;

							float diff = max(Vec3::dot(hitInfo.normal, lightDir), 0.0f);
							Vec3 diffuse = mat.diffuse * diff;

							Vec3 viewDir = (-ray.direction).normalized();
							Vec3 halfDir = (lightDir + viewDir).normalized();
							float spec = powf(max(Vec3::dot(hitInfo.normal, halfDir), 0.0f), mat.shininess);
							Vec3 specular = spec * mat.specular;

							Vec3 res = mat.ambient + diffuse + specular;
							colorForLight += res.mulComp(light->color);
						}
						if (sampleHits > 0)
						{
							colorForLight/=sampleHits;
							++lightHits;
							color += colorForLight;
						}
					}
					if (lightHits > 0)
						color /= lightHits;
					break;
				}
			}
			else break;
		}

		float maxVal = max(color.x, max(color.y, color.z));
		if (maxVal > 1.0f)
			color /= maxVal;
		color *= 255.0f;
		val.x = color.x;
		val.y = color.y;
		val.z = color.z;
		val.w = 255;
		// the error on the next line is a lie
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
}

__global__ void setupRandomState(curandState_t * state, uint64_t seed)
{
	int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__device__ float randUniform(curandState_t * state)
{
	int tid = gridDim.x*blockDim.x*(blockDim.y*blockIdx.y+threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	return curand_uniform(&state[tid]);
}

__device__ Vec3 randomInSphere(curandState_t * state)
{
	constexpr float pi = 3.14159265358979323846f;
	float pitch = randUniform(state)*pi*2.f;
	float yaw = randUniform(state)*pi*2.f;
	Vec3 res;
	res.x = cosf(yaw) * cosf(pitch);
	res.y = sinf(pitch);
	res.z = sinf(yaw) * cosf(pitch);
	res.normalize();
	float distance = sqrtf(randUniform(state));
	return res*distance;
}

__global__ void testFillFramebuffer(unsigned int w, unsigned int h, cudaSurfaceObject_t surface)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<w && y<h)
	{
		uchar4 val = {255, 255, 0, 255};
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
}

__global__ void renderStraight(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface)
{
	float x = w/2.0f;
	float y = h/2.0f;
	float ndc_x = (2.f*(float)x / (float)w)-1.f;
	float ndc_y = 1.f - (2.f*(float)y / (float)h);
	Vec4 vec{ndc_x, ndc_y, camNear, 1.f};
	vec = rayTransform*vec;
	Vec3 dir = Vec3{vec[0], vec[1], vec[2]};
	dir.normalize();

	Ray ray{camPos, dir.normalized()};

	Hit out;
	uchar4 val;
	if (scene->hit(ray, 0.1f, 50.0f, out))
	{
		val = {255, 0, 0, 255};
	}
}