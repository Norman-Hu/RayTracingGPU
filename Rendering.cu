#include <Rendering.cuh>
#include <surface_indirect_functions.h>
#include <cstdio>


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 rayTransform, cudaSurfaceObject_t surface)
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

		Vec3 directionalLight{-0.5f, -0.5f, -0.5f};
		directionalLight.normalize();

		Vec3 color(1.0f, 1.0f, 1.0f);
		constexpr int maxBounces = 10;

		for (int i=0; i<maxBounces; ++i)
		{
			Hit hitInfo;
			bool hit = scene->hit(ray, 0.1f, 50.0f, hitInfo);
			if (hit)
			{
				BlinnPhongMaterial & mat = scene->materials[hitInfo.materialId];
				if (mat.mirror > 0.f)
					ray = {hitInfo.p, Vec3::reflect(ray.direction, hitInfo.normal)};
				else
				{
					Vec3 lightDir = -directionalLight;

					float diff = max(Vec3::dot(hitInfo.normal, lightDir), 0.0f);
					Vec3 diffuse = mat.diffuse*diff;

					Vec3 viewDir = (-ray.direction).normalized();
					Vec3 halfDir = (lightDir+viewDir).normalized();
					float spec = powf(max(Vec3::dot(hitInfo.normal, halfDir), 0.0f), mat.shininess);
					Vec3 specular = spec * mat.specular;

					Vec3 res = mat.ambient + diffuse + specular;
					float maxVal = max(res.x, max(res.y, res.z));
					if (maxVal > 1.0f)
						res /= maxVal;

					color[0] *= res[0];
					color[1] *= res[1];
					color[2] *= res[2];
					break;
				}
			}
			else break;
		}

		Hit out;
		color *= 255.0f;
		uchar4 val;
		val.x = color.x;
		val.y = color.y;
		val.z = color.z;
		val.w = 255;
		// the error on the next line is a lie
		surf2Dwrite<uchar4>(val, surface, (int)sizeof(uchar4)*x, y, cudaBoundaryModeClamp);
	}
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

__global__ void renderStraight(Scene * scene, float camNear, Vec3 camPos, Matrix4x4 invViewProj)
{
	float u = 0.5f;
	float v = 0.5f;
	Vec4 vec{2.0f*u - 1.0f, -(2.f*v - 1.f), camNear, 1.0f};
	vec = vec * invViewProj;
	Vec3 dir = Vec3{vec[0], vec[1], vec[2]}/vec[3];

	Hit out;
	uchar4 val;
	if (scene->hit({camPos, dir}, 0.1f, 50.0f, out))
	{
		val = {255, 0, 0, 255};
	}
	else
	{
		val = {0, 0, 0, 255};
	}
}