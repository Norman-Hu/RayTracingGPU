#include <Rendering.cuh>
#include <surface_indirect_functions.h>


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

		uchar4 val;
		Vec3 colorVal = Vec3(0.0f, 0.0f, 0.0f);
		int lightsCount = scene->pointLightsCount;

		for (int l=0; l<scene->pointLightsCount; l++)
		{
			Vec3 pointLight = scene->pointLights[l];

			Vec3 color(1.0f, 1.0f, 1.0f);
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
					{
						Vec3 lightDir = (pointLight - hitInfo.p);
						lightDir.normalize();

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
			colorVal.x += color.x;
			colorVal.y += color.y;
			colorVal.z += color.z;
		}

		val.x = colorVal.x / lightsCount;
		val.y = colorVal.y / lightsCount;
		val.z = colorVal.z / lightsCount;
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