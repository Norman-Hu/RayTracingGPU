#include <Rendering.cuh>
#include <surface_indirect_functions.h>


__global__ void render(Scene * scene, unsigned int w, unsigned int h, float camNear, Vec3 camPos, Matrix4x4 invViewProj, cudaSurfaceObject_t surface)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x<w && y<h)
	{
		float u = (float)x / (float)w;
		float v = (float)y / (float)h;
		Vec4 vec{2.0f*u - 1.0f, -(2.f*v - 1.f), camNear, 1.0f};
		vec = vec * invViewProj;
		Vec3 dir = Vec3{vec[0], vec[1], vec[2]}/vec[3];

		Hit out;
		uchar4 val;
		if (scene->hit({camPos, dir}, 0.1f, 50.0f, out))
			val = {255, 0, 0, 255};
		else
			val = {0, 0, 0, 255};
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