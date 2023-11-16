#include <Scene.cuh>
#include <CudaHelpers.h>


__device__ Scene::Scene()
: objectList(nullptr)
, size(0)
{

}

__device__ Scene::Scene(Hitable ** _objectList, int _size)
: objectList(_objectList)
, size(_size)
{
}

__device__ Scene::~Scene()
{
	for (int i=0; i<size; ++i)
		delete objectList[i];
	delete [] objectList;
}

__device__ bool Scene::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	bool hasHit = false;
	float closest_t = tmax;

	Hit tmpHit;
	for (int i=0; i<size; ++i)
	{
		if (objectList[i]->hit(ray, tmin, tmax, tmpHit))
		{
			hasHit = true;
			if (tmpHit.t < closest_t)
			{
				closest_t = tmpHit.t;
				out = tmpHit;
			}
		}
	}
	return hasHit;
}

// initialization and destruction
Scene * createScene()
{
	Scene * d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	initScene<<<1, 1>>>(d_scene);
	syncAndCheckErrors();

	return d_scene;
}

void destroyScene(Scene * d_scene)
{
	deleteScene<<<1, 1>>>(d_scene);
	cudaFree(d_scene);
	syncAndCheckErrors();
}


// Kernels
__global__ void initScene(Scene * ptrScene)
{
	new (ptrScene) Scene(new Hitable*[2], 2);
	Sphere * pSphere = new Sphere();
	ptrScene->objectList[0] = pSphere;
	pSphere->c = {1.0f, 0.0f, -10.0f};
	pSphere->r = 1.0f;
	pSphere = new Sphere();
	ptrScene->objectList[1] = pSphere;
	pSphere->c = {-1.0f, 0.0f, -13.0f};
	pSphere->r = 0.5f;
}

__global__ void deleteScene(Scene * ptrScene)
{
	ptrScene->~Scene();
}
