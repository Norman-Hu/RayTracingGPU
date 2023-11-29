#include <Scene.cuh>
#include <CudaHelpers.h>


__device__ Scene::Scene()
: objectList(nullptr)
, objectCount(0)
, lights(nullptr)
, lightCount(0)
, materials(nullptr)
, materialCount(0)
{

}

__device__ Scene::Scene(Hitable ** _objectList, int _size, Light ** _lights, int _lightCount, PBRMaterial * _materials, int _materialCount)
: objectList(_objectList)
, objectCount(_size)
, lights(_lights)
, lightCount(_lightCount)
, materials(_materials)
, materialCount(_materialCount)
{
}

__device__ Scene::~Scene()
{
	for (int i=0; i < objectCount; ++i)
		delete objectList[i];
	delete [] objectList;
	for (int i=0; i < lightCount; ++i)
		delete lights[i];
	delete [] lights;
	delete [] materials;
}

__device__ bool Scene::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	bool hasHit = false;
	float closest_t = tmax;

	Hit tmpHit;
	for (int i=0; i < objectCount; ++i)
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

// helpers
Scene * createScene()
{
	Scene * d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	d_createScene<<<1, 1>>>(d_scene);
//	initScene<<<1, 1>>>(d_scene);
//	initCornellBox<<<1, 1>>>(d_scene);
	syncAndCheckErrors();

	return d_scene;
}

__global__ void d_createScene(Scene * ptr)
{
	new (ptr) Scene();
}

void setHitableCount(Scene * d_scene, unsigned int count)
{
	d_setHitableCount<<<1, 1>>>(d_scene, count);
}

__global__ void d_setHitableCount(Scene * d_scene, unsigned int count)
{
	for (int i=0; i < d_scene->objectCount; ++i)
		delete d_scene->objectList[i];
	delete [] d_scene->objectList;

	d_scene->objectList = new Hitable*[count];
	d_scene->objectCount = count;
}

void setHitable(Scene * d_scene, unsigned int id, Hitable * hitable)
{
	d_setHitable<<<1, 1>>>(d_scene, id, hitable);
}

__global__ void d_setHitable(Scene * d_scene, unsigned int id, Hitable * hitable)
{
	d_scene->objectList[id] = hitable;
}

void setLightCount(Scene * d_scene, unsigned int count)
{
	d_setLightCount<<<1, 1>>>(d_scene, count);
}

__global__ void d_setLightCount(Scene * d_scene, unsigned int count)
{
	for (int i=0; i < d_scene->lightCount; ++i)
		delete d_scene->lights[i];
	delete [] d_scene->lights;

	d_scene->lights = new Light*[count];
	d_scene->lightCount = count;
}

void setLight(Scene * d_scene, unsigned int id, Light * light)
{
	d_setLight<<<1, 1>>>(d_scene, id, light);
}

__global__ void d_setLight(Scene * d_scene, unsigned int id, Light * light)
{
	d_scene->lights[id] = light;
}

void setMaterialCount(Scene * d_scene, unsigned int count)
{
	d_setMaterialCount<<<1, 1>>>(d_scene, count);
}

__global__ void d_setMaterialCount(Scene * d_scene, unsigned int count)
{
	delete [] d_scene->materials;
	d_scene->materials = new PBRMaterial[count];
	d_scene->materialCount = count;
}

void setMaterial(Scene * d_scene, unsigned int id, const PBRMaterial & material)
{
	d_setMaterial<<<1, 1>>>(d_scene, id, material);
}

__global__ void d_setMaterial(Scene * d_scene, unsigned int id, PBRMaterial material)
{
	d_scene->materials[id] = material;
}

void destroyScene(Scene * d_scene)
{
	deleteScene<<<1, 1>>>(d_scene);
	syncAndCheckErrors();
	cudaFree(d_scene);
	syncAndCheckErrors();
}

__global__ void deleteScene(Scene * ptrScene)
{
	ptrScene->~Scene();
}
