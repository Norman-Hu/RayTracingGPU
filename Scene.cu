#include <Scene.cuh>
#include <CudaHelpers.h>


__device__ Scene::Scene()
: tlas()
, BVHList(nullptr)
, bvhCount(0)
, instances(nullptr)
, instanceCount(0)
, meshes(nullptr)
, meshCount(0)
, lights(nullptr)
, lightCount(0)
, materials(nullptr)
, materialCount(0)
{

}

__device__ Scene::~Scene()
{
	delete [] BVHList;
	delete [] instances;
	delete [] meshes;
	for (int i=0; i < lightCount; ++i)
		delete lights[i];
	delete [] lights;
	delete [] materials;
}

__device__ bool Scene::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	// TODO: trace with TLAS
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

__host__ BVH * Scene::createBVHList(Scene * d_scene, unsigned int count)
{
	// alloc pointer to the bvh list
	BVH ** ptr;
	errchk(cudaMalloc(&ptr, sizeof(BVH**)));

	// instantiate list
	d_createBVHList<<<1, 1>>>(d_scene, count, ptr);
	syncAndCheckErrors();

	// copy result
	BVH * res;
	errchk(cudaMemcpy(res, ptr, sizeof(BVH*), cudaMemcpyDeviceToHost));

	errchk(cudaFree(ptr));
	return res;
}

__global__ void d_createBVHList(Scene * d_scene, unsigned int count, BVH ** out)
{
	delete [] d_scene->BVHList;
	d_scene->BVHList = new BVH[count];
	*out = d_scene->BVHList;
}

__host__ static BVHInstance * createBVHInstanceList(Scene * d_scene, unsigned int count)
{
    // alloc pointer to the bvh list
    BVHInstance ** ptr;
    errchk(cudaMalloc(&ptr, sizeof(BVHInstance**)));

    // instantiate list
    d_createBVHInstanceList<<<1, 1>>>(d_scene, count, ptr);
    syncAndCheckErrors();

    // copy result
    BVHInstance * res;
    errchk(cudaMemcpy(res, ptr, sizeof(BVHInstance*), cudaMemcpyDeviceToHost));

    errchk(cudaFree(ptr));
    return res;
}

__global__ static void d_createBVHInstanceList(Scene * d_scene, unsigned int count, BVHInstance ** out)
{
    delete [] d_scene->instances;
    d_scene->instances = new BVHInstance[count];
    *out = d_scene->instances;
}

// helpers
Scene * createScene()
{
	Scene * d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
	d_createScene<<<1, 1>>>(d_scene);
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
