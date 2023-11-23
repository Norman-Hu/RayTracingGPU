#include <Scene.cuh>
#include <CudaHelpers.h>


__device__ Scene::Scene()
: objectList(nullptr)
, objectCount(0)
, materials(nullptr)
, materialCount(0)
{

}

__device__ Scene::Scene(Hitable ** _objectList, int _size, BlinnPhongMaterial * _materials, int _materialCount)
: objectList(_objectList)
, objectCount(_size)
, materials(_materials)
, materialCount(_materialCount)
{
}

__device__ Scene::~Scene()
{
	for (int i=0; i < objectCount; ++i)
		delete objectList[i];
	delete [] objectList;
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

// initialization and destruction
Scene * createScene()
{
	Scene * d_scene;
	cudaMalloc(&d_scene, sizeof(Scene));
//	initScene<<<1, 1>>>(d_scene);
//	initDebugScene<<<1, 1>>>(d_scene);
	initCornellBox<<<1, 1>>>(d_scene);
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
	new (ptrScene) Scene(new Hitable*[2], 2, new BlinnPhongMaterial[2], 2);
	Sphere * pSphere = new Sphere();
	ptrScene->objectList[0] = pSphere;
	pSphere->c = {1.f, 0.0f, -10.0f};
	pSphere->r = 1.f;
	pSphere->materialId = 1;
	pSphere = new Sphere();
	ptrScene->objectList[1] = pSphere;
	pSphere->c = {-1.f, 0.0f, -10.0f};
	pSphere->r = 1.f;
	pSphere->materialId = 0;

	BlinnPhongMaterial * mat = &ptrScene->materials[0];
	mat->ambient = {0.1f, 0.1f, 0.1f};
	mat->diffuse = {0.6f, 0.f, 0.f};
	mat->specular = {1.f, 1.f, 1.f};
	mat->shininess = 32.f;
	mat->mirror = 0.f;
	mat = &ptrScene->materials[1];
	mat->ambient = {0.0f, 0.0f, 0.0f};
	mat->diffuse = {1.f, 1.0f, 1.f};
	mat->specular = {1.f, 1.f, 1.f};
	mat->shininess = 32.f;
	mat->mirror = 1.f;
}

__global__ void initCornellBox(Scene * ptrScene)
{
	new (ptrScene) Scene(new Hitable*[8], 4, new BlinnPhongMaterial[5], 5);

	// Floor
	Square * pSquare = new Square();
	ptrScene->objectList[0] = pSquare;
	pSquare->p = Vec3(-1.f, -1.f, 1.f);
	pSquare->right = Vec3(2.f, 0.f, 0.f);
	pSquare->up = Vec3(0.f, 0.f, -2.f);
	pSquare->n = Vec3(0.f, 1.f, 0.f);
	pSquare->materialId = 0;

	// Ceiling
	pSquare = new Square();
	ptrScene->objectList[1] = pSquare;
	pSquare->p = Vec3(-1.f, 1.f, -1.f);
	pSquare->right = Vec3(2.f, 0.f, 0.f);
	pSquare->up = Vec3(0.f, 0.f, 2.f);
	pSquare->n = Vec3(0.f, -1.f, 0.f);
	pSquare->materialId = 0;

	// Left wall
	pSquare = new Square();
	ptrScene->objectList[2] = pSquare;
	pSquare->p = Vec3(-1.f, -1.f, 1.f);
	pSquare->right = Vec3(0.f, 0.f, -2.f);
	pSquare->up = Vec3(0.f, 2.f, 0.f);
	pSquare->n = Vec3(1.f, 0.f, 0.f);
	pSquare->materialId = 0;

	// Right wall
	pSquare = new Square();
	ptrScene->objectList[3] = pSquare;
	pSquare->p = Vec3(1.f, -1.f, -1.f);
	pSquare->right = Vec3(0.f, 0.f, 2.f);
	pSquare->up = Vec3(0.f, 2.f, 0.f);
	pSquare->n = Vec3(-1.f, 0.f, 0.f);
	pSquare->materialId = 0;
	
	BlinnPhongMaterial * mat = &ptrScene->materials[0];
	mat->ambient = {0.1f, 0.1f, 0.1f};
	mat->diffuse = {0.6f, 0.f, 0.f};
	mat->specular = {1.f, 1.f, 1.f};
	mat->shininess = 32.f;
	mat->mirror = 0.f;
}

__global__ void deleteScene(Scene * ptrScene)
{
	ptrScene->~Scene();
}
