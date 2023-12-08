#ifndef SCENE_H
#define SCENE_H


#include <Object.cuh>
#include <Material.cuh>
#include <Light.cuh>
#include <BVH.cuh>


class Scene : public Hitable
{
public:
	__device__ Scene();
	__device__ ~Scene() override;
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;

public:
	TLAS tlas;
	BVH * BVHList;
	unsigned int bvhCount;
	BVHInstance * instances;
	unsigned int instanceCount;
	Mesh * meshes;
	unsigned int meshCount;

	Light ** lights;
	int lightCount;
	PBRMaterial * materials;
	int materialCount;


public:
	__host__ static BVH * createBVHList(Scene * d_scene, unsigned int count);
};

__global__ static void d_createBVHList(Scene * d_scene, unsigned int count, BVH ** out);


__global__ void deleteScene(Scene * ptrScene);

// helpers
Scene * createScene();
__global__ void d_createScene(Scene * ptr);
void destroyScene(Scene * d_scene);

void setHitableCount(Scene * d_scene, unsigned int count);
__global__ void d_setHitableCount(Scene * d_scene, unsigned int count);
void setHitable(Scene * d_scene, unsigned int id, Hitable * hitable);
__global__ void d_setHitable(Scene * d_scene, unsigned int id, Hitable * hitable);

void setLightCount(Scene * d_scene, unsigned int count);
__global__ void d_setLightCount(Scene * d_scene, unsigned int count);
void setLight(Scene * d_scene, unsigned int id, Light * light);
__global__ void d_setLight(Scene * d_scene, unsigned int id, Light * light);

void setMaterialCount(Scene * d_scene, unsigned int count);
__global__ void d_setMaterialCount(Scene * d_scene, unsigned int count);
void setMaterial(Scene * d_scene, unsigned int id, const PBRMaterial & material);
__global__ void d_setMaterial(Scene * d_scene, unsigned int id, PBRMaterial material);

#endif // SCENE_H
