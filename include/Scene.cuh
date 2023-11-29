#ifndef SCENE_H
#define SCENE_H


#include <Object.cuh>
#include <Material.cuh>
#include <Light.cuh>


class Scene : public Hitable
{
public:
	__device__ Scene();
	__device__ ~Scene() override;
	__device__ Scene(Hitable ** _objectList, int _size, Light ** _lights, int _lightCount, PBRMaterial * _materials, int _materialCount);
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;

public:
	Hitable ** objectList;
	int objectCount;
	Light ** lights;
	int lightCount;
	PBRMaterial * materials;
	int materialCount;
};

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
