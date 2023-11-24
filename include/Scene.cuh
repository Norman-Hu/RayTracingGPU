#ifndef SCENE_H
#define SCENE_H


#include <Object.cuh>
#include <Material.cuh>


class Scene : public Hitable
{
public:
	__device__ Scene();
	__device__ ~Scene();
	__device__ Scene(Hitable ** _objectList, int _size, Vec3 * _pointLights, int _pointLightsCount, BlinnPhongMaterial * _materials, int _materialCount);
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;

public:
	Hitable ** objectList;
	int objectCount;
	Vec3 * pointLights;
	int pointLightsCount;
	BlinnPhongMaterial * materials;
	int materialCount;
};

__global__ void initScene(Scene * ptrScene);
__global__ void initCornellBox(Scene * ptrScene);
__global__ void deleteScene(Scene * ptrScene);

Scene * createScene();
void destroyScene(Scene * d_scene);


#endif // SCENE_H
