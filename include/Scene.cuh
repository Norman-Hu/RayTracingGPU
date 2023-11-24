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
	__device__ Scene(Hitable ** _objectList, int _size, Light ** _lights, int _lightCount, BlinnPhongMaterial * _materials, int _materialCount);
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;

public:
	Hitable ** objectList;
	int objectCount;
	Light ** lights;
	int lightCount;
	BlinnPhongMaterial * materials;
	int materialCount;
};

__global__ void initScene(Scene * ptrScene);
__global__ void initCornellBox(Scene * ptrScene);
__global__ void deleteScene(Scene * ptrScene);

Scene * createScene();
void destroyScene(Scene * d_scene);


#endif // SCENE_H
