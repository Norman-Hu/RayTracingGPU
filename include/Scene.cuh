#ifndef SCENE_H
#define SCENE_H


#include <vector>
#include <Object.cuh>


class Scene : public Hitable
{
public:
	__device__ Scene();
	__device__ Scene(Hitable ** _objectList, int _size);
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;

public:
	Hitable ** objectList;
	int size;
};

#endif // SCENE_H
