#ifndef OBJECT_H
#define OBJECT_H


#include <Vec3.cuh>
#include <Ray.cuh>


struct Hit
{
	float t;
	Vec3 p;
	Vec3 normal;
};

class Hitable
{
public:
	__device__ virtual bool hit(const Ray & ray, float tmin, float tmax, Hit & out) = 0;
};

class Sphere : public Hitable
{
public:
	Vec3 c;
	float r;

	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;
};

#endif // OBJECT_H
