#ifndef OBJECT_H
#define OBJECT_H


#include <Vec3.cuh>
#include <Ray.cuh>


struct Hit
{
	float t;
	Vec3 p;
	Vec3 normal;
	int materialId;
};

class Hitable
{
public:
	int materialId = -1;

	__device__ virtual ~Hitable() {};
	__device__ virtual bool hit(const Ray & ray, float tmin, float tmax, Hit & out) = 0;
};

class Sphere : public Hitable
{
public:
	Vec3 c;
	float r;

	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;
};

class Square : public Hitable
{
public:
	Vec3 p;
	Vec3 n;
	Vec3 right;
	Vec3 up;

	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;
};

class Mesh : public Hitable
{
public:
	Vec3 p;
	Vec3* vertices;
	unsigned int vertices_count;
	Vec3* normals;
	unsigned int normals_count;
	unsigned int* indices;
	unsigned int indices_count;

	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;
};

#endif // OBJECT_H