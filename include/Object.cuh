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

struct AABB
{
	Vec3 min;
	Vec3 max;
};

__host__ __device__ bool rayAABB(const Ray & ray, const AABB & aabb, float tmin, float tmax);

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
	Vec3 * vertices;
	unsigned int vertices_count;

	Vec3 * normals;
	unsigned int normals_count;

	unsigned int * indices;
	unsigned int indices_count;

	AABB aabb;

    __device__ Mesh();
    __device__ ~Mesh() override;
	__device__ bool hit(const Ray & ray, float tmin, float tmax, Hit & out) override;
};

AABB computeAABB(Vec3 * vertices, unsigned int count);

Mesh * createMesh();
__global__ void d_createMesh(Mesh ** ptr_d_mesh);
void setMeshAABB(Mesh * d_mesh, const AABB & bounds);
__global__ void d_setMeshAABB(Mesh * d_mesh, AABB bounds);
void setMeshMaterial(Mesh * mesh, unsigned int index);
__global__ void d_setMeshMaterial(Mesh * mesh, unsigned int index);
void setMeshVertices(Mesh * mesh, Vec3 * vertices, unsigned int verticesCount);
__global__ void d_copyVertices(Mesh * mesh, Vec3 * d_vertices, unsigned int verticesCount);
__global__ void d_setMeshVerticesCount(Mesh * mesh, unsigned int verticesCount);
void setMeshIndices(Mesh * mesh, unsigned int * indices, unsigned int indicesCount);
__global__ void d_copyIndices(Mesh * mesh, unsigned int * d_indices, unsigned int indicesCount);
__global__ void d_setMeshIndicesCount(Mesh * mesh, unsigned int indicesCount);
void setMeshNormals(Mesh * mesh, Vec3 * normals, unsigned int normalsCount);
__global__ void d_copyNormals(Mesh * mesh, Vec3 * d_normals, unsigned int normalsCount);
__global__ void d_setMeshNormalsCount(Mesh * mesh, unsigned int normalsCount);

#endif // OBJECT_H
