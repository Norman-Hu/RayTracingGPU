#ifndef BVH_H
#define BVH_H


#include <Vec3.cuh>
#include <Object.cuh>
#include <Matrix.cuh>


struct AABB
{
	Vec3 min;
	Vec3 max;

    void grow_vec3(const Vec3 & p);
    void grow_aabb(const AABB & bounds);
};

/******** BVH ********/
struct BVHNode
{
	AABB aabb;
	unsigned int nodeLeft;
	unsigned int nodeRight;
	int triangleIndex;
	int triangleCount;
};

class BVH
{
public:
	__host__ __device__ BVH();
	BVH(unsigned int _meshID, const Mesh & mesh);

	~BVH();
	BVH(BVH && other) noexcept;
	BVH & operator=(BVH && other) noexcept;
private:
	void build();

public:
	BVHNode * nodes;
	unsigned int nodeCount;

	unsigned int * primitives;
	unsigned int primitive_count;

	// references
	unsigned int meshID;


// gpu
public:
	static void copyToGPU(const BVH & instance, BVH * gpuMemory);
};

// kernels
__global__ static void d_BVH_copyToGPU(BVH * d_instance, BVHNode * d_nodeBuffer, unsigned int nodeCount, unsigned int * d_primitivesBuffer, unsigned int primitive_count, unsigned int meshID);

class BVHInstance
{
public:
	AABB bounds; // world space
	unsigned int instanceID = 0;

	// references
	unsigned int bvhID = 0;
	// transforms
	Matrix4x4 transform;
	Matrix4x4 invTransform;

public:
    static void copyToGPU(const BVHInstance & instance, BVHInstance * gpuMemory);
};

__global__ static void d_BVHInstance_copyToGPU(BVHInstance * d_instance, BVHInstance instanceToCopy);

/******** TLAS ********/
struct TLASNode
{
	AABB aabb;
	unsigned int nodeLeft;
	unsigned int nodeRight;
	unsigned int BLAS;
    __host__ __device__ bool isLeaf();
};

class TLAS
{
public:
	__host__ __device__ TLAS();
	TLAS(BVHInstance * instances, unsigned int count);

	~TLAS();
	TLAS(TLAS && other) noexcept;
	TLAS & operator=(TLAS && other) noexcept;

private:
	void build();

public:
	TLASNode * nodes;
	unsigned int nodeCount;

	BVHInstance * blas;
	unsigned int blasCount;

	unsigned int * nodeIds;
};

#endif // BVH_H
