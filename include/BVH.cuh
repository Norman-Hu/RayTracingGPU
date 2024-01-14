#ifndef BVH_H
#define BVH_H


#include <Vec3.cuh>
#include <Object.cuh>
#include <Matrix.cuh>
Vec3 max_comp(const Vec3 & v1, const Vec3 & v2);
Vec3 min_comp(const Vec3 & v1, const Vec3 & v2);
int smin(int a, int b);
int smax(int a, int b);

struct AABB
{
	Vec3 min;
	Vec3 max;

    void grow_vec3(const Vec3 & p);
    void grow_aabb(const AABB & bounds);
	float area() const;
};

__device__ bool intersect_aabb(const Ray & ray, const AABB & bounds, float & distance, float tmin, float tmax, Hit & out);
__device__ bool intersect_aabb(const Ray & ray, const AABB & bounds, const Vec3 & inverseDir, float & distance, float tmin, float tmax, Hit & out);

/******** BVH ********/
struct BVHNode
{
	AABB aabb;
	unsigned int nodeLeft;
	unsigned int nodeRight;
	int triangleIndex;
	int triangleCount;

	__host__ __device__ bool isLeaf() const {return triangleCount > 0;};
	__host__ float calculateNodeCost()
	{
		Vec3 e = aabb.max - aabb.min; // extent of the node
		return (e.x * e.y + e.y * e.z + e.z * e.x) * triangleCount;
	}
};

__device__ bool intersectTri(const Ray & ray, Vec3 v0, Vec3 v1, Vec3 v2, float tmin, float tmax, Hit & out);

class BVH
{
public:
	__host__ __device__ BVH();
	BVH(unsigned int _meshID, const Mesh & mesh);

	__host__ __device__ ~BVH();
	BVH(BVH && other) noexcept;
	BVH & operator=(BVH && other) noexcept;

	__device__ bool intersect(const Ray & ray, float tmin, float tmax, Hit & out, Mesh * meshList, const Vec3 & invDir) const;

private:
	void build(const Mesh & mesh);
	void updateNodeBounds(unsigned int nodeIdx, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, Vec3 * centroids);
	void subdivide(unsigned int nodeIdx, unsigned int depth, unsigned int & nodePtr, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, Vec3 * centroids);
	float findBestSplitPlane(BVHNode & node, int & axis, int & splitPos, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, const Vec3 * centroids);
	static const int BINS = 8;
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

	__device__ bool intersect(const Ray & ray, float tmin, float tmax, Hit & out, BVH * blasList, Mesh * meshList) const;

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

    static void copyToGPU(const TLASNode & node, TLASNode * gpuMemory);
};
__global__ static void d_TLASNode_copyToGPU(TLASNode * d_instance, TLASNode nodeToCopy);

class TLAS
{
public:
	__host__ __device__ TLAS();
	TLAS(BVHInstance * instances, unsigned int count);

	__host__ __device__ ~TLAS();
	TLAS(TLAS && other) noexcept;
	TLAS & operator=(TLAS && other) noexcept;

	__device__ bool intersect(const Ray & ray, float tmin, float tmax, Hit & out, BVH * blasList, Mesh * meshList) const;

private:
	void build();
	int findBestMatch(int * list, int N, int A) const;

public:
	TLASNode * nodes;
	unsigned int nodeCount;

	BVHInstance * blas;
	unsigned int blasCount;

public:
	__host__ static BVHInstance * createBVHInstanceList(TLAS * d_tlas, unsigned int count);
	__host__ static TLASNode * createTLASNodeList(TLAS * d_tlas, unsigned int count);
};

__global__ static void d_createBVHInstanceList(TLAS * d_tlas, unsigned int count, BVHInstance ** out);
__global__ static void d_createTLASNodeList(TLAS * d_tlas, unsigned int count, TLASNode ** out);

#endif // BVH_H
