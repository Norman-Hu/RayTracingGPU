#include <BVH.cuh>
#include <Memory.cuh>
#include <CudaHelpers.h>

int smin(int a, int b)
{
	return a < b ? a : b;
}

int smax(int a, int b)
{
	return a > b ? a : b;
}

void AABB::grow_vec3(const Vec3 & p)
{
    min = Vec3(fminf(p.x, min.x), fminf(p.y, min.y), fminf(p.z, min.z));
    max = Vec3(fmaxf(p.x, max.x), fmaxf(p.y, max.y), fmaxf(p.z, max.z));
}

void AABB::grow_aabb(const AABB & bounds)
{
    grow_vec3(bounds.min);
    grow_vec3(bounds.max);
}

float AABB::area() const
{
	Vec3 e = max - min;
	return e.x * e.y + e.y * e.z + e.z * e.x;
}

Vec3 max_comp(const Vec3 & v1, const Vec3 & v2)
{
	return {fmaxf(v1.x, v2.x), fmaxf(v1.y, v2.y), fmaxf(v1.z, v2.z)};
}

Vec3 min_comp(const Vec3 & v1, const Vec3 & v2)
{
	return {fminf(v1.x, v2.x), fminf(v1.y, v2.y), fminf(v1.z, v2.z)};
}

//int min(int a, int b)
//{
//	return a < b ? a : b;
//}
//
//int max(int a, int b)
//{
//	return a > b ? a : b;
//}

__device__ bool intersect_aabb(const Ray & ray, const AABB & bounds, float & distance, float tmin, float tmax)
{
	Vec3 rD = {1.0f/ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};
	return intersect_aabb(ray, bounds, rD, distance, tmin, tmax);
}

__device__ bool intersect_aabb(const Ray & ray, const AABB & bounds, const Vec3 & inverseDir, float & distance, float tmin, float tmax)
{
	float tx1 = (bounds.min.x - ray.origin.x) * inverseDir.x, tx2 = (bounds.max.x - ray.origin.x) * inverseDir.x;
	float tlower = fminf(tx1, tx2), thigher = fmaxf(tx1, tx2);
	float ty1 = (bounds.min.y - ray.origin.y) * inverseDir.y, ty2 = (bounds.max.y - ray.origin.y) * inverseDir.y;
	tlower = fmaxf(tlower, fminf(ty1, ty2)), thigher = fminf(thigher, fmaxf(ty1, ty2));
	float tz1 = (bounds.min.z - ray.origin.z) * inverseDir.z, tz2 = (bounds.max.z - ray.origin.z) * inverseDir.z;
	tlower = fmaxf(tlower, fminf(tz1, tz2)), thigher = fminf(thigher, fmaxf(tz1, tz2));
	if (thigher >= tlower && thigher > 0.0f && tlower > tmin && tlower < tmax)
	{
		distance = tlower;
		return true;
	}
	return false;
}

__device__ bool intersectTri(const Ray & ray, Vec3 v0, Vec3 v1, Vec3 v2, float tmin, float tmax, Hit & out)
{
	// Möller–Trumbore intersection algorithm
	// From https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	constexpr float EPSILON = 1e-7f;
	Vec3 e1 = v1 - v0;
	Vec3 e2 = v2 - v0;
	Vec3 crossRayE2 = Vec3::cross(ray.direction, e2);
	float det = Vec3::dot(e1, crossRayE2);
	if (det > -EPSILON && det < EPSILON) // parallel
		return false;

	float invDet = 1.0f/det;
	Vec3 s = ray.origin - v0;
	float u = invDet * Vec3::dot(s, crossRayE2);
	if (u < 0.f || u > 1.f)
		return false;

	Vec3 crossSE1 = Vec3::cross(s, e1);
	float v = invDet * Vec3::dot(ray.direction, crossSE1);
	if (v < 0.f || u+v > 1.f)
		return false;

	float t = invDet * Vec3::dot(e2, crossSE1);
	if (t < tmin || t > tmax)
		return false;

	out.t = t;
	out.p = ray.origin + t * ray.direction;
	out.normal = Vec3::cross(e1, e2); // FIXME: Interpolate normals using barycentric coordinates
	return true;
}

__host__ __device__ BVH::BVH()
: nodes(nullptr)
, nodeCount(0)
, primitives(nullptr)
, primitive_count(0)
, meshID(0)
{
}

BVH::BVH(unsigned int _meshID, const Mesh & mesh)
: nodes(nullptr)
, nodeCount(0)
, primitives(nullptr)
, primitive_count(mesh.indices_count/3)
, meshID(_meshID)
{
	primitives = new unsigned int[primitive_count];
	build(mesh);
}

__host__ __device__ BVH::~BVH()
{
	delete [] nodes;
	delete [] primitives;
}

BVH::BVH(BVH && other) noexcept
: nodes(exchange<BVHNode*>(other.nodes, nullptr))
, nodeCount(other.nodeCount)
, primitives(exchange<unsigned int *>(other.primitives, nullptr))
, primitive_count(other.primitive_count)
, meshID(other.meshID)
{
}

BVH & BVH::operator=(BVH &&other) noexcept
{
	swap(other.nodes, nodes);
	nodeCount = other.nodeCount;
	swap(other.primitives, primitives);
	primitive_count = other.primitive_count;
	meshID = other.meshID;
	return *this;
}

__device__ bool BVH::intersect(const Ray & ray, float tmin, float tmax, Hit & out, Mesh * meshList, const Vec3 & invDir) const
{
	BVHNode * node = &nodes[0], *stack[64];
	unsigned int stackPtr = 0;
	Mesh * mesh = &meshList[meshID];
	bool intersects = false;
	Hit tmp;
	while (true)
	{
		if (node->isLeaf())
		{
			for (unsigned int i = 0; i < node->triangleCount; i++)
			{
				if (intersectTri(ray, mesh->vertices[mesh->indices[node->nodeLeft+3*i]], mesh->vertices[mesh->indices[node->nodeLeft+3*i+1]], mesh->vertices[mesh->indices[node->nodeLeft+3*i+2]], tmin, tmax, tmp))
				{
					if (!intersects || tmp.t < out.t)
					{
						intersects = true;
						out = tmp;
					}
				}
			}
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
			continue;
		}

		BVHNode * child1 = &nodes[node->nodeLeft];
		BVHNode * child2 = &nodes[node->nodeRight];

		float dist1 = 1e30f, dist2 = 1e30f;
		bool hit1 = intersect_aabb(ray, child1->aabb, invDir, dist1, tmin, tmax);
		bool hit2 = intersect_aabb(ray, child2->aabb, invDir, dist2, tmin, tmax);
		if (dist1 > dist2)
		{
			swap(dist1, dist2);
			swap(child1, child2);
			swap(hit1, hit2);
		}
		if (!hit1)
		{
			// missed both child nodes; pop a node from the stack
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
		}
		else
		{
			// visit near node; push the far node if the ray intersects it
			node = child1;
			if (hit2)
				stack[stackPtr++] = child2;
		}
	}
	return intersects;
}

void BVH::build(const Mesh & mesh)
{
	// reset node pool
	nodes = new BVHNode[primitive_count*2];
	nodeCount = 2;
	int triCount = (mesh.indices_count/3);
	memset(nodes, 0, triCount * 2 * sizeof(BVHNode));
	// populate triangle index array
	for (int i = 0; i < triCount; i++)
		primitives[i] = i;
	// calculate triangle centroids for partitioning
	Vec3 * centroids = new Vec3[triCount];
	for (int i = 0; i < mesh.indices_count; i+=3)
		centroids[i/3] = (mesh.vertices[mesh.indices[i]] + mesh.vertices[mesh.indices[i+1]] + mesh.vertices[mesh.indices[i+2]]) * 0.3333f;
	// assign all triangles to root node
	BVHNode & root = nodes[0];
	root.triangleIndex = 0;
	root.triangleCount = triCount;
	Vec3 centroidMin, centroidMax;
	updateNodeBounds( 0, centroidMin, centroidMax, mesh, centroids);
	// subdivide
	subdivide(0, 0, nodeCount, centroidMin, centroidMax, mesh, centroids);

	delete [] centroids;
}

void BVH::updateNodeBounds(unsigned int nodeIdx, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, Vec3 * centroids)
{
	BVHNode & node = nodes[nodeIdx];
	node.aabb.min = {1e30f, 1e30f, 1e30f};
	node.aabb.max = {-1e30f, -1e30f, -1e30f};
	centroidMin = node.aabb.min;
	centroidMax = node.aabb.max;
	for (unsigned int first = node.triangleIndex, i = 0; i < node.triangleCount; i++)
	{
		unsigned int leafTriIdx = primitives[first + i];
		node.aabb.min = min_comp(node.aabb.min, mesh.vertices[mesh.indices[leafTriIdx*3]]);
		node.aabb.min = min_comp(node.aabb.min, mesh.vertices[mesh.indices[leafTriIdx*3+1]]);
		node.aabb.min = min_comp(node.aabb.min, mesh.vertices[mesh.indices[leafTriIdx*3+2]]);
		node.aabb.max = max_comp(node.aabb.max, mesh.vertices[mesh.indices[leafTriIdx*3]]);
		node.aabb.max = max_comp(node.aabb.max, mesh.vertices[mesh.indices[leafTriIdx*3+1]]);
		node.aabb.max = max_comp(node.aabb.max, mesh.vertices[mesh.indices[leafTriIdx*3+2]]);

		centroidMin = min_comp(centroidMin, centroids[leafTriIdx]);
		centroidMax = max_comp(centroidMax, centroids[leafTriIdx]);
	}
}

void BVH::subdivide(unsigned int nodeIdx, unsigned int depth, unsigned int & nodePtr, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, Vec3 * centroids)
{
	BVHNode & node = nodes[nodeIdx];
	// determine split axis using SAH
	int axis, splitPos;
	float splitCost = findBestSplitPlane(node, axis, splitPos, centroidMin, centroidMax, mesh, centroids);
	// terminate recursion
//	if (subdivToOnePrim)
	if (false)
	{
		if (node.triangleCount == 1) return;
	}
	else
	{
		float nosplitCost = node.calculateNodeCost();
		if (splitCost >= nosplitCost)
			return;
	}
	// in-place partition
	int i = node.triangleIndex;
	int j = i + node.triangleCount - 1;
	float scale = BINS / (centroidMax[axis] - centroidMin[axis]);
	while (i <= j)
	{
		// use the exact calculation we used for binning to prevent rare inaccuracies
		int binIdx = smin( BINS - 1, static_cast<int>((centroids[primitives[i]][axis] - centroidMin[axis]) * scale));
		if (binIdx < splitPos)
		{
			i++;
		}
		else
		{
			swap(primitives[i], primitives[j--]);
//			swap(centroids[i], centroids[j--]);
		}
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.triangleIndex;
	if (leftCount == 0 || leftCount == node.triangleCount)
		return; // never happens for dragon mesh, nice
	// create child nodes
	int leftChildIdx = nodePtr++;
	int rightChildIdx = nodePtr++;
	nodes[leftChildIdx].triangleIndex = node.triangleIndex;
	nodes[leftChildIdx].triangleCount = leftCount;
	nodes[rightChildIdx].triangleIndex = i;
	nodes[rightChildIdx].triangleCount = node.triangleCount - leftCount;
	node.nodeLeft = leftChildIdx;
	node.nodeRight = rightChildIdx;
	node.triangleCount = 0;
	// recurse
	updateNodeBounds(leftChildIdx, centroidMin, centroidMax, mesh, centroids);
	subdivide(leftChildIdx, depth + 1, nodePtr, centroidMin, centroidMax, mesh, centroids);
	updateNodeBounds(rightChildIdx, centroidMin, centroidMax, mesh, centroids);
	subdivide(rightChildIdx, depth + 1, nodePtr, centroidMin, centroidMax, mesh, centroids);
}

float BVH::findBestSplitPlane(BVHNode & node, int & axis, int & splitPos, Vec3 & centroidMin, Vec3 & centroidMax, const Mesh & mesh, const Vec3 * centroids)
{
	float bestCost = 1e30f;
	for (int a = 0; a < 3; a++)
	{
		float boundsMin = centroidMin[a], boundsMax = centroidMax[a];
		if (boundsMin == boundsMax) continue;
		// populate the bins
		float scale = BINS / (boundsMax - boundsMin);
		float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
		int leftSum = 0, rightSum = 0;
		struct Bin {
			AABB bounds;
			int triCount = 0;
		} bin[BINS];
		for (unsigned int i = 0; i < node.triangleCount; i++)
		{
			unsigned int triId = node.triangleIndex + i;
			int binIdx = smin(BINS - 1, (int)((centroids[primitives[triId]][a] - boundsMin) * scale));
			bin[binIdx].triCount++;
			bin[binIdx].bounds.grow_vec3(mesh.vertices[mesh.indices[primitives[triId]*3]]);
			bin[binIdx].bounds.grow_vec3(mesh.vertices[mesh.indices[primitives[triId]*3+1]]);
			bin[binIdx].bounds.grow_vec3(mesh.vertices[mesh.indices[primitives[triId]*3+2]]);
		}
		// gather data for the 7 planes between the 8 bins
		AABB leftBox, rightBox;
		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bin[i].triCount;
			leftBox.grow_aabb(bin[i].bounds);
			leftCountArea[i] = leftSum * leftBox.area();
			rightSum += bin[BINS - 1 - i].triCount;
			rightBox.grow_aabb(bin[BINS - 1 - i].bounds);
			rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
		}
		// calculate SAH cost for the 7 planes
//		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			const float planeCost = leftCountArea[i] + rightCountArea[i];
			if (planeCost < bestCost)
				axis = a, splitPos = i + 1, bestCost = planeCost;
		}
	}
	return bestCost;
}

// GPU
void BVH::copyToGPU(const BVH & instance, BVH * gpuMemory)
{
	BVHNode * gpuNodes;
	errchk(cudaMalloc(&gpuNodes, sizeof(BVHNode)*instance.nodeCount));
	errchk(cudaMemcpy(gpuNodes, instance.nodes, sizeof(BVHNode)*instance.nodeCount, cudaMemcpyHostToDevice));

	unsigned int * gpuPrimitives;
	errchk(cudaMalloc(&gpuPrimitives, sizeof(unsigned int)*instance.primitive_count));
	errchk(cudaMemcpy(gpuPrimitives, instance.primitives, sizeof(unsigned int)*instance.primitive_count, cudaMemcpyHostToDevice));

	d_BVH_copyToGPU<<<1, 1>>>(gpuMemory, gpuNodes, instance.nodeCount, gpuPrimitives, instance.primitive_count, instance.meshID);
	syncAndCheckErrors();

	errchk(cudaFree(gpuNodes));
	errchk(cudaFree(gpuPrimitives));
}

__global__ void d_BVH_copyToGPU(BVH * d_instance, BVHNode * d_nodeBuffer, unsigned int nodeCount, unsigned int * d_primitivesBuffer, unsigned int primitive_count, unsigned int meshID)
{
	d_instance->nodeCount = nodeCount;
	memcpy(d_instance->nodes, d_nodeBuffer, sizeof(BVHNode)*nodeCount);
	d_instance->primitive_count = primitive_count;
	memcpy(d_instance->primitives, d_primitivesBuffer, sizeof(unsigned int)*primitive_count);
	d_instance->meshID = meshID;
}


__device__ bool BVHInstance::intersect(const Ray & ray, float tmin, float tmax, Hit & out, BVH * blasList, Mesh * meshList) const
{
	// transform ray
	Ray newRay = ray;
	newRay.origin = (invTransform * Vec4(ray.origin, 1)).vec3();
	newRay.direction = (invTransform * Vec4(ray.direction, 0)).vec3();
	Vec3 invDir = {1.0f/ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};

	return blasList[bvhID].intersect(newRay, tmin, tmax, out, meshList, invDir);
}

void BVHInstance::copyToGPU(const BVHInstance & instance, BVHInstance * gpuMemory)
{
    d_BVHInstance_copyToGPU<<<1, 1>>>(gpuMemory, instance);
    syncAndCheckErrors();
}

__global__ static void d_BVHInstance_copyToGPU(BVHInstance * d_instance, BVHInstance instanceToCopy)
{
    *d_instance = instanceToCopy;
}

__host__ __device__ bool TLASNode::isLeaf()
{
    return (nodeLeft == 0) && (nodeRight == 0);
}

__host__ __device__ TLAS::TLAS()
: nodes(nullptr)
, nodeCount(0)
, blas(nullptr)
, blasCount(0)
{
}

TLAS::TLAS(BVHInstance *instances, unsigned int count)
: nodes(nullptr)
, nodeCount(0)
, blas(nullptr)
, blasCount(count)
{
	blas = new BVHInstance[count];
	for (int i=0; i<count; ++i)
		blas[i] = instances[i];

	build();
}

__host__ __device__ TLAS::~TLAS()
{
	delete [] nodes;
	delete [] blas;
}

TLAS::TLAS(TLAS &&other) noexcept
: nodes(exchange<TLASNode*>(other.nodes, nullptr))
, nodeCount(other.nodeCount)
, blas(exchange<BVHInstance*>(other.blas, nullptr))
, blasCount(other.blasCount)
{
}

TLAS &TLAS::operator=(TLAS &&other) noexcept
{
	swap(other.nodes, nodes);
	nodeCount = other.nodeCount;
	swap(other.blas, blas);
	blasCount = other.blasCount;
	return *this;
}

__device__ bool TLAS::intersect(const Ray & ray, float tmin, float tmax, Hit & out, BVH * blasList, Mesh * meshList) const
{
	Vec3 rD = {1.0f/ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z};
	TLASNode * node = &nodes[0], *stack[64];
	unsigned int stackPtr = 0;

	bool intersects = false;
	while (true)
	{
		if (node->isLeaf())
		{
			// current node is a leaf: intersect BLAS
			Hit tmp;
			bool hitBlas = blas[node->BLAS].intersect(ray, tmin, tmax, tmp, blasList, meshList);
			if (hitBlas && (!intersects || tmp.t < out.t))
			{
				intersects = true;
				out = tmp;
			}
			// pop a node from the stack; terminate if none left
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
			continue;
		}
		// current node is an interior node: visit child nodes, ordered
		TLASNode * child1 = &nodes[node->nodeLeft];
		TLASNode * child2 = &nodes[node->nodeRight];
		float dist1 = 1e30f, dist2 = 1e30f;
		bool hit1 = intersect_aabb(ray, child1->aabb, rD, dist1, tmin, tmax);
		bool hit2 = intersect_aabb(ray, child2->aabb, rD, dist2, tmin, tmax);
		if (dist1 > dist2)
		{
			swap(dist1, dist2);
			swap(child1, child2);
			swap(hit1, hit2);
		}
		if (!hit1)
		{
			// missed both child nodes; pop a node from the stack
			if (stackPtr == 0)
				break;
			else
				node = stack[--stackPtr];
		}
		else
		{
			// visit near node; push the far node if the ray intersects it
			node = child1;
			if (hit2)
				stack[stackPtr++] = child2;
		}
	}
	return intersects;
}

int TLAS::findBestMatch(int * list, int N, int A) const
{
	// find BLAS B that, when joined with A, forms the smallest AABB
	float smallest = 1e30f;
	int bestB = -1;
	for (int B = 0; B < N; B++) if (B != A)
	{
		Vec3 bmax = max_comp(nodes[list[A]].aabb.max, nodes[list[B]].aabb.max);
		Vec3 bmin = min_comp(nodes[list[A]].aabb.min, nodes[list[B]].aabb.min);
		Vec3 e = bmax - bmin;
		float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
		if (surfaceArea < smallest)
		{
			smallest = surfaceArea;
			bestB = B;
		}
	}
	return bestB;
}

void TLAS::build()
{
	// assign a TLAS leaf node to each BLAS
	nodeCount = 1;
	int nodeIds[256];
	int nodeIndices = blasCount;
	for (unsigned int i = 0; i < blasCount; i++)
	{
		nodeIds[i] = nodeCount;
		nodes[nodeCount].aabb = blas[i].bounds;
		nodes[nodeCount].BLAS = i;

		// leaf
		nodes[nodeCount++].nodeLeft = 0;
		nodes[nodeCount++].nodeRight = 0;
	}

	// use agglomerative clustering to build the TLAS
	int A = 0, B = findBestMatch(nodeIds, nodeIndices, A);
	while (nodeIndices > 1)
	{
		int C = findBestMatch(nodeIds, nodeIndices, B);
		if (A == C)
		{
			int nodeIdxA = nodeIds[A], nodeIdxB = nodeIds[B];
			TLASNode& nodeA = nodes[nodeIdxA];
			TLASNode& nodeB = nodes[nodeIdxB];
			TLASNode& newNode = nodes[nodeCount];
			newNode.nodeLeft = nodeIdxB;
			newNode.nodeRight = nodeIdxA;
			newNode.aabb.min = min_comp(nodeA.aabb.min, nodeB.aabb.min);
			newNode.aabb.max = max_comp(nodeA.aabb.max, nodeB.aabb.max);
			nodeIds[A] = nodeCount++;
			nodeIds[B] = nodeIds[nodeIndices - 1];
			B = findBestMatch(nodeIds, --nodeIndices, A);
		}
		else A = B, B = C;
	}
	// copy last remaining node to the root node
	nodes[0] = nodes[nodeIds[A]];
}


__host__ BVHInstance * TLAS::createBVHInstanceList(TLAS * d_tlas, unsigned int count)
{
	// alloc pointer to the bvh list
	BVHInstance ** ptr;
	errchk(cudaMalloc(&ptr, sizeof(BVHInstance**)));

	// instantiate list
	d_createBVHInstanceList<<<1, 1>>>(d_tlas, count, ptr);
	syncAndCheckErrors();

	// copy result
	BVHInstance * res = nullptr;
	errchk(cudaMemcpy(&res, ptr, sizeof(BVHInstance*), cudaMemcpyDeviceToHost));

	errchk(cudaFree(ptr));
	return res;
}

__global__ static void d_createBVHInstanceList(TLAS * d_tlas, unsigned int count, BVHInstance ** out)
{
	delete [] d_tlas->blas;
	d_tlas->blas = new BVHInstance[count];
	*out = d_tlas->blas;
}

__host__ TLASNode * TLAS::createTLASNodeList(TLAS * d_tlas, unsigned int count)
{
	// alloc pointer to the bvh list
	TLASNode ** ptr;
	errchk(cudaMalloc(&ptr, sizeof(TLASNode**)));

	// instantiate list
	d_createTLASNodeList<<<1, 1>>>(d_tlas, count, ptr);
	syncAndCheckErrors();

	// copy result
	TLASNode * res = nullptr;
	errchk(cudaMemcpy(&res, ptr, sizeof(TLASNode*), cudaMemcpyDeviceToHost));

	errchk(cudaFree(ptr));
	return res;
}

__global__ static void d_createTLASNodeList(TLAS * d_tlas, unsigned int count, TLASNode ** out)
{
	delete [] d_tlas->nodes;
	d_tlas->nodes = new TLASNode[count];
	*out = d_tlas->nodes;
}