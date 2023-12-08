#include <BVH.cuh>
#include <Memory.cuh>
#include <CudaHelpers.h>


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
	build();
}

BVH::~BVH()
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

void BVH::build()
{
	nodes = new BVHNode[primitive_count*2-1];
	nodeCount = 2;
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

__host__ __device__ TLAS::TLAS()
: nodes(nullptr)
, nodeCount(0)
, blas(nullptr)
, blasCount(0)
, nodeIds(nullptr)
{
}

TLAS::TLAS(BVHInstance *instances, unsigned int count)
: nodes(nullptr)
, nodeCount(0)
, blas(nullptr)
, blasCount(0)
, nodeIds(nullptr)
{
	blas = new BVHInstance[count];
	for (int i=0; i<count; ++i)
		blas[i] = instances[i];
}

TLAS::~TLAS()
{
	delete [] nodes;
	delete [] blas;
	delete [] nodeIds;
}

TLAS::TLAS(TLAS &&other) noexcept
: nodes(exchange<TLASNode*>(other.nodes, nullptr))
, nodeCount(other.nodeCount)
, blas(exchange<BVHInstance*>(other.blas, nullptr))
, blasCount(other.blasCount)
, nodeIds(exchange<unsigned int*>(other.nodeIds, nullptr))
{
}

TLAS &TLAS::operator=(TLAS &&other) noexcept
{
	swap(other.nodes, nodes);
	nodeCount = other.nodeCount;
	swap(other.blas, blas);
	blasCount = other.blasCount;
	swap(other.nodeIds, nodeIds);
}

void TLAS::build()
{
	// TODO
}
