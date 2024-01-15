#include <Object.cuh>
#include <CudaHelpers.h>


__device__ bool Sphere::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	Vec3 centerToOrigin = ray.origin - c;
	float a = ray.direction.sqLength();
	float b = Vec3::dot(centerToOrigin, ray.direction);
	float c = centerToOrigin.sqLength() - r*r;

	float delta = b*b - a*c;
	if (delta < 0)
	{
		return false;
	}
	float sqrtDelta = sqrtf(delta);

	float root = (-b-sqrtDelta/a);
	if (root < tmin || root > tmax)
	{
		root = (-b+sqrtDelta)/a;
		if (root < tmin || root > tmax)
			return false;
	}

	out.t = root;
	out.p = ray.origin+ray.direction*root;
	out.normal = (out.p - this->c)/r;
	out.materialId = materialId;
	return true;
}

__device__ bool Square::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
    if (Vec3::dot(ray.direction, n) > 0.f)
        return false;

	float t = Vec3::dot(p - ray.origin, n)/Vec3::dot(ray.direction, n);
	if (t < tmin || t > tmax)
		return false;

	Vec3 inter = ray.origin + t * ray.direction;

	Vec3 toIntersection = inter - p;
	float dotRight = Vec3::dot(toIntersection, right);
	float dotUp = Vec3::dot(toIntersection, up);

	if (!(dotRight >= 0 && dotRight <= Vec3::dot(right, right) && dotUp >= 0 && dotUp <= Vec3::dot(up, up)))
		return false;

	out.t = t;
	out.p = inter;
	out.normal = n;
	out.materialId = materialId;

	return true;
}

__host__ __device__ bool rayAABB(const Ray & ray, const AABB & aabb, float tmin, float tmax)
{
	Vec3 invD = {1.0f/ray.direction.x, 1.0f/ray.direction.y, 1.0f/ray.direction.z};
	Vec3 t0s = compwise_mul((aabb.min - ray.origin), invD);
	Vec3 t1s = compwise_mul((aabb.max - ray.origin), invD);

	Vec3 tsmaller = compwise_min(t0s, t1s);
	Vec3 tbigger  = compwise_max(t0s, t1s);

	tmin = fmaxf(tmin, fmaxf(tsmaller[0], fmaxf(tsmaller[1], tsmaller[2])));
	tmax = fminf(tmax, fminf(tbigger[0], fminf(tbigger[1], tbigger[2])));

	return (tmin < tmax);
}

__device__ bool Mesh::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
    bool intersects = false;
    if (indices_count > 18 && !rayAABB(ray, aabb, tmin, tmax)) return false;
    for (unsigned int i = 0; i < indices_count; i += 3)
    {
        // Möller–Trumbore intersection algorithm
        // From https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        constexpr float EPSILON = 1e-7f;
        Vec3 v0 = vertices[indices[i]], v1 = vertices[indices[i + 1]], v2 = vertices[indices[i + 2]];

//        Vec3 centroid = (v0+v1+v2)*0.33333f;
//        float maxSqDist = fmaxf(fmaxf((v0 - centroid).sqLength(), (v1 - centroid).sqLength()), (v2 - centroid).sqLength());
//        Vec3 oc = ray.origin - centroid;
//        float b = Vec3::dot(oc, ray.direction);
//        float c = Vec3::dot(oc, oc) - maxSqDist;
//        float h = b*b - c;
//        if(h < 0.0f)
//            continue;

        Vec3 v1v0 = v1 - v0;
        Vec3 v2v0 = v2 - v0;
        Vec3 rov0 = ray.origin - v0;

        Vec3  n = Vec3::cross(v1v0, v2v0);
        Vec3  q = Vec3::cross(rov0, ray.direction);
        float d = 1.0f/Vec3::dot(ray.direction, n);
        float u = d*Vec3::dot(-q, v2v0);
        float v = d*Vec3::dot( q, v1v0);
        float t = d*Vec3::dot(-n, rov0);

        if (u < 0.0f || v < 0.0f || (u+v) > 1.0f || t < tmin || t > tmax)
            continue;

        if (!intersects || t < out.t)
        {
            out.t = t;
            out.p = ray.origin + t * ray.direction;
            out.normal = -Vec3::cross(v2v0, v1v0); // FIXME: Interpolate normals using barycentric coordinates
            out.materialId = materialId;
            intersects = true;
        }
    }
    return intersects;
}

/*
 * bool intersects = false;
	if (!rayAABB(ray, aabb, tmin, tmax)) return false;
	for (unsigned int i = 0; i < indices_count; i += 3)
	{
		// Möller–Trumbore intersection algorithm
		// From https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
		constexpr float EPSILON = 1e-7f;
		Vec3 v0 = vertices[indices[i]], v1 = vertices[indices[i + 1]], v2 = vertices[indices[i + 2]];

//        Vec3 centroid = (v0+v1+v2)*0.33333f;
//        float maxSqDist = fmaxf(fmaxf((v0 - centroid).sqLength(), (v1 - centroid).sqLength()), (v2 - centroid).sqLength());
//        Vec3 oc = ray.origin - centroid;
//        float b = Vec3::dot(oc, ray.direction);
//        float c = Vec3::dot(oc, oc) - maxSqDist;
//        float h = b*b - c;
//        if(h < 0.0f)
//            continue;

        Vec3 v1v0 = v1 - v0;
        Vec3 v2v0 = v2 - v0;
        Vec3 rov0 = ray.origin - v0;

        Vec3  n = Vec3::cross(v1v0, v2v0);
        Vec3  q = Vec3::cross(rov0, ray.direction);
        float d = 1.0f/Vec3::dot(ray.direction, n);
        float u = d*Vec3::dot(-q, v2v0);
        float v = d*Vec3::dot( q, v1v0);
        float t = d*Vec3::dot(-n, rov0);

        if (u < 0.0f || v < 0.0f || (u+v) > 1.0f)
            continue;

		if (!intersects || t < out.t)
		{
			out.t = t;
			out.p = ray.origin + t * ray.direction;
			out.normal = Vec3::cross(v2v0, v1v0); // FIXME: Interpolate normals using barycentric coordinates
			out.materialId = materialId;
			intersects = true;
		}
	}
	return intersects;*/

__device__ Mesh::Mesh()
: vertices(nullptr)
, normals(nullptr)
, indices(nullptr)
, vertices_count(0)
, normals_count(0)
, indices_count(0)
{
}

__device__ Mesh::~Mesh()
{
	delete[] vertices;
	delete[] normals;
	delete[] indices;
}

AABB computeAABB(Vec3 * vertices, unsigned int count)
{
	AABB res {vertices[0], vertices[0]};
	for (int i=1; i<count; ++i)
	{
		res.min = compwise_min(res.min, vertices[i]);
		res.max = compwise_max(res.max, vertices[i]);
	}
	return res;
}

Mesh * createMesh()
{
	Mesh * res;
	Mesh ** ptr_d_Mesh;
	cudaMalloc(&ptr_d_Mesh, sizeof(Mesh *));
	d_createMesh<<<1, 1>>>(ptr_d_Mesh);
	cudaMemcpy(&res, ptr_d_Mesh, sizeof(Mesh *), cudaMemcpyDeviceToHost);
	cudaFree(ptr_d_Mesh);
	return res;
}

__global__ void d_createMesh(Mesh ** ptr_d_mesh)
{
	Mesh * mesh = new Mesh();
	*ptr_d_mesh = mesh;
}

void setMeshAABB(Mesh * d_mesh, const AABB & bounds)
{
	d_setMeshAABB<<<1, 1>>>(d_mesh, bounds);
}

__global__ void d_setMeshAABB(Mesh * d_mesh, AABB bounds)
{
	d_mesh->aabb = bounds;
}

void setMeshMaterial(Mesh * mesh, unsigned int index)
{
	d_setMeshMaterial<<<1, 1>>>(mesh, index);
}

__global__ void d_setMeshMaterial(Mesh * mesh, unsigned int index)
{
	mesh->materialId = index;
}

void setMeshVertices(Mesh * mesh, Vec3 * vertices, unsigned int verticesCount)
{
	d_setMeshVerticesCount<<<1, 1>>>(mesh, verticesCount);
	syncAndCheckErrors();
	Vec3 * d_vertices;
	cudaMalloc(&d_vertices, sizeof(Vec3)*verticesCount);
	cudaMemcpy(d_vertices, vertices, sizeof(Vec3)*verticesCount, cudaMemcpyHostToDevice);
	d_copyVertices<<<1, 1>>>(mesh, d_vertices, verticesCount);
	syncAndCheckErrors();
	cudaFree(d_vertices);
}

__global__ void d_copyVertices(Mesh * mesh, Vec3 * d_vertices, unsigned int verticesCount)
{
	memcpy((void*)mesh->vertices, (void*)d_vertices, sizeof(Vec3)*verticesCount);
}

__global__ void d_setMeshVerticesCount(Mesh * mesh, unsigned int verticesCount)
{
	delete [] mesh->vertices;
	mesh->vertices = new Vec3[verticesCount];
	mesh->vertices_count = verticesCount;
}

void setMeshIndices(Mesh * mesh, unsigned int * indices, unsigned int indicesCount)
{
	d_setMeshIndicesCount<<<1, 1>>>(mesh, indicesCount);
	syncAndCheckErrors();
	unsigned int * d_indices;
	cudaMalloc(&d_indices, sizeof(unsigned int)*indicesCount);
	cudaMemcpy(d_indices, indices, sizeof(unsigned int)*indicesCount, cudaMemcpyHostToDevice);
	d_copyIndices<<<1, 1>>>(mesh, d_indices, indicesCount);
	syncAndCheckErrors();
	cudaFree(d_indices);
}

__global__ void d_copyIndices(Mesh * mesh, unsigned int * d_indices, unsigned int indicesCount)
{
	memcpy((void*)mesh->indices, (void*)d_indices, sizeof(unsigned int)*indicesCount);
}

__global__ void d_setMeshIndicesCount(Mesh * mesh, unsigned int indicesCount)
{
	delete [] mesh->indices;
	mesh->indices = new unsigned int[indicesCount];
	mesh->indices_count = indicesCount;
}

void setMeshNormals(Mesh * mesh, Vec3 * normals, unsigned int normalsCount)
{
	d_setMeshNormalsCount<<<1, 1>>>(mesh, normalsCount);
	syncAndCheckErrors();
	Vec3 * d_normals;
	cudaMalloc(&d_normals, sizeof(Vec3)*normalsCount);
	cudaMemcpy(d_normals, normals, sizeof(Vec3)*normalsCount, cudaMemcpyHostToDevice);
	d_copyNormals<<<1, 1>>>(mesh, d_normals, normalsCount);
	syncAndCheckErrors();
	cudaFree(d_normals);
}

__global__ void d_copyNormals(Mesh * mesh, Vec3 * d_normals, unsigned int normalsCount)
{
	memcpy((void*)mesh->normals, (void*)d_normals, sizeof(Vec3)*normalsCount);
}

__global__ void d_setMeshNormalsCount(Mesh * mesh, unsigned int normalsCount)
{
	delete [] mesh->normals;
	mesh->normals = new Vec3[normalsCount];
	mesh->normals_count = normalsCount;
}
