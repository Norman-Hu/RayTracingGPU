#include <Object.cuh>


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

__device__ bool Mesh::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	for (unsigned int v = 0; v < indices_count; v += 3)
	{
		Vec3 v0 = vertices[indices[v]], v1 = vertices[indices[v + 1]], v2 = vertices[indices[v + 2]];
		Vec3 n = Vec3::cross(v1 - v0, v2 - v0);
		float D = -Vec3::dot(n, v0);

		float dot_N_R = Vec3::dot(n, ray.direction);
		if (dot_N_R == 0.f)
			return false;
		float t = -(Vec3::dot(n, ray.origin) + D) / dot_N_R;
		if (t < 0.f)
			return false;
		Vec3 p = ray.origin + t * ray.direction;

		Vec3 c;

		Vec3 e0 = v1 - v0;
		Vec3 v0p = p - v0;
		c = Vec3::cross(e0, v0p);
		if (Vec3::dot(n, c) < 0.f) continue;

		Vec3 e1 = v2 - v1;
		Vec3 v1p = p - v1;
		c = Vec3::cross(e1, v1p);
		if (Vec3::dot(n, c) < 0.f) continue;

		Vec3 e2 = v0 - v2;
		Vec3 v2p = p - v2;
		c = Vec3::cross(e2, v2p);
		if (Vec3::dot(n, c) < 0.f) continue;

		out.t = t;
		out.p = p;
		out.normal = n;
		out.materialId = materialId;
		return true;
	}
	return false;
}

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
    delete [] vertices;
    delete [] normals;
    delete [] indices;
}
