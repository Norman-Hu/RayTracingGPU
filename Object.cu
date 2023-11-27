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
