#include <Object.cuh>

__device__ bool Sphere::hit(const Ray & ray, float tmin, float tmax, Hit & out)
{
	Vec3 centerToOrigin = ray.origin - c;
	float a = ray.direction.sqLength();
	float halfb = Vec3::dot(centerToOrigin, ray.direction);
	float c = centerToOrigin.sqLength() - r*r;

	float delta = halfb*halfb - 4*a*c;
	if (delta < 0)
		return false;
	float sqrtDelta = sqrt(delta);

	float root = (-halfb-sqrtDelta/a);
	if (root < tmin || root > tmax)
	{
		root = (-halfb+sqrtDelta)/a;
		if (root < tmin || root > tmax)
			return false;
	}

	out.t = root;
	out.p = ray.origin+ray.direction*root;
	out.normal = (out.p - this->c)/r;
	return true;
}