#ifndef RAY_H
#define RAY_H

#include <Vec3.cuh>

struct Ray
{
	Vec3 origin;
	Vec3 direction;
};

#endif // RAY_H