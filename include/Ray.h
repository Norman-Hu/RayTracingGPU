#ifndef RAY_H
#define RAY_H

#include <Vec3.h>

class Ray
{
public:
	inline Ray(const Vec3 & _origin, const Vec3 & _direction)
	: origin(_origin), direction(_direction)
	{
	}

public:
	Vec3 origin, direction;
};

#endif // RAY_H