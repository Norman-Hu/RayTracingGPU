#ifndef RADIANS_H
#define RADIANS_H

#include <cmath>


inline float radians(float degrees)
{
	return degrees*(static_cast<float>(M_PI)/180.0f);
}

inline float degrees(float radians)
{
	return radians*(180.0f/static_cast<float>(M_PI));
}

#endif // RADIANS_H
