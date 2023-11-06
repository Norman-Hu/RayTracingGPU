#ifndef RADIANS_H
#define RADIANS_H


#include <cmath>


inline float radians(float degrees)
{
	return degrees*(M_PIf/180.0f);
}

inline float degrees(float radians)
{
	return radians*(180.0f/M_PIf);
}

#endif // RADIANS_H
