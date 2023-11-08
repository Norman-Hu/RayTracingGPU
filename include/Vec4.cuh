#ifndef VEC4_H
#define VEC4_H


#include <cmath>


class Vec4
{
public:
	__host__ __device__ Vec4();
	__host__ __device__ Vec4(float _x, float _y, float _z, float _w);
	__host__ __device__ Vec4(const Vec4 & other);

	__host__ __device__ ~Vec4() = default;

	__host__ __device__ float sqLength() const;
	__host__ __device__ float length() const;

	__host__ __device__ void normalize();
	__host__ __device__ Vec4 normalized() const;

	__host__ __device__ float & operator[](unsigned int i);
	__host__ __device__ const float & operator[](unsigned int i) const;

public:
	float x, y, z, w;

// static
public:
	__host__ __device__ static float dot(const Vec4 & a, const Vec4 & b);

// operators
public:
	__host__ __device__ Vec4 & operator+=(const Vec4 & other);
	__host__ __device__ Vec4 & operator-=(const Vec4 & other);
	__host__ __device__ Vec4 & operator*=(float a);
	__host__ __device__ Vec4 & operator/=(float a);
};

__host__ __device__ Vec4 operator+(const Vec4 & a, const Vec4 & b);
__host__ __device__ Vec4 operator-(const Vec4 & a, const Vec4 & b);
__host__ __device__ Vec4 operator*(float a, const Vec4 & v);
__host__ __device__ Vec4 operator*(const Vec4 & v, float a);
__host__ __device__ Vec4 operator/(const Vec4 & v, float a);


#endif // VEC4_H
