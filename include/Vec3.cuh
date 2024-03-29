#ifndef VEC3_H
#define VEC3_H


#include <cmath>


class Vec3
{
public:
	__host__ __device__ Vec3();

	__host__ __device__ Vec3(float _x, float _y, float _z);

	__host__ __device__ Vec3(const Vec3 & other);

	__host__ __device__ ~Vec3() = default;

	__host__ __device__ float sqLength() const;
	__host__ __device__ float length() const;

	__host__ __device__ void normalize();

	__host__ __device__ Vec3 normalized() const;

	__host__ __device__ float & operator[](unsigned int i);
	__host__ __device__ const float & operator[](unsigned int i) const;

    __host__ __device__ float max() const;
    __host__ __device__ float min() const;

	__host__ __device__ Vec3 mulComp(const Vec3 & other) const;
public:
	float x, y, z;

// static
public:
	__host__ __device__ static float dot(const Vec3 & a, const Vec3 & b);
	__host__ __device__ static Vec3 cross(const Vec3 & a, const Vec3 & b);
	__host__ __device__ static Vec3 reflect(const Vec3 & incident, const Vec3 & normal);
	__host__ __device__ static Vec3 refract(const Vec3 & incident, const Vec3 & normal, float ratio);
	__host__ __device__ static Vec3 mix(const Vec3 & v1, const Vec3 & v2, float val);

// operators
public:
	__host__ __device__ Vec3 & operator+=(const Vec3 & other);
	__host__ __device__ Vec3 & operator-=(const Vec3 & other);
	__host__ __device__ Vec3 & operator*=(float a);
	__host__ __device__ Vec3 & operator/=(float a);
};

__host__ __device__ Vec3 operator-(const Vec3 & a);
__host__ __device__ Vec3 operator+(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 operator-(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 operator*(float a, const Vec3 & v);
__host__ __device__ Vec3 operator*(const Vec3 & v, float a);
__host__ __device__ Vec3 operator/(const Vec3 & v, float a);

__host__ __device__ Vec3 compwise_max(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 compwise_min(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 compwise_mul(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 compwise_div(const Vec3 & a, const Vec3 & b);
__host__ __device__ Vec3 compwise_pow(const Vec3 & a, const Vec3 & b);


#endif // VEC3_H
