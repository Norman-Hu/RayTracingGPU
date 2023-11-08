#ifndef MATRIX_H
#define MATRIX_H


#include <array>
#include <Vec3.cuh>
#include <Vec4.cuh>

class Matrix4x4
{
public:
	__host__ __device__ Matrix4x4();

	__host__ __device__ Matrix4x4(const Matrix4x4 & other);

	__host__ __device__ ~Matrix4x4() = default;

	__host__ __device__ const Vec4 & operator[](unsigned int col) const;
	__host__ __device__ Vec4 & operator[](unsigned int col);

	__host__ __device__ const float * data() const;

	__host__ __device__ float * data();

public:
	Vec4 cols[4];

// static
public:
	__host__ __device__ static Matrix4x4 lookAt(const Vec3 & eye, const Vec3 & center, const Vec3 & up);

	__host__ __device__ static Matrix4x4 perspective(float fovy, float aspect, float zNear, float zFar);
	__host__ __device__ static bool invertMatrix(const Matrix4x4 & mat, Matrix4x4 & invOut);
};

__host__ __device__ Vec4 operator*(const Matrix4x4 & m, const Vec4 & v);
__host__ __device__ Vec4 operator*(const Vec4 & v, const Matrix4x4 & m);
__host__ __device__ Matrix4x4 operator*(const Matrix4x4 & m1, const Matrix4x4 & m2);


#endif // MATRIX_H
