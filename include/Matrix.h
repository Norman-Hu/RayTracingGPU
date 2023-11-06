#ifndef MATRIX_H
#define MATRIX_H


#include <array>
#include <Vec3.h>


using Matrix4x4 = std::array<float, 16>;
using Vec4 = std::array<float, 4>;

Matrix4x4 lookAt(const Vec3 & eye, const Vec3 & center, const Vec3 & up)
{
	const Vec3 f((center - eye).normalized());
	const Vec3 s((Vec3::cross(f, up)).normalized());
	const Vec3 u(Vec3::cross(s, f));

	Matrix4x4 res;
	res.fill(1.0f);
	res[0 * 4 + 0] = s.x;
	res[1 * 4 + 0] = s.y;
	res[2 * 4 + 0] = s.z;
	res[0 * 4 + 1] = u.x;
	res[1 * 4 + 1] = u.y;
	res[2 * 4 + 1] = u.z;
	res[0 * 4 + 2] =-f.x;
	res[1 * 4 + 2] =-f.y;
	res[2 * 4 + 2] =-f.z;
	res[3 * 4 + 0] =-Vec3::dot(s, eye);
	res[3 * 4 + 1] =-Vec3::dot(u, eye);
	res[3 * 4 + 2] = Vec3::dot(f, eye);
	return res;
}

Matrix4x4 perspective(float fovy, float aspect, float zNear, float zFar)
{
	float const tanHalfFovy = std::tan(fovy / 2.0f);

	Matrix4x4 res;
	res.fill(0.0f);
	res[4 * 0 + 0] = 1.0f / (aspect * tanHalfFovy);
	res[4 * 1 + 1] = 1.0f / (tanHalfFovy);
	res[4 * 2 + 2] = - (zFar + zNear) / (zFar - zNear);
	res[4 * 2 + 3] = - 1.0f;
	res[4 * 3 + 2] = - (2.0f * zFar * zNear) / (zFar - zNear);
	return res;
}

bool invertMatrix(const Matrix4x4 & m, Matrix4x4 & invOut)
{
	float det;
	Matrix4x4 inv;

	inv[0] = m[5]  * m[10] * m[15] -
			 m[5]  * m[11] * m[14] -
			 m[9]  * m[6]  * m[15] +
			 m[9]  * m[7]  * m[14] +
			 m[13] * m[6]  * m[11] -
			 m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] +
			 m[4]  * m[11] * m[14] +
			 m[8]  * m[6]  * m[15] -
			 m[8]  * m[7]  * m[14] -
			 m[12] * m[6]  * m[11] +
			 m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] -
			 m[4]  * m[11] * m[13] -
			 m[8]  * m[5] * m[15] +
			 m[8]  * m[7] * m[13] +
			 m[12] * m[5] * m[11] -
			 m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] +
			  m[4]  * m[10] * m[13] +
			  m[8]  * m[5] * m[14] -
			  m[8]  * m[6] * m[13] -
			  m[12] * m[5] * m[10] +
			  m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] +
			 m[1]  * m[11] * m[14] +
			 m[9]  * m[2] * m[15] -
			 m[9]  * m[3] * m[14] -
			 m[13] * m[2] * m[11] +
			 m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] -
			 m[0]  * m[11] * m[14] -
			 m[8]  * m[2] * m[15] +
			 m[8]  * m[3] * m[14] +
			 m[12] * m[2] * m[11] -
			 m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] +
			 m[0]  * m[11] * m[13] +
			 m[8]  * m[1] * m[15] -
			 m[8]  * m[3] * m[13] -
			 m[12] * m[1] * m[11] +
			 m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] -
			  m[0]  * m[10] * m[13] -
			  m[8]  * m[1] * m[14] +
			  m[8]  * m[2] * m[13] +
			  m[12] * m[1] * m[10] -
			  m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] -
			 m[1]  * m[7] * m[14] -
			 m[5]  * m[2] * m[15] +
			 m[5]  * m[3] * m[14] +
			 m[13] * m[2] * m[7] -
			 m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] +
			 m[0]  * m[7] * m[14] +
			 m[4]  * m[2] * m[15] -
			 m[4]  * m[3] * m[14] -
			 m[12] * m[2] * m[7] +
			 m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] -
			  m[0]  * m[7] * m[13] -
			  m[4]  * m[1] * m[15] +
			  m[4]  * m[3] * m[13] +
			  m[12] * m[1] * m[7] -
			  m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] +
			  m[0]  * m[6] * m[13] +
			  m[4]  * m[1] * m[14] -
			  m[4]  * m[2] * m[13] -
			  m[12] * m[1] * m[6] +
			  m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
			 m[1] * m[7] * m[10] +
			 m[5] * m[2] * m[11] -
			 m[5] * m[3] * m[10] -
			 m[9] * m[2] * m[7] +
			 m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
			 m[0] * m[7] * m[10] -
			 m[4] * m[2] * m[11] +
			 m[4] * m[3] * m[10] +
			 m[8] * m[2] * m[7] -
			 m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
			  m[0] * m[7] * m[9] +
			  m[4] * m[1] * m[11] -
			  m[4] * m[3] * m[9] -
			  m[8] * m[1] * m[7] +
			  m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
			  m[0] * m[6] * m[9] -
			  m[4] * m[1] * m[10] +
			  m[4] * m[2] * m[9] +
			  m[8] * m[1] * m[6] -
			  m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0f / det;

	for (int i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

void multVec4Vec4(float * v1, float * v2, float * out)
{
	*out = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3];
}

void multMat4Vec4(float * m, float * v, float * out)
{
	multVec4Vec4(m+0, v, out+0);
	multVec4Vec4(m+4, v, out+1);
	multVec4Vec4(m+8, v, out+2);
	multVec4Vec4(m+12, v, out+3);
}


// gpu :
//Eigen::Vector4d vec(2.f*u - 1.f , -(2.f*v - 1.f) , nearAndFarPlanes[0] , 1.0);
//Eigen::Vector4d resInt = projectionInverse * vec;
//Eigen::Vector4d res = modelviewInverse * resInt;
//return Vec3( res[0] / res[3] , res[1] / res[3] , res[2] / res[3] );

#endif // MATRIX_H
