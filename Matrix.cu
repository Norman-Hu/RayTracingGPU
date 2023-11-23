#include <Matrix.cuh>


__host__ __device__ Matrix4x4::Matrix4x4()
		: cols()
{
}

__host__ __device__ Matrix4x4::Matrix4x4(const Matrix4x4 & other)
		: cols()
{
	for (int i=0; i<4; ++i)
		cols[i] = other.cols[i];
}

__host__ __device__ const Vec4 & Matrix4x4::operator[](unsigned int col) const
{
	return cols[col];
}

__host__ __device__ Vec4 & Matrix4x4::operator[](unsigned int col)
{
	return cols[col];
}

__host__ __device__ const float * Matrix4x4::data() const
{
	return reinterpret_cast<const float *>(&cols[0][0]);
}

__host__ __device__ float * Matrix4x4::data()
{
	return reinterpret_cast<float *>(&cols[0][0]);
}

// static

__host__ __device__ Matrix4x4 Matrix4x4::lookAt(const Vec3 & eye, const Vec3 & center, const Vec3 & up)
{
	const Vec3 f((center - eye).normalized());
	const Vec3 s(Vec3::cross(f, up).normalized());
	const Vec3 u(Vec3::cross(s, f));

	Matrix4x4 res;
	res[3][3] = 1.0f; // identity

	res[0][0] = s.x;
	res[1][0] = s.y;
	res[2][0] = s.z;
	res[0][1] = u.x;
	res[1][1] = u.y;
	res[2][1] = u.z;
	res[0][2] = -f.x;
	res[1][2] = -f.y;
	res[2][2] = -f.z;
	res[3][0] = -Vec3::dot(s, eye);
	res[3][1] = -Vec3::dot(u, eye);
	res[3][2] = Vec3::dot(f, eye);

	return res;
}

__host__ __device__ Matrix4x4 Matrix4x4::perspective(float fovy, float aspect, float zNear, float zFar)
{
	float const tanHalfFovy = std::tan(fovy / 2.0f);
	Matrix4x4 res;
	res[0][0] = 1.0f / (aspect * tanHalfFovy);
	res[1][1] = 1.0f / (tanHalfFovy);
	res[2][2] = - (zFar + zNear) / (zFar - zNear);
	res[2][3] = - 1.0f;
	res[3][2] = - (2.0f * zFar * zNear) / (zFar - zNear);
	return res;
}


__host__ __device__ bool Matrix4x4::invertMatrix(const Matrix4x4 & mat, Matrix4x4 & invOut)
{
	float det;
	Matrix4x4 inv;
	const float * m = mat.data();

	inv.data()[0] = m[5]  * m[10] * m[15] -
					m[5]  * m[11] * m[14] -
					m[9]  * m[6]  * m[15] +
					m[9]  * m[7]  * m[14] +
					m[13] * m[6]  * m[11] -
					m[13] * m[7]  * m[10];

	inv.data()[4] = -m[4]  * m[10] * m[15] +
					m[4]  * m[11] * m[14] +
					m[8]  * m[6]  * m[15] -
					m[8]  * m[7]  * m[14] -
					m[12] * m[6]  * m[11] +
					m[12] * m[7]  * m[10];

	inv.data()[8] = m[4]  * m[9] * m[15] -
					m[4]  * m[11] * m[13] -
					m[8]  * m[5] * m[15] +
					m[8]  * m[7] * m[13] +
					m[12] * m[5] * m[11] -
					m[12] * m[7] * m[9];

	inv.data()[12] = -m[4]  * m[9] * m[14] +
					 m[4]  * m[10] * m[13] +
					 m[8]  * m[5] * m[14] -
					 m[8]  * m[6] * m[13] -
					 m[12] * m[5] * m[10] +
					 m[12] * m[6] * m[9];

	inv.data()[1] = -m[1]  * m[10] * m[15] +
					m[1]  * m[11] * m[14] +
					m[9]  * m[2] * m[15] -
					m[9]  * m[3] * m[14] -
					m[13] * m[2] * m[11] +
					m[13] * m[3] * m[10];

	inv.data()[5] = m[0]  * m[10] * m[15] -
					m[0]  * m[11] * m[14] -
					m[8]  * m[2] * m[15] +
					m[8]  * m[3] * m[14] +
					m[12] * m[2] * m[11] -
					m[12] * m[3] * m[10];

	inv.data()[9] = -m[0]  * m[9] * m[15] +
					m[0]  * m[11] * m[13] +
					m[8]  * m[1] * m[15] -
					m[8]  * m[3] * m[13] -
					m[12] * m[1] * m[11] +
					m[12] * m[3] * m[9];

	inv.data()[13] = m[0]  * m[9] * m[14] -
					 m[0]  * m[10] * m[13] -
					 m[8]  * m[1] * m[14] +
					 m[8]  * m[2] * m[13] +
					 m[12] * m[1] * m[10] -
					 m[12] * m[2] * m[9];

	inv.data()[2] = m[1]  * m[6] * m[15] -
					m[1]  * m[7] * m[14] -
					m[5]  * m[2] * m[15] +
					m[5]  * m[3] * m[14] +
					m[13] * m[2] * m[7] -
					m[13] * m[3] * m[6];

	inv.data()[6] = -m[0]  * m[6] * m[15] +
					m[0]  * m[7] * m[14] +
					m[4]  * m[2] * m[15] -
					m[4]  * m[3] * m[14] -
					m[12] * m[2] * m[7] +
					m[12] * m[3] * m[6];

	inv.data()[10] = m[0]  * m[5] * m[15] -
					 m[0]  * m[7] * m[13] -
					 m[4]  * m[1] * m[15] +
					 m[4]  * m[3] * m[13] +
					 m[12] * m[1] * m[7] -
					 m[12] * m[3] * m[5];

	inv.data()[14] = -m[0]  * m[5] * m[14] +
					 m[0]  * m[6] * m[13] +
					 m[4]  * m[1] * m[14] -
					 m[4]  * m[2] * m[13] -
					 m[12] * m[1] * m[6] +
					 m[12] * m[2] * m[5];

	inv.data()[3] = -m[1] * m[6] * m[11] +
					m[1] * m[7] * m[10] +
					m[5] * m[2] * m[11] -
					m[5] * m[3] * m[10] -
					m[9] * m[2] * m[7] +
					m[9] * m[3] * m[6];

	inv.data()[7] = m[0] * m[6] * m[11] -
					m[0] * m[7] * m[10] -
					m[4] * m[2] * m[11] +
					m[4] * m[3] * m[10] +
					m[8] * m[2] * m[7] -
					m[8] * m[3] * m[6];

	inv.data()[11] = -m[0] * m[5] * m[11] +
					 m[0] * m[7] * m[9] +
					 m[4] * m[1] * m[11] -
					 m[4] * m[3] * m[9] -
					 m[8] * m[1] * m[7] +
					 m[8] * m[3] * m[5];

	inv.data()[15] = m[0] * m[5] * m[10] -
					 m[0] * m[6] * m[9] -
					 m[4] * m[1] * m[10] +
					 m[4] * m[2] * m[9] +
					 m[8] * m[1] * m[6] -
					 m[8] * m[2] * m[5];

	det = m[0] * inv[0][0] + m[1] * inv[1][0] + m[2] * inv[2][0] + m[3] * inv[3][0];

	if (det == 0)
		return false;

	det = 1.0f / det;

	for (int i = 0; i < 16; i++)
		invOut.data()[i] = inv.data()[i] * det;

	return true;
}

// operators

__host__ __device__ Vec4 operator*(const Matrix4x4 & m, const Vec4 & v)
{
	Vec4 out;
	out[0] = Vec4::dot(m[0], v);
	out[1] = Vec4::dot(m[1], v);
	out[2] = Vec4::dot(m[2], v);
	out[3] = Vec4::dot(m[3], v);
	return out;
}

__host__ __device__ Vec4 operator*(const Vec4 & v, const Matrix4x4 & m)
{
	Vec4 out;
	out[0] = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + m[0][3] * v[3];
	out[1] = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + m[1][3] * v[3];
	out[2] = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + m[2][3] * v[3];
	out[3] = m[3][0] * v[0] + m[3][1] * v[1] + m[3][2] * v[2] + m[3][3] * v[3];
	return out;
}

__host__ __device__ Matrix4x4 operator*(const Matrix4x4 & m1, const Matrix4x4 & m2)
{
	Matrix4x4 out;

	const Vec4 & A0 = m1[0];
	const Vec4 & A1 = m1[1];
	const Vec4 & A2 = m1[2];
	const Vec4 & A3 = m1[3];

	const Vec4 & B0 = m2[0];
	const Vec4 & B1 = m2[1];
	const Vec4 & B2 = m2[2];
	const Vec4 & B3 = m2[3];

	out[0] = A0 * B0[0] + A1 * B0[1] + A2 * B0[2] + A3 * B0[3];
	out[1] = A0 * B1[0] + A1 * B1[1] + A2 * B1[2] + A3 * B1[3];
	out[2] = A0 * B2[0] + A1 * B2[1] + A2 * B2[2] + A3 * B2[3];
	out[3] = A0 * B3[0] + A1 * B3[1] + A2 * B3[2] + A3 * B3[3];

	return out;
}
