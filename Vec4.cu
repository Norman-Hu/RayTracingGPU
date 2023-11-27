#include <Vec4.cuh>
#include <cassert>

__host__ __device__ Vec4::Vec4()
		: x(0.0f), y(0.0f), z(0.0f), w(0.0f)
{
};

__host__ __device__ Vec4::Vec4(float _x, float _y, float _z, float _w)
		: x(_x), y(_y), z(_z), w(_w)
{
};

__host__ __device__ Vec4::Vec4(const Vec4 & other)
		: x(other.x), y(other.y), z(other.z), w(other.w)
{
}

__host__ __device__ float Vec4::sqLength() const {return x*x+y*y+z*z+w*w;}
__host__ __device__ float Vec4::length() const {return sqrtf(sqLength());};

__host__ __device__ void Vec4::normalize()
{
	float l = length();
	x/=l;
	y/=l;
	z/=l;
	w/=l;
}

__host__ __device__ Vec4 Vec4::normalized() const
{
	Vec4 other(*this);
	other.normalize();
	return other;
}

__host__ __device__ float & Vec4::operator[](unsigned int i)
{
	switch(i)
	{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		default:
			assert(false && "Vec4: Index out of bounds.");
	}
}

__host__ __device__ const float & Vec4::operator[](unsigned int i) const
{
	switch(i)
	{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		case 3:
			return w;
		default:
			assert(false && "Vec4: Index out of bounds.");
	}
}

// static
__host__ __device__ float Vec4::dot(const Vec4 & a, const Vec4 & b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
}

// operators
__host__ __device__ Vec4 & Vec4::operator+=(const Vec4 & other)
{
	x += other.x;
	y += other.y;
	z += other.z;
	w += other.w;
	return *this;
}

__host__ __device__ Vec4 & Vec4::operator-=(const Vec4 & other)
{
	x -= other.x;
	y -= other.y;
	z -= other.z;
	w -= other.w;
	return *this;
}

__host__ __device__ Vec4 & Vec4::operator*=(float a)
{
	x *= a;
	y *= a;
	z *= a;
	w *= a;
	return *this;
}

__host__ __device__ Vec4 & Vec4::operator/=(float a)
{
	x /= a;
	y /= a;
	z /= a;
	w /= a;
	return *this;
}



__host__ __device__ Vec4 operator-(const Vec4 & a)
{
	return {-a.x, -a.y, -a.z, -a.w};
}

__host__ __device__ Vec4 operator+(const Vec4 & a, const Vec4 & b)
{
	return {a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}

__host__ __device__ Vec4 operator-(const Vec4 & a, const Vec4 & b)
{
	return {a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w};
}

__host__ __device__ Vec4 operator*(float a, const Vec4 & v)
{
	return {a*v.x, a*v.y, a*v.z, a*v.w};
}

__host__ __device__ Vec4 operator*(const Vec4 & v, float a)
{
	return {a*v.x, a*v.y, a*v.z, a*v.w};
}

__host__ __device__ Vec4 operator/(const Vec4 & v, float a)
{
	return {v.x/a, v.y/a, v.z/a, v.w/a};
}