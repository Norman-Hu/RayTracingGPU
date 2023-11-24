#include <Vec3.cuh>
#include <cassert>


__host__ __device__ Vec3::Vec3()
		: x(0.0f), y(0.0f), z(0.0f)
{
};

__host__ __device__ Vec3::Vec3(float _x, float _y, float _z)
		: x(_x), y(_y), z(_z)
{
};

__host__ __device__ Vec3::Vec3(const Vec3 & other)
		: x(other.x), y(other.y), z(other.z)
{
}

__host__ __device__ float Vec3::sqLength() const {return x*x+y*y+z*z;}
__host__ __device__ float Vec3::length() const {return std::sqrt(sqLength());};

__host__ __device__ void Vec3::normalize()
{
	float l = length();
	x/=l;
	y/=l;
	z/=l;
}

__host__ __device__ Vec3 Vec3::normalized() const
{
	Vec3 other(*this);
	other.normalize();
	return other;
}

__host__ __device__ float & Vec3::operator[](unsigned int i)
{
	switch(i)
	{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			assert(false && "Vec4: Index out of bounds.");
	}
}

__host__ __device__ const float & Vec3::operator[](unsigned int i) const
{
	switch(i)
	{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		default:
			assert(false && "Vec4: Index out of bounds.");
	}
}

__host__ __device__ Vec3 Vec3::mulComp(const Vec3 & other) const
{
	return {x*other.x, y*other.y, z*other.z};
}

// static
__host__ __device__ float Vec3::dot(const Vec3 & a, const Vec3 & b)
{
	return a.x*b.x+a.y*b.y+a.z*b.z;
}
__host__ __device__ Vec3 Vec3::cross(const Vec3 & a, const Vec3 & b)
{
	return {
			a.y*b.z-a.z*b.y,
			a.z*b.x-a.x*b.z,
			a.x*b.y-a.y*b.x
	};
}

__host__ __device__ Vec3 Vec3::reflect(const Vec3 & incident, const Vec3 & normal)
{
	return incident - 2.0f * dot(normal, incident) * normal;
}

__host__ __device__ Vec3 Vec3::refract(const Vec3 & incident, const Vec3 & normal, float ratio)
{
	auto cos_theta = fmin(dot(-incident, normal), 1.0f);
	Vec3 r_out_perp = ratio * (incident + cos_theta * normal);
	Vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.sqLength())) * normal;
	return r_out_perp + r_out_parallel;
}

__host__ __device__ static Vec3 mix(const Vec3 & v1, const Vec3 & v2, float val)
{
	return v1*(1-val) + v2*val;
}

// operators
__host__ __device__ Vec3 & Vec3::operator+=(const Vec3 & other)
{
	x += other.x;
	y += other.y;
	z += other.z;
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator-=(const Vec3 & other)
{
	x -= other.x;
	y -= other.y;
	z -= other.z;
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator*=(float a)
{
	x *= a;
	y *= a;
	z *= a;
	return *this;
}

__host__ __device__ Vec3 & Vec3::operator/=(float a)
{
	x /= a;
	y /= a;
	z /= a;
	return *this;
}



__host__ __device__ Vec3 operator-(const Vec3 & a)
{
	return {-a.x, -a.y, -a.z};
}

__host__ __device__ Vec3 operator+(const Vec3 & a, const Vec3 & b)
{
	return {a.x+b.x, a.y+b.y, a.z+b.z};
}

__host__ __device__ Vec3 operator-(const Vec3 & a, const Vec3 & b)
{
	return {a.x-b.x, a.y-b.y, a.z-b.z};
}

__host__ __device__ Vec3 operator*(float a, const Vec3 & v)
{
	return {a*v.x, a*v.y, a*v.z};
}

__host__ __device__ Vec3 operator*(const Vec3 & v, float a)
{
	return {a*v.x, a*v.y, a*v.z};
}

__host__ __device__ Vec3 operator/(const Vec3 & v, float a)
{
	return {v.x/a, v.y/a, v.z/a};
}
