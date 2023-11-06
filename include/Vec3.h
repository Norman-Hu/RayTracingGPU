#ifndef VEC3_H
#define VEC3_H


#include <cmath>


class Vec3
{
public:
	inline Vec3()
	: x(0.0f), y(0.0f), z(0.0f)
	{
	};

	inline Vec3(float _x, float _y, float _z)
	: x(_x), y(_y), z(_z)
	{
	};

	inline Vec3(const Vec3 & other)
	: x(other.x), y(other.y), z(other.z)
	{
	}

	inline ~Vec3() = default;

	inline float sqLength() const {return x*x+y*y+z*z;}
	inline float length() const {return std::sqrt(sqLength());};

	inline void normalize()
	{
		float l = length();
		x/=l;
		y/=l;
		z/=l;
	}

	inline Vec3 normalized() const
	{
		Vec3 other(*this);
		float l = other.length();
		other.x/=l;
		other.y/=l;
		other.z/=l;
		return other;
	}

public:
	union {
		float val[3];
		struct {
			float x, y, z;
		};
	};

// static
public:
	static inline float dot(const Vec3 & a, const Vec3 & b)
	{
		return a.x*b.x+a.y*b.y+a.z*b.z;
	}
	static inline Vec3 cross(const Vec3 & a, const Vec3 & b)
	{
		return {
			a.y*b.z-a.z*b.y,
			a.z*b.x-a.x*b.z,
			a.x*b.y-a.y*b.x
		};
	}

// operators
public:
	inline Vec3 & operator+=(const Vec3 & other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	inline Vec3 & operator-=(const Vec3 & other)
	{
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	inline Vec3 & operator*=(float a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}

	inline Vec3 & operator/=(float a)
	{
		x /= a;
		y /= a;
		z /= a;
		return *this;
	}
};

inline Vec3 operator+(const Vec3 & a, const Vec3 & b)
{
	return {a.x+b.x, a.y+b.y, a.z+b.z};
}

inline Vec3 operator-(const Vec3 & a, const Vec3 & b)
{
	return {a.x-b.x, a.y-b.y, a.z-b.z};
}

inline Vec3 operator*(float a, const Vec3 & v)
{
	return {a*v.x, a*v.y, a*v.z};
}

inline Vec3 operator*(const Vec3 & v, float a)
{
	return {a*v.x, a*v.y, a*v.z};
}

inline Vec3 operator/(const Vec3 & v, float a)
{
	return {v.x/a, v.y/a, v.z/a};
}


#endif // VEC3_H
