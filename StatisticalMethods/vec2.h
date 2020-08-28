#pragma once
#include <iostream>
#include <cmath>

class vec2
{
public:
	vec2() : x(0.0), y(0.0) {}
	vec2(double x, double y) : x(x), y(y) {}
	vec2(double a) : x(a), y(a) {}
	vec2(const vec2& v) : x(v.x), y(v.y) {}


	vec2 operator+(const vec2& v) const
	{
		return vec2(v.x + x, v.y + y);
	}

	vec2 operator-(const vec2& v) const
	{
		return vec2(x - v.x, y - v.y);
	}

	vec2 operator-() const
	{
		return vec2(-x, -y);
	}

	double operator*(const vec2& v) const
	{
		return v.x * x + v.y * y;
	}

	double norm() const
	{
		return sqrt(x * x + y * y);
	}

	vec2 normalize() const
	{
		double n = norm();
		return vec2(x / n, y / n);
	}

public:
	double x = 0.0, y = 0.0;
};

template<typename T>
vec2 operator*(T a, const vec2& v)
{
	return vec2(v.x * a, v.y * a);
}

template<typename T>
vec2 operator*(const vec2& v, T a)
{
	return vec2(v.x * a, v.y * a);
}

template<typename T>
vec2 operator/(T a, const vec2& v)
{
	return vec2(v.x / a, v.y / a);
}

template<typename T>
vec2 operator/(const vec2& v, T a)
{
	return vec2(v.x / a, v.y / a);
}

inline std::ostream& operator<<(std::ostream& os, const vec2& x)
{
	os << "(" << x.x << "," << x.y << ")";
	return os;
}