#pragma once

#include <functional>

struct grad {
	std::function<double(class vec2)> df1, df2;
};

struct hesse {
	std::function<double(class vec2)> df11, df12, df21, df22;
};

using func = std::function<double(class vec2)>;

class vec2
{
public:
	vec2() : x(0.0), y(0.0) {}
	vec2(double x, double y) : x(x), y(y) {}
	vec2(double a) : x(a), y(a) {}
	vec2(const vec2& v) : x(v.x), y(v.y) {}
	vec2(grad g, vec2 x): x(g.df1(x)), y(g.df2(x)) {}


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
	double x = 0.0 , y = 0.0;
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


class mat2
{
public:
	mat2(double a11, double a12, double a21, double a22) : a11(a11), a12(a12), a21(a21), a22(a22) {}
	mat2(double a): a11(a), a12(0.0), a21(0.0), a22(a) {}
	mat2(const mat2& m): a11(m.a11), a12(m.a12), a21(m.a21), a22(m.a22) {}
	mat2(hesse h, vec2 x): 
		a11(h.df11(x)),
		a12(h.df12(x)),
		a21(h.df21(x)),
		a22(h.df22(x)) 
	{}

	mat2 operator*(const mat2& m) const
	{
		return mat2(
			a11 * m.a11 + a12 * m.a21,
			a11 * m.a12 + a12 * m.a22,
			a21 * m.a11 + a22 * m.a12,
			a21 * m.a12 + a22 * m.a22);
	}

	vec2 operator*(const vec2& v) const
	{
		return vec2(a11 * v.x + a12 * v.y, a21 * v.x + a22 * v.y);
	}

	mat2 operator+(const mat2& m) const
	{
		return mat2(a11 + m.a11, a12 + m.a12, a21 + m.a21, a22 + m.a22);
	}

	mat2 operator-(const mat2& m) const
	{
		return mat2(a11 - m.a11, a12 - m.a12, a21 - m.a21, a22 - m.a22);
	}

	mat2 operator-() const
	{
		return mat2(-a11, -a12, -a21, -a22);
	}

	mat2 inverse() const 
	{
		double det = a11 * a22 - a12 * a21;
		return mat2(a22 / det, -a12 / det, -a21 / det, a11 / det);
	}

public:
	double a11, a12, a21, a22;
};

template<typename T>
mat2 operator*(T a, const mat2& m)
{
	return mat2(m.a11 * a, m.a12 * a, m.a21 * a, m.a22 * a);
}

template<typename T>
vec2 operator*(const mat2& m, T a)
{
	return mat2(m.a11 * a, m.a12 * a, m.a21 * a, m.a22 * a);
}

template<typename T>
mat2 operator/(T a, const mat2& m)
{
	return mat2(m.a11 / a, m.a12 / a, m.a21 / a, m.a22 / a);
}

template<typename T>
mat2 operator/(const mat2& m, T a)
{
	return mat2(m.a11 / a, m.a12 / a, m.a21 / a, m.a22 / a);
}