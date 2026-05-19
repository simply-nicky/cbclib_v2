#ifndef  GEOMETRY_CUDA_
#define  GEOMETRY_CUDA_
#include "include.hpp"

namespace cbclib::cuda {

typedef unsigned csize_t;

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Math trait for floating point operations
template <typename T>
struct math_traits;

template <>
struct math_traits<float>
{
    static HOST_DEVICE float acos(float x) { return acosf(x); }
    static HOST_DEVICE float asin(float x) { return asinf(x); }
    static HOST_DEVICE float atan(float x) { return atanf(x); }
    static HOST_DEVICE float atan2(float y, float x) { return atan2f(y, x); }
    static HOST_DEVICE float abs(float x) { return fabsf(x); }
    static HOST_DEVICE float ceil(float x) { return ceilf(x); }
    static HOST_DEVICE float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
    static HOST_DEVICE float cos(float x) { return cosf(x); }
    static HOST_DEVICE float exp(float x) { return expf(x); }
    static HOST_DEVICE float floor(float x) { return floorf(x); }
    static HOST_DEVICE float fmod(float x, float y) { return fmodf(x, y); }
    static HOST_DEVICE float lgamma(float x) { return lgammaf(x); }
    static HOST_DEVICE float log(float x) { return logf(x); }
    static HOST_DEVICE float log1p(float x) { return log1pf(x); }
    static HOST_DEVICE float min(float x, float y) { return fminf(x, y); }
    static HOST_DEVICE float max(float x, float y) { return fmaxf(x, y); }
    static HOST_DEVICE float pow(float x, float y) { return powf(x, y); }
    static HOST_DEVICE float round(float x) { return roundf(x); }
    static HOST_DEVICE float sin(float x) { return sinf(x); }
    static HOST_DEVICE float sqrt(float x) { return sqrtf(x); }
    static HOST_DEVICE float tan(float x) { return tanf(x); }
};

template <>
struct math_traits<double>
{
    static HOST_DEVICE double acos(double x) { return ::acos(x); }
    static HOST_DEVICE double asin(double x) { return ::asin(x); }
    static HOST_DEVICE double atan(double x) { return ::atan(x); }
    static HOST_DEVICE double atan2(double y, double x) { return ::atan2(y, x); }
    static HOST_DEVICE double abs(double x) { return ::fabs(x); }
    static HOST_DEVICE double ceil(double x) { return ::ceil(x); }
    static HOST_DEVICE double cos(double x) { return ::cos(x); }
    static HOST_DEVICE double clamp(double x, double lo, double hi) { return ::fmin(::fmax(x, lo), hi); }
    static HOST_DEVICE double exp(double x) { return ::exp(x); }
    static HOST_DEVICE double floor(double x) { return ::floor(x); }
    static HOST_DEVICE double fmod(double x, double y) { return ::fmod(x, y); }
    static HOST_DEVICE double lgamma(double x) { return ::lgamma(x); }
    static HOST_DEVICE double log(double x) { return ::log(x); }
    static HOST_DEVICE double log1p(double x) { return ::log1p(x); }
    static HOST_DEVICE double min(double x, double y) { return ::fmin(x, y); }
    static HOST_DEVICE double max(double x, double y) { return ::fmax(x, y); }
    static HOST_DEVICE double pow(double x, double y) { return ::pow(x, y); }
    static HOST_DEVICE double round(double x) { return ::round(x); }
    static HOST_DEVICE double sin(double x) { return ::sin(x); }
    static HOST_DEVICE double sqrt(double x) { return ::sqrt(x); }
    static HOST_DEVICE double tan(double x) { return ::tan(x); }
};

template <typename T>
struct numbers;

template <>
struct numbers<float>
{
    static constexpr HOST_DEVICE float pi() { return 3.14159265358979323846f; }
    static constexpr HOST_DEVICE float pi_2() { return 1.57079632679489661923f; }
    static constexpr HOST_DEVICE float e() { return 2.71828182845904523536f; }
    static constexpr HOST_DEVICE float sqrt2() { return 1.41421356237309504880f; }
    static constexpr HOST_DEVICE float log2() { return 0.69314718055994530942f; }
};

template <>
struct numbers<double>
{
    static constexpr HOST_DEVICE double pi() { return 3.14159265358979323846; }
    static constexpr HOST_DEVICE double pi_2() { return 1.57079632679489661923; }
    static constexpr HOST_DEVICE double e() { return 2.71828182845904523536; }
    static constexpr HOST_DEVICE double sqrt2() { return 1.41421356237309504880; }
    static constexpr HOST_DEVICE double log2() { return 0.69314718055994530942; }
};

namespace detail {

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
HOST_DEVICE constexpr auto modulo(T a, U b) -> decltype(a % b)
{
    return (a % b + b) % b;
}

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_floating_point_v<T> || std::is_floating_point_v<U>>>
HOST_DEVICE constexpr auto modulo(T a, U b) -> decltype(math_traits<T>::fmod(a, b))
{
    return math_traits<T>::fmod(math_traits<T>::fmod(a, b) + b, b);
}

}

// Kernel functions
template <typename T>
struct kernel_traits
{
    static constexpr double M_1_SQRT2PI = 0.3989422804014327;

    HOST_DEVICE static T biweight(T x)
    {
        return T(0.9375) * math_traits<T>::pow(math_traits<T>::max(1 - math_traits<T>::pow(x, 2), T()), 2);
    }
    HOST_DEVICE static T gaussian(T x)
    {
        if (math_traits<T>::abs(x) <= T(1.0)) return M_1_SQRT2PI * math_traits<T>::exp(-math_traits<T>::pow(3 * x, 2) / 2);
        return T();
    }
    HOST_DEVICE static T parabolic(T x)
    {
        return T(0.75) * math_traits<T>::max(1 - math_traits<T>::pow(x, 2), T());
    }
    HOST_DEVICE static T rectangular(T x)
    {
        return (math_traits<T>::abs(x) <= T(1.0)) ? T(1.0) : T();
    }
    HOST_DEVICE static T triangular(T x)
    {
        return math_traits<T>::max(T(1.0) - math_traits<T>::abs(x), T());
    }

    // Device-friendly stateless functors (wrap static methods)
    struct Biweight    { HOST_DEVICE inline T operator()(T x) const { return biweight(x); } };
    struct Gaussian    { HOST_DEVICE inline T operator()(T x) const { return gaussian(x); } };
    struct Parabolic   { HOST_DEVICE inline T operator()(T x) const { return parabolic(x); } };
    struct Rectangular { HOST_DEVICE inline T operator()(T x) const { return rectangular(x); } };
    struct Triangular  { HOST_DEVICE inline T operator()(T x) const { return triangular(x); } };
};

// Epsilon trait for numerical stability
template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float>
{
    static constexpr HOST_DEVICE float epsilon() { return 1e-8f; }
    static constexpr HOST_DEVICE float infinity() { return INFINITY; }
};

template <>
struct numeric_limits<double>
{
    static constexpr HOST_DEVICE double epsilon() { return 1e-16; }
    static constexpr HOST_DEVICE double infinity() { return INFINITY; }
};

// Minimal device-friendly N-D point
template <typename T, csize_t N>
struct PointND
{
    static_assert(std::is_arithmetic_v<T>, "PointND requires arithmetic type");
    static_assert(!std::is_const_v<T>, "PointND does not support const element types");

    using value_type = T;
    using size_type = csize_t;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    using iterator = T *;
    using const_iterator = const T *;

    T _M_elems[N];

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<V, T>>>
    HOST_DEVICE operator PointND<V, N>() const
    {
        PointND<V, N> res;
        for (csize_t i = 0; i < N; i++) res[i] = static_cast<V>(_M_elems[i]);
        return res;
    }

    HOST_DEVICE PointND()
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] = T();
    }

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    HOST_DEVICE PointND(const V * ptr)
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] = static_cast<T>(ptr[i]);
    }

    template <typename... Args,  typename = std::enable_if_t<
        sizeof...(Args) == N && (std::is_constructible_v<T, Args> && ...)
    >>
    HOST_DEVICE constexpr PointND(Args... args) : _M_elems{static_cast<T>(args)...} {}

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    HOST_DEVICE explicit PointND(V (&arr)[N]) : PointND(&arr[0]) {}

    HOST_DEVICE reference operator[](size_type i) { return _M_elems[i]; }
    HOST_DEVICE const_reference operator[](size_type i) const { return _M_elems[i]; }

    HOST_DEVICE constexpr size_type size() const { return N; }

    HOST_DEVICE iterator begin() { return &_M_elems[0]; }
    HOST_DEVICE const_iterator begin() const { return &_M_elems[0]; }

    HOST_DEVICE iterator end() { return &_M_elems[0] + N; }
    HOST_DEVICE const_iterator end() const { return &_M_elems[0] + N; }

    HOST_DEVICE pointer data() { return &_M_elems[0]; }
    HOST_DEVICE const_pointer data() const { return &_M_elems[0]; }

    HOST_DEVICE reference front() { return _M_elems[0]; }
    HOST_DEVICE const_reference front() const { return _M_elems[0]; }

    HOST_DEVICE reference back() { return _M_elems[N - 1]; }
    HOST_DEVICE const_reference back() const { return _M_elems[N - 1]; }

    // In-place operators

    HOST_DEVICE bool operator==(const PointND & rhs) const
    {
        for (csize_t i = 0; i < N; i++) if (_M_elems[i] != rhs._M_elems[i]) return false;
        return true;
    }

    HOST_DEVICE bool operator!=(const PointND & rhs) const { return !(*this == rhs); }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator+=(const PointND<V, N> & rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] += rhs._M_elems[i];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator+=(V rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] += rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator-=(const PointND<V, N> & rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] -= rhs._M_elems[i];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator-=(V rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] -= rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator*=(const PointND<V, N> & rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] *= rhs._M_elems[i];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator*=(V rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] *= rhs;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator/=(const PointND<V, N> & rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] /= rhs._M_elems[i];
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    HOST_DEVICE PointND & operator/=(V rhs) &
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] /= rhs;
        return *this;
    }

    //  friend operators

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator+(const PointND<T, N> & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result += rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator+(const PointND<T, N> & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result += rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator+(V lhs, const PointND<T, N> & rhs)
    {
        PointND<U, N> result = rhs;
        result += lhs;
        return result;
    }

    HOST_DEVICE friend PointND<T, N> operator-(const PointND & rhs)
    {
        PointND<T, N> result;
        for (csize_t i = 0; i < N; i++) result[i] = -rhs[i];
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator-(const PointND<T, N> & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result -= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator-(const PointND<T, N> & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result -= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator-(V lhs, const PointND<T, N> & rhs)
    {
        PointND<U, N> result;
        for (csize_t i = 0; i < N; i++) result[i] = lhs - rhs[i];
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator*(const PointND<T, N> & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result *= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator*(const PointND<T, N> & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result *= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator*(V lhs, const PointND<T, N> & rhs)
    {
        PointND<U, N> result = rhs;
        result *= lhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator/(const PointND<T, N> & lhs, const PointND<V, N> & rhs)
    {
        PointND<U, N> result = lhs;
        result /= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator/(const PointND<T, N> & lhs, V rhs)
    {
        PointND<U, N> result = lhs;
        result /= rhs;
        return result;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    HOST_DEVICE friend PointND<U, N> operator/(V lhs, const PointND<T, N> & rhs)
    {
        PointND<U, N> result;
        for (csize_t i = 0; i < N; ++i) result[i] = lhs / rhs[i];
        return result;
    }

    friend std::ostream & operator<<(std::ostream & os, const PointND & pt)
    {
        os << "{";
        std::copy(pt.begin(), pt.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "}";
        return os;
    }

    // methods

    HOST_DEVICE PointND<T, N> clamp(const PointND<T, N> & lo, const PointND<T, N> & hi) const
    {
        PointND<T, N> result;
        for (csize_t i = 0; i < N; i++) result[i] = min(max(_M_elems[i], lo[i]), hi[i]);
        return result;
    }

    HOST_DEVICE PointND<T, N> round() const
    {
        PointND<T, N> result;
        for (csize_t i = 0; i < N; i++) result[i] = math_traits<T>::round(_M_elems[i]);
        return result;
    }

    template <size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    HOST_DEVICE reference x() { return _M_elems[0]; }
    template <size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    HOST_DEVICE reference y() { return _M_elems[1]; }
    template <size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    HOST_DEVICE reference z() { return _M_elems[2]; }

    template <size_t M = N, typename = std::enable_if_t<(M >= 1)>>
    HOST_DEVICE const_reference x() const { return _M_elems[0]; }
    template <size_t M = N, typename = std::enable_if_t<(M >= 2)>>
    HOST_DEVICE const_reference y() const { return _M_elems[1]; }
    template <size_t M = N, typename = std::enable_if_t<(M >= 3)>>
    HOST_DEVICE const_reference z() const { return _M_elems[2]; }


};

// Return a point from a raw pointer (assumes contiguous memory and correct size)
template <csize_t N, typename T>
HOST_DEVICE inline PointND<T, N> to_point(const T * ptr)
{
    PointND<T, N> result;
    for (csize_t i = 0; i < N; i++) result[i] = ptr[i];
    return result;
}

// Dot product
template <typename T, typename V, csize_t N, typename U = std::common_type_t<T, V>>
HOST_DEVICE inline U dot(const PointND<T, N> & a, const PointND<V, N> & b)
{
    U r = U();
    for (csize_t i = 0; i < N; i++) r += a[i] * b[i];
    return r;
}

// Magnitude squared
template <typename T, csize_t N>
HOST_DEVICE inline T magnitude(const PointND<T, N> & v)
{
    T s = T();
    for (csize_t i = 0; i < N; i++) s += v[i] * v[i];
    return s;
}

// Euclidean amplitude
template <typename T, csize_t N, typename U = decltype(math_traits<T>::sqrt(std::declval<T &>()))>
HOST_DEVICE inline U amplitude(const PointND<T, N> & v)
{
    return math_traits<U>::sqrt(magnitude(v));
}

template <typename T, csize_t N>
struct LineND
{
    PointND<T, N> pt0 {}, pt1 {};

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    HOST_DEVICE operator LineND<V, N>() const
    {
        return LineND<V, N>(static_cast<PointND<V, N>>(pt0), static_cast<PointND<V, N>>(pt1));
    }

    LineND() = default;

    HOST_DEVICE LineND(const PointND<T, N> & p0, const PointND<T, N> & p1) : pt0(p0), pt1(p1) {}

    HOST_DEVICE bool operator==(const LineND<T, N> & rhs) const {return pt0 == rhs.pt0 && pt1 == rhs.pt1;}
    HOST_DEVICE bool operator!=(const LineND<T, N> & rhs) const {return !operator==(rhs);}

    HOST_DEVICE PointND<T, N> tangent() const { return pt1 - pt0; }
    HOST_DEVICE PointND<T, N> center() const { return (pt0 + pt1) / T(2); }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(math_traits<U>::sqrt(std::declval<U &>()))>
    HOST_DEVICE PointND<W, N> project_to_streak(const PointND<V, N> & point) const
    {
        auto tau = tangent();
        auto mag = magnitude(tau);

        if (mag)
        {
            auto ctr = center();
            auto r = point - ctr;
            auto r_tau = static_cast<W>(dot(tau, r)) / mag;
            return math_traits<W>::clamp(r_tau, -0.5, 0.5) * tau + ctr;
        }
        return pt0;
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(math_traits<U>::sqrt(std::declval<U &>()))>
    HOST_DEVICE W distance(const PointND<V, N> & point) const
    {
        return amplitude(point - project_to_streak(point));
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<U &>()))>
    HOST_DEVICE PointND<W, N> project_to_line(const PointND<V, N> & point) const
    {
        auto tau = tangent();
        auto mag = magnitude(tau);

        if (mag)
        {
            auto ctr = center();
            auto r = point - ctr;
            auto r_tau = static_cast<W>(dot(tau, r)) / mag;
            return r_tau * tau + ctr;
        }
        return pt0;
    }

    template <typename V, typename U = std::common_type_t<T, V>, typename W = decltype(std::sqrt(std::declval<U &>()))>
    HOST_DEVICE W normal_distance(const PointND<V, N> & point) const
    {
        return amplitude(point - project_to_line(point));
    }

    friend std::ostream & operator<<(std::ostream & os, const LineND & line)
    {
        os << "{" << line.pt0 << " -> " << line.pt1 << "}";
        return os;
    }
};

#undef HOST_DEVICE

} // namespace cuda

#endif //  GEOMETRY_CUDA_
