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
    static HOST_DEVICE float abs(float x) { return fabsf(x); }
    static HOST_DEVICE float ceil(float x) { return ceilf(x); }
    static HOST_DEVICE float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
    static HOST_DEVICE float cos(float x) { return cosf(x); }
    static HOST_DEVICE float exp(float x) { return expf(x); }
    static HOST_DEVICE float floor(float x) { return floorf(x); }
    static HOST_DEVICE float log(float x) { return logf(x); }
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
    static HOST_DEVICE double abs(double x) { return ::fabs(x); }
    static HOST_DEVICE double ceil(double x) { return ::ceil(x); }
    static HOST_DEVICE double cos(double x) { return ::cos(x); }
    static HOST_DEVICE double clamp(double x, double lo, double hi) { return ::fmin(::fmax(x, lo), hi); }
    static HOST_DEVICE double exp(double x) { return ::exp(x); }
    static HOST_DEVICE double floor(double x) { return ::floor(x); }
    static HOST_DEVICE double log(double x) { return ::log(x); }
    static HOST_DEVICE double min(double x, double y) { return ::fmin(x, y); }
    static HOST_DEVICE double max(double x, double y) { return ::fmax(x, y); }
    static HOST_DEVICE double pow(double x, double y) { return ::pow(x, y); }
    static HOST_DEVICE double round(double x) { return ::round(x); }
    static HOST_DEVICE double sin(double x) { return ::sin(x); }
    static HOST_DEVICE double sqrt(double x) { return ::sqrt(x); }
    static HOST_DEVICE double tan(double x) { return ::tan(x); }
};

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
};

template <>
struct numeric_limits<double>
{
    static constexpr HOST_DEVICE double epsilon() { return 1e-16; }
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

    HOST_DEVICE PointND & operator+=(const PointND & rhs)
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] += rhs._M_elems[i];
        return *this;
    }
    HOST_DEVICE PointND & operator-=(const PointND & rhs)
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] -= rhs._M_elems[i];
        return *this;
    }
    HOST_DEVICE PointND & operator*=(T s)
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] *= s;
        return *this;
    }
    HOST_DEVICE PointND & operator/=(T s)
    {
        for (csize_t i = 0; i < N; i++) _M_elems[i] /= s;
        return *this;
    }

    HOST_DEVICE friend PointND operator+(PointND lhs, const PointND & rhs)
    {
        lhs += rhs; return lhs;
    }
    HOST_DEVICE friend PointND operator-(PointND lhs, const PointND & rhs)
    {
        lhs -= rhs; return lhs;
    }
    HOST_DEVICE friend PointND operator*(PointND lhs, T s)
    {
        lhs *= s; return lhs;
    }
    HOST_DEVICE friend PointND operator*(T s, PointND rhs)
    {
        rhs *= s; return rhs;
    }
    HOST_DEVICE friend PointND operator/(PointND lhs, T s)
    {
        lhs /= s; return lhs;
    }

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

    friend std::ostream & operator<<(std::ostream & os, const PointND & pt)
    {
        os << "{";
        std::copy(pt.begin(), pt.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "}";
        return os;
    }
};

// Dot product
template <typename T, csize_t N>
HOST_DEVICE inline T dot(const PointND<T, N> & a, const PointND<T, N> & b)
{
    T r = T();
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
template <typename T, csize_t N>
HOST_DEVICE inline T amplitude(const PointND<T, N> & v)
{
    return math_traits<T>::sqrt(magnitude(v));
}

#undef HOST_DEVICE

} // namespace cuda

#endif //  GEOMETRY_CUDA_
