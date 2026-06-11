#ifndef NEW_LABEL_H_
#define NEW_LABEL_H_
#include "geometry.hpp"
#include "numpy.hpp"

namespace cbclib {

using LabelResult = std::tuple<py::array_t<int>, py::array_t<int>>;

namespace detail {

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
constexpr auto modulo(T a, U b) -> decltype(a % b)
{
    return (a % b + b) % b;
}

/* Returns a positive remainder of division */
template <typename T, typename U, typename = std::enable_if_t<std::is_floating_point_v<T> || std::is_floating_point_v<U>>>
constexpr auto modulo(T a, U b) -> decltype(std::fmod(a, b))
{
    return std::fmod(std::fmod(a, b) + b, b);
}

template <typename I, typename Shift, typename Shape, typename = std::enable_if_t<
    std::is_integral_v<typename Shift::value_type> &&
    std::is_integral_v<typename Shape::value_type>
>>
long shift_index(I index, const Shift & shift, const Shape & shape)
{
    size_t delta = shape.size() - shift.size();

    long new_index = 0;
    long stride = 1;
    for (size_t n = shape.size(); n > 0; --n)
    {
        long coord = index % shape[n - 1] + shift[n - 1 - delta];
        if (coord < 0 || coord >= static_cast<long>(shape[n - 1])) return -1;

        new_index += coord * stride;
        index /= shape[n - 1];
        stride *= shape[n - 1];
    }
    return new_index;
}

// Return log(binomial_tail(n, k, p))
// binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
// bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

template <typename T>
T logaddexp(T a, T b)
{
    const T neg_inf = -std::numeric_limits<T>::infinity();

    if (a == neg_inf) return b;
    if (b == neg_inf) return a;

    T m = (a > b) ? a : b;
    return m + std::log(std::exp(a - m) + std::exp(b - m));
};

template <typename I, typename T>
T logbinom(I n, I k, T p)
{
    const T neg_inf = -std::numeric_limits<T>::infinity();

    if (k <= 0) return T(0.0);
    if (k > n) return neg_inf;

    if (p <= T(0.0)) return (k <= 0) ? T(0.0) : neg_inf;
    if (p >= T(1.0)) return (k <= n) ? T(0.0) : neg_inf;

    T log_p = std::log(p);
    T log_q = std::log1p(-p);

    T log_term = std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1) +
                 k * log_p + (n - k) * log_q;
    T log_tail = log_term;

    for (I i = k + 1; i < n + 1; ++i)
    {
        log_term += std::log(n - i + 1) - std::log(i) + log_p - log_q;
        log_tail = logaddexp(log_tail, log_term);
    }

    return log_tail;
}

}

template <size_t N, typename Container, typename I, typename = std::enable_if_t<
    std::is_integral_v<typename Container::value_type> && std::is_integral_v<I>
>>
PointND<I, N> make_point(I index, const Container & shape)
{
    PointND<I, N> point;
    for (size_t n = N; n > 0; --n)
    {
        size_t zyx_n = shape.size() - N + n - 1, xyz_n = N - n;
        point[xyz_n] = index % shape[zyx_n];
        index /= shape[zyx_n];
    }
    return point;
}

// Connectivity structure class
// Structure offsets follow zyx ordering

template <bool IsConst>
struct chunk_traits {};

template <>
struct chunk_traits<false>
{
    using size_type = size_t;
    using iterator = long *;
    using const_iterator = const long *;
    using reference = long &;
    using const_reference = const long &;
    using pointer = long *;
};

template <>
struct chunk_traits<true>
{
    using size_type = size_t;
    using iterator = const long *;
    using const_iterator = const long *;
    using reference = const long &;
    using const_reference = const long &;
    using pointer = const long *;
};

struct Structure
{
protected:
    template <bool IsConst>
    class chunk_iterator;

    template <bool IsConst>
    class chunk
    {
    public:
        using size_type = typename chunk_traits<IsConst>::size_type;
        using value_type = long;

        using iterator = typename chunk_traits<IsConst>::iterator;
        using const_iterator = typename chunk_traits<IsConst>::const_iterator;

        using reference = typename chunk_traits<IsConst>::reference;
        using const_reference = typename chunk_traits<IsConst>::const_reference;

        iterator begin() {return m_first;}
        const_iterator begin() const {return m_first;}

        iterator end() {return m_last;}
        const_iterator end() const {return m_last;}

        size_type size() const {return m_last - m_first;}

        reference operator[](size_type index) {return m_first[index];}
        const_reference operator[](size_type index) const {return m_first[index];}
    protected:
        using pointer = typename chunk_traits<IsConst>::pointer;
        pointer m_first, m_last;

        chunk(pointer first, pointer last) : m_first(first), m_last(last) {}

        friend class chunk_iterator<IsConst>;
    };

    template <bool IsConst>
    class chunk_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = chunk<IsConst>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = value_type;

        chunk_iterator() = default;

        template <bool RHIsConst, typename = std::enable_if_t<IsConst || !RHIsConst>>
        chunk_iterator(const chunk_iterator<RHIsConst> & rhs) : m_ptr(rhs.m_ptr), m_chunk_size(rhs.m_chunk_size) {}

        bool operator==(const chunk_iterator & rhs) const {return m_ptr == rhs.m_ptr;}
        bool operator!=(const chunk_iterator & rhs) const {return !(*this == rhs);}
        bool operator<(const chunk_iterator & rhs) const {return m_ptr < rhs.m_ptr;}
        bool operator>(const chunk_iterator & rhs) const {return m_ptr > rhs.m_ptr;}
        bool operator<=(const chunk_iterator & rhs) const {return !(*this > rhs);}
        bool operator>=(const chunk_iterator & rhs) const {return !(*this < rhs);}

        chunk_iterator & operator+=(difference_type offset)
        {
            m_ptr += offset * m_chunk_size;
            return *this;
        }
        chunk_iterator & operator-=(difference_type offset)
        {
            m_ptr -= offset * m_chunk_size;
            return *this;
        }
        chunk_iterator & operator++()
        {
            m_ptr += m_chunk_size;
            return *this;
        }
        chunk_iterator & operator--()
        {
            m_ptr -= m_chunk_size;
            return *this;
        }
        chunk_iterator operator++(int)
        {
            auto saved = *this;
            ++(*this);
            return saved;
        }
        chunk_iterator operator--(int)
        {
            auto saved = *this;
            --(*this);
            return saved;
        }
        chunk_iterator operator+(difference_type offset) const
        {
            auto saved = *this;
            return saved += offset;
        }
        chunk_iterator operator-(difference_type offset) const
        {
            auto saved = *this;
            return saved -= offset;
        }
        difference_type operator-(const chunk_iterator & rhs) const
        {
            return (m_ptr - rhs.m_ptr) / static_cast<difference_type>(m_chunk_size);
        }

        reference operator[] (size_t offset) const
        {
            return *(*this + offset);
        }
        reference operator*() const
        {
            return chunk<IsConst>(m_ptr, m_ptr + m_chunk_size);
        }
    protected:
        using ptr_t = typename chunk_traits<IsConst>::pointer;
        ptr_t m_ptr;
        size_t m_chunk_size = 0;

        chunk_iterator(ptr_t ptr, size_t chunk_size) : m_ptr(ptr), m_chunk_size(chunk_size) {}

        friend class Structure;
    };

    template <bool IsConst>
    class chunk_range
    {
    public:
        using iterator = chunk_iterator<IsConst>;
        using size_type = typename chunk_traits<IsConst>::size_type;

        chunk_range() = default;
        chunk_range(iterator begin, iterator end) : m_begin(begin), m_end(end) {}

        iterator begin() const {return m_begin;}
        iterator end() const {return m_end;}

        size_type size() const {return m_end - m_begin;}

    private:
        iterator m_begin, m_end;
    };

public:

    using iterator = chunk_iterator<false>;
    using const_iterator = chunk_iterator<true>;
    using range = chunk_range<false>;
    using const_range = chunk_range<true>;
    using size_type = size_t;
    using reference = chunk<false>;
    using const_reference = chunk<true>;

    int connectivity;

    template <typename Radii>
    Structure(const Radii & radii, int connectivity) : connectivity(connectivity)
    {
        for (size_t n = 0; n < radii.size(); n++) m_shape.push_back(2 * radii[n] + 1);

        // Fill in the center
        for (size_t n = 0; n < radii.size(); n++) m_ctr.push_back(0);

        // Fill out shifts within the connectivity
        for (auto point : rectangle_range<std::vector<long>>(std::vector<long>(m_shape.begin(), m_shape.end())))
        {
            long abs = 0;
            for (size_t n = 0; n < radii.size(); n++)
            {
                point[n] -= radii[n];
                abs += std::abs(point[n]);
            }
            if (abs > 0 && abs <= connectivity)
            {
                m_ctr.insert(m_ctr.end(), point.begin(), point.end());
            }
        }
    }

    iterator begin() {return iterator(m_ctr.data(), rank());}
    const_iterator begin() const {return const_iterator(m_ctr.data(), rank());}

    iterator end() {return iterator(m_ctr.data() + rank() * size(), rank());}
    const_iterator end() const {return const_iterator(m_ctr.data() + rank() * size(), rank());}

    range shifts() {return range(std::next(begin()), end());}
    const_range shifts() const {return const_range(std::next(begin()), end());}

    size_type size() const {return m_ctr.size() / rank();}
    size_type rank() const {return m_shape.size();}

    const std::vector<size_t> & shape() const {return m_shape;}
    size_t shape(size_t index) const {return m_shape[index];}

    std::string info() const
    {
        return "<Structure, connectivity = " + std::to_string(connectivity) +
               ", rank = " + std::to_string(rank()) + ", size = " +  std::to_string(size()) + ">";
    }

protected:
    std::vector<long> m_ctr;
    std::vector<size_t> m_shape;
};

namespace detail {

struct ShiftOffset
{
    std::vector<long> delta;
    long offset = 0;
};

template <typename Shape, typename = std::enable_if_t<std::is_integral_v<typename Shape::value_type>>>
bool is_inbound_shift(const std::vector<size_t> & coord, const ShiftOffset & shift, const Shape & shape)
{
    for (size_t dim = 0; dim < shape.size(); ++dim)
    {
        long shifted = static_cast<long>(coord[dim]) + shift.delta[dim];
        if (shifted < 0 || shifted >= static_cast<long>(shape[dim])) return false;
    }
    return true;
}

template <typename Shape, typename = std::enable_if_t<std::is_integral_v<typename Shape::value_type>>>
void next_coord(std::vector<size_t> & coord, const Shape & shape)
{
    for (size_t dim = shape.size(); dim-- > 0;)
    {
        if (++coord[dim] < shape[dim]) return;
        coord[dim] = 0;
    }
}

template <typename Shape, typename = std::enable_if_t<std::is_integral_v<typename Shape::value_type>>>
std::vector<ShiftOffset> shift_offsets(const Structure & structure, const Shape & shape,
                                       bool reverse = false, bool negative_only = false)
{
    auto strides = c_strides(shape, typename Shape::value_type{1});
    std::vector<ShiftOffset> shifts;
    shifts.reserve(structure.size() - 1);

    for (const auto & shift : structure.shifts())
    {
        ShiftOffset result;
        result.delta.reserve(structure.rank());
        for (size_t dim = 0; dim < structure.rank(); ++dim)
        {
            long delta = reverse ? -shift[dim] : shift[dim];
            result.delta.push_back(delta);
            result.offset += delta * static_cast<long>(strides[dim]);
        }

        if (!negative_only || result.offset < 0) shifts.emplace_back(std::move(result));
    }
    return shifts;
}

}

// Image moments class

template <typename T, size_t N>
class MomentsND;

template <typename T, size_t N>
class CentralMomentsND
{
public:
    // Returns the centroid of the object
    // Coordinates are in the same order as the input data (zyx for 3D)
    std::array<T, N> first() const
    {
        return (mu_x + origin).coordinate();
    }

    // Returns the covariance matrix of the object in the form of a flat array in row-major order
    // Matrix follows the input data order (zyx for 3D)
    std::array<T, N * N> second() const
    {
        std::array<T, N * N> cmat;
        for (size_t n = 0; n < N; n++) cmat[n + N * n] = mu_xx[N - n - 1];
        for (size_t n = 0; n < NumPairs; n++)
        {
            // We need to reverse the order of indices since moments are stored in xyz order
            // but covariance matrix is in zyx order
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            i = N - i - 1; j = N - j - 1;
            cmat[i + N * j] = mu_xy[n]; cmat[j + N * i] = mu_xy[n];
        }
        return cmat;
    }

    // Angle between the largest eigenvector of the covariance matrix and x-axis
    // Can return nan if mu_xx[0] == mu_xx[1]
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    T theta() const
    {
        return 0.5 * std::atan2(2 * mu_xy[0], mu_xx[0] - mu_xx[1]);
    }

    // Return line segment representing the major axis of the object
    // Returns a zero-length line if mu_xx[0] == mu_xx[1]
    template <size_t M = N, typename = std::enable_if_t<(M > 1)>>
    LineND<T, N> line() const
    {
        T angle = theta();
        if (std::isnan(angle)) return LineND<T, N>{mu_x + origin, mu_x + origin};
        PointND<T, N> tau {std::cos(angle), std::sin(angle)};
        T delta = std::sqrt(4 * mu_xy[0] * mu_xy[0] + (mu_xx[0] - mu_xx[1]) * (mu_xx[0] - mu_xx[1]));
        T hw = std::sqrt(2 * std::log(2) * (mu_xx[0] + mu_xx[1] + delta));
        return LineND<T, N>{mu_x + origin + hw * tau, mu_x + origin - hw * tau};
    }

    friend std::ostream & operator<<(std::ostream & os, const CentralMomentsND & m)
    {
        os << "{origin = " << m.origin << ", mu_x = " << m.mu_x
           << ", mu_xx = " << m.mu_xx << ", mu_xy = " << m.mu_xy << "}";
        return os;
    }

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

    // All moments follow the point order (xyz), reversed from the input data (zyx)
    PointND<T, N> origin;               // centroid
    PointND<T, N> mu_x {}, mu_xx {};    // mu_x: first central moments, mu_xx: second central moments
    PointND<T, NumPairs> mu_xy {};      // cross second central moments

    friend class MomentsND<T, N>;

    CentralMomentsND(PointND<T, N> pt) : origin(std::move(pt)) {}
    CentralMomentsND(PointND<T, N> pt, PointND<T, N> mx, PointND<T, N> mxx, PointND<T, NumPairs> mxy) :
        origin(std::move(pt)), mu_x(std::move(mx)), mu_xx(std::move(mxx)), mu_xy(std::move(mxy)) {}
};

template <typename T, size_t N>
class MomentsND
{
public:
    MomentsND() = default;

    template <typename Pt, typename = std::enable_if_t<std::is_base_of_v<PointND<T, N>, remove_cvref_t<Pt>>>>
    MomentsND(Pt && pt) : org(std::forward<Pt>(pt)) {}

    // In-place operators

    MomentsND & operator+=(MomentsND rhs)
    {
        rhs.move(org);
        mu += rhs.mu;
        mu_x += rhs.mu_x;
        mu_xx += rhs.mu_xx;
        mu_xy += rhs.mu_xy;
        return *this;
    }

    MomentsND & operator-=(MomentsND rhs)
    {
        rhs.move(org);
        mu -= rhs.mu;
        mu_x -= rhs.mu_x;
        mu_xx -= rhs.mu_xx;
        mu_xy -= rhs.mu_xy;
        return *this;
    }

    void insert(const PointND<T, N> & point, T value)
    {
        auto r = point - org;

        value = std::max(value, T());
        mu += value;
        mu_x += r * value;
        mu_xx += r * r * value;
        for (size_t n = 0; n < NumPairs; n++)
        {
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            mu_xy[n] += r[i] * r[j] * value;
        }
    }

    void insert(long index, array<T> data)
    {
        auto point = make_point<N>(index, data.shape());
        auto val = data[index];

        insert(point, val);
    }

    void move(const PointND<T, N> & point)
    {
        if (org != point)
        {
            auto r = org - point;
            mu_xx += 2 * r * mu_x + r * r * mu;
            for (size_t n = 0; n < NumPairs; n++)
            {
                auto [i, j] = UniquePairs<N>::instance().pairs(n);
                mu_xy[n] += r[i] * mu_x[j] + r[j] * mu_x[i] + r[i] * r[j] * mu;
            }
            mu_x += r * mu;
            org = point;
        }
    }

    // Friend members

    friend MomentsND operator+(const MomentsND & lhs, const MomentsND & rhs)
    {
        MomentsND result = lhs;
        result += rhs;
        return result;
    }

    friend MomentsND operator-(const MomentsND & lhs, const MomentsND & rhs)
    {
        MomentsND result = lhs;
        result += rhs;
        return result;
    }

    friend std::ostream & operator<<(std::ostream & os, const MomentsND & m)
    {
        os << "{origin = " << m.org << ", mu = " << m.mu << ", mu_x = " << m.mu_x
           << ", mu_xx = " << m.mu_xx << ", mu_xy = " << m.mu_xy << "}";
        return os;
    }

    // Other members

    CentralMomentsND<T, N> central() const
    {
        if (mu)
        {
            auto M_X = mu_x / mu;
            auto M_XX = mu_xx / mu - M_X * M_X;
            PointND<T, NumPairs> M_XY {};
            for (size_t n = 0; n < NumPairs; n++)
            {
                auto [i, j] = UniquePairs<N>::instance().pairs(n);
                M_XY[n] = mu_xy[n] / mu - M_X[i] * M_X[j];
            }
            return {org, std::move(M_X), std::move(M_XX), std::move(M_XY)};
        }
        return {org};
    }

    const PointND<T, N> & origin() const {return org;}

    T zeroth() const {return mu;}

    // Coordinates of the centroid in the same order as the input data (zyx for 3D)
    std::array<T, N> first() const {return (mu_x + org * mu).coordinate();}

    // Covariance matrix of the object in the form of a flat array in row-major order
    // Matrix follows the input data order (zyx for 3D)
    std::array<T, N * N> second() const
    {
        std::array<T, N * N> matrix {};
        for (size_t n = 0; n < N; n++)
        {
            auto zyx_n = N - n - 1;
            matrix[n + N * n] = mu_xx[zyx_n] + 2 * org[zyx_n] * mu_x[zyx_n] + org[zyx_n] * org[zyx_n] * mu;
        }
        for (size_t n = 0; n < NumPairs; n++)
        {
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            i = N - i - 1; j = N - j - 1;
            auto m_xy = mu_xy[n] + org[i] * mu_x[j] + org[j] * mu_x[i] + org[i] * org[j] * mu;
            matrix[i + N * j] = m_xy; matrix[j + N * i] = m_xy;
        }
        return matrix;
    }

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

    PointND<T, N> org {};               // origin
    T mu = T();                         // zeroth moment
    PointND<T, N> mu_x {}, mu_xx {};    // mu_x: first central moments, mu_xx: second central moments
    PointND<T, NumPairs> mu_xy {};      // cross second central moments
};

template <typename T>
using Moments = MomentsND<T, 2>;

// PyBind11 helper functions to wrap an std::vector derived classes

template <typename List, typename Element = typename List::value_type, typename = std::enable_if_t<std::is_base_of_v<std::vector<Element>, List>>>
void declare_list(py::class_<List> & cls, const std::string & str)
{
    cls.def(py::init<>())
        .def(py::init([](py::iterable elems)
        {
            List list;
            for (auto item : elems) list.push_back(item.cast<Element>());
            return list;
        }), py::arg("elements"))
        .def("__delitem__", [str](List & list, py::ssize_t index)
        {
            list.erase(std::next(list.begin(), compute_index(index, list.size(), str)));
        }, py::arg("index"))
        .def("__delitem__", [](List & list, py::slice & slice)
        {
            auto range = slice_range(slice, list.size());
            auto iter = std::next(list.begin(), range.start());
            for (size_t i = 0; i < static_cast<size_t>(range.size()); ++i, iter += range.step() - 1) iter = list.erase(iter);
        }, py::arg("index"))
        .def("__getitem__", [str](const List & list, py::ssize_t index)
        {
            return list[compute_index(index, list.size(), str)];
        }, py::arg("index"))
        .def("__getitem__", [](const List & list, py::slice & slice)
        {
            List sliced;
            for (auto [_, py_index] : slice_range(slice, list.size())) sliced.push_back(list[py_index]);
            return sliced;
        }, py::arg("index"))
        .def("__getitem__", [str](const List & list, py::array_t<py::ssize_t> indices)
        {
            List sliced;
            for (auto index : array<py::ssize_t>(indices.request())) sliced.push_back(list[compute_index(index, list.size(), str)]);
            return sliced;
        }, py::arg("index"))
        .def("__getitem__", [](const List & list, py::array_t<bool> mask)
        {
            if (mask.ndim() != 1 || mask.size() != static_cast<py::ssize_t>(list.size()))
                throw std::invalid_argument("Mask must be a 1D array of the same size as the list");

            List sliced;
            auto mask_ptr = mask.data();
            for (size_t i = 0; i < list.size(); i++) if (mask_ptr[i]) sliced.push_back(list[i]);
            return sliced;
        }, py::arg("mask"))
        .def("__setitem__", [str](List & list, py::ssize_t index, Element elem)
        {
            list[compute_index(index, list.size(), str)] = std::move(elem);
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__setitem__", [](List & list, py::slice & slice, List & elems)
        {
            for (auto [index, py_index] : slice_range(slice, list.size())) list[py_index] = elems[index];
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__iter__", [](const List & list){return py::make_iterator(list.begin(), list.end());}, py::keep_alive<0, 1>())
        .def("__len__", [](const List & list){return list.size();})
        .def("__repr__", [str](const List & list)
        {
            return "<" + str + ", size = " + std::to_string(list.size()) + ">";
        })
        .def("append", [](List & list, Element elem){list.emplace_back(std::move(elem));}, py::arg("value"), py::keep_alive<1, 2>())
        .def("extend", [](List & list, const List & elems)
        {
            for (const auto & elem : elems) list.push_back(elem);
        }, py::arg("values"), py::keep_alive<1, 2>());
}

}

#endif
