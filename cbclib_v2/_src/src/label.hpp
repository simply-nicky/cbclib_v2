#ifndef NEW_LABEL_H_
#define NEW_LABEL_H_
#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

// Image moments class
template <typename T, size_t N>
using PixelND = std::pair<PointND<long, N>, T>;

template <typename T, size_t N>
using PixelSetND = std::set<PixelND<T, N>>;

template <typename T>
using PixelSet = PixelSetND<T, 2>;

namespace detail {

template <typename Container, typename Element = typename Container::value_type, typename T = typename Element::value_type>
std::vector<T> get_x(const Container & c, size_t index)
{
    std::vector<T> x;
    std::transform(c.begin(), c.end(), std::back_inserter(x), [index](const Element & elem){return elem[index];});
    return x;
}

}

template <typename T, typename I, size_t N, typename = std::enable_if_t<std::is_integral_v<I>>>
PixelND<T, N> make_pixel(const PointND<I, N> & point, const array<T> & data)
{
    return std::make_pair(point, data.at(point.coordinate()));
}

template <typename T, typename I, size_t N, typename = std::enable_if_t<std::is_integral_v<I>>>
PixelND<T, N> make_pixel(PointND<I, N> && point, const array<T> & data)
{
    return std::make_pair(std::move(point), data.at(point.coordinate()));
}

template <typename... Ix, typename T, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
PixelND<T, sizeof...(Ix)> make_pixel(T value, Ix... xs)
{
    return std::make_pair(PointND<long, sizeof...(Ix)>{xs...}, value);
}

// Image moments class

template <typename T, size_t N>
class MomentsND;

template <typename T, size_t N>
class CentralMomentsND
{
public:
    std::array<T, N> first() const
    {
        return mu_x + origin;
    }

    std::array<T, N * N> second() const
    {
        std::array<T, N * N> cmat;
        for (size_t n = 0; n < N; n++) cmat[n + N * n] = mu_xx[n];
        for (size_t n = 0; n < NumPairs; n++)
        {
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            cmat[i + N * j] = mu_xy[n]; cmat[j + N * i] = mu_xy[n];
        }
        return cmat;
    }

    // Angle between the largest eigenvector of the covariance matrix and x-axis
    // Can return nan if mu_xx[0] == mu_xx[1]
    template <size_t M = N, typename = std::enable_if_t<(M == 2)>>
    T theta() const
    {
        T theta = 0.5 * std::atan(2 * mu_xy[0] / (mu_xx[0] - mu_xx[1]));
        if (mu_xx[1] > mu_xx[0]) theta += M_PI_2;
        return detail::modulo(theta, M_PI);
    }

    // Return line segment representing the major axis of the object
    // Returns a zero-length line if mu_xx[0] == mu_xx[1]
    template <size_t M = N, typename = std::enable_if_t<(M == 2)>>
    Line<T> line() const
    {
        T angle = theta();
        if (std::isnan(angle)) return Line<T>{origin, origin};
        Point<T> tau {std::cos(angle), std::sin(angle)};
        T delta = std::sqrt(4 * mu_xy[0] * mu_xy[0] + (mu_xx[0] - mu_xx[1]) * (mu_xx[0] - mu_xx[1]));
        T hw = std::sqrt(2 * std::log(2) * (mu_xx[0] + mu_xx[1] + delta));
        return Line<T>{mu_x + origin + hw * tau, mu_x + origin - hw * tau};
    }

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;
    PointND<T, N> origin;
    PointND<T, N> mu_x, mu_xx;
    PointND<T, NumPairs> mu_xy;

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
    MomentsND(Pt && pt) : org(std::forward<Pt>(pt)), mu(), mu_x(), mu_xx(), mu_xy() {}

    MomentsND(const PixelSetND<T, N> & pset) : MomentsND()
    {
        if (pset.size())
        {
            org = std::next(pset.begin(), pset.size() / 2)->first;
            insert(pset.begin(), pset.end());
        }
    }

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

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void insert(const PointND<V, N> & point, T val)
    {
        auto r = point - org;

        val = std::max(val, T());
        mu += val;
        mu_x += r * val;
        mu_xx += r * r * val;
        for (size_t n = 0; n < NumPairs; n++)
        {
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            mu_xy[n] += r[i] * r[j] * val;
        }
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void insert(const PixelND<V, N> & pixel)
    {
        insert(std::get<0>(pixel), std::get<1>(pixel));
    }

    template <typename InputIt, typename Value = iter_value_t<InputIt>, typename V = typename Value::second_type,
        typename = std::enable_if_t<std::is_same_v<PixelND<V, N>, Value> && std::is_convertible_v<T, V>>
    >
    void insert(InputIt first, InputIt last)
    {
        for (; first != last; ++first) insert(*first);
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
    std::array<T, N> first() const {return mu_x + org * mu;}
    std::array<T, N * N> second() const
    {
        std::array<T, N * N> matrix {};
        for (size_t n = 0; n < N; n++)
        {
            matrix[n + N * n] = mu_xx[n] + 2 * org[n] * mu_x[n] + org[n] * org[n] * mu;
        }
        for (size_t n = 0; n < NumPairs; n++)
        {
            auto [i, j] = UniquePairs<N>::instance().pairs(n);
            auto m_xy = mu_xy[n] + org[i] * mu_x[j] + org[j] * mu_x[i] + org[i] * org[j] * mu;
            matrix[i + N * j] = m_xy; matrix[j + N * i] = m_xy;
        }
        return matrix;
    }

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

    PointND<T, N> org;
    T mu;
    PointND<T, N> mu_x, mu_xx;
    PointND<T, NumPairs> mu_xy;
};

template <typename T>
using Moments = MomentsND<T, 2>;

// Connectivity structure class

template <size_t N>
struct StructureND : public WrappedContainer<std::vector<PointND<long, N>>>
{
public:
    using WrappedContainer<std::vector<PointND<long, N>>>::begin;
    using WrappedContainer<std::vector<PointND<long, N>>>::end;
    using WrappedContainer<std::vector<PointND<long, N>>>::size;

    int radius, rank;

    StructureND(int radius, int rank) : radius(radius), rank(rank)
    {
        PointND<long, N> shape;
        for (size_t i = 0; i < N; i++) shape[i] = 2 * radius + 1;
        for (auto point : rectangle_range<PointND<long, N>>{std::move(shape)})
        {
            point -= radius;
            long abs = 0;
            for (size_t i = 0; i < N; i++) abs += std::abs(point[i]);
            if (abs <= rank) m_ctr.emplace_back(std::move(point));
        }
    }

    StructureND & sort()
    {
        auto compare = [](const PointND<long, N> & a, const PointND<long, N> & b)
        {
            return magnitude(a) < magnitude(b);
        };
        std::sort(begin(), end(), compare);
        return *this;
    }

    std::string info() const
    {
        return "<StructureND, N = " + std::to_string(N) + ", radius = " + std::to_string(radius) +
               ", rank = " + std::to_string(rank) + ", points = <PointsND, size = " +  std::to_string(size()) + ">>";
    }

protected:
    using WrappedContainer<std::vector<PointND<long, N>>>::m_ctr;
};

using Structure = StructureND<2>;

// Extended interface of set of points - needed for Regions

template <size_t N>
class PointSetND : public WrappedContainer<std::set<PointND<long, N>>>
{
public:
    using WrappedContainer<std::set<PointND<long, N>>>::WrappedContainer;
    using WrappedContainer<std::set<PointND<long, N>>>::begin;
    using WrappedContainer<std::set<PointND<long, N>>>::end;
    using WrappedContainer<std::set<PointND<long, N>>>::size;

    PointSetND() = default;

    template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<bool, remove_cvref_t<Func>, PointND<long, N>>>>
    void dilate(Func && func, const StructureND<N> & structure)
    {
        std::vector<PointND<long, N>> last_pixels {begin(), end()};
        std::unordered_set<PointND<long, N>, detail::PointHasher<long, N>> new_pixels;

        while (last_pixels.size())
        {
            for (const auto & point: last_pixels)
            {
                for (const auto & shift: structure)
                {
                    new_pixels.insert(point + shift);
                }
            }
            last_pixels.clear();

            for (auto && point: new_pixels)
            {
                if (std::forward<Func>(func)(point))
                {
                    auto [iter, is_added] = m_ctr.insert(std::forward<decltype(point)>(point));
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename Func, typename Stop, typename = std::enable_if_t<
        std::is_invocable_r_v<bool, remove_cvref_t<Func>, PointND<long, N>> &&
        std::is_invocable_r_v<bool, remove_cvref_t<Stop>, const PointSetND<N> &>
    >>
    void dilate(Func && func, const StructureND<N> & structure, Stop && stop)
    {
        std::vector<PointND<long, N>> last_pixels {begin(), end()};
        std::unordered_set<PointND<long, N>, detail::PointHasher<long, N>> new_pixels;

        while (last_pixels.size() && std::forward<Stop>(stop)(*this))
        {
            for (const auto & point: last_pixels)
            {
                for (const auto & shift: structure)
                {
                    new_pixels.insert(point + shift);
                }
            }
            last_pixels.clear();

            for (auto && point: new_pixels)
            {
                if (std::forward<Func>(func)(point))
                {
                    auto [iter, is_added] = m_ctr.insert(std::forward<decltype(point)>(point));
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<bool, remove_cvref_t<Func>, PointND<long, N>>>>
    void dilate(Func && func, const StructureND<N> & structure, size_t n_iter)
    {
        std::vector<PointND<long, N>> last_pixels {begin(), end()};
        std::unordered_set<PointND<long, N>, detail::PointHasher<long, N>> new_pixels;

        for (size_t n = 0; n < n_iter; n++)
        {
            for (const auto & point: last_pixels)
            {
                for (const auto & shift: structure)
                {
                    new_pixels.insert(point + shift);
                }
            }
            last_pixels.clear();

            for (auto && point: new_pixels)
            {
                if (std::forward<Func>(func)(point))
                {
                    auto [iter, is_added] = m_ctr.insert(std::forward<decltype(point)>(point));
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> & array, bool value) const
    {
        for (const auto & pt : m_ctr)
        {
            if (array.is_inbound(pt.coordinate())) array.at(pt.coordinate()) = value;
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> && array, bool value) const
    {
        mask(array, value);
    }

    std::string info() const
    {
        return "<PointsSetND, N = " + std::to_string(N) + ", size = " + std::to_string(m_ctr.size()) + ">";
    }

protected:
    using WrappedContainer<std::set<PointND<long, N>>>::m_ctr;
};

using PointSet = PointSetND<2>;

// Set of [point, value] pairs

template <typename T, size_t N>
class PixelsND
{
public:
    PixelsND() = default;

    PixelsND(const PixelSetND<T, N> & pset) : m_mnt(pset), m_pset(pset) {}
    PixelsND(PixelSetND<T, N> && pset) : m_mnt(pset), m_pset(std::move(pset)) {}

    PixelsND(const PointSetND<N> & points, const array<T> & data)
    {
        for (auto && pt : points)
        {
            if (data.is_inbound(pt.coordinate()))
            {
                m_pset.emplace_hint(m_pset.end(), make_pixel(std::forward<decltype(pt)>(pt), data));
            }
        }
        m_mnt = m_pset;
    }

    void merge(PixelsND & source)
    {
        auto first1 = m_pset.begin(), last1 = m_pset.end();
        auto first2 = source.m_pset.begin(), last2 = source.m_pset.end();
        for (; first1 != last1 && first2 != last2;)
        {
            if (*first2 < *first1)
            {
                m_mnt.insert(*first2);
                m_pset.insert(first1, source.m_pset.extract(first2++));
            }
            else if (*first2 > *first1) ++first1;
            else
            {
                ++first1; ++first2;
            }
        }
        for (; first2 != last2;)
        {
            m_mnt.insert(*first2);
            m_pset.insert(first1, source.m_pset.extract(first2++));
        }
    }

    void merge(PixelsND && source)
    {
        merge(source);
    }

    template <size_t M = N, typename = std::enable_if_t<(M == 2)>>
    Line<T> line() const
    {
        if (m_mnt.zeroth()) return m_mnt.central().line();
        return {m_mnt.origin(), m_mnt.origin()};
    }

    const PixelSetND<T, N> & pixels() const {return m_pset;}
    const MomentsND<T, N> & moments() const {return m_mnt;}

protected:
    MomentsND<T, N> m_mnt;
    PixelSetND<T, N> m_pset;
};

template <typename T>
using Pixels = PixelsND<T, 2>;

template <typename InputIt, typename I, size_t N, typename = std::enable_if_t<std::is_integral_v<I>>>
std::vector<PointSetND<N>> labelise(InputIt first, InputIt last, array<I> & mask, const StructureND<N> & structure, size_t npts)
{
    std::vector<PointSetND<N>> regions;
    auto func = [&mask](const PointND<long, N> & pt)
    {
        return mask.is_inbound(pt.coordinate()) && mask.at(pt.coordinate());
    };

    for (; first != last; ++first)
    {
        size_t index = mask.index_at(first->coordinate());
        if (mask[index])
        {
            PointSetND<N> points;
            points->insert(*first);
            points.dilate(func, structure);
            points.mask(mask, false);
            if (points.size() >= npts) regions.emplace_back(std::move(points));
        }
    }

    return regions;
}

template <typename InputIt, typename I, size_t N, typename = std::enable_if_t<std::is_integral_v<I>>>
std::vector<PointSetND<N>> labelise(InputIt first, InputIt last, array<I> && mask, const StructureND<N> & structure, size_t npts)
{
    return labelise(first, last, mask, structure, npts);
}

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
