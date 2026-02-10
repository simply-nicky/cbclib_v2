#ifndef NEW_LABEL_H_
#define NEW_LABEL_H_
#include "geometry.hpp"
#include "numpy.hpp"

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

    friend std::ostream & operator<<(std::ostream & os, const CentralMomentsND & m)
    {
        os << "{origin = " << m.origin << ", mu_x = " << m.mu_x
           << ", mu_xx = " << m.mu_xx << ", mu_xy = " << m.mu_xy << "}";
        return os;
    }

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;
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

    MomentsND(const PixelSetND<T, N> & pset)
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

    PointND<T, N> org {};               // origin
    T mu = T();                         // zeroth moment
    PointND<T, N> mu_x {}, mu_xx {};    // mu_x: first central moments, mu_xx: second central moments
    PointND<T, NumPairs> mu_xy {};      // cross second central moments
};

template <typename T>
using Moments = MomentsND<T, 2>;

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
public:

    using iterator = chunk_iterator<false>;
    using const_iterator = chunk_iterator<true>;
    using size_type = size_t;
    using reference = chunk<false>;
    using const_reference = chunk<true>;

    int connectivity;

    template <typename Radii>
    Structure(const Radii & radii, int connectivity) : connectivity(connectivity)
    {
        for (size_t n = 0; n < radii.size(); n++) m_shape.push_back(2 * radii[n] + 1);
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

// Extended interface of set of points - needed for Regions
class Region
{
public:
    using size_type = size_t;
    using iterator = typename std::set<long>::iterator;
    using const_iterator = typename std::set<long>::const_iterator;

    iterator begin() {return m_ctr.begin();}
    const_iterator begin() const {return m_ctr.begin();}

    iterator end() {return m_ctr.end();}
    const_iterator end() const {return m_ctr.end();}

    size_type size() const {return m_ctr.size();}

    std::pair<iterator, bool> insert(long index)
    {
        return m_ctr.insert(index);
    }

    iterator insert(iterator hint, long index)
    {
        return m_ctr.insert(hint, index);
    }

    template <typename Func, typename I, typename = std::enable_if_t<std::is_invocable_r_v<bool, remove_cvref_t<Func>, long>>>
    void dilate(Func && func, const Structure & structure, const array_indexer_view<I> & indexer)
    {
        std::vector<long> last_pixels {begin(), end()};
        std::set<long> new_pixels;

        while (last_pixels.size())
        {
            for (auto index : last_pixels)
            {
                for (const auto & shift: structure)
                {
                    // Add new index if in bounds
                    long new_index = shift_index(index, shift, indexer);
                    if (new_index >= 0) new_pixels.insert(new_index);
                }
            }
            last_pixels.clear();

            for (auto index : new_pixels)
            {
                if (std::forward<Func>(func)(index))
                {
                    auto [iter, is_added] = m_ctr.insert(index);
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename Func, typename Stop, typename I, typename = std::enable_if_t<
        std::is_invocable_r_v<bool, remove_cvref_t<Func>, long> &&
        std::is_invocable_r_v<bool, remove_cvref_t<Stop>, const Region &>
    >>
    void dilate(Func && func, const Structure & structure, Stop && stop, const array_indexer_view<I> & indexer)
    {
        std::vector<long> last_pixels {begin(), end()};
        std::set<long> new_pixels;

        while (last_pixels.size() && std::forward<Stop>(stop)(*this))
        {
            for (auto index : last_pixels)
            {
                for (const auto & shift: structure)
                {
                    // Add new index if in bounds
                    long new_index = shift_index(index, shift, indexer);
                    if (new_index >= 0) new_pixels.insert(new_index);
                }
            }
            last_pixels.clear();

            for (auto index : new_pixels)
            {
                if (std::forward<Func>(func)(index))
                {
                    auto [iter, is_added] = m_ctr.insert(index);
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename Func, typename I, typename = std::enable_if_t<std::is_invocable_r_v<bool, remove_cvref_t<Func>, long>>>
    void dilate(Func && func, const Structure & structure, size_t n_iter, const array_indexer_view<I> & indexer)
    {
        std::vector<long> last_pixels {begin(), end()};
        std::set<long> new_pixels;

        for (size_t n = 0; n < n_iter; n++)
        {
            for (auto index : last_pixels)
            {
                for (const auto & shift: structure)
                {
                    // Add new index if in bounds
                    long new_index = shift_index(index, shift, indexer);
                    if (new_index >= 0) new_pixels.insert(new_index);
                }
            }
            last_pixels.clear();

            for (auto index : new_pixels)
            {
                if (std::forward<Func>(func)(index))
                {
                    auto [iter, is_added] = m_ctr.insert(index);
                    if (is_added) last_pixels.push_back(*iter);
                }
            }
            new_pixels.clear();
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> & array, I value) const
    {
        for (auto index : m_ctr) array[index] = value;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> && array, I value) const
    {
        mask(array, value);
    }

    // Region follow xyz ordering
    template <size_t N>
    PointND<long, N> to_point(long index, const array_indexer & indexer) const
    {
        PointND<long, N> point;
        for (size_t n = N; n > 0; --n)
        {
            size_t zyx_n = n - 1, xyz_n = N - n;
            point[xyz_n] = index % indexer.shape(zyx_n);
            index /= indexer.shape(zyx_n);
        }
        return point;
    }

    std::string info() const
    {
        return "<Region, size = " +  std::to_string(size()) + ">";
    }

protected:
    std::set<long> m_ctr;

    template <typename Container, typename I>
    long shift_index(long index, const Container & shift, const array_indexer_view<I> & indexer) const
    {
        long new_index = 0;
        long stride = 1;
        for (size_t n = shift.size(); n > 0; --n)
        {
            long coord = index % indexer.shape(n - 1) + shift[n - 1];
            if (coord < 0 || coord >= indexer.shape(n - 1)) return -1;

            new_index += coord * stride;
            index /= indexer.shape(n - 1);
            stride *= indexer.shape(n - 1);
        }
        return new_index;
    }
};

class LabelResult
{
public:
    LabelResult(array_indexer && indexer, std::vector<Region> && regions) :
        m_indexer(std::move(indexer)), m_regions(std::move(regions)) {}

    LabelResult(const array_indexer & indexer, const std::vector<Region> & regions) :
        m_indexer(indexer), m_regions(regions) {}

    const std::vector<Region> & regions() const {return m_regions;}
    std::vector<Region> & regions() {return m_regions;}

    size_t coord_along_dim(long index, size_t dim) const {return m_indexer.coord_along_dim(index, dim);}

    const std::vector<size_t> & shape() const {return m_indexer.shape();}
    size_t shape(size_t index) const {return m_indexer.shape(index);}

protected:
    array_indexer m_indexer;
    std::vector<Region> m_regions;
};

// Set of [point, value] pairs

template <typename T, size_t N>
class PixelsND
{
public:
    PixelsND() = default;

    PixelsND(const PixelSetND<T, N> & pset) : m_mnt(pset), m_pset(pset) {}
    PixelsND(PixelSetND<T, N> && pset) : m_mnt(pset), m_pset(std::move(pset)) {}

    PixelsND(const Region & points, const array<T> & data)
    {
        for (auto index : points)
        {
            m_pset.emplace_hint(m_pset.end(), make_pixel(points.to_point<N>(index, data), data));
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
