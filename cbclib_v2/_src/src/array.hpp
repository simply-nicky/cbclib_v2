#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

namespace cbclib {

template <typename T, typename ... Types>
concept is_all_same = (... && std::is_same_v<T, Types>);

template <typename ... Types>
concept is_all_integral = (... && std::is_integral_v<Types>);

namespace detail{

static const size_t GOLDEN_RATIO = 0x9e3779b9;

template <typename T>
inline constexpr int signum(T x, std::false_type is_signed)
{
    return T(0) < x;
}

template <typename T>
inline constexpr int signum(T x, std::true_type is_signed)
{
    return (T(0) < x) - (x < T(0));
}

template <typename T>
inline constexpr int signum(T x)
{
    return signum(x, std::is_signed<T>());
}

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

/* Returns a quotient: a = quotient * b + modulo(a, b) */
template <typename T, typename U>
constexpr auto quotient(T a, U b) -> decltype(modulo(a, b))
{
    return (a - modulo(a, b)) / b;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> mirror(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min) - 1;
    if (modulo(quotient(val, period), 2)) return period - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> reflect(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    if (modulo(quotient(val, period), 2)) return period - 1 - modulo(val, period) + min;
    else return modulo(val, period) + min;
}

template <typename T, typename U, typename V>
constexpr std::make_signed_t<T> wrap(T a, U min, V max)
{
    using F = std::make_signed_t<T>;
    F val = std::minus<F>()(a, min);
    F period = std::minus<F>()(max, min);
    return modulo(val, period) + min;
}

template <typename InputIt1, typename InputIt2>
auto index_offset_impl(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    using value_t = decltype(+*std::declval<InputIt1 &>());
    value_t index = value_t();
    for (; cfirst != clast; cfirst++, ++sfirst) index += *cfirst * *sfirst;
    return index;
}

template <size_t Dim = 0, typename Strides>
size_t index_offset_unsafe(const Strides & strides)
{
    return 0;
}

template <size_t Dim = 0, typename Strides, typename... Ix>
size_t index_offset_unsafe(const Strides & strides, size_t i, Ix... index)
{
    return i * strides[Dim] + index_offset_unsafe<Dim + 1>(strides, index...);
}

template <typename InputIt, typename OutputIt, typename T>
OutputIt unravel_index_unsafe(InputIt sfirst, InputIt slast, size_t itemsize, T index, OutputIt cfirst)
{
    for (; sfirst != slast; ++sfirst)
    {
        if (*sfirst)
        {
            auto coord = index / (*sfirst / itemsize);
            index -= coord * *sfirst;
            *cfirst++ = coord;
        }
        else *cfirst++ = 0;
    }
    return cfirst;
}

template <typename Strides>
size_t offset_along_dim(const Strides & strides, size_t index, size_t dim)
{
    if (dim == 0) return index;
    if (dim >= strides.size()) return 0;

    size_t offset = offset_along_dim(strides, index, dim - 1);
    return offset - (offset / strides[dim - 1]) * strides[dim - 1];
}

class shape_handler
{
protected:
    using ShapeContainer = AnyContainer<size_t>;

public:
    using size_type = size_t;

    shape_handler() = default;

    shape_handler(ShapeContainer sh, ShapeContainer st) : m_ndim(sh.size()), m_shape(std::move(sh))
    {
        for (size_t n = 0; n < m_ndim; n++) m_strides.push_back(st[n]);
    }

    shape_handler(ShapeContainer sh) : m_ndim(sh.size()), m_shape(std::move(sh)), m_strides(m_ndim, 1)
    {
        if (m_ndim)
        {
            for (size_t n = m_ndim - 1; n > 0; --n) m_strides[n - 1] = m_strides[n] * m_shape[n];
        }
    }

    size_t index_along_dim(size_t index, size_t dim) const
    {
        if (dim >= m_ndim) fail_dim_check(dim, "invalid axis");
        return offset_along_dim(m_strides, index, dim) / m_strides[dim];
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator_v<CoordIter>>>
    bool is_inbound(CoordIter first, CoordIter last) const
    {
        bool flag = true;
        for (size_t i = 0; first != last; ++first, ++i)
        {
            flag &= *first >= 0 && *first < static_cast<decltype(+*std::declval<CoordIter &>())>(m_shape[i]);
        }
        return flag;
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    bool is_inbound(const Container & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    bool is_inbound(const std::initializer_list<T> & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    auto index_at(CoordIter first, CoordIter last) const
    {
        return offset_at(first, last) / itemsize();
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto index_at(const Container & coord) const
    {
        return offset_at(coord) / itemsize();
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto index_at(const std::initializer_list<T> & coord) const
    {
        return offset_at(coord) / itemsize();
    }

    template <typename ... Ix> requires is_all_integral<Ix ...>
    auto index_at(Ix... index) const
    {
        return offset_at(index...) / itemsize();
    }

    template <typename OutputIt> requires std::output_iterator<OutputIt, size_t>
    OutputIt unravel_index(OutputIt first, size_t index) const
    {
        return unravel_index_unsafe(m_strides.begin(), m_strides.end(), itemsize(), index, first);
    }

    size_t ndim() const {return m_ndim;}
    size_t size() const {return std::reduce(m_shape.begin(), m_shape.end(), size_t(1), std::multiplies());}

    const std::vector<size_t> & shape() const {return m_shape;}
    size_t shape(size_t dim) const
    {
        if (dim >= m_ndim) fail_dim_check(dim, "invalid axis");
        return m_shape[dim];
    }

    const std::vector<size_t> & strides() const {return m_strides;}
    size_t strides(size_t dim) const
    {
        if (dim >= m_ndim) fail_dim_check(dim, "invalid axis");
        return m_strides[dim];
    }

protected:
    size_t m_ndim;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator<CoordIter>::value>>
    auto offset_at(CoordIter first, CoordIter last) const
    {
        return index_offset_impl(first, last, m_strides.begin());
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto offset_at(const Container & coord) const
    {
        return index_offset_impl(coord.begin(), coord.end(), m_strides.begin());
    }

    template <typename ... Ix> requires is_all_integral<Ix ...>
    auto offset_at(Ix... index) const
    {
        if (sizeof...(index) > m_ndim) fail_dim_check(sizeof...(index), "too many indices for an array");

        return index_offset_unsafe(m_strides, size_t(index)...);
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto offset_at(const std::initializer_list<T> & coord) const
    {
        return index_offset_impl(coord.begin(), coord.end(), m_strides.begin());
    }

    size_t itemsize() const
    {
        if (m_ndim) return strides(m_ndim - 1);
        return size_t(1);
    }

    void fail_dim_check(size_t dim, const std::string & msg) const
    {
        throw std::out_of_range(msg + ": " + std::to_string(dim) + " (ndim = " + std::to_string(m_ndim) + ')');
    }
};

// Taken from the boost::hash_combine: https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
template <class T>
inline size_t hash_combine(size_t seed, const T & v)
{
    //  Golden Ratio constant used for better hash scattering
    //  See https://softwareengineering.stackexchange.com/a/402543
    return seed ^ (std::hash<T>()(v) + GOLDEN_RATIO + (seed << 6) + (seed >> 2));
}

template <typename T, size_t N>
struct ArrayHasher
{
    size_t operator()(const std::array<T, N> & arr) const
    {
        size_t h = 0;
        for (auto elem : arr) h = hash_combine(h, elem);
        return h;
    }
};

// Recursive template code derived from Matthieu M.
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl
{
    static size_t apply(size_t seed, const Tuple & tuple)
    {
        seed = HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        return hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0>
{
    static size_t apply(size_t seed, const Tuple & tuple)
    {
        return hash_combine(seed, std::get<0>(tuple));
    }
};

template <typename ... Ts>
struct TupleHasher
{
    size_t operator()(const std::tuple<Ts...> & tt) const
    {
        return HashValueImpl<std::tuple<Ts...>>::apply(0, tt);
    }
};

template <typename T1, typename T2>
struct PairHasher
{
    size_t operator()(const std::pair<T1, T2> & tt) const
    {
        return HashValueImpl<std::pair<T1, T2>>::apply(0, tt);
    }
};

}

template <typename T, bool IsConst>
struct StrideIteratorTraits;

template <typename T>
struct StrideIteratorTraits<T, false>
{
    using value_type = T;
    using pointer = T *;
    using reference = T &;
};

template <typename T>
struct StrideIteratorTraits<T, true>
{
    using value_type = const T;
    using pointer = const T *;
    using reference = const T &;
};

template <typename T>
class array;

template <typename T, bool IsConst>
class StrideIterator
{
private:
    friend class StrideIterator<T, !IsConst>;
    friend class array<T>;
    using traits = StrideIteratorTraits<T, IsConst>;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename traits::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename traits::pointer;
    using reference = typename traits::reference;

    StrideIterator() : ptr(nullptr), stride(1) {}

    // This is templated so that we can allow constructing a const iterator from
    // a nonconst iterator...
    template <bool RHIsConst, typename = std::enable_if_t<IsConst || !RHIsConst>>
    StrideIterator(const StrideIterator<T, RHIsConst> & rhs) : ptr(rhs.ptr), stride(rhs.stride) {}

    operator bool() const {return bool(ptr);}

    bool operator==(const StrideIterator & rhs) const {return ptr == rhs.ptr;}
    bool operator!=(const StrideIterator & rhs) const {return ptr != rhs.ptr;}
    bool operator<=(const StrideIterator & rhs) const {return ptr <= rhs.ptr;}
    bool operator>=(const StrideIterator & rhs) const {return ptr >= rhs.ptr;}
    bool operator<(const StrideIterator & rhs) const {return ptr < rhs.ptr;}
    bool operator>(const StrideIterator & rhs) const {return ptr > rhs.ptr;}

    StrideIterator & operator+=(const difference_type & step) {ptr += step * stride; return *this;}
    StrideIterator & operator-=(const difference_type & step) {ptr -= step * stride; return *this;}
    StrideIterator & operator++() {ptr += stride; return *this;}
    StrideIterator & operator--() {ptr -= stride; return *this;}
    StrideIterator operator++(int) {StrideIterator temp = *this; ++(*this); return temp;}
    StrideIterator operator--(int) {StrideIterator temp = *this; --(*this); return temp;}
    StrideIterator operator+(const difference_type & step) const
    {
        return {ptr + step * stride, stride};
    }
    StrideIterator operator-(const difference_type & step) const
    {
        return {ptr - step * stride, stride};
    }

    difference_type operator-(const StrideIterator & rhs) const {return (ptr - rhs.ptr) / stride;}

    reference operator[] (size_t index) const {return *(ptr + index * stride);}
    reference operator*() const {return *ptr;}
    pointer operator->() const {return ptr;}

private:
    T * ptr;
    size_t stride;

    StrideIterator(T * ptr, size_t stride = 1) : ptr(ptr), stride(stride) {}
};

template <typename T>
class array : public detail::shape_handler
{
public:
    using value_type = T;
    using size_type = typename detail::shape_handler::size_type;
    using iterator = StrideIterator<T, false>;
    using const_iterator = StrideIterator<T, true>;

    operator py::array_t<T>() const {return {m_shape, m_strides, ptr};}

    array() : shape_handler(), ptr(nullptr) {}

    array(ShapeContainer shape, ShapeContainer strides, T * ptr) :
        shape_handler(std::move(shape), std::move(strides)), ptr(ptr) {}

    array(shape_handler handler, T * ptr) : shape_handler(std::move(handler)), ptr(ptr) {}

    array(ShapeContainer shape, T * ptr) : shape_handler(std::move(shape)), ptr(ptr) {}

    array(size_t count, T * ptr) : shape_handler({count}) , ptr(ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, static_cast<T *>(buf.ptr)) {}

    T & operator[] (size_t index) {return *(ptr + itemsize() * index);}
    const T & operator[] (size_t index) const {return *(ptr + itemsize() * index);}

    iterator begin() {return {ptr, itemsize()};}
    iterator end() {return {ptr + size() * itemsize(), itemsize()};}
    const_iterator begin() const {return {ptr, itemsize()};}
    const_iterator end() const {return {ptr + size() * itemsize(), itemsize()};}

    array<T> reshape(ShapeContainer new_shape) const
    {
        return {std::move(new_shape), ptr};
    }

    /* Slice sub-array:
        Take a slice of an array 'array' as follows:
        - array[..., :, ...] slice, where ':' is at axis
    */
    array<T> slice(size_t index, size_t axis) const
    {
        if (!m_ndim) return *this;

        axis = axis  % m_ndim;
        size_t offset = size_t();
        if (size())
        {
            index = index % size();
            for (size_t n = m_ndim; n > 0; --n)
            {
                if (n - 1 != axis)
                {
                    auto coord = index % m_shape[n - 1];
                    index /= m_shape[n - 1];
                    offset += m_strides[n - 1] * coord;
                }
            }
        }
        return array<T>{std::vector<size_t>{m_shape[axis]},
                        std::vector<size_t>{m_strides[axis]},
                        ptr + offset};
    }

    /* Slice sub-array:
        Take a slice of an array 'array' as follows:
        - array[..., :, :], where array[..., :, :].ndim() = ndim
    */
    array<T> slice_back(size_t index, size_t ndim) const
    {
        if (!m_ndim) return *this;

        if (!ndim) return array<T>{std::vector<size_t>{}, ptr};
        if (ndim < m_ndim)
        {
            size_t offset = size_t();
            if (size())
            {
                index = index % size();
                for (size_t n = m_ndim - ndim; n > 0; --n)
                {
                    auto coord = index % m_shape[n - 1];
                    index /= m_shape[n - 1];
                    offset += m_strides[n - 1] * coord;
                }
            }
            return array<T>{std::vector<size_t>{std::prev(m_shape.end(), ndim), m_shape.end()},
                            std::vector<size_t>{std::prev(m_strides.end(), ndim), m_strides.end()},
                            ptr + offset};
        }
        return *this;
    }

    /* Slice sub-array:
        Take a slice of an array 'array' as follows:
        - array[:, :, ...], where array[:, :, ...].ndim() = ndim
    */
    array<T> slice_front(size_t index, size_t ndim) const
    {
        if (!m_ndim) return *this;

        if (!ndim) return array<T>{std::vector<size_t>{}, ptr};
        if (ndim < m_ndim)
        {
            size_t offset = size_t();
            if (size())
            {
                index = index % size();
                for (size_t n = m_ndim; n > ndim; --n)
                {
                    auto coord = index % m_shape[n - 1];
                    index /= m_shape[n - 1];
                    offset += m_strides[n - 1] * coord;
                }
            }
            return array<T>{std::vector<size_t>{m_shape.begin(), std::next(m_shape.begin(), ndim)},
                            std::vector<size_t>{m_strides.begin(), std::next(m_strides.begin(), ndim)},
                            ptr + offset};
        }
        return *this;
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator_v<CoordIter>>>
    const T & at(CoordIter first, CoordIter last) const
    {
        return *(ptr + offset_at(first, last));
    }

    template <typename CoordIter, typename = std::enable_if_t<is_input_iterator_v<CoordIter>>>
    T & at(CoordIter first, CoordIter last)
    {
        return *(ptr + offset_at(first, last));
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    const T & at(const Container & coord) const
    {
        return *(ptr + offset_at(coord));
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T & at(const Container & coord)
    {
        return *(ptr + offset_at(coord));
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    const T & at(const std::initializer_list<I> & coord) const
    {
        return *(ptr + offset_at(coord));
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    T & at(const std::initializer_list<I> & coord)
    {
        return *(ptr + offset_at(coord));
    }

    template <typename ... Ix> requires is_all_integral<Ix ...>
    const T & at(Ix... index) const
    {
        return *(ptr + offset_at(index...));
    }

    template <typename ... Ix> requires is_all_integral<Ix ...>
    T & at(Ix... index)
    {
        return *(ptr + offset_at(index...));
    }

    const T * data() const {return ptr;}
    T * data() {return ptr;}

protected:
    T * ptr;

    void set_data(T * new_ptr) {ptr = new_ptr;}
};

template <typename T>
class vector_array : public array<T>
{
protected:
    using ShapeContainer = detail::shape_handler::ShapeContainer;
    using array<T>::set_data;

    std::vector<T> m_data;

public:
    using array<T>::size;

    vector_array() = default;

    template <typename Vector, typename = std::enable_if_t<std::is_base_of_v<std::vector<T>, std::remove_cvref_t<Vector>>>>
    vector_array(ShapeContainer shape, Vector && v) : array<T>(std::move(shape), v.data()), m_data(std::forward<Vector>(v))
    {
        if (m_data.size() != size()) m_data.resize(size());
    }

    vector_array(ShapeContainer shape, T value = T()) : array<T>(std::move(shape), nullptr), m_data(size(), value)
    {
        set_data(m_data.data());
    }
};

/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangular iterator ---------------------------*/
/*----------------------------------------------------------------------------*/

template <typename Container = std::vector<size_t>, bool IsPoint = false>
struct rectangle_range
{
public:
    class rectangle_iterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Container;
        using difference_type = std::ptrdiff_t;
        using pointer = const Container *;
        using reference = const Container &;

        size_t index() const {return m_index;}

        rectangle_iterator & operator++()
        {
            m_index++;
            update();
            return *this;
        }

        rectangle_iterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        rectangle_iterator & operator--()
        {
            m_index--;
            update();
            return *this;
        }

        rectangle_iterator operator--(int)
        {
            auto saved = *this;
            operator--();
            return saved;
        }

        rectangle_iterator & operator+=(difference_type offset)
        {
            m_index += offset;
            update();
            return *this;
        }

        rectangle_iterator operator+(difference_type offset) const
        {
            auto saved = *this;
            return saved += offset;
        }

        rectangle_iterator & operator-=(difference_type offset)
        {
            m_index -= offset;
            update();
            return *this;
        }

        rectangle_iterator operator-(difference_type offset) const
        {
            auto saved = *this;
            return saved -= offset;
        }

        difference_type operator-(const rectangle_iterator & rhs) const
        {
            return m_index - rhs.m_index;
        }

        reference operator[](difference_type offset) const
        {
            return *(*this + offset);
        }

        bool operator==(const rectangle_iterator & rhs) const {return m_coord == rhs.m_coord;}
        bool operator!=(const rectangle_iterator & rhs) const {return !(*this == rhs);}

        bool operator<(const rectangle_iterator & rhs) const {return m_index < rhs.m_index;}
        bool operator>(const rectangle_iterator & rhs) const {return m_index > rhs.m_index;}

        bool operator<=(const rectangle_iterator & rhs) const {return !(*this > rhs);}
        bool operator>=(const rectangle_iterator & rhs) const {return !(*this < rhs);}

        reference operator*() const {return m_coord;}
        pointer operator->() const {return &m_coord;}

    private:
        Container m_coord, m_strides;
        size_t m_index;


        rectangle_iterator(Container st, size_t idx) : m_coord(st), m_strides(std::move(st)), m_index(idx)
        {
            update();
        }

        void update()
        {
            if constexpr(IsPoint)
            {
                detail::unravel_index_unsafe(m_strides.begin(), m_strides.end(), 1, m_index, m_coord.rbegin());
            }
            else
            {
                detail::unravel_index_unsafe(m_strides.begin(), m_strides.end(), 1, m_index, m_coord.begin());
            }
        }

        friend class rectangle_range;
    };

    using iterator = rectangle_iterator;
    using reverse_iterator = std::reverse_iterator<rectangle_iterator>;

    rectangle_range(Container sh) : strides(sh), shape(std::move(sh)), size(1)
    {
        for (auto length : shape) size *= length;

        for (size_t stride = size, i = 0; i < shape.size(); i++)
        {
            stride = (shape[i]) ? stride / shape[i] : stride;
            strides[i] = stride;
        }
    }

    iterator begin() const {return iterator(strides, 0);}
    iterator end() const {return iterator(strides, size);}

    reverse_iterator rbegin() const {return reverse_iterator(strides, 0);}
    reverse_iterator rend() const {return reverse_iterator(strides, size);}

private:
    Container strides, shape;
    size_t size;
};

/* Iterator adapter for point containers for pybind11 */
/* python_point_iterator dereferences to an std::array instead of PointND */

template <typename Iterator, typename = decltype(std::declval<Iterator &>()->to_array())>
class python_point_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename std::remove_reference_t<decltype(std::declval<Iterator &>()->to_array())>;
    using difference_type = typename std::iter_difference_t<Iterator>;
    using reference = const value_type &;
    using pointer = const value_type *;

    python_point_iterator() = default;
    python_point_iterator(Iterator && iter) : m_iter(std::move(iter)) {}
    python_point_iterator(const Iterator & iter) : m_iter(iter) {}

    python_point_iterator & operator++() requires (std::forward_iterator<Iterator>)
    {
        ++m_iter;
        return *this;
    }

    python_point_iterator operator++(int) requires (std::forward_iterator<Iterator>)
    {
        return python_point_iterator(m_iter++);
    }

    python_point_iterator & operator--() requires (std::bidirectional_iterator<Iterator>)
    {
        --m_iter;
        return *this;
    }

    python_point_iterator operator--(int) requires (std::bidirectional_iterator<Iterator>)
    {
        return python_point_iterator(m_iter--);
    }

    python_point_iterator & operator+=(difference_type offset) requires (std::random_access_iterator<Iterator>)
    {
        m_iter += offset;
        return *this;
    }

    python_point_iterator operator+(difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return python_point_iterator(m_iter + offset);
    }

    python_point_iterator & operator-=(difference_type offset) requires (std::random_access_iterator<Iterator>)
    {
        m_iter -= offset;
        return *this;
    }

    python_point_iterator operator-(difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return python_point_iterator(m_iter - offset);
    }

    difference_type operator-(const python_point_iterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter - rhs;
    }

    reference operator[](difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return (m_iter + offset)->to_array();
    }

    bool operator==(const python_point_iterator & rhs) const requires (std::forward_iterator<Iterator>)
    {
        return m_iter == rhs.m_iter;
    }
    bool operator!=(const python_point_iterator & rhs) const requires (std::forward_iterator<Iterator>)
    {
        return !(*this == rhs);
    }

    bool operator<(const python_point_iterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter < rhs.m_iter;
    }
    bool operator>(const python_point_iterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter > rhs.m_iter;
    }

    bool operator<=(const python_point_iterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return !(*this > rhs);
    }
    bool operator>=(const python_point_iterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return !(*this < rhs);
    }

    reference operator*() const {return m_iter->to_array();}
    pointer operator->() const {return &(m_iter->to_array());}

private:
    Iterator m_iter;
};

template <typename Iterator, typename = decltype(std::declval<Iterator &>()->to_array())>
python_point_iterator<Iterator> make_python_iterator(Iterator && iterator)
{
    return python_point_iterator(std::forward<Iterator>(iterator));
}

/*----------------------------------------------------------------------------*/
/*--------------------------- Extend line modes ------------------------------*/
/*----------------------------------------------------------------------------*/
/*
    constant: kkkkkkkk|abcd|kkkkkkkk
    nearest:  aaaaaaaa|abcd|dddddddd
    mirror:   cbabcdcb|abcd|cbabcdcb
    reflect:  abcddcba|abcd|dcbaabcd
    wrap:     abcdabcd|abcd|abcdabcd
*/
enum class extend
{
    constant = 0,
    nearest = 1,
    mirror = 2,
    reflect = 3,
    wrap = 4
};

static std::unordered_map<std::string, extend> const modes = {{"constant", extend::constant},
                                                              {"nearest", extend::nearest},
                                                              {"mirror", extend::mirror},
                                                              {"reflect", extend::reflect},
                                                              {"wrap", extend::wrap}};

/*----------------------------------------------------------------------------*/
/*-------------------------------- Kernels -----------------------------------*/
/*----------------------------------------------------------------------------*/
/* All kernels defined with the support of [-1, 1]. */
namespace detail {

// 1 / sqrt(2 * pi)
static constexpr double M_1_SQRT2PI = 0.3989422804014327;

template <typename T>
T rectangular(T x) {return (std::abs(x) <= T(1.0)) ? T(1.0) : T();}

template <typename T>
T gaussian(T x)
{
    if (std::abs(x) <= T(1.0)) return M_1_SQRT2PI * std::exp(-std::pow(3 * x, 2) / 2);
    return T();
}

template <typename T>
T gaussian_grad(T x) {return -9 * x * gaussian(x);}

template <typename T>
T triangular(T x) {return std::max<T>(T(1.0) - std::abs(x), T());}

template <typename T>
T triangular_grad(T x)
{
    if (std::abs(x) < T(1.0)) return -signum(x);
    return T();
}

template <typename T>
T parabolic(T x) {return T(0.75) * std::max<T>(1 - std::pow(x, 2), T());}

template <typename T>
T parabolic_grad(T x)
{
    if (std::abs(x) < T(1.0)) return T(0.75) * -2 * x;
    return T();
}

template <typename T>
T biweight(T x) {return T(0.9375) * std::pow(std::max<T>(1 - std::pow(x, 2), T()), 2);}

template <typename T>
T biweight_grad(T x)
{
    if (std::abs(x) < T(1.0)) return T(0.9375) * -4 * x * (1 - std::pow(x, 2));
    return T();
}

}

template <typename T>
struct kernels
{
    enum kernel_type
    {
        biweight = 0,
        gaussian = 1,
        parabolic = 2,
        rectangular = 3,
        triangular = 4
    };

    using kernel = T (*)(T);
    using gradient = T (*)(T);

    static inline std::map<std::string, kernel_type> kernel_names =
    {
        {"biweight", kernel_type::biweight},
        {"gaussian", kernel_type::gaussian},
        {"parabolic", kernel_type::parabolic},
        {"rectangular", kernel_type::rectangular},
        {"triangular", kernel_type::triangular}
    };

    static inline std::map<kernel_type, std::pair<kernel, kernel>> registered_kernels =
    {
        {kernel_type::biweight, {detail::biweight<T>, detail::biweight_grad<T>}},
        {kernel_type::gaussian, {detail::gaussian<T>, detail::gaussian_grad<T>}},
        {kernel_type::parabolic, {detail::parabolic<T>, detail::parabolic_grad<T>}},
        {kernel_type::rectangular, {detail::rectangular<T>, nullptr}},
        {kernel_type::triangular, {detail::triangular<T>, detail::triangular_grad<T>}}
    };

    static kernel get_kernel(kernel_type k, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(k);
        if (it != registered_kernels.end()) return it->second.first;
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + std::to_string(k));
        return nullptr;
    }

    static kernel get_kernel(std::string name, bool throw_if_missing = true)
    {
        auto it = kernel_names.find(name);
        if (it != kernel_names.end()) return get_kernel(it->second, throw_if_missing);
        if (throw_if_missing)
            throw std::invalid_argument("kernel is missing for " + name);
        return nullptr;
    }

    static kernel get_grad(kernel_type k, bool throw_if_missing = true)
    {
        auto it = registered_kernels.find(k);
        if (it != registered_kernels.end() && it->second.second) return it->second.second;
        if (throw_if_missing)
            throw std::invalid_argument("gradient is missing for " + std::to_string(k));
        return nullptr;
    }

    static kernel get_grad(std::string name, bool throw_if_missing = true)
    {
        auto it = kernel_names.find(name);
        if (it != kernel_names.end()) return get_grad(it->second, throw_if_missing);
        if (throw_if_missing)
            throw std::invalid_argument("gradient is missing for " + name);
        return nullptr;
    }


};

/*----------------------------------------------------------------------------*/
/*-------------- Compile-time to_array, to_tuple, and to_tie -----------------*/
/*----------------------------------------------------------------------------*/
namespace detail {
template <size_t... I>
constexpr auto integral_sequence_impl(std::index_sequence<I...>)
{
  return std::make_tuple(std::integral_constant<size_t, I>{}...);
}

template <typename T, T... I, size_t... J>
constexpr auto reverse_impl(std::integer_sequence<T, I...>, std::index_sequence<J...>)
{
    return std::integer_sequence<T, std::get<sizeof...(J) - J - 1>(std::make_tuple(I...))...>{};
}

}

template <size_t N>
constexpr auto integral_sequence()
{
    return detail::integral_sequence_impl(std::make_index_sequence<N>{});
}

template <typename T, T... I>
constexpr auto reverse_sequence(std::integer_sequence<T, I...> seq)
{
    return detail::reverse_impl(seq, std::make_index_sequence<sizeof...(I)>{});
}

template <size_t N, class Func>
constexpr decltype(auto) apply_to_sequence(Func && func)
{
    return std::apply(std::forward<Func>(func), integral_sequence<N>());
}

template <size_t N, class Container, typename T = Container::value_type>
constexpr std::array<T, N> to_array(const Container & a, size_t start)
{
    auto impl = [&a, start](auto... idxs) -> std::array<T, N> {return {{a[start + idxs]...}};};
    return apply_to_sequence<N>(impl);
}

template <size_t N, class Container>
constexpr auto to_tuple(const Container & a, size_t start)
{
    return apply_to_sequence<N>([&a, start](auto... idxs){return std::make_tuple(a[start + idxs]...);});
}

template <typename T, size_t N>
constexpr auto to_tuple(const std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return std::make_tuple(a[idxs]...);});
}

template <size_t N, class Container>
constexpr auto to_tie(Container & a, size_t start)
{
    return apply_to_sequence<N>([&a, start](auto... idxs){return std::tie(a[start + idxs]...);});
}

template <typename T, size_t N>
constexpr auto to_tie(std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return std::tie(a[idxs]...);});
}

}

#endif
