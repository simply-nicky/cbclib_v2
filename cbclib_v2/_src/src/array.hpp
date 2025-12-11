#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"

// ---------------------------------------------------------------------------
// cbclib array utilities
//
// Lightweight header-only helpers for multi-dimensional array views,
// indexing utilities, iterators and kernel helpers used by the C++/pybind11
// bindings. All additions here are implementation-level helpers and are
// intentionally minimal to avoid heavy dependencies.
// ---------------------------------------------------------------------------

namespace cbclib {

template <typename T, typename ... Types>
struct is_all_same
{
    static constexpr bool value = (... && std::is_same_v<T, Types>);
};

template <typename T, typename ... Types>
constexpr bool is_all_same_v = is_all_same<T, Types...>::value;

template <typename ... Types>
struct is_all_integral
{
    static constexpr bool value = (... && std::is_integral_v<Types>);
};

template <typename ... Types>
constexpr bool is_all_integral_v = is_all_integral<Types...>::value;

// Internal helper utilities (implementation details).
// Placed in the `detail` namespace to indicate they are not part of the
// public API.
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

// Compute C-contiguous strides from a shape sequence.

template <typename Container>
Container c_strides(const Container & shape, size_t itemsize)
{
    Container strides (shape);
    if (shape.size() > 0)
    {
        strides.back() = itemsize;
        for (size_t n = shape.size() - 1; n > 0; --n) strides[n - 1] = strides[n] * shape[n];
    }
    return strides;
}

// Compute a linear offset by summing coordinate * stride values. This is a
// low-level inner-loop helper used by shape handling and indexing helpers.

template <typename InputIt1, typename InputIt2, typename I1 = iter_value_t<InputIt1>>
I1 coord_to_offset(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    I1 offset = I1();
    for (; cfirst != clast; cfirst++, ++sfirst) offset += *cfirst * *sfirst;
    return offset;
}

template <size_t Dim = 0, typename Strides>
size_t coord_to_offset_var(const Strides & strides)
{
    return 0;
}

template <size_t Dim = 0, typename Strides, typename... Ix>
size_t coord_to_offset_var(const Strides & strides, size_t i, Ix... index)
{
    return i * strides[Dim] + coord_to_offset_var<Dim + 1>(strides, index...);
}

// Compute a flat index by summing coordinate * cumsum(shape) values. This is a
// low-level inner-loop helper used by shape handling and indexing helpers.

template <typename InputIt1, typename InputIt2, typename I1 = iter_value_t<InputIt1>,
          typename I2 = iter_value_t<InputIt2>>
I1 coord_to_index(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst, I2 stride)
{
    I1 index = I1();
    for (; cfirst != clast; ++cfirst, ++sfirst)
    {
        stride /= *sfirst;
        index += *cfirst * stride;
    }
    return index;
}

template <size_t Dim = 0, typename Shape>
std::pair<size_t, size_t> coord_to_index_var(const Shape & shape)
{
    return std::make_pair(0, 1);
}

template <size_t Dim = 0, typename Shape, typename... Ix>
std::pair<size_t, size_t> coord_to_index_var(const Shape & shape, size_t i, Ix... index)
{
    auto [sub_index, sub_strides] = coord_to_index_var<Dim + 1>(shape, index...);
    return std::make_pair(i * sub_strides + sub_index, sub_strides * shape[Dim]);
}

// Compute a coordinate sequence from a flat index and a shape sequence.

template <typename InputIt, typename OutputIt, typename I = iter_value_t<InputIt>>
OutputIt index_to_coord(InputIt sfirst, InputIt slast, I & index, I & stride, OutputIt cfirst)
{
    for (; sfirst != slast; ++sfirst)
    {
        if (*sfirst)
        {
            stride /= *sfirst;
            I coord = index / stride;
            index -= coord * stride;
            *cfirst++ = coord;
        }
        else *cfirst++ = 0;
    }
    return cfirst;
}

// Convert a linear offset into multiple coordinates using a strides sequence.
// WORKS ONLY FOR A C CONTIGUOUS ARRAY LAYOUT.

template <typename InputIt, typename OutputIt, typename I = iter_value_t<InputIt>>
OutputIt offset_to_coord(InputIt sfirst, InputIt slast, I offset, OutputIt cfirst)
{
    for (; sfirst != slast; ++sfirst)
    {
        if (*sfirst)
        {
            auto coord = offset / *sfirst;
            offset -= coord * *sfirst;
            *cfirst++ = coord;
        }
        else *cfirst++ = 0;
    }
    return cfirst;
}

// ------------------------------------------------------------------
// shape_handler
// Stores shape and stride metadata and provides helpers for converting
// between linear indices and multi-dimensional coordinates, bounds
// checking and stride calculations. Used by the lightweight `array`
// view wrapper below.
// ------------------------------------------------------------------
class shape_handler
{
protected:
    using ShapeContainer = AnyContainer<size_t>;

public:
    using size_type = size_t;

    shape_handler() = default;

    shape_handler(ShapeContainer sh, ShapeContainer st, size_t itemsize) :
        m_ndim(sh.size()), m_shape(std::move(sh)), m_strides(std::move(st)), m_itemsize(itemsize) {}

    shape_handler(ShapeContainer sh, size_t itemsize) :
        m_ndim(sh.size()), m_shape(std::move(sh)), m_itemsize(itemsize)
    {
        m_strides = c_strides(m_shape, m_itemsize);
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
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
        return is_inbound(coord.begin(), coord.end(), size());
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    auto index_at(CoordIter first, CoordIter last) const
    {
        return coord_to_index(first, last, m_shape.begin(), size());
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto index_at(const Container & coord) const
    {
        return coord_to_index(coord.begin(), coord.end(), m_shape.begin(), size());
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto index_at(const std::initializer_list<T> & coord) const
    {
        return coord_to_index(coord.begin(), coord.end(), m_shape.begin(), size());
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    auto index_at(Ix... index) const
    {
        return coord_to_index_var(m_shape, index...).first;
    }

    template <typename OutputIt, typename I, typename = std::enable_if_t<
        std::is_integral_v<I> && output_iterator_v<OutputIt, I>
    >>
    OutputIt coord_at(OutputIt first, I index) const
    {
        auto stride = size();
        return index_to_coord(m_shape.begin(), m_shape.end(), index, stride, first);
    }

    template <typename OutputIt, typename I, typename = std::enable_if_t<
        std::is_integral_v<I> && output_iterator_v<OutputIt, I>
    >>
    std::pair<I, I> coord_at(OutputIt first, I index, I size, size_t n0, size_t n1) const
    {
        index_to_coord(std::next(m_shape.begin(), n0), std::next(m_shape.begin(), n1), index, size, first);
        return std::make_pair(index, size);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    I coord_along_dim(I index, size_t dim) const
    {
        if (dim >= m_ndim) fail_dim_check(dim, "invalid axis");
        I coord = 0;
        for (size_t n = m_shape.size(); n > dim; --n)
        {
            coord = index % m_shape[n - 1];
            index /= m_shape[n - 1];
        }
        return coord;
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    auto offset_at(CoordIter first, CoordIter last) const
    {
        return coord_to_offset(first, last, m_strides.begin());
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto offset_at(const Container & coord) const
    {
        return coord_to_offset(coord.begin(), coord.end(), m_strides.begin());
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    auto offset_at(Ix... index) const
    {
        if (sizeof...(index) > m_ndim) fail_dim_check(sizeof...(index), "too many indices for an array");

        return coord_to_offset_var(m_strides, size_t(index)...);
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto offset_at(const std::initializer_list<T> & coord) const
    {
        return coord_to_offset(coord.begin(), coord.end(), m_strides.begin());
    }

    size_t ndim() const {return m_ndim;}
    size_t size() const {return partial_size(0, m_ndim);}

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

    size_t itemsize() const {return m_itemsize;}

protected:
    size_t m_ndim;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    size_t m_itemsize;

    size_t partial_size(size_t n0, size_t n1) const
    {
        return std::reduce(std::next(m_shape.begin(), n0), std::next(m_shape.begin(), n1), size_t(1), std::multiplies());
    }

    size_t index_to_offset(size_t index, size_t n0, size_t n1) const
    {
        size_t offset = size_t();
        for (size_t n = n1; n > n0; --n)
        {
            auto coord = index % m_shape[n - 1];
            index /= m_shape[n - 1];
            offset += m_strides[n - 1] * coord;
        }
        return offset;
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

// ------------------------------------------------------------------
// Hashing helpers
// Small utilities to combine value hashes into a single hash. These are
// used by fixed-size array/tuple hashers implemented below.
// ------------------------------------------------------------------

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

} // namespace detail

template <typename T>
class array;

template <typename T, bool IsConst>
struct array_iterator_traits;

template <typename T>
struct array_iterator_traits<T, false>
{
    using array_pointer = array<T> *;
    using value_type = T;
    using pointer = T *;
    using reference = T &;
};

template <typename T>
struct array_iterator_traits<T, true>
{
    using array_pointer = const array<T> *;
    using value_type = const T;
    using pointer = const T *;
    using reference = const T &;
};

// ------------------------------------------------------------------
// array_iterator
// Random-access iterator that uses an array pointer to do the
// indexing and iteration
// ------------------------------------------------------------------
template <typename T, bool IsConst>
class array_iterator
{
private:
    friend class array_iterator<T, !IsConst>;
    friend class array<T>;
    using traits = array_iterator_traits<T, IsConst>;
    using array_pointer = typename array_iterator_traits<T, IsConst>::array_pointer;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename traits::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename traits::pointer;
    using reference = typename traits::reference;

    array_iterator() : m_parent(nullptr), m_index() {}

    // This is templated so that we can allow constructing a const iterator from
    // a nonconst iterator...
    template <bool RHIsConst, typename = std::enable_if_t<IsConst || !RHIsConst>>
    array_iterator(const array_iterator<T, RHIsConst> & rhs) : m_parent(rhs.m_parent), m_index(rhs.m_index) {}

    operator bool() const {return bool(m_parent);}

    bool operator==(const array_iterator & rhs) const {return m_index == rhs.m_index;}
    bool operator!=(const array_iterator & rhs) const {return m_index != rhs.m_index;}
    bool operator<=(const array_iterator & rhs) const {return m_index <= rhs.m_index;}
    bool operator>=(const array_iterator & rhs) const {return m_index >= rhs.m_index;}
    bool operator<(const array_iterator & rhs) const {return m_index < rhs.m_index;}
    bool operator>(const array_iterator & rhs) const {return m_index > rhs.m_index;}

    array_iterator & operator+=(const difference_type & step) {m_index += step; return *this;}
    array_iterator & operator-=(const difference_type & step) {m_index -= step; return *this;}
    array_iterator & operator++() {m_index++; return *this;}
    array_iterator & operator--() {m_index--; return *this;}
    array_iterator operator++(int) {array_iterator temp = *this; ++(*this); return temp;}
    array_iterator operator--(int) {array_iterator temp = *this; --(*this); return temp;}
    array_iterator operator+(const difference_type & step) const
    {
        return {m_parent, m_index + step};
    }
    array_iterator operator-(const difference_type & step) const
    {
        return {m_parent, m_index - step};
    }

    difference_type operator-(const array_iterator & rhs) const {return m_index - rhs.m_index;}

    reference operator[] (size_t index) const {return m_parent->operator[](m_index + index);}
    reference operator*() const {return m_parent->operator[](m_index);}
    pointer operator->() const {return &(m_parent->operator[](m_index));}

    size_t index() const {return m_index;}

private:
    array_pointer m_parent;
    size_t m_index;

    array_iterator(array_pointer parent, size_t index) : m_parent(parent), m_index(index) {}
};

// ------------------------------------------------------------------
// array (non-owning view)
// Lightweight wrapper around a raw pointer with associated shape/stride
// information. Does not own memory; provides slicing and indexing helpers
// suitable for exposing C++ memory to Python (pybind11) and for internal
// algorithms.
// ------------------------------------------------------------------
template <typename T>
class array : public detail::shape_handler
{
private:
    friend class array_iterator<T, false>;
    friend class array_iterator<T, true>;
public:
    using value_type = T;
    using size_type = typename detail::shape_handler::size_type;
    using iterator = array_iterator<T, false>;
    using const_iterator = array_iterator<T, true>;

    operator py::array_t<T>() const {return {m_shape, m_strides, m_ptr};}

    array() : shape_handler(), m_ptr(nullptr) {}

    array(ShapeContainer shape, ShapeContainer strides, T * ptr) :
        shape_handler(std::move(shape), std::move(strides), sizeof(T)), m_ptr(ptr) {}

    array(shape_handler handler, T * ptr) : shape_handler(std::move(handler)), m_ptr(ptr) {}

    array(ShapeContainer shape, T * ptr) : shape_handler(std::move(shape), sizeof(T)), m_ptr(ptr) {}

    array(size_t count, T * ptr) : shape_handler({count}, sizeof(T)), m_ptr(ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, buf.strides, static_cast<T *>(buf.ptr)) {}

    operator bool() const {return bool(m_ptr);}

    T & operator[] (size_t index)
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }
    const T & operator[] (size_t index) const
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }

    iterator begin() {return {this, 0};}
    iterator end() {return {this, size()};}
    const_iterator begin() const {return {this, 0};}
    const_iterator end() const {return {this, size()};}

    /* Slice sub-array:
        Take a slice of an array 'array' as follows:
        - array[..., :, ...] slice, where ':' is at axis
    */
    array<T> slice(size_t index, size_t axis) const
    {
        if (!m_ndim) return *this;

        axis = axis % m_ndim;
        size_t offset = size_t();
        if (size() && m_shape[axis])
        {
            index = index % (size() / m_shape[axis]);
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
                        m_ptr + offset / m_itemsize};
    }

    /* Slice sub-array:
        Take a slice of an array 'array' as follows:
        - array[..., :, :], where array[..., :, :].ndim() = ndim
    */
    array<T> slice_back(size_t index, size_t ndim) const
    {
        if (!m_ndim) return *this;

        if (!ndim) return array<T>{std::vector<size_t>{}, m_ptr};
        if (ndim < m_ndim)
        {
            size_t offset = size_t();
            if (size()) offset = index_to_offset(index % partial_size(0, m_ndim - ndim), 0, m_ndim - ndim);
            return array<T>{std::vector<size_t>{std::prev(m_shape.end(), ndim), m_shape.end()},
                            std::vector<size_t>{std::prev(m_strides.end(), ndim), m_strides.end()},
                            m_ptr + offset / m_itemsize};
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

        if (!ndim) return array<T>{std::vector<size_t>{}, m_ptr};
        if (ndim < m_ndim)
        {
            size_t offset = size_t();
            if (size()) offset = index_to_offset(index % partial_size(ndim, m_ndim), ndim, m_ndim);
            return array<T>{std::vector<size_t>{m_shape.begin(), std::next(m_shape.begin(), ndim)},
                            std::vector<size_t>{m_strides.begin(), std::next(m_strides.begin(), ndim)},
                            m_ptr + offset / m_itemsize};
        }
        return *this;
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    const T & at(CoordIter first, CoordIter last) const
    {
        return *(m_ptr + offset_at(first, last) / m_itemsize);
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    T & at(CoordIter first, CoordIter last)
    {
        return *(m_ptr + offset_at(first, last) / m_itemsize);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    const T & at(const Container & coord) const
    {
        return *(m_ptr + offset_at(coord) / m_itemsize);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T & at(const Container & coord)
    {
        return *(m_ptr + offset_at(coord) / m_itemsize);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    const T & at(const std::initializer_list<I> & coord) const
    {
        return *(m_ptr + offset_at(coord) / m_itemsize);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    T & at(const std::initializer_list<I> & coord)
    {
        return *(m_ptr + offset_at(coord) / m_itemsize);
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    const T & at(Ix... index) const
    {
        return *(m_ptr + offset_at(index...) / m_itemsize);
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    T & at(Ix... index)
    {
        return *(m_ptr + offset_at(index...) / m_itemsize);
    }

    const T * data() const {return m_ptr;}
    T * data() {return m_ptr;}

protected:
    T * m_ptr;

    void set_data(T * ptr) {m_ptr = ptr;}
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

    template <typename Vector, typename = std::enable_if_t<std::is_base_of_v<std::vector<T>, remove_cvref_t<Vector>>>>
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
                detail::offset_to_coord(m_strides.begin(), m_strides.end(), m_index, m_coord.rbegin());
            }
            else
            {
                detail::offset_to_coord(m_strides.begin(), m_strides.end(), m_index, m_coord.begin());
            }
        }

        friend class rectangle_range;
    };

    using iterator = rectangle_iterator;
    using reverse_iterator = std::reverse_iterator<rectangle_iterator>;

    rectangle_range(Container sh) : m_strides(detail::c_strides(sh, 1)), m_size(std::reduce(sh.begin(), sh.end(), size_t(1), std::multiplies())) {}

    size_t size() const {return m_size;}

    iterator begin() const {return iterator(m_strides, 0);}
    iterator end() const {return iterator(m_strides, m_size);}

    reverse_iterator rbegin() const {return reverse_iterator(m_strides, m_size);}
    reverse_iterator rend() const {return reverse_iterator(m_strides, 0);}

private:
    Container m_strides;
    size_t m_size;
};

/*----------------------------------------------------------------------------*/
/*------------------------------ Python helpers ------------------------------*/
/*----------------------------------------------------------------------------*/

/* Iterator adapter for point containers for pybind11 */
/* python_point_iterator dereferences to an std::array instead of PointND */

template <typename Iterator, typename = decltype(std::declval<Iterator &>()->to_array())>
class python_point_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename std::remove_reference_t<decltype(std::declval<Iterator &>()->to_array())>;
    using difference_type = iter_difference_t<Iterator>;
    using reference = const value_type &;
    using pointer = const value_type *;

    python_point_iterator() = default;
    python_point_iterator(Iterator && iter) : m_iter(std::move(iter)) {}
    python_point_iterator(const Iterator & iter) : m_iter(iter) {}

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    python_point_iterator & operator++()
    {
        ++m_iter;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    python_point_iterator operator++(int)
    {
        return python_point_iterator(m_iter++);
    }

    template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
    python_point_iterator & operator--()
    {
        --m_iter;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
    python_point_iterator operator--(int)
    {
        return python_point_iterator(m_iter--);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    python_point_iterator & operator+=(difference_type offset)
    {
        m_iter += offset;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    python_point_iterator operator+(difference_type offset) const
    {
        return python_point_iterator(m_iter + offset);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    python_point_iterator & operator-=(difference_type offset)
    {
        m_iter -= offset;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    python_point_iterator operator-(difference_type offset) const
    {
        return python_point_iterator(m_iter - offset);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    difference_type operator-(const python_point_iterator & rhs) const
    {
        return m_iter - rhs;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    reference operator[](difference_type offset) const
    {
        return (m_iter + offset)->to_array();
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    bool operator==(const python_point_iterator & rhs) const
    {
        return m_iter == rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    bool operator!=(const python_point_iterator & rhs) const
    {
        return !(*this == rhs);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator<(const python_point_iterator & rhs) const
    {
        return m_iter < rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator>(const python_point_iterator & rhs) const
    {
        return m_iter > rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator<=(const python_point_iterator & rhs) const
    {
        return !(*this > rhs);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator>=(const python_point_iterator & rhs) const
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

/* C++ container to NumPy array converters */

template <typename Container, typename Shape, typename = std::enable_if_t<
    std::is_rvalue_reference_v<Container &&> && std::is_integral_v<typename remove_cvref_t<Shape>::value_type>
>>
inline py::array_t<typename Container::value_type> as_pyarray(Container && seq, Shape && shape)
{
    Container * seq_ptr = new Container(std::move(seq));
    auto capsule = py::capsule(seq_ptr, [](void * p) {delete reinterpret_cast<Container *>(p);});
    return py::array(std::forward<Shape>(shape),  // shape of array
                     seq_ptr->data(),  // c-style contiguous strides for Container
                     capsule           // numpy array references this parent
    );
}

template <typename Container, typename = std::enable_if_t<std::is_rvalue_reference_v<Container &&>>>
inline py::array_t<typename Container::value_type> as_pyarray(Container && seq)
{
    Container * seq_ptr = new Container(std::move(seq));
    auto capsule = py::capsule(seq_ptr, [](void * p) {delete reinterpret_cast<Container *>(p);});
    return py::array(seq_ptr->size(),  // shape of array
                     seq_ptr->data(),  // c-style contiguous strides for Container
                     capsule           // numpy array references this parent
    );
}

template <typename Container, typename Shape, typename = std::enable_if_t<
    std::is_integral_v<typename remove_cvref_t<Shape>::value_type>
>>
inline py::array_t<typename Container::value_type> to_pyarray(const Container & seq, Shape && shape)
{
    return py::array(std::forward<Shape>(shape), seq.data());
}

template <typename Container>
inline py::array_t<typename Container::value_type> to_pyarray(const Container & seq)
{
    return py::array(seq.size(), seq.data());
}

/* Python index value processing */

py::ssize_t compute_index(py::ssize_t index, py::ssize_t length, std::string cls)
{
    index = (index >= 0) ? index : index + length;
    if (index < 0 || index >= length) throw std::out_of_range(cls + " index is out of range");
    return index;
}

/* Python slice to C++ iterator */

class slice_range
{
public:
    class slice_iterator;

    class slice_sentinel
    {
    private:
        size_t m_index;

        slice_sentinel(size_t index) : m_index(index) {}

        friend bool operator==(const slice_iterator & lhs, const slice_sentinel & rhs);
        friend bool operator!=(const slice_iterator & lhs, const slice_sentinel & rhs);
        friend class slice_range;
    };

    class slice_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<size_t, py::ssize_t>;
        using pointer = const value_type *;
        using reference = const value_type &;

        slice_iterator & operator++()
        {
            m_index.first++;
            m_index.second += m_step;
            return *this;
        }

        slice_iterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        reference operator*() const {return m_index;}
        pointer operator->() const {return &m_index;}

        friend bool operator==(const slice_iterator & lhs, const slice_sentinel & rhs) {return lhs.m_index.first == rhs.m_index;}
        friend bool operator!=(const slice_iterator & lhs, const slice_sentinel & rhs) {return lhs.m_index.first != rhs.m_index;}

    private:
        std::pair<size_t, py::ssize_t> m_index;
        py::ssize_t m_step;

        slice_iterator(size_t index, py::ssize_t py_index, py::ssize_t step) : m_index(index, py_index), m_step(step) {}

        friend class slice_range;
    };

    using iterator = slice_iterator;
    using sentinel = slice_sentinel;

    slice_range() = default;

    slice_range(const py::slice & slice, py::ssize_t length)
    {
        if (!slice.compute(length, &m_start, &m_stop, &m_step, &m_slicelength))
            throw py::error_already_set();
    }

    iterator begin() const {return {0, m_start, m_step};}
    sentinel end() const {return {static_cast<size_t>(m_slicelength)};}

    py::ssize_t size() const {return m_slicelength;}
    py::ssize_t start() const {return m_start;}
    py::ssize_t stop() const {return m_stop;}
    py::ssize_t step() const {return m_step;}

private:
    py::ssize_t m_start, m_stop, m_step, m_slicelength;
};

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

template <size_t N, class Container, typename T = typename Container::value_type>
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
