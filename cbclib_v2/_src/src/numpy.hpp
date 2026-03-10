#ifndef ARRAY_
#define ARRAY_
#include "include.hpp"
#include "array_view.hpp"

// ---------------------------------------------------------------------------
// cbclib array utilities
//
// Lightweight header-only helpers for multi-dimensional array views,
// indexing utilities, iterators and kernel helpers used by the C++/pybind11
// bindings. All additions here are implementation-level helpers and are
// intentionally minimal to avoid heavy dependencies.
// ---------------------------------------------------------------------------

namespace cbclib {

// Internal helper utilities (implementation details).
// Placed in the `detail` namespace to indicate they are not part of the
// public API.
namespace detail {

// Compute a linear offset by summing coordinate * stride values. This is a
// low-level inner-loop helper used by shape handling and indexing helpers.

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

} // namespace detail

// ------------------------------------------------------------------
// array_indexer
// Stores shape and stride metadata and provides helpers for converting
// between linear indices and multi-dimensional coordinates, bounds
// checking and stride calculations.
// ------------------------------------------------------------------
class array_indexer : public array_indexer_view<size_t>
{
protected:
    using ShapeContainer = AnyContainer<size_t>;

public:
    array_indexer() = default;

    array_indexer(ShapeContainer sh, ShapeContainer st, size_t itemsize) :
        m_shape(std::move(sh)), m_strides(std::move(st))
    {
        this->m_shape_ptr = m_shape.data();
        this->m_strides_ptr = m_strides.data();
        this->m_ndim = m_shape.size();
        this->m_itemsize = itemsize;
    }

    array_indexer(ShapeContainer sh, size_t itemsize) :
        m_shape(std::move(sh))
    {
        m_strides = detail::c_strides(m_shape, itemsize);
        this->m_shape_ptr = m_shape.data();
        this->m_strides_ptr = m_strides.data();
        this->m_ndim = m_shape.size();
        this->m_itemsize = itemsize;
    }

    array_indexer(const py::buffer_info & buf) : array_indexer(buf.shape, buf.strides, buf.itemsize) {}

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    bool is_inbound(CoordIter first, CoordIter last) const
    {
        return array_indexer_view<size_t>::is_inbound(first, last);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    bool is_inbound(const Container & coord) const
    {
        return array_indexer_view<size_t>::is_inbound(coord);
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    bool is_inbound(const std::initializer_list<T> & coord) const
    {
        return array_indexer_view<size_t>::is_inbound(coord);
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    auto index_at(CoordIter first, CoordIter last) const
    {
        return array_indexer_view<size_t>::index_at(first, last);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto index_at(const Container & coord) const
    {
        return array_indexer_view<size_t>::index_at(coord);
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto index_at(const std::initializer_list<T> & coord) const
    {
        return array_indexer_view<size_t>::index_at(coord);
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    auto index_at(Ix... index) const
    {
        return detail::coord_to_index_var(m_shape, index...).first;
    }

    template <typename OutputIt, typename = std::enable_if_t<
        output_iterator_v<OutputIt, size_t>
    >>
    OutputIt coord_at(OutputIt first, size_t index) const
    {
        return array_indexer_view<size_t>::coord_at(first, index);
    }

    template <typename OutputIt, typename = std::enable_if_t<output_iterator_v<OutputIt, size_t>>>
    std::pair<size_t, size_t> coord_at(OutputIt first, size_t index, size_t size, size_t n0, size_t n1) const
    {
        detail::index_to_coord(std::next(m_shape.begin(), n0), std::next(m_shape.begin(), n1), index, size, first);
        return std::make_pair(index, size);
    }

    size_t coord_along_dim(size_t index, size_t dim) const
    {
        if (dim >= m_ndim) fail_dim_check(dim, "invalid axis");
        return array_indexer_view<size_t>::coord_along_dim(index, dim);
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    auto offset_at(CoordIter first, CoordIter last) const
    {
        return array_indexer_view<size_t>::offset_at(first, last);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto offset_at(const Container & coord) const
    {
        return array_indexer_view<size_t>::offset_at(coord);
    }

    template <typename ... Ix, typename = std::enable_if_t<is_all_integral_v<Ix...>>>
    auto offset_at(Ix... index) const
    {
        if (sizeof...(index) > m_ndim) fail_dim_check(sizeof...(index), "too many indices for an array");

        return detail::coord_to_offset_var(m_strides, size_t(index)...);
    }

    // initializer_list's aren't deducible, so don't get matched by the above template;
    // we need this to explicitly allow implicit conversion from one:
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    auto offset_at(const std::initializer_list<T> & coord) const
    {
        return array_indexer_view<size_t>::offset_at(coord);
    }

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
    using array_indexer_view<size_t>::m_ndim;

    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;

    void fail_dim_check(size_t dim, const std::string & msg) const
    {
        throw std::out_of_range(msg + ": " + std::to_string(dim) + " (ndim = " + std::to_string(m_ndim) + ')');
    }
};

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
class array : public array_indexer
{
protected:
    friend class array_iterator<T, false>;
    friend class array_iterator<T, true>;
    using array_indexer::ShapeContainer;

public:
    using value_type = T;
    using iterator = array_iterator<T, false>;
    using const_iterator = array_iterator<T, true>;

    operator py::array_t<T>() const {return {m_shape, m_strides, m_ptr};}

    array() = default;

    array(ShapeContainer shape, ShapeContainer strides, T * ptr) :
        array_indexer(std::move(shape), std::move(strides), sizeof(T)), m_ptr(ptr) {}

    array(ShapeContainer shape, T * ptr) : array_indexer(std::move(shape), sizeof(T)), m_ptr(ptr) {}

    array(size_t count, T * ptr) : array(std::vector<size_t>{count}, ptr) {}

    array(const py::buffer_info & buf) : array(buf.shape, buf.strides, static_cast<T *>(buf.ptr)) {}

    operator bool() const {return m_ptr != nullptr;}

    iterator begin() {return {this, 0};}
    iterator end() {return {this, size()};}
    const_iterator begin() const {return {this, 0};}
    const_iterator end() const {return {this, size()};}

    T & operator[] (size_t index)
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }
    const T & operator[] (size_t index) const
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }

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
        return *(m_ptr + offset_at(coord.begin(), coord.end()) / m_itemsize);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T & at(const Container & coord)
    {
        return *(m_ptr + offset_at(coord.begin(), coord.end()) / m_itemsize);
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
    using array_indexer_view<size_t>::m_ndim;
    using array_indexer_view<size_t>::m_itemsize;
    using array_indexer::m_shape;
    using array_indexer::m_strides;

    T * m_ptr;
};

template <typename T>
class vector_array : public array<T>
{
protected:
    using ShapeContainer = typename array<T>::ShapeContainer;

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
        this->m_ptr = m_data.data();
    }

protected:
    std::vector<T> m_data;
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
    size_t m_size = 0;
};

/*----------------------------------------------------------------------------*/
/*------------------------------ Python helpers ------------------------------*/
/*----------------------------------------------------------------------------*/

/* Transform iterator implementation */

template <typename Iterator, typename Func, typename Value = typename std::iterator_traits<Iterator>::value_type, typename Return = std::remove_reference_t<decltype(std::declval<Func &>()(std::declval<Value &>()))>>
class transform_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Return;
    using difference_type = iter_difference_t<Iterator>;
    using reference = value_type;

    transform_iterator() = default;
    transform_iterator(Iterator && iter, Func && func) : m_iter(std::forward<Iterator>(iter)), m_func(std::forward<Func>(func)) {}

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    transform_iterator & operator++()
    {
        ++m_iter;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    transform_iterator operator++(int)
    {
        return transform_iterator(m_iter++, m_func);
    }

    template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
    transform_iterator & operator--()
    {
        --m_iter;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
    transform_iterator operator--(int)
    {
        return transform_iterator(m_iter--, m_func);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    transform_iterator & operator+=(difference_type offset)
    {
        m_iter += offset;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    transform_iterator operator+(difference_type offset) const
    {
        return transform_iterator(m_iter + offset, m_func);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    transform_iterator & operator-=(difference_type offset)
    {
        m_iter -= offset;
        return *this;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    transform_iterator operator-(difference_type offset) const
    {
        return transform_iterator(m_iter - offset, m_func);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    difference_type operator-(const transform_iterator & rhs) const
    {
        return m_iter - rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    reference operator[](difference_type offset) const
    {
        return m_func(*(m_iter + offset));
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    bool operator==(const transform_iterator & rhs) const
    {
        return m_iter == rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
    bool operator!=(const transform_iterator & rhs) const
    {
        return !(*this == rhs);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator<(const transform_iterator & rhs) const
    {
        return m_iter < rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator>(const transform_iterator & rhs) const
    {
        return m_iter > rhs.m_iter;
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator<=(const transform_iterator & rhs) const
    {
        return !(*this > rhs);
    }

    template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
    bool operator>=(const transform_iterator & rhs) const
    {
        return !(*this < rhs);
    }

    reference operator*() const {return m_func(*m_iter);}

private:
    Iterator m_iter;
    Func m_func;
};

template <typename Iterator, typename Func, typename = std::enable_if_t<
    input_iterator_v<Iterator> && std::is_invocable_v<Func, typename std::iterator_traits<Iterator>::value_type>
>>
transform_iterator<Iterator, Func> make_transform_iterator(Iterator && iterator, Func && func)
{
    return transform_iterator<Iterator, Func>(std::forward<Iterator>(iterator), std::forward<Func>(func));
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
        size_t m_index = 0;

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
        std::pair<size_t, py::ssize_t> m_index {};
        py::ssize_t m_step = 0;

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
    py::ssize_t m_start = 0, m_stop = 0, m_step = 0, m_slicelength = 0;
};

} // namespace cbclib

#endif
