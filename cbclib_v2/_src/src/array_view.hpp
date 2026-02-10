#ifndef ARRAY_VIEW_
#define ARRAY_VIEW_
#include "include.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace cbclib {

namespace detail {

// Compute C-contiguous strides from a shape sequence.

template <typename Container, typename I = typename Container::value_type>
Container c_strides(const Container & shape, I itemsize)
{
    Container strides (shape);
    if (shape.size() > 0)
    {
        strides.back() = itemsize;
        for (I n = shape.size() - 1; n > 0; --n) strides[n - 1] = strides[n] * shape[n];
    }
    return strides;
}

// Compute a linear offset by summing coordinate * stride values. This is a
// low-level inner-loop helper used by shape handling and indexing helpers.

template <typename InputIt1, typename InputIt2, typename I1 = iter_value_t<InputIt1>>
HOST_DEVICE I1 coord_to_offset(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst)
{
    I1 offset = I1();
    for (; cfirst != clast; cfirst++, ++sfirst) offset += *cfirst * *sfirst;
    return offset;
}

// Compute a flat index by summing coordinate * cumsum(shape) values. This is a
// low-level inner-loop helper used by shape handling and indexing helpers.

template <typename InputIt1, typename InputIt2, typename I1 = iter_value_t<InputIt1>,
          typename I2 = iter_value_t<InputIt2>>
HOST_DEVICE I1 coord_to_index(InputIt1 cfirst, InputIt1 clast, InputIt2 sfirst, I2 stride)
{
    I1 index = I1();
    for (; cfirst != clast; ++cfirst, ++sfirst)
    {
        stride /= *sfirst;
        index += *cfirst * stride;
    }
    return index;
}

// Compute a coordinate sequence from a flat index and a shape sequence.

template <typename InputIt, typename OutputIt, typename I = iter_value_t<InputIt>>
HOST_DEVICE OutputIt index_to_coord(InputIt sfirst, InputIt slast, I & index, I & stride, OutputIt cfirst)
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

// Convert linear index to offset given shape and strides
template <typename I1, typename I2>
HOST_DEVICE inline I1 index_to_offset(I1 idx, const I2 * shape, const I2 * strides, I2 n0, I2 n1)
{
    I1 offset = 0;
    for (I2 n = n1; n > n0; --n)
    {
        I2 coord = idx % shape[n - 1];
        idx /= shape[n - 1];
        offset += static_cast<I1>(coord * strides[n - 1]);
    }
    return offset;
}

// Convert a linear offset into multiple coordinates using a strides sequence.
// WORKS ONLY FOR A C CONTIGUOUS ARRAY LAYOUT.

template <typename InputIt, typename OutputIt, typename I = iter_value_t<InputIt>>
HOST_DEVICE OutputIt offset_to_coord(InputIt sfirst, InputIt slast, I offset, OutputIt cfirst)
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

} // namespace detail

// Array indexer view: provides shape/stride metadata and indexing utilities.
// Used by both numpy and cupy array wrappers.
// For CuPy only: array_indexer_view works on host side only.
template <typename I>
class array_indexer_view
{
public:
    using size_type = I;

    array_indexer_view() : m_shape_ptr(nullptr), m_strides_ptr(nullptr) {}

    array_indexer_view(const I * shape, const I * strides, I ndim, I itemsize) :
        m_ndim(ndim), m_shape_ptr(shape), m_strides_ptr(strides), m_itemsize(itemsize) {}

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    bool is_inbound(CoordIter first, CoordIter last) const
    {
        bool flag = true;
        for (I i = 0; first != last; ++first, ++i)
        {
            flag &= *first >= 0 && *first < static_cast<I>(m_shape_ptr[i]);
        }
        return flag;
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    bool is_inbound(const Container & coord) const
    {
        return is_inbound(coord.begin(), coord.end());
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    I index_at(CoordIter first, CoordIter last) const
    {
        return detail::coord_to_index(first, last, m_shape_ptr, size());
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    I index_at(const Container & coord) const
    {
        return detail::coord_to_index(coord.begin(), coord.end(), m_shape_ptr, size());
    }

    template <typename OutputIt, typename = std::enable_if_t<output_iterator_v<OutputIt, I>>>
    OutputIt coord_at(OutputIt first, I index) const
    {
        auto stride = size();
        return detail::index_to_coord(m_shape_ptr, m_shape_ptr + m_ndim, index, stride, first);
    }

    I coord_along_dim(I index, I dim) const
    {
        I coord = 0;
        for (I n = m_ndim; n > dim; --n)
        {
            coord = index % m_shape_ptr[n - 1];
            index /= m_shape_ptr[n - 1];
        }
        return coord;
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    auto offset_at(CoordIter first, CoordIter last) const
    {
        return detail::coord_to_offset(first, last, m_strides_ptr);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    auto offset_at(const Container & coord) const
    {
        return detail::coord_to_offset(coord.begin(), coord.end(), m_strides_ptr);
    }

    I ndim() const {return m_ndim;}
    I size() const {return partial_size(0, m_ndim);}

    const I * shape() const {return m_shape_ptr;}
    I shape(I dim) const
    {
        return m_shape_ptr[dim];
    }

    const I * strides() const {return m_strides_ptr;}
    I strides(I dim) const
    {
        return m_strides_ptr[dim];
    }

    I itemsize() const {return m_itemsize;}

protected:
    const I * m_shape_ptr;
    const I * m_strides_ptr;
    I m_itemsize = 0;
    I m_ndim = 0;

    I partial_size(I n0, I n1) const
    {
        I s = I(1);
        for (I n = n0; n < n1; ++n) s *= m_shape_ptr[n];
        return s;
    }

    I index_to_offset(I index, I n0, I n1) const
    {
        return detail::index_to_offset(index, m_shape_ptr, m_strides_ptr, n0, n1);
    }
};

// Array view: lightweight non-owning multi-dimensional array wrapper
template <typename T, typename I>
class array_view : public array_indexer_view<I>
{
public:
    array_view() : m_ptr(nullptr) {}

    array_view(T * ptr, const I * shape, const I * strides, I ndim) :
        array_indexer_view<I>(shape, strides, ndim, sizeof(T)), m_ptr(ptr) {}

    T & operator[] (I index)
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }
    const T & operator[] (I index) const
    {
        return *(m_ptr + index_to_offset(index, 0, m_ndim) / m_itemsize);
    }

    // Direct data pointer access, works for both host and device code
    HOST_DEVICE T * data() {return m_ptr;}
    HOST_DEVICE const T * data() const {return m_ptr;}

    template <typename CoordIter>
    const T * data(CoordIter first, CoordIter last) const
    {
        return m_ptr + offset_at(first, last) / m_itemsize;
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    T * data(CoordIter first, CoordIter last)
    {
        return m_ptr + offset_at(first, last) / m_itemsize;
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    const T * data(const Container & coord) const
    {
        return m_ptr + offset_at(coord) / m_itemsize;
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T * data(const Container & coord)
    {
        return m_ptr + offset_at(coord) / m_itemsize;
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    const T & at(CoordIter first, CoordIter last) const
    {
        return *data(first, last);
    }

    template <typename CoordIter, typename = std::enable_if_t<input_iterator_v<CoordIter>>>
    T & at(CoordIter first, CoordIter last)
    {
        return *data(first, last);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    const T & at(const Container & coord) const
    {
        return *data(coord);
    }

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    T & at(const Container & coord)
    {
        return *data(coord);
    }

protected:
    using array_indexer_view<I>::index_to_offset;
    using array_indexer_view<I>::m_ndim;
    using array_indexer_view<I>::m_itemsize;

    T * m_ptr;
};

#undef HOST_DEVICE

} // namespace cbclib

#endif // ARRAY_VIEW_
