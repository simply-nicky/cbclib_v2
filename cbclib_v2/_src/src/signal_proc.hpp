#ifndef SIGNAL_PROC_
#define SIGNAL_PROC_
#include "array.hpp"

namespace cbclib {

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

template <typename T, bool IsConst>
struct InterpTraits;

template <typename T>
struct InterpTraits<T, false>
{
    using value_type = std::pair<std::vector<size_t>, T>;
    using reference = std::pair<std::vector<size_t> &, T &>;
};

template <typename T>
struct InterpTraits<T, true>
{
    using value_type = std::pair<const std::vector<size_t>, const T>;
    using reference = std::pair<const std::vector<size_t> &, const T &>;
};

template <typename T>
class InterpValues;

template <typename T, bool IsConst>
class InterpIterator
{
private:
    friend class InterpIterator<T, !IsConst>;
    friend class InterpValues<T>;
    using traits = InterpTraits<T, IsConst>;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename traits::value_type;
    using difference_type = std::ptrdiff_t;
    using reference = typename traits::reference;

    template <bool RHIsConst, typename = std::enable_if_t<IsConst || !RHIsConst>>
    InterpIterator(const InterpIterator<T, RHIsConst> & rhs) : m_cptr(rhs.m_cptr), m_fptr(rhs.m_fptr) {}

    operator bool() const {return bool(m_cptr) && bool(m_fptr);}

    bool operator==(const InterpIterator & rhs) const {return m_cptr == rhs.m_cptr && m_fptr == rhs.m_fptr;}
    bool operator!=(const InterpIterator & rhs) const {return m_cptr != rhs.m_cptr && m_fptr != rhs.m_fptr;}
    bool operator<=(const InterpIterator & rhs) const {return m_cptr <= rhs.m_cptr && m_fptr <= rhs.m_fptr;}
    bool operator>=(const InterpIterator & rhs) const {return m_cptr >= rhs.m_cptr && m_fptr >= rhs.m_fptr;}
    bool operator<(const InterpIterator & rhs) const {return m_cptr < rhs.m_cptr && m_fptr < rhs.m_fptr;}
    bool operator>(const InterpIterator & rhs) const {return m_cptr > rhs.m_cptr && m_fptr > rhs.m_fptr;}

    InterpIterator & operator+=(const difference_type & step) {m_cptr += step; m_fptr += step; return *this;}
    InterpIterator & operator-=(const difference_type & step) {m_cptr -= step; m_fptr -= step; return *this;}
    InterpIterator & operator++() {m_cptr++; m_fptr++; return *this;}
    InterpIterator & operator--() {m_cptr--; m_fptr--; return *this;}
    InterpIterator operator++(int) {InterpIterator temp = *this; ++(*this); return temp;}
    InterpIterator operator--(int) {InterpIterator temp = *this; --(*this); return temp;}
    InterpIterator operator+(const difference_type & step) const
    {
        return {m_cptr + step, m_fptr + step};
    }
    InterpIterator operator-(const difference_type & step) const
    {
        return {m_cptr - step, m_fptr - step};
    }

    difference_type operator-(const InterpIterator & rhs) const {return m_cptr - rhs.m_cptr;}

    reference operator*() const {return {*m_cptr, *m_fptr};}

private:
    std::vector<size_t> * m_cptr;
    T * m_fptr;

    InterpIterator(std::vector<size_t> * cptr, T * fptr) : m_cptr(cptr), m_fptr(fptr) {}
};

template <typename T>
class InterpValues
{
public:
    using value_type = std::pair<std::vector<size_t>, T>;
    using size_type = size_t;
    using iterator = InterpIterator<T, false>;
    using const_iterator = InterpIterator<T, true>;

    InterpValues(size_t ndim) :
        m_crd(1ul << ndim, std::vector<size_t>(ndim, size_t())), m_fct(1ul << ndim, T(1.0)) {}

    size_t size() const {return m_crd.size();}

    const std::vector<size_t> & coordinate(size_t idx) const {return m_crd[idx];}
    std::vector<size_t> & coordinate(size_t idx) {return m_crd[idx];}

    T factor(size_t idx) const {return m_fct[idx];}
    T & factor(size_t idx) {return m_fct[idx];}

    iterator begin() {return {m_crd.data(), m_fct.data()};}
    iterator end() {return {m_crd.data() + m_crd.size(), m_fct.data() + m_fct.size()};}
    const_iterator begin() const {return {m_crd.data(), m_fct.data()};}
    const_iterator end() const {return {m_crd.data() + m_crd.size(), m_fct.data() + m_fct.size()};}

private:
    std::vector<std::vector<size_t>> m_crd;
    std::vector<T> m_fct;
};

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

template <typename T, typename Coord, typename U = typename Coord::value_type>
InterpValues<T> bilinear(const std::vector<array<U>> & grid, const Coord & coord)
{
    std::vector<size_t> lbound, ubound;
    std::vector<T> dx;

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = n;
        // liter is GREATER OR EQUAL
        auto liter = std::lower_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // uiter is GREATER
        auto uiter = std::upper_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // lbound is LESS OR EQUAL
        lbound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), uiter) - 1, 0, grid[index].size() - 1));
        // rbound is GREATER OR EQUAL
        ubound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), liter), 0, grid[index].size() - 1));
    }

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = n;
        if (lbound[index] != ubound[index])
        {
            dx.push_back(T(coord[n] - grid[n][lbound[index]]) / T(grid[n][ubound[index]] - grid[n][lbound[index]]));
        }
        else dx.push_back(T());
    }

    InterpValues<T> values {coord.size()};

    // Iterating over a square around coord
    size_t i = 0;
    for (auto [v_coord, v_factor] : values)
    {
        for (size_t n = 0; n < coord.size(); n++)
        {
            // If the index is odd
            if ((i >> n) & 1)
            {
                v_coord[n] = ubound[n];
                v_factor *= dx[n];
            }
            else
            {
                v_coord[n] = lbound[n];
                v_factor *= 1.0 - dx[n];
            }
        }

        i++;
    }

    return values;
}

template <typename T>
class MaximaND
{
public:
    MaximaND(array<T> arr, std::vector<size_t> axes) : m_arr(std::move(arr))
    {
        for (auto axis : axes)
        {
            auto & offset = m_offsets.emplace_back();
            for (size_t n = 0; n < m_arr.ndim(); n++)
            {
                if (n == axis) offset.push_back(1);
                else offset.push_back(0);
            }
        }
    }

    template <typename InputIt, typename UnaryFunction, typename = std::enable_if_t<
        (std::is_base_of_v<typename array<T>::const_iterator, InputIt> ||
         std::is_base_of_v<typename array<T>::iterator, InputIt>) &&
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, size_t>
    >>
    UnaryFunction find(InputIt first, InputIt last, size_t index, size_t axis, UnaryFunction && unary_op)
    {
        FindContext context {m_arr, index, axis};
        return find_with_context(first, last, std::forward<UnaryFunction>(unary_op), context);
    }

    template <typename UnaryFunction, typename = std::enable_if_t<
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, size_t>
    >>
    UnaryFunction find(size_t index, size_t axis, UnaryFunction && unary_op)
    {
        FindContext context {m_arr, index, axis};
        return find_with_context(context.first, context.last, std::forward<UnaryFunction>(unary_op), context);
    }

    const array<T> & data() const {return m_arr;}

private:
    array<T> m_arr;
    std::vector<std::vector<long>> m_offsets;

    std::vector<long> to_coord(long index, size_t axis) const
    {
        long size = m_arr.size() / m_arr.shape()[axis];
        std::vector<long> coord;
        std::tie(index, size) = m_arr.coord_at(std::back_inserter(coord), index, size, 0, axis);
        coord.push_back(long());
        m_arr.coord_at(std::back_inserter(coord), index, size, axis + 1, m_arr.ndim());
        return coord;
    }

    struct FindContext
    {
        using const_iterator = typename array<T>::const_iterator;

        size_t index, axis;
        array<T> line;
        const_iterator first, last;

        FindContext(const array<T> & arr, size_t idx, size_t ax) : index(idx), axis(ax), line(arr.slice(idx, ax))
        {
            first = line.begin(); last = line.end();
        }
    };

    template <typename InputIt, typename UnaryFunction, typename = std::enable_if_t<
        (std::is_base_of_v<typename array<T>::const_iterator, InputIt> ||
         std::is_base_of_v<typename array<T>::iterator, InputIt>) &&
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, size_t>
    >>
    UnaryFunction find_with_context(InputIt first, InputIt last, UnaryFunction && unary_op, const FindContext & context)
    {
        auto coord = to_coord(context.index, context.axis);
        for (auto iter = first; iter < last; ++iter)
        {
            if (iter != context.first && *std::prev(iter) < *iter)
            {
                // ahead can be last, must not be end
                auto ahead = std::next(iter);
                if (ahead == context.last) break;

                while (ahead != context.last && *ahead == *iter) ++ahead;

                if (*ahead < *iter)
                {
                    // It will return an offset relative to the arr.begin() since it's stride is smaller
                    coord[context.axis] = iter.index();
                    size_t maximum = m_arr.index_at(coord);

                    size_t n = 0;
                    std::vector<long> left, right;
                    for (auto offset : m_offsets)
                    {
                        left.clear(); right.clear();
                        for (size_t n = 0; n < m_arr.ndim(); n++)
                        {
                            left.push_back(coord[n] + offset[n]);
                            right.push_back(coord[n] - offset[n]);
                        }

                        if (m_arr.is_inbound(left) && m_arr.is_inbound(right))
                        {
                            if (*iter > m_arr.at(left) && *iter > m_arr.at(right)) n++;
                        }
                    }

                    if (n == m_offsets.size()) std::forward<UnaryFunction>(unary_op)(maximum);

                    // Skip samples that can't be maximum
                    iter = ahead;
                }
            }
        }

        return std::forward<UnaryFunction>(unary_op);
    }
};

}

#endif
