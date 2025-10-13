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

    T out = T();
    InterpValues<T> values {coord.size()};

    // Iterating over a square around coord
    for (size_t i = 0; auto [v_coord, v_factor] : values)
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

template <typename InputIt, typename T, typename Axes, class UnaryFunction, typename = std::enable_if_t<
    (std::is_base_of_v<typename array<T>::iterator, InputIt> || std::is_base_of_v<typename array<T>::const_iterator, InputIt>) &&
    std::is_invocable_v<std::remove_cvref_t<UnaryFunction>, size_t> && std::is_integral_v<typename Axes::value_type>
>>
UnaryFunction maxima_nd(InputIt first, InputIt last, UnaryFunction && unary_op, const array<T> & arr, const Axes & axes, size_t order)
{
    // First element can't be a maximum
    auto iter = (first != last) ? std::next(first) : first;
    last = (iter != last) ? std::prev(last) : last;

    while (iter != last)
    {
        if (*std::prev(iter) < *iter)
        {
            // ahead can be last
            auto ahead = std::next(iter);

            while (ahead != last && *ahead == *iter) ++ahead;

            if (*ahead < *iter)
            {
                // It will return an index relative to the arr.begin() since it's stride is smaller
                auto index = std::addressof(*iter) - arr.data();

                size_t n = 1;
                for (; n < axes.size(); n++)
                {
                    auto coord = arr.index_along_dim(index, axes[n]);
                    if (coord > 1 && coord < arr.shape(axes[n]) - 1)
                    {
                        if (arr[index - arr.strides(axes[n])] < *iter && arr[index + arr.strides(axes[n])] < *iter)
                        {
                            continue;
                        }
                    }

                    break;
                }

                if (n >= order) std::forward<UnaryFunction>(unary_op)(index);

                // Skip samples that can't be maximum, check if it's not last
                if (ahead != last) iter = ahead;
            }
        }

        iter = std::next(iter);
    }

    return std::forward<UnaryFunction>(unary_op);
}

}

#endif
