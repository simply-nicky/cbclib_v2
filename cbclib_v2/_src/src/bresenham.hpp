#ifndef BRESENHAM_
#define BRESENHAM_
#include "numpy.hpp"
#include "geometry.hpp"

namespace cbclib {

namespace detail {

template <class... Types>
class ImageBuffer : public array_indexer
{
public:
    using value_type = std::tuple<size_t, Types...>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    ImageBuffer() : array_indexer(), data() {}
    ImageBuffer(ShapeContainer shape) : array_indexer(std::move(shape), 1), data() {}

    iterator begin() {return data.begin();}
    const_iterator begin() const {return data.begin();}
    iterator end() {return data.end();}
    const_iterator end() const {return data.begin();}

    template <typename I, size_t N, class... Args, typename = std::enable_if_t<std::is_integral_v<I>>>
    void emplace_back(const PointND<I, N> & pt, Args &&... args)
    {
        data.emplace_back(index_at(pt.rbegin(), pt.rend()), std::forward<Args>(args)...);
    }

private:
    std::vector<value_type> data;
};

}

template <typename T, size_t N>
struct BresenhamError
{
    // Signed error to one coordinate plane that contains the continuous line.
    PointND<T, N> normal {};
    T error = T();

    BresenhamError() = default;

    BresenhamError(PointND<T, N> normal, const PointND<long, N> & point,
                   const PointND<T, N> & origin) :
        normal(std::move(normal)), error(dot(this->normal, point - origin)) {}

    T error_at() const {return error;}

    BresenhamError & increment(long x, size_t axis)
    {
        error += x * normal[axis];
        return *this;
    }

    std::pair<bool, bool> is_next(const PointND<long, N> & step, size_t axis1, size_t axis2) const
    {
        // Choose whether the next Bresenham step advances axis1, axis2, or both by
        // taking the move that keeps the line closest to this coordinate plane.
        auto e_x = error_at() + step[axis1] * normal[axis1];
        auto e_y = error_at() + step[axis2] * normal[axis2];
        auto e_xy = e_x + step[axis2] * normal[axis2];

        if (std::abs(e_y) < std::abs(e_x))
        {
            if (std::abs(e_xy) < std::abs(e_y)) return std::make_pair(true, true);
            return std::make_pair(false, true);
        }
        if (std::abs(e_xy) < std::abs(e_x)) return std::make_pair(true, true);
        return std::make_pair(true, false);
    }
};

template <typename T, size_t N, bool IsForward>
class BresenhamPlotter;

template <typename T, size_t N>
class LineIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = PointND<long, N>;
    using pointer = PointND<long, N> *;
    using reference = const PointND<long, N> &;

    LineIterator & operator++()
    {
        for (size_t i = 0; i < N; i++) if (next[i]) increment(step[i], i);
        update();
        return *this;
    }

    LineIterator operator++(int)
    {
        auto saved = *this;
        operator++();
        return saved;
    }

    bool operator==(const LineIterator & rhs) const
    {
        bool is_equal = false;
        for (size_t i = 0; i < N; i++) is_equal |= current[i] == rhs.current[i];
        return is_equal;
    }
    bool operator!=(const LineIterator & rhs) const {return !operator==(rhs);}

    reference operator*() const {return current;}
    pointer operator->() const {return &current;}

private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

    PointND<long, N> step {}, current {};
    // next marks which axes should advance on the next central-line step.
    PointND<bool, N> next {};
    std::array<BresenhamError<T, N>, NumPairs> errors {};

    LineIterator(PointND<long, N> current) : current(std::move(current)) {}

    LineIterator(PointND<long, N> step, PointND<long, N> current,
                 std::array<BresenhamError<T, N>, NumPairs> errors) :
        step(std::move(step)), current(std::move(current)), next(), errors(std::move(errors))
    {
        update();
    }

    LineIterator & increment(long x, size_t axis)
    {
        current[axis] += x;
        for (auto index : axes().indices(axis)) errors[index].increment(x, axis);

        return *this;
    }

    void update()
    {
        for (size_t i = 0; i < N; i++) next[i] = step[i];

        // Intersect all pairwise 2D Bresenham decisions into one N-D step mask.
        for (size_t i = 0; i < NumPairs; i++)
        {
            auto [axis1, axis2] = axes().pairs(i);
            if (step[axis1] && step[axis2])
            {
                auto [first, second] = errors[i].is_next(step, axis1, axis2);
                next[axis1] &= first;
                next[axis2] &= second;
            }
        }
    }

    static const UniquePairs<N> & axes()
    {
        return UniquePairs<N>::instance();
    }

    friend class BresenhamPlotter<T, N, true>;
    friend class BresenhamPlotter<T, N, false>;
};

template <typename T, size_t N, bool IsForward>
class BresenhamPlotter
{
private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

public:
    using const_iterator = LineIterator<T, N>;
    using iterator = const_iterator;

    BresenhamPlotter(LineND<T, N> l) :
        line(std::move(l)), m_pt0(pt0().round()), m_pt1(pt1().round() + step()), m_axis(long_axis()) {}

    BresenhamPlotter(LineND<T, N> l, long offset) :
        line(std::move(l)), m_axis(long_axis())
    {
        // Extend the central-line walk past both endpoints so the thick capsule ends
        // are covered before exact segment-distance filtering trims candidates.
        auto tau = (pt1() - pt0()) / amplitude(pt1() - pt0());
        m_pt0 = (pt0() - std::abs(offset / tau[m_axis]) * tau).round();
        m_pt1 = (pt1() + std::abs(offset / tau[m_axis]) * tau).round();
        m_pt1 += step();
    }

    iterator begin() const
    {
        return iterator(step(), m_pt0, errors(m_pt0));
    }

    iterator end() const
    {
        return iterator(m_pt1);
    }

    size_t axis() const {return m_axis;}
    size_t axis(size_t offset) const {return (axis() + offset) % N;}

    bool is_next(const iterator & iter, size_t axis) const
    {
        return iter.next[axis];
    }

private:
    LineND<T, N> line;
    PointND<long, N> m_pt0, m_pt1;
    size_t m_axis;

    size_t long_axis() const
    {
        auto tau = line.tangent();
        auto compare = [](const T & a, const T & b){return std::abs(a) < std::abs(b);};
        return std::distance(tau.begin(), std::max_element(tau.begin(), tau.end(), compare));
    }

    PointND<T, N> normal(const std::pair<size_t, size_t> & pair) const
    {
        PointND<T, N> norm {};
        norm[pair.first % N] = line.pt1[pair.second % N] - line.pt0[pair.second % N];
        norm[pair.second % N] = line.pt0[pair.first % N] - line.pt1[pair.first % N];
        return norm;
    }

    const PointND<T, N> & pt0() const
    {
        if constexpr(IsForward) return line.pt0;
        else return line.pt1;
    }

    const PointND<T, N> & pt1() const
    {
        if constexpr(IsForward) return line.pt1;
        else return line.pt0;
    }

    PointND<long, N> step() const
    {
        PointND<long, N> point;
        for (size_t i = 0; i < N; i++) point[i] = (pt1()[i] >= pt0()[i]) ? 1 : -1;
        return point;
    }

    std::array<BresenhamError<T, N>, NumPairs> errors(const PointND<long, N> & point) const
    {
        std::array<BresenhamError<T, N>, NumPairs> result;
        for (size_t i = 0; i < NumPairs; i++)
            result[i] = BresenhamError<T, N>(normal(iterator::axes().pairs(i)), point, pt0());
        return result;
    }
};

namespace detail {

template <typename T, class Func, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PointND<long, 2> &, T>
>>
void draw_line_2d(const LineND<T, 2> & line, T width, Func && func)
{
    if (width <= T()) return;

    // Bresenham chooses dominant-axis slices; exact projection decides inclusion.
    BresenhamPlotter<T, 2, true> p {line, long(std::ceil(width) + 1)};
    auto projector = line.projector();
    auto radius = long(std::ceil(width)) + 1;
    auto inv_width2 = T(1) / (width * width);

    auto ax0 = p.axis();
    auto ax1 = p.axis(1);

    auto emit = [&projector, &func, inv_width2](const PointND<long, 2> & ipt)
    {
        auto closest = projector.project_to_streak(ipt);
        T error = magnitude(closest - ipt) * inv_width2;
        if (error <= T(1)) std::forward<Func>(func)(ipt, error);
    };

    for (auto iter = p.begin(); iter != p.end(); ++iter)
    {
        // Emit one orthogonal interval per dominant-axis step.
        if (p.is_next(iter, ax0))
        {
            for (long d1 = -radius; d1 <= radius; d1++)
            {
                auto ipt = *iter;
                ipt[ax1] += d1;
                emit(ipt);
            }
        }
    }
}

/* Point and bound conventions:
   bound = {X, Y, Z}
   line  = {pt0, pt1}       pt0, pt1 = {x, y, z}
 */
template <typename T, class Func, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PointND<long, 3> &, T>
>>
void draw_line_3d(const LineND<T, 3> & line, T width, Func && func)
{
    if (width <= T()) return;

    // Bresenham chooses dominant-axis slices; exact projection decides inclusion.
    BresenhamPlotter<T, 3, true> p {line, long(std::ceil(width) + 1)};
    auto projector = line.projector();
    auto radius = long(std::ceil(width)) + 1;
    auto inv_width2 = T(1) / (width * width);

    auto ax0 = p.axis();
    auto ax1 = p.axis(1);
    auto ax2 = p.axis(2);

    auto emit = [&projector, &func, inv_width2](const PointND<long, 3> & ipt)
    {
        auto closest = projector.project_to_streak(ipt);
        T error = magnitude(closest - ipt) * inv_width2;
        if (error <= T(1)) std::forward<Func>(func)(ipt, error);
    };

    for (auto iter = p.begin(); iter != p.end(); ++iter)
    {
        // Emit one orthogonal box per dominant-axis step.
        if (p.is_next(iter, ax0))
        {
            for (long d1 = -radius; d1 <= radius; d1++)
            {
                for (long d2 = -radius; d2 <= radius; d2++)
                {
                    auto ipt = *iter;
                    ipt[ax1] += d1;
                    ipt[ax2] += d2;
                    emit(ipt);
                }
            }
        }
    }
}

}

template <typename T, class Func, size_t N, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PointND<long, N> &, T>
>>
void draw_line_nd(const LineND<T, N> & line, T width, Func && func)
{
    static_assert(N == 2 || N == 3);

    if constexpr(N == 2) detail::draw_line_2d(line, width, std::forward<Func>(func));
    else detail::draw_line_3d(line, width, std::forward<Func>(func));
}

namespace detail {

template <typename T, size_t N, class Func, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PointND<long, N> &, T>
>>
void draw_segment_box_nd(const LineND<T, N> & line, T width, Func && func)
{
    if (width <= T()) return;

    PointND<long, N> lo, hi;
    for (size_t n = 0; n < N; n++)
    {
        lo[n] = std::floor(std::min(line.pt0[n], line.pt1[n]) - width);
        hi[n] = std::ceil(std::max(line.pt0[n], line.pt1[n]) + width);
    }

    T inv_width = T(1) / (width * width);
    auto projector = line.projector();

    auto emit = [&projector, &func, inv_width](const PointND<long, N> & ipt)
    {
        auto closest = projector.project_to_streak(ipt);
        T error = magnitude(closest - ipt) * inv_width;
        if (error <= T(1)) std::forward<Func>(func)(ipt, error);
    };

    if constexpr(N == 2)
    {
        for (long y = lo[1]; y <= hi[1]; y++)
        {
            for (long x = lo[0]; x <= hi[0]; x++)
            {
                emit(PointND<long, 2>{x, y});
            }
        }
    }
    else
    {
        for (long z = lo[2]; z <= hi[2]; z++)
        {
            for (long y = lo[1]; y <= hi[1]; y++)
            {
                for (long x = lo[0]; x <= hi[0]; x++)
                {
                    emit(PointND<long, 3>{x, y, z});
                }

            }

        }
    }
}

}

template <typename T, size_t N, class Func, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PointND<long, N> &, T>
>>
void draw_curve_nd(const CurveND<T, N> & curve, T width, Func && func)
{
    static_assert(N == 2 || N == 3);

    if (curve.size() < 2)
        throw std::invalid_argument("Curve must contain at least two points");

    for (size_t i = 0; i + 1 < curve.size(); i++)
    {
        detail::draw_segment_box_nd(curve.segment(i), width, std::forward<Func>(func));
    }
}

}

#endif
