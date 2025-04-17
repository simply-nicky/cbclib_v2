#ifndef BRESENHAM_
#define BRESENHAM_
#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

namespace detail {

template <class... Types>
class ImageBuffer : public shape_handler
{
public:
    using value_type = std::tuple<size_t, Types...>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    ImageBuffer() : shape_handler(), data() {}
    ImageBuffer(ShapeContainer shape) : shape_handler(std::move(shape)), data() {}

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

template <typename T, size_t N, bool IsForward>
class BresenhamPlotter
{
private:
    constexpr static size_t NumPairs = UniquePairs<N>::NumPairs;

    struct BaseError
    {
        PointND<T, N> derror;
        T error;

        BaseError() = default;

        BaseError(PointND<T, N> derr, T err) : derror(std::move(derr)), error(err) {}

        BaseError(PointND<T, N> derr, const PointND<long, N> & point, const PointND<T, N> & origin) :
            derror(std::move(derr)), error(dot(derror, point - origin)) {}

        T error_at() const {return error;}

        // Return e(x + sx, y + sy)
        T error_at(const PointND<T, N> & step) const
        {
            return error_at() + dot(step, derror);
        }

        BaseError & increment(long x, size_t axis)
        {
            error += x * derror[axis];
            return *this;
        }
    };

    struct TangentError : public BaseError
    {
        using BaseError::error;
        T length;

        TangentError() = default;

        TangentError(PointND<T, N> derr, T err) : BaseError(std::move(derr), err), length(amplitude(derr)) {}

        TangentError(PointND<T, N> derr, const PointND<long, N> & point, const PointND<T, N> & origin) :
            BaseError(std::move(derr), point, origin), length(amplitude(derr)) {}

        T error_at() const {return std::max(std::max(-error, error - length * length), T());}
    };

    struct NormalError : public BaseError
    {
        using BaseError::derror;

        NormalError() = default;

        NormalError(PointND<T, N> derr, const PointND<long, N> & point, const PointND<T, N> & origin) :
            BaseError(std::move(derr), point, origin) {}

        using BaseError::error_at;

        // We need to choose one out of three option:
        //      1) increment only x, error e_x
        //      2) increment only y, error e_y
        //      3) increment both, error e_xy
        // We need to choose an option with the minimal absolute error
        std::pair<bool, bool> is_next(const PointND<long, N> & step, size_t axis1, size_t axis2) const
        {
            auto e_x = error_at() + step[axis1] * derror[axis1];
            auto e_y = error_at() + step[axis2] * derror[axis2];
            auto e_xy = e_x + step[axis2] * derror[axis2];

            if (std::abs(e_y) < std::abs(e_x))
            {
                if (std::abs(e_xy) < std::abs(e_y)) return std::make_pair(true, true);
                return std::make_pair(false, true);
            }
            if (std::abs(e_xy) < std::abs(e_x)) return std::make_pair(true, true);
            return std::make_pair(true, false);
        }
    };

public:
    class LineIterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = PointND<long, N>;
        using pointer = PointND<long, N> *;
        using reference = const PointND<long, N> &;

        LineIterator & flip(size_t axis)
        {
            step[axis] *= -1;
            update();
            return *this;
        }

        LineIterator & flip(const std::array<bool, N> & to_flip)
        {
            for (size_t i = 0; i < N; i++) if (to_flip[i]) step[i] *= -1;
            update();
            return *this;
        }

        LineIterator & move(size_t axis)
        {
            increment(step[axis], axis);
            update();
            return *this;
        }

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
        PointND<long, N> step, current;
        PointND<bool, N> next;
        TangentError terror;
        std::array<NormalError, NumPairs> nerrors;

        LineIterator(PointND<long, N> current) :
            step(), current(std::move(current)), next(), terror(), nerrors() {}

        LineIterator(PointND<long, N> step, PointND<long, N> current, TangentError terror, std::array<NormalError, NumPairs> nerrors) :
            step(std::move(step)), current(std::move(current)), next(), terror(std::move(terror)), nerrors(std::move(nerrors))
        {
            update();
        }

        template <typename ... Ix> requires is_all_integral<Ix ...>
        LineIterator(const LineIterator & p, Ix ... axes) :
            step(p.step), current(p.current), next(), terror(p.terror), nerrors(p.nerrors)
        {
            (step[axes] = ... = 0);
            update();
        }

        LineIterator & increment(long x, size_t axis)
        {
            current[axis] += x;

            terror.increment(x, axis);
            for (auto index : axes().indices(axis)) nerrors[index].increment(x, axis);

            return *this;
        }

        void update()
        {
            for (size_t i = 0; i < N; i++) next[i] = step[i];

            for (size_t i = 0; i < NumPairs; i++)
            {
                auto [axis1, axis2] = axes().pairs(i);
                if (step[axis1] && step[axis2])
                {
                    auto [first, second] = nerrors[i].is_next(step, axis1, axis2);
                    next[axis1] &= first;
                    next[axis2] &= second;
                }
            }
        }

        static const UniquePairs<N> & axes()
        {
            return UniquePairs<N>::instance();
        }

        friend class BresenhamPlotter;
    };

    using const_iterator = LineIterator;
    using iterator = const_iterator;

    BresenhamPlotter(LineND<T, N> l) :
        line(std::move(l)), m_pt0(pt0().round()), m_pt1(pt1().round() + step()), m_axis(long_axis()) {}

    BresenhamPlotter(LineND<T, N> l, long offset) :
        line(std::move(l)), m_axis(long_axis())
    {
        auto tau = (pt1() - pt0()) / amplitude(pt1() - pt0());
        m_pt0 = (pt0() - std::abs(offset / tau[m_axis]) * tau).round();
        m_pt1 = (pt1() + std::abs(offset / tau[m_axis]) * tau).round();
        m_pt1 += step();
    }

    iterator begin(PointND<long, N> point) const
    {
        return iterator(step(), std::move(point), tangent_error(point), normal_errors(point));
    }

    iterator begin() const
    {
        return iterator(step(), m_pt0, tangent_error(m_pt0), normal_errors(m_pt0));
    }

    iterator end() const
    {
        return iterator(m_pt1);
    }

    template <typename ... Ix> requires is_all_integral<Ix ...>
    iterator collapse(const iterator & iter, Ix ... axes) const
    {
        return iterator(iter, axes...);
    }

    size_t axis() const {return m_axis;}
    size_t axis(size_t offset) const {return (axis() + offset) % N;}

    T normal_error(const iterator & iter) const
    {
        T error = T();
        for (size_t i = 0; i < NumPairs; i++) error += std::pow(iter.nerrors[i].error_at() / iter.terror.length, 2);
        return error;
    }

    T error(const iterator & iter, T width) const
    {
        if (width <= T()) return std::numeric_limits<T>::infinity();
        auto error = std::pow(iter.terror.error_at() / iter.terror.length, 2) + normal_error(iter);
        return error / (width * width);
    }

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

    std::array<NormalError, NumPairs> normal_errors(const PointND<long, N> & point) const
    {
        std::array<NormalError, NumPairs> errors;
        for (size_t i = 0; i < NumPairs; i++) errors[i] = NormalError(normal(iterator::axes().pairs(i)), point, pt0());
        return errors;
    }

    TangentError tangent_error(const PointND<long, N> & point) const
    {
        return TangentError(line.tangent(), point, pt0());
    }
};

namespace detail {

template <typename T, class Func, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, const PointND<long, 2> &, T>
>>
void draw_line_2d(const LineND<T, 2> & line, T width, Func && func)
{
    // Initialize with a line and an offset from the start and end points
    BresenhamPlotter<T, 2, true> p {line, long(std::ceil(width) + 1)};

    // Walking along the central line
    for (auto iter = p.begin(); iter != p.end(); ++iter)
    {
        // We fill the orthogonal plane if we stepped the longest axis
        if (p.is_next(iter, p.axis()))
        {
            std::forward<Func>(func)(*iter, p.error(iter, width));

            // Filling the first half
            for (auto iter_x = std::next(p.collapse(iter, p.axis()));
                 p.normal_error(iter_x) < width * width; ++iter_x)
            {
                std::forward<Func>(func)(*iter_x, p.error(iter_x, width));
            }

            // Filling the second half
            for (auto iter_x = std::next(p.collapse(iter, p.axis()).flip(p.axis(1)));
                 p.normal_error(iter_x) < width * width; ++iter_x)
            {
                std::forward<Func>(func)(*iter_x, p.error(iter_x, width));
            }
        }
    }
}

/* Point and bound conventions:
   bound = {X, Y, Z}
   line  = {pt0, pt1}       pt0, pt1 = {x, y, z}
 */
template <typename T, class Func, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, const PointND<long, 3> &, T>
>>
void draw_line_3d(const LineND<T, 3> & line, T width, Func && func)
{
    // Initialize with a line and an offset from the start and end points
    BresenhamPlotter<T, 3, true> p {line, long(std::ceil(width) + 1)};

    // Walking along the central line
    for (auto iter = p.begin(); iter != p.end(); ++iter)
    {
        // We fill the orthogonal plane if we stepped the longest axis
        if (p.is_next(iter, p.axis()))
        {
            std::forward<Func>(func)(*iter, p.error(iter, width));

            // The orthogonal plane is divided in four quadrants
            for (size_t i = 0; i < 4; i++)
            {
                // The flipping mask of the size N
                std::array<bool, 3> flip {};
                flip[p.axis(1)] |= i & 1;
                flip[p.axis(2)] |= i >> 1;

                size_t counter = 0;
                for (auto iter_xy = p.collapse(iter, p.axis()).flip(flip).move(p.axis(1 + ((i & 1) ^ (i >> 1))));
                     p.normal_error(iter_xy) < width * width && counter < 100; ++iter_xy, ++counter)
                {
                    std::forward<Func>(func)(*iter_xy, p.error(iter_xy, width));

                    for (size_t j = 1; j < 3; j++) if (p.is_next(iter_xy, p.axis(j)))
                    {
                        for (auto iter_x = std::next(p.collapse(iter_xy, p.axis(j)));
                             p.normal_error(iter_x) < width * width; ++iter_x)
                        {
                            std::forward<Func>(func)(*iter_x, p.error(iter_x, width));
                        }
                    }
                }
            }
        }
    }
}

}

template <typename T, class Func, size_t N, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, const PointND<long, N> &, T>
>>
void draw_line_nd(const LineND<T, N> & line, T width, Func && func)
{
    static_assert(N == 2 || N == 3);

    if constexpr(N == 2) detail::draw_line_2d(line, width, std::forward<Func>(func));
    else detail::draw_line_3d(line, width, std::forward<Func>(func));
}

}

#endif
