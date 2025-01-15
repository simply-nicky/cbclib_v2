#ifndef BRESENHAM_
#define BRESENHAM_
#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

template <typename T>
using table_t = std::map<std::pair<long, long>, T>;

namespace detail {

template <typename T>
class ImageBuffer : public shape_handler
{
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    ImageBuffer() : shape_handler(), data() {}
    ImageBuffer(ShapeContainer shape) : shape_handler(std::move(shape)), data() {}

    iterator begin() {return data.begin();}
    const_iterator begin() const {return data.begin();}
    iterator end() {return data.end();}
    const_iterator end() const {return data.begin();}

    template <typename Container, class... Args, typename =
        std::enable_if_t<std::is_integral_v<typename Container::value_type>>
    >
    void emplace_back(const Container & coord, Args &&... args)
    {
        data.emplace_back(ravel_index(coord), std::forward<Args>(args)...);
    }

private:
    std::vector<T> data;
};

}

template <size_t N>
class UniquePairs
{
public:
    constexpr static size_t NumPairs = N * (N - 1) / 2;

    static const UniquePairs & instance()
    {
        static UniquePairs axes;
        return axes;
    }

    UniquePairs(const UniquePairs &)        = delete;
    void operator=(const UniquePairs &)     = delete;

    const std::array<std::pair<size_t, size_t>, NumPairs> & pairs() const {return m_pairs;}
    const std::array<std::array<size_t, N - 1>, N> & lookup() const {return m_lookup;}

private:
    std::array<std::pair<size_t, size_t>, NumPairs> m_pairs;
    std::array<std::array<size_t, N - 1>, N> m_lookup;

    UniquePairs()
    {
        std::pair<size_t, size_t> pair {};
        for (size_t i = 0; i < NumPairs; i++)
        {
            ++pair.second;
            if (pair.second == N)
            {
                ++pair.first;
                pair.second = pair.first + 1;
            }
            m_pairs[i] = pair;
        }

        for (size_t i = 0; i < N; i++)
        {
            size_t index = 0;
            for (size_t j = 0; j < NumPairs; j++)
            {
                if (m_pairs[j].first == i || m_pairs[j].second == i) m_lookup[i][index++] = j;
            }
        }
    }
};

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

        BaseError & increment(long step, size_t axis)
        {
            error += step * derror[axis];
            return *this;
        }
    };

    struct TangentError : public BaseError
    {
        T length;

        TangentError() = default;

        TangentError(PointND<T, N> derr, T err) : BaseError(std::move(derr), err), length(amplitude(derr)) {}

        TangentError(PointND<T, N> derr, const PointND<long, N> & point, const PointND<T, N> & origin) :
            BaseError(std::move(derr), point, origin), length(amplitude(derr)) {}

        T error_at() const {return std::max(std::max(-this->error, this->error - length * length), T());}
    };

    struct NormalError : public BaseError
    {
        NormalError() = default;

        NormalError(PointND<T, N> derr, const PointND<long, N> & point, const PointND<T, N> & origin) :
            BaseError(std::move(derr), point, origin) {}

        bool is_next(const PointND<long, N> & step, size_t axis) const
        {
            if (step[axis] * this->derror[axis] == 0) return true;
            if (step[axis] * this->derror[axis] > 0) return 2 * this->error_at(step) <= step[axis] * this->derror[axis];
            return 2 * this->error_at(step) >= step[axis] * this->derror[axis];
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

        LineIterator & operator++()
        {
            for (size_t i = 0; i < N; i++) if (next[i]) increment(i);
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

        LineIterator(PointND<long, N> current) : current(std::move(current)) {}

        LineIterator(PointND<long, N> step, PointND<long, N> current, TangentError terror, std::array<NormalError, NumPairs> nerrors) :
            step(std::move(step)), current(std::move(current)), terror(std::move(terror)), nerrors(std::move(nerrors))
        {
            update();
        }

        LineIterator(const LineIterator & p, size_t axis) : step(p.step), current(p.current), terror(p.terror), nerrors(p.nerrors)
        {
            step[axis] = 0;
            update();
        }

        LineIterator & increment(size_t axis)
        {
            current[axis] += step[axis];
            terror.increment(step[axis], axis);
            for (auto index : axes().lookup()[axis]) nerrors[index].increment(step[axis], axis);
            return *this;
        }

        void update()
        {
            for (size_t i = 0; i < N; i++) next[i] = step[i];

            for (size_t i = 0; i < NumPairs; i++)
            {
                if (step[axes().pairs()[i].first] && step[axes().pairs()[i].second])
                {
                    next[axes().pairs()[i].first]  &= nerrors[i].is_next(step, axes().pairs()[i].first);
                    next[axes().pairs()[i].second] &= nerrors[i].is_next(step, axes().pairs()[i].second);
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

    BresenhamPlotter(LineND<T, N> l, PointND<long, N> p0, PointND<long, N> p1) :
        line(std::move(l)), m_pt0(std::move(p0)), m_pt1(p1 + step()) {}

    BresenhamPlotter(LineND<T, N> l, long offset) :
        line(std::move(l)), m_pt0(pt0().round() - step() * offset), m_pt1(pt1().round() + step() * (offset + 1)) {}

    BresenhamPlotter(LineND<T, N> l) :
        line(std::move(l)), m_pt0(pt0().round()), m_pt1(pt1().round() + step()) {}

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

    iterator begin(const iterator & iter, size_t axis) const
    {
        return iterator(iter, axis);
    }

    T normal_error(const iterator & iter) const
    {
        T error = T();
        for (size_t i = 0; i < NumPairs; i++) error += std::pow(iter.nerrors[i].error_at() / iter.terror.length, 2);
        return error;
    }

    T error(const iterator & iter, T width) const
    {
        if (width <= T()) return std::numeric_limits<T>::infinity();
        return (std::pow(iter.terror.error_at() / iter.terror.length, 2) + normal_error(iter)) / (width * width);
    }

    bool is_next(const iterator & iter, size_t axis) const
    {
        return iter.next[axis];
    }

private:
    LineND<T, N> line;
    PointND<long, N> m_pt0, m_pt1;

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
        auto tau = line.tangent();

        for (size_t i = 0; i < N; i++)
        {
            if constexpr(IsForward) point[i] = (tau[i] >= 0) ? 1 : -1;
            else point[i] = (tau[i] >= 0) ? -1 : 1;
        }

        return point;
    }

    std::array<NormalError, NumPairs> normal_errors(const PointND<long, N> & point) const
    {
        std::array<NormalError, NumPairs> errors;
        for (size_t i = 0; i < NumPairs; i++) errors[i] = NormalError(normal(iterator::axes().pairs()[i]), point, pt0());
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
    BresenhamPlotter<T, 2, true> plotter {line, long(std::ceil(width) + 1)};

    for (auto iter = plotter.begin(); iter != plotter.end(); ++iter)
    {
        std::forward<Func>(func)(*iter, plotter.error(iter, width));

        if (plotter.is_next(iter, 0))
        {
            for (auto iter_y = std::next(plotter.begin(iter, 0));
                 plotter.normal_error(iter_y) <= width * width && iter_y != plotter.end();
                 ++iter_y)
            {
                std::forward<Func>(func)(*iter_y, plotter.error(iter_y, width));
            }
        }

        if (plotter.is_next(iter, 1))
        {
            for (auto iter_x = std::next(plotter.begin(iter, 1));
                 plotter.normal_error(iter_x) <= width * width && iter_x != plotter.end();
                 ++iter_x)
            {
                std::forward<Func>(func)(*iter_x, plotter.error(iter_x, width));
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
    BresenhamPlotter<T, 3, true> plotter {line, long(std::ceil(width) + 1)};

    for (auto iter = plotter.begin(); iter != plotter.end(); ++iter)
    {
        std::forward<Func>(func)(*iter, plotter.error(iter, width));

        for (size_t n = 0; n < 3; n++) if (plotter.is_next(iter, n))
        {
            auto iter_xy = plotter.begin(iter, n);

            for (size_t m = n + 1; m < n + 3; m++)
            {
                if (plotter.is_next(iter_xy, m % 3) && (!plotter.is_next(iter, m % 3) || n < m % 3))
                {
                    for (auto iter_x = std::next(plotter.begin(iter_xy, m % 3));
                         iter_x != plotter.end(); ++iter_x)
                    {
                        auto error = plotter.normal_error(iter_x);
                        if (error > width * width)
                        {
                            if (error > plotter.normal_error(std::next(iter_x))) continue;
                            else break;
                        }

                        std::forward<Func>(func)(*iter_x, plotter.error(iter_x, width));
                    }
                }
            }

            for (iter_xy = std::next(iter_xy); iter_xy != plotter.end(); ++iter_xy)
            {
                auto error = plotter.normal_error(iter_xy);
                if (error > width * width)
                {
                    if (error > plotter.normal_error(std::next(iter_xy))) continue;
                    else break;
                }

                std::forward<Func>(func)(*iter_xy, plotter.error(iter_xy, width));

                for (size_t m = n + 1; m < n + 3; m++) if (plotter.is_next(iter_xy, m % 3))
                {
                    for (auto iter_x = std::next(plotter.begin(iter_xy, m % 3));
                         iter_x != plotter.end(); ++iter_x)
                    {
                        auto error = plotter.normal_error(iter_x);
                        if (error > width * width)
                        {
                            if (error > plotter.normal_error(std::next(iter_x))) continue;
                            else break;
                        }

                        std::forward<Func>(func)(*iter_x, plotter.error(iter_x, width));
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
