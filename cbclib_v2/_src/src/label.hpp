#ifndef LABEL_H_
#define LABEL_H_
#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

template <typename T>
using Pixel = std::pair<Point<long>, T>;

template <typename T>
using PixelSet = std::set<Pixel<T>>;

namespace detail {

template <typename Container, typename Element = typename Container::value_type, typename T = typename Element::value_type,
    typename = std::enable_if_t<std::is_same_v<Element, Point<T>>>
>
std::vector<T> get_x(const Container & c)
{
    std::vector<T> x;
    std::transform(c.begin(), c.end(), std::back_inserter(x), [](const Point<T> & elem){return elem.x();});
    return x;
}

template <typename Container, typename Element = typename Container::value_type, typename T = typename Element::value_type,
    typename = std::enable_if_t<std::is_same_v<Element, Point<T>>>
>
std::vector<T> get_y(const Container & c)
{
    std::vector<T> y;
    std::transform(c.begin(), c.end(), std::back_inserter(y), [](const Point<T> & elem){return elem.y();});
    return y;
}

}

template <typename T, typename Pt, typename I = std::remove_cvref_t<Pt>::value_type,
    typename = std::enable_if_t<
        std::is_base_of_v<Point<I>, std::remove_cvref_t<Pt>> && std::is_integral_v<I>
    >
>
Pixel<T> make_pixel(Pt && point, const array<T> & data)
{
    return std::make_pair(std::forward<Pt>(point), data.at(point.coordinate()));
}

template <typename T, typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
Pixel<T> make_pixel(I x, I y, T value)
{
    return std::make_pair(Point<long>{x, y}, value);
}

// Image moments class

template <typename T>
class Moments;

template <typename T>
class CentralMoments
{
public:
    std::array<T, 2> first() const {return mu_x + origin;}
    std::array<T, 4> second() const {return {mu_xx[0], mu_xy, mu_xy, mu_xx[1]};}

    // Angle between the largest eigenvector of the covariance matrix and x-axis
    T theta() const
    {
        T theta = 0.5 * std::atan(2 * mu_xy / (mu_xx[0] - mu_xx[1]));
        if (mu_xx[1] > mu_xx[0]) theta += M_PI_2;
        return detail::modulo(theta, M_PI);
    }

    Line<T> line() const
    {
        T angle = theta();
        Point<T> tau {std::cos(angle), std::sin(angle)};
        T delta = std::sqrt(4 * mu_xy * mu_xy + (mu_xx[0] - mu_xx[1]) * (mu_xx[0] - mu_xx[1]));
        T hw = std::sqrt(2 * std::log(2) * (mu_xx[0] + mu_xx[1] + delta));
        return Line<T>{mu_x + origin + hw * tau, mu_x + origin - hw * tau};
    }

    bool is_positive() const
    {
        return mu_xx[0] > T() && mu_xx[1] > T();
    }

private:
    Point<T> origin;
    Point<T> mu_x, mu_xx;
    T mu_xy;

    friend class Moments<T>;

    CentralMoments(Point<T> pt) : origin(std::move(pt)), mu_x(), mu_xx(), mu_xy() {}
    CentralMoments(Point<T> pt, Point<T> mx, Point<T> mxx, T mxy) :
        origin(std::move(pt)), mu_x(std::move(mx)), mu_xx(std::move(mxx)), mu_xy(mxy) {}
};

template <typename T>
class Moments
{
public:
    Moments() = default;

    template <typename Pt, typename = std::enable_if_t<std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt>>>>
    Moments(Pt && pt) : org(std::forward<Pt>(pt)), mu(), mu_x(), mu_xx(), mu_xy() {}

    Moments(const PixelSet<T> & pset) : Moments()
    {
        if (pset.size())
        {
            org = std::next(pset.begin(), pset.size() / 2)->first;
            insert(pset.begin(), pset.end());
        }
    }

    // In-place operators

    Moments & operator+=(Moments rhs)
    {
        rhs.move(org);
        mu += rhs.mu;
        mu_x += rhs.mu_x;
        mu_xx += rhs.mu_xx;
        mu_xy += rhs.mu_xy;
        return *this;
    }

    Moments & operator-=(Moments rhs)
    {
        rhs.move(org);
        mu -= rhs.mu;
        mu_x -= rhs.mu_x;
        mu_xx -= rhs.mu_xx;
        mu_xy -= rhs.mu_xy;
        return *this;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void insert(const Point<V> & point, T val)
    {
        auto r = point - org;
        mu += val;
        mu_x += r * val;
        mu_xx += r * r * val;
        mu_xy += r.x() * r.y() * val;
    }

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    void insert(const Pixel<V> & pixel)
    {
        insert(std::get<0>(pixel), std::get<1>(pixel));
    }

    template <typename InputIt, typename Value = typename std::iterator_traits<InputIt>::value_type, typename V = typename Value::second_type,
        typename = std::enable_if_t<std::is_same_v<Pixel<V>, Value> && std::is_convertible_v<T, V>>
    >
    void insert(InputIt first, InputIt last)
    {
        for (; first != last; ++first) insert(*first);
    }

    void move(const Point<T> & point) const
    {
        if (org != point)
        {
            auto r = org - point;
            mu_xx += 2 * r * mu_x + r * r * mu;
            mu_xy += r.x() * mu_x.y() + r.y() * mu_x.x() + r.x() * r.y() * mu;
            mu_x += r * mu;
            org = point;
        }
    }

    // Friend members

    friend Moments operator+(const Moments & lhs, const Moments & rhs)
    {
        Moments result = lhs;
        result += rhs;
        return result;
    }

    friend Moments operator-(const Moments & lhs, const Moments & rhs)
    {
        Moments result = lhs;
        result += rhs;
        return result;
    }

    friend std::ostream & operator<<(std::ostream & os, const Moments & m)
    {
        os << "{origin = " << m.org << ", mu = " << m.mu << ", mu_x = " << m.mu_x
           << ", mu_xx = " << m.mu_xx << ", mu_xy = " << m.mu_xy << "}";
        return os;
    }

    // Other members

    CentralMoments<T> central() const
    {
        if (mu)
        {
            auto M_X = mu_x / mu;
            auto M_XX = mu_xx / mu - M_X * M_X;
            auto M_XY = mu_xy / mu - M_X[0] * M_X[1];
            return CentralMoments<T>{org, std::move(M_X), std::move(M_XX), M_XY};
        }
        return CentralMoments<T>{org};
    }

    const Point<T> & origin() const {return org;}

    T zeroth() const {return mu;}
    std::array<T, 2> first() const {return mu_x + org * mu;}
    std::array<T, 4> second() const
    {
        auto m_xx = mu_xx + 2 * org * mu_x + org * org * mu;
        auto m_xy = mu_xy + org.x() * mu_x.y() + org.y() * mu_x.x() + org.x() * org.y() * mu;
        return {m_xx[0], m_xy, m_xy, m_xx[1]};
    }

private:
    Point<T> org;
    T mu;
    Point<T> mu_x, mu_xx;
    T mu_xy;
};

// Connectivity structure class

struct Structure : public WrappedContainer<std::vector<Point<long>>>
{
    int radius, rank;

    Structure(int radius, int rank) : radius(radius), rank(rank)
    {
        if (rank > 2 * radius) rank = 2 * radius;
        for (int i = -radius; i <= radius; i++)
        {
            for (int j = -radius; j <= radius; j++)
            {
                if (std::abs(i) + std::abs(j) <= rank) m_ctr.emplace_back(Point<long>{j, i});
            }
        }
    }

    Structure & sort()
    {
        auto compare = [](const Point<long> & a, const Point<long> & b)
        {
            return a.x() * a.x() + a.y() * a.y() < b.x() * b.x() + b.y() * b.y();
        };
        std::sort(begin(), end(), compare);
        return *this;
    }

    std::string info() const
    {
        return "<Structure, radius = " + std::to_string(radius) + ", rank = " + std::to_string(rank) +
                         ", points = <Points, size = " + std::to_string(size()) + ">>";
    }
};

// Extended interface of set of points - needed for Regions

struct PointsSet : public WrappedContainer<std::set<Point<long>>>
{
    using WrappedContainer<std::set<Point<long>>>::WrappedContainer;

    template <typename Func, typename = std::enable_if_t<std::is_invocable_r_v<bool, std::remove_cvref_t<Func>, Point<long>>>>
    PointsSet(Point<long> seed, Func && func, const Structure & structure)
    {
        if (std::forward<Func>(func)(seed))
        {
            std::vector<Point<long>> last_pixels;
            std::unordered_set<Point<long>, detail::PointHasher<long>> new_pixels;

            last_pixels.emplace_back(std::move(seed));

            while (last_pixels.size())
            {
                for (const auto & point: last_pixels)
                {
                    for (const auto & shift: structure)
                    {
                        new_pixels.insert(point + shift);
                    }
                }
                last_pixels.clear();

                for (auto && point: new_pixels)
                {
                    if (std::forward<Func>(func)(point))
                    {
                        auto [iter, is_added] = m_ctr.insert(std::forward<decltype(point)>(point));
                        if (is_added) last_pixels.push_back(*iter);
                    }
                }
                new_pixels.clear();
            }
        }
    }

    PointsSet(Point<long> seed, const array<bool> & mask, const Structure & structure) :
        PointsSet(seed, [&mask](Point<long> pt){return mask.is_inbound(pt.coordinate()) && mask.at(pt.coordinate());}, structure) {}

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> & array, bool value) const
    {
        for (const auto & pt : m_ctr)
        {
            if (array.is_inbound(pt.coordinate())) array.at(pt.coordinate()) = value;
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> && array, bool value) const
    {
        mask(array, value);
    }

    std::string info() const
    {
        return "<PointsSet, size = " + std::to_string(m_ctr.size()) + ">";
    }
};

// Set of [point, value] pairs

template <typename T>
class Pixels
{
public:
    Pixels() = default;

    Pixels(const PixelSet<T> & pset) : m_mnt(pset), m_pset(pset) {}
    Pixels(PixelSet<T> && pset) : m_mnt(pset), m_pset(std::move(pset)) {}

    Pixels(const PointsSet & points, const array<T> & data)
    {
        for (auto && pt : points)
        {
            if (data.is_inbound(pt.coordinate()))
            {
                m_pset.insert(m_pset.end(), make_pixel(std::forward<decltype(pt)>(pt), data));
            }
        }
        m_mnt = m_pset;
    }

    void merge(Pixels & source)
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

    void merge(Pixels && source)
    {
        merge(source);
    }

    Line<T> line() const
    {
        auto m_ctr = m_mnt.central();
        if (m_ctr.is_positive()) return m_mnt.central().line();
        return {m_mnt.origin(), m_mnt.origin()};
    }

    const PixelSet<T> & pixels() const {return m_pset;}
    const Moments<T> & moments() const {return m_mnt;}

protected:
    Moments<T> m_mnt;
    PixelSet<T> m_pset;
};

struct Regions : public WrappedContainer<std::vector<PointsSet>>
{
    using WrappedContainer<std::vector<PointsSet>>::WrappedContainer;

    std::string info() const
    {
        return "<Regions, regions = <List[PointsSet], size = " + std::to_string(size()) + ">>";
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> & array, bool value) const
    {
        for (const auto & region: m_ctr) region.mask(array, value);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    void mask(array<I> && array, bool value) const
    {
        mask(array, value);
    }
};

}

#endif
