#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "bresenham.hpp"
#include "label.hpp"
#include "signal_proc.hpp"

namespace cbclib {

namespace detail{

// Return log(binomial_tail(n, k, p))
// binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
// bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

template <typename I, typename T>
T logbinom(I n, I k, T p)
{
    if (n == k) return n * std::log(p);

    auto term = std::exp(std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1) +
                         k * std::log(p) + (n - k) * std::log(T(1.0) - p));
    auto bin_tail = term;
    auto p_term = p / (T(1.0) - p);

    for (I i = k + 1; i < n + 1; i++)
    {
        term *= (n - i + 1) / i * p_term;
        bin_tail += term;
    }

    return std::log(bin_tail);
}

}

// Sparse 2D peaks

struct Peaks;

template <typename Iterator, typename = std::iterator_traits<Iterator>::value_type::second_type>
class ValueIterator
{
public:
    using iterator_category = std::iterator_traits<Iterator>::iterator_category;
    using value_type = std::iterator_traits<Iterator>::value_type::second_type;
    using difference_type = typename std::iter_difference_t<Iterator>;
    using reference = const value_type &;
    using pointer = const value_type *;

    ValueIterator() = default;
    ValueIterator(Iterator && iter) : m_iter(std::move(iter)) {}
    ValueIterator(const Iterator & iter) : m_iter(iter) {}

    ValueIterator & operator++() requires (std::forward_iterator<Iterator>)
    {
        ++m_iter;
        return *this;
    }

    ValueIterator operator++(int) requires (std::forward_iterator<Iterator>)
    {
        return ValueIterator(m_iter++);
    }

    ValueIterator & operator--() requires (std::bidirectional_iterator<Iterator>)
    {
        --m_iter;
        return *this;
    }

    ValueIterator operator--(int) requires (std::bidirectional_iterator<Iterator>)
    {
        return ValueIterator(m_iter--);
    }

    ValueIterator & operator+=(difference_type offset) requires (std::random_access_iterator<Iterator>)
    {
        m_iter += offset;
        return *this;
    }

    ValueIterator operator+(difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return ValueIterator(m_iter + offset);
    }

    ValueIterator & operator-=(difference_type offset) requires (std::random_access_iterator<Iterator>)
    {
        m_iter -= offset;
        return *this;
    }

    ValueIterator operator-(difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return ValueIterator(m_iter - offset);
    }

    difference_type operator-(const ValueIterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter - rhs;
    }

    reference operator[](difference_type offset) const requires (std::random_access_iterator<Iterator>)
    {
        return (m_iter + offset)->to_array();
    }

    bool operator==(const ValueIterator & rhs) const requires (std::forward_iterator<Iterator>)
    {
        return m_iter == rhs.m_iter;
    }
    bool operator!=(const ValueIterator & rhs) const requires (std::forward_iterator<Iterator>)
    {
        return !(*this == rhs);
    }

    bool operator<(const ValueIterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter < rhs.m_iter;
    }
    bool operator>(const ValueIterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return m_iter > rhs.m_iter;
    }

    bool operator<=(const ValueIterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return !(*this > rhs);
    }
    bool operator>=(const ValueIterator & rhs) const requires (std::random_access_iterator<Iterator>)
    {
        return !(*this < rhs);
    }

    reference operator*() const {return m_iter->second;}
    pointer operator->() const {return &(m_iter->second);}

private:
    Iterator m_iter;

    friend class Peaks;
};

struct Peaks
{
protected:
    std::map<Point<long>, Point<long>> m_ctr;
    long m_radius;

public:
    using container_type = std::map<Point<long>, Point<long>>;
    using value_type = Point<long>;
    using size_type = typename container_type::size_type;

    using iterator = ValueIterator<container_type::iterator>;
    using const_iterator = ValueIterator<container_type::const_iterator>;

    Peaks(long radius) : m_ctr(), m_radius(radius) {}

    void clear() {m_ctr.clear();}

    const_iterator find(const Point<long> & key) const
    {
        return m_ctr.find(Point<long>{key.x() / m_radius, key.y() / m_radius});
    }

    iterator find(const Point<long> & key)
    {
        return m_ctr.find(Point<long>{key.x() / m_radius, key.y() / m_radius});
    }

    const_iterator find_range(const Point<long> & key, long range) const
    {
        auto start = (key - range) / m_radius;
        auto end = (key + range) / m_radius + 1;
        const_iterator iter = m_ctr.end();
        for (long x = start[0]; x != end[0]; x++)
        {
            for (long y = start[1]; y != end[1]; y++)
            {
                iter = choose(iter, m_ctr.find(Point<long>{x, y}), key);
            }
        }

        if (iter != m_ctr.end() && distance(key, *iter) < range * range) return iter;
        return m_ctr.end();
    }

    std::pair<iterator, bool> insert(const Point<long> & value)
    {
        auto [iter, is_inserted] = m_ctr.emplace(Point<long>{value.x() / m_radius, value.y() / m_radius}, value);
        return std::make_pair(ValueIterator(iter), is_inserted);
    }

    std::pair<iterator, bool> insert(Point<long> && value)
    {
        auto [iter, is_inserted] = m_ctr.emplace(Point<long>{value.x() / m_radius, value.y() / m_radius}, std::move(value));
        return std::make_pair(ValueIterator(iter), is_inserted);
    }

    iterator erase(const_iterator pos)
    {
        return m_ctr.erase(pos.m_iter);
    }

    iterator erase(iterator pos)
    {
        return m_ctr.erase(pos.m_iter);
    }

    void merge(Peaks & source)
    {
        if (source.m_radius == m_radius) m_ctr.merge(source.m_ctr);
    }

    void merge(Peaks && source)
    {
        if (source.m_radius == m_radius) m_ctr.merge(std::move(source.m_ctr));
    }

    const_iterator begin() const {return ValueIterator(m_ctr.begin());}
    iterator begin() {return ValueIterator(m_ctr.begin());}
    const_iterator end() const {return ValueIterator(m_ctr.end());}
    iterator end() {return ValueIterator(m_ctr.end());}

    size_type size() const {return m_ctr.size();}

    long radius() const {return m_radius;}

    std::string info() const
    {
        return "<Peaks, points = <Points, size = " + std::to_string(m_ctr.size()) + ">>";
    }

private:
    const_iterator choose(const_iterator first, const_iterator second, const Point<long> & point) const
    {
        if (first == end()) return second;
        if (second == end()) return first;

        if (distance(*first, point) < distance(*second, point)) return first;
        return second;
    }
};

template <typename T>
class PeaksData
{
public:
    PeaksData(array<T> data, array<bool> mask) : m_data(std::move(data)), m_mask(std::move(mask)) {}

    template <typename InputIt, typename = std::enable_if_t<
        std::is_base_of_v<typename array<T>::iterator, InputIt> || std::is_base_of_v<typename array<T>::const_iterator, InputIt>
    >>
    void insert(InputIt first, InputIt last, Peaks & peaks, T vmin, size_t order)
    {
        auto insert = [this, vmin, &peaks](size_t index)
        {
            long y = m_data.index_along_dim(index, 0);
            long x = m_data.index_along_dim(index, 1);

            if (m_mask.at(y, x) && m_data.at(y, x) > vmin)
            {
                auto iter = peaks.find(Point<long>{x, y});
                if (iter == peaks.end()) peaks.insert(Point<long>{x, y});
                else if (m_data.at(y, x) > m_data.at(iter->coordinate())) peaks.insert(Point<long>{x, y});
            }
        };

        maxima_nd(first, last, insert, m_data, Axes, order);
    }

    const array<T> & data() const {return m_data;}
    const array<bool> & mask() const {return m_data;}

protected:
    constexpr static std::array<size_t, 2> Axes = {0, 1};

    array<T> m_data;
    array<bool> m_mask;

};

template <typename T>
struct FilterData : PeaksData<T>
{
public:
    FilterData(array<T> data, array<bool> mask) : PeaksData<T>(std::move(data), std::move(mask))
    {
        m_good = vector_array<unsigned char>{m_mask.shape(), 0};
    }

    template <typename InputIt, typename = std::enable_if_t<
        std::is_base_of_v<typename Peaks::iterator, InputIt>
    >>
    void filter(InputIt first, InputIt last, std::vector<InputIt> & output, const Structure & srt, T vmin, size_t npts)
    {
        auto func = [this, vmin](const Point<long> & pt)
        {
            if (m_data.is_inbound(pt.coordinate()))
            {
                auto idx = m_data.index_at(pt.coordinate());
                return m_mask[idx] && m_data[idx] > vmin;
            }
            return false;
        };
        auto stop = [npts](const PointSet & support)
        {
            return support.size() < npts;
        };

        for (auto iter = first; iter != last; ++iter)
        {
            if (!m_good[m_good.index_at(iter->coordinate())])
            {
                PointSet support;
                support->insert(*iter);
                support.dilate(func, srt, stop);

                if (support.size() < npts) output.push_back(iter);
                else support.mask(m_good, true);
            }
        }
    }

protected:
    using PeaksData<T>::m_data;
    using PeaksData<T>::m_mask;
    vector_array<unsigned char> m_good;
};

// Streak class

template <typename T>
class Streak
{
public:
    template <typename PSet, typename Pt, typename = std::enable_if_t<
        std::is_same_v<PixelSet<T>, std::remove_cvref_t<PSet>> &&
        std::is_constructible_v<Point<long>, std::remove_cvref_t<Pt>>
    >>
    Streak(PSet && pset, Pt && ctr) : m_pxls(std::forward<PSet>(pset))
    {
        auto [pt0, pt1] = line().to_pair();
        m_ctrs.emplace_back(std::forward<Pt>(ctr));
        m_ends.emplace_back(std::move(pt0));
        m_ends.emplace_back(std::move(pt1));
        update_minmax();
    }

    void merge(Streak & streak)
    {
        m_pxls.merge(streak.m_pxls);
        m_ctrs.insert(m_ctrs.end(), streak.m_ctrs.begin(), streak.m_ctrs.end());
        m_ends.insert(m_ends.end(), streak.m_ends.begin(), streak.m_ends.end());
        update_minmax();
    }

    void merge(Streak && streak)
    {
        merge(streak);
    }

    const Point<long> & center() const {return m_ctrs.front();}
    long id() const {return center().x() + center().y();}

    Line<long> central_line() const {return Line<long>{m_ctrs[m_min], m_ctrs[m_max]};}

    Line<T> line() const {return m_pxls.line();}
    const PixelSet<T> & pixels() const {return m_pxls.pixels();}
    const Moments<T> & moments() const {return m_pxls.moments();}
    const std::vector<Point<long>> & centers() const {return m_ctrs;}
    const std::vector<Point<T>> & ends() const {return m_ends;}

protected:
    using iterator_t = std::vector<Point<long>>::const_iterator;

    Pixels<T> m_pxls;
    std::vector<Point<long>> m_ctrs;
    std::vector<Point<T>> m_ends;
    size_t m_min, m_max;

    void update_minmax()
    {
        auto tau = line().tangent();
        auto ctr = center();
        auto compare = [tau, ctr](const Point<long> & pt0, const Point<long> & pt1)
        {
            return dot(tau, pt0 - ctr) < dot(tau, pt1 - ctr);
        };
        const auto [min, max] = std::minmax_element(m_ctrs.begin(), m_ctrs.end(), compare);
        m_min = std::distance(m_ctrs.begin(), min), m_max = std::distance(m_ctrs.begin(), max);
    }
};

class StreakMask
{
public:
    enum flags
    {
        bad = -1,
        not_used = 0
    };

    StreakMask() = default;

    StreakMask(array<bool> mask) : m_mask(std::move(mask))
    {
        m_flags = std::vector<int>(m_mask.size(), not_used);
    }

    template <typename T>
    void remove(const Streak<T> & streak)
    {
        for (auto [pt, _] : streak.pixels())
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                auto index = m_mask.index_at(pt.coordinate());
                if (m_mask[index]) m_flags[index] -= streak.id();
            }
        }
    }

    template <typename T>
    void add(const Streak<T> & streak)
    {
        for (auto [pt, _] : streak.pixels())
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                auto index = m_mask.index_at(pt.coordinate());
                if (m_mask[index]) m_flags[index] += streak.id();
            }
        }
    }

    bool is_bad(const Point<long> & point) const
    {
        if (m_mask.is_inbound(point.coordinate())) return !m_mask.at(point.coordinate());
        return true;
    }

    bool is_free(const Point<long> & point) const
    {
        if (m_mask.is_inbound(point.coordinate()))
        {
            size_t index = m_mask.index_at(point.coordinate());
            if (m_mask[index]) return m_flags[index] == not_used;
        }
        return false;
    }

    template <typename T>
    T p_value(const Streak<T> & streak, T xtol, T vmin, T p, int flag) const
    {
        size_t n = 0, k = 0;
        auto line = streak.line();
        for (auto [pt, val] : streak.pixels())
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                size_t index = m_mask.index_at(pt.coordinate());
                if (m_mask[index] && m_flags[index] == flag)
                {
                    if (val > T() && line.distance(pt) < xtol)
                    {
                        n++;
                        if (val > vmin) k++;
                    }
                }
            }
        }

        return detail::logbinom(n, k, p);
    }

    const array<bool> & mask() const {return m_mask;}
    bool mask(size_t index) const {return m_mask[index];}

    const std::vector<int> & flags() const {return m_flags;}
    int flags(size_t index) const {return m_flags[index];}

    void clear()
    {
        std::fill(m_flags.begin(), m_flags.end(), not_used);
    }

protected:
    array<bool> m_mask;
    std::vector<int> m_flags;
};

template <typename T>
struct StreakFinderInput
{
public:
    using const_iterator = std::vector<Point<long>>::const_iterator;
    using iterator = std::vector<Point<long>>::iterator;

    StreakFinderInput() = default;

    StreakFinderInput(Peaks peaks, array<T> data, const Structure & structure, unsigned lookahead, unsigned nfa) :
        m_peaks(std::move(peaks)), m_data(std::move(data)), m_structure(structure), m_lookahead(lookahead), m_nfa(nfa) {}

    const array<T> & data() const {return m_data;}
    const Peaks & peaks() const {return m_peaks;}

    Streak<T> get_streak(const Point<long> & seed, const StreakMask & mask) const
    {
        PixelSet<T> pset;
        for (auto shift : m_structure)
        {
            Point<long> pt {seed.x() + shift.x(), seed.y() + shift.y()};

            if (!mask.is_bad(pt)) pset.emplace(make_pixel(std::move(pt), m_data));
        }
        return Streak<T>{std::move(pset), seed};
    }

    Streak<T> get_streak(const Point<long> & seed, const StreakMask & mask, T xtol) const
    {
        auto streak = get_streak(seed, mask);

        Line<long> old_line = Line<long>{}, line = streak.central_line();
        while (old_line != line)
        {
            old_line = line;

            streak = grow_streak<false>(std::move(streak), old_line.pt0, mask, xtol);
            streak = grow_streak<true>(std::move(streak), old_line.pt1, mask, xtol);
            line = streak.central_line();
        }

        return streak;
    }

    std::vector<Point<long>> points(size_t index, size_t size) const
    {
        std::vector<Point<long>> points;
        auto first = std::next(m_peaks.begin(), index);
        auto last = std::next(first, size);
        for (auto iter = first; iter != last; iter++) points.push_back(*iter);

        auto compare = [this](const Point<long> & a, const Point<long> & b)
        {
            return m_data.at(a.coordinate()) > m_data.at(b.coordinate());
        };
        std::sort(points.begin(), points.end(), compare);
        return points;
    }

protected:
    Peaks m_peaks;
    array<T> m_data;
    Structure m_structure;
    unsigned m_lookahead;
    unsigned m_nfa;

    std::pair<bool, Streak<T>> add_point_to_streak(Streak<T> && streak, const StreakMask & mask, const Point<long> & pt, T xtol) const
    {
        auto new_streak = streak;
        new_streak.merge(get_streak(pt, mask));
        auto new_line = new_streak.line();

        auto is_unaligned = [&new_line, xtol](const Point<T> & pt)
        {
            return new_line.distance(pt) >= xtol;
        };
        auto num_unaligned = std::transform_reduce(new_streak.ends().begin(), new_streak.ends().end(), unsigned(), std::plus(), is_unaligned);

        if (num_unaligned <= m_nfa)
        {
            return std::make_pair(true, std::move(new_streak));
        }

        return std::make_pair(false, std::move(streak));
    }

    template <bool IsForward>
    Point<long> find_next_step(const Streak<T> & streak, const Point<long> & point, int n_steps) const
    {
        auto iter = BresenhamPlotter<T, 2, IsForward>{streak.line()}.begin(point);
        for (int i = 0; i < n_steps; i++) iter++;
        return *iter;
    }

    template <bool IsForward>
    Streak<T> grow_streak(Streak<T> && streak, Point<long> point, const StreakMask & mask, T xtol) const
    {
        unsigned tries = 0;
        while (tries <= m_lookahead)
        {
            Point<long> pt = find_next_step<IsForward>(streak, point, m_structure.rank);

            // Find the closest peak in structure vicinity
            auto iter = m_peaks.find_range(pt, m_structure.rank);
            if (iter != m_peaks.end() && mask.is_free(*iter) && *iter != point) pt = *iter;

            if (!mask.is_bad(pt) && pt != point)
            {
                auto [is_add, new_streak] = add_point_to_streak(std::move(streak), mask, pt, xtol);

                if (is_add) return new_streak;
                else
                {
                    streak = std::move(new_streak);
                    tries++;
                }
            }
            else tries++;

            point = pt;
        }

        return streak;
    }
};

}

#endif
