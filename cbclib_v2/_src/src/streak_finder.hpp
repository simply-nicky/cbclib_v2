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

struct Peaks : public PointsSet
{
protected:
    using PointsSet::m_ctr;

public:
    Peaks() = default;

    template <typename Pts, typename = std::enable_if_t<std::is_same_v<container_type, std::remove_cvref_t<Pts>>>>
    Peaks(Pts && pts) : PointsSet(std::forward<Pts>(pts))
    {
        std::vector<std::pair<Point<long>, std::nullptr_t>> items;
        std::transform(m_ctr.begin(), m_ctr.end(), std::back_inserter(items), [](Point<long> pt){return std::make_pair(pt, nullptr);});
    }

    template <typename T>
    Peaks(const array<T> & data, const array<bool> & mask, size_t radius, T vmin)
    {
        // Inserting maxima in a grid of {x / radius, y / radius} coordinates
        std::array<size_t, 2> axes {0, 1};
        std::unordered_map<Point<long>, Point<long>, detail::PointHasher<long>> peak_map;
        auto add_peak = [&data, &mask, &peak_map, radius, vmin](size_t index)
        {
            long y = data.index_along_dim(index, 0);
            long x = data.index_along_dim(index, 1);
            long r = radius;
            if (mask.at(y, x) && data.at(y, x) > vmin)
            {
                peak_map.try_emplace(Point<long>{x / r, y / r}, Point<long>{x, y});
            }
        };

        for (auto axis : axes)
        {
            for (size_t i = radius / 2; i < data.shape(1 - axis); i += radius)
            {
                auto dline = data.slice(i, axis);
                maxima_nd(dline.begin(), dline.end(), add_peak, data, axes, 1);
            }
        }

        // Moving a set of m_ctr into the container
        for (auto && [_, point] : peak_map)
        {
            m_ctr.emplace_hint(m_ctr.end(), std::forward<decltype(point)>(point));
        }
    }

    template <typename T>
    Peaks(py::array_t<T> data, py::array_t<bool> mask, size_t radius, T vmin)
        : Peaks(array<T>(data.request()), array<bool>(mask.request()), radius, vmin) {}

    template <typename T>
    void filter(const array<T> & data, const array<bool> & mask, const Structure & srt, T vmin, size_t npts)
    {
        auto func = [&data, &mask, vmin](Point<long> pt)
        {
            if (data.is_inbound(pt.coordinate()))
            {
                auto idx = data.index_at(pt.coordinate());
                return mask[idx] && data[idx] > vmin;
            }
            return false;
        };

        vector_array<unsigned char> is_good ({mask.size()}, false);
        for (auto iter = begin(); iter != end();)
        {
            if (!is_good[is_good.index_at(iter->coordinate())])
            {
                PointsSet support {*iter, func, srt};

                if (support.size() < npts) iter = m_ctr.erase(iter);
                else
                {
                    support.mask(is_good, true);
                    ++iter;
                }
            }
            else ++iter;
        }
    }

    template <typename T>
    std::list<iterator> sort(const array<T> & data) const
    {
        // list container is used to enable deletion inside the loop
        std::list<iterator> result;
        for (auto iter = m_ctr.begin(); iter != m_ctr.end(); ++iter) result.push_back(iter);

        // Sorting peaks in descending order
        auto compare = [&data](iterator a, iterator b)
        {
            return data.at(a->coordinate()) > data.at(b->coordinate());
        };
        result.sort(compare);

        return result;
    }

    std::string info() const
    {
        return "<Peaks, points = <Points, size = " + std::to_string(m_ctr.size()) + ">>";
    }
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

    Line<long> central_line() const {return Line<long>{*m_min, *m_max};}

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
    iterator_t m_min, m_max;

    void update_minmax()
    {
        auto tau = line().tangent();
        auto ctr = center();
        auto compare = [tau, ctr](const Point<long> & pt0, const Point<long> & pt1)
        {
            return dot(tau, pt0 - ctr) < dot(tau, pt1 - ctr);
        };
        std::tie(m_min, m_max) = std::minmax_element(m_ctrs.begin(), m_ctrs.end(), compare);
    }
};

template <typename T>
class StreakFinderResult
{
public:
    enum flags
    {
        bad = -1,
        not_used = 0
    };

    using iterator = std::map<int, Streak<T>>::const_iterator;
    using const_iterator = iterator;

    StreakFinderResult(const array<T> & d, const array<bool> & m)
    {
        if (m.ndim() != 2)
            throw std::invalid_argument("StreakFinder mask array has invalid number of dimensions (" +
                                        std::to_string(m.ndim()) + ")");
        check_equal("data and mask have incompatible shapes",
                    d.shape().begin(), d.shape().end(), m.shape().begin(), m.shape().end());

        std::vector<int> mvec;
        for (auto flag: m)
        {
            if (flag) mvec.push_back(flags::not_used);
            else mvec.push_back(flags::bad);
        }
        m_mask = vector_array<int>(m.shape(), std::move(mvec));
    }

    iterator begin() const {return m_streaks.begin();}
    iterator end() const {return m_streaks.end();}

    size_t size() const {return m_streaks.size();}
    iterator find(int key) const {return m_streaks.find(key);}

    iterator erase(iterator pos)
    {
        erase_mask(pos->second.pixels(), pos->first);
        return m_streaks.erase(pos);
    }

    std::pair<iterator, bool> insert(std::pair<int, Streak<T>> && elem)
    {
        auto [iter, is_added] = m_streaks.emplace(std::move(elem));
        if (is_added)
        {
            insert_mask(iter->second.pixels(), iter->first);
        }
        return std::make_pair(iter, is_added);
    }

    bool is_bad(const Point<long> & point) const
    {
        if (m_mask.is_inbound(point.coordinate()))
        {
            return m_mask.at(point.coordinate()) == flags::bad;
        }
        return true;
    }

    bool is_free(const Point<long> & point) const
    {
        if (m_mask.is_inbound(point.coordinate()))
        {
            return m_mask.at(point.coordinate()) == flags::not_used;
        }
        return false;
    }

    Streak<T> new_streak(const array<T> & data, const Structure & structure, long x, long y) const
    {
        PixelSet<T> pset;
        for (auto shift : structure)
        {
            Point<long> pt {x + shift.x(), y + shift.y()};

            if (!is_bad(pt)) pset.emplace(make_pixel(std::move(pt), data));
        }
        return Streak<T>{std::move(pset), Point<long>{x, y}};
    }

    Streak<T> new_streak(const array<T> & data, const Structure & structure, const Point<long> & point) const
    {
        return new_streak(data, structure, point.x(), point.y());
    }

    T probability(const array<T> & data, T vmin) const
    {
        std::vector<T> values;
        for (size_t i = 0; i < m_mask.size(); i++)
        {
            if (m_mask[i] != flags::bad) values.push_back(data[i]);
        }

        std::sort(values.begin(), values.end());
        auto index = std::distance(values.begin(), std::lower_bound(values.begin(), values.end(), vmin));
        return T(1.0) - T(index) / values.size();
    }

    T p_value(const Streak<T> & streak, T xtol, T vmin, T p) const
    {
        return p_value(streak.pixels(), streak.line(), xtol, vmin, p, flags::not_used);
    }

    T p_value(iterator iter, T xtol, T vmin, T p) const
    {
        return p_value(iter->second.pixels(), iter->second.line(), xtol, vmin, p, iter->first);
    }

    const vector_array<int> & mask() const {return m_mask;}

protected:
    vector_array<int> m_mask;
    std::map<int, Streak<T>> m_streaks;

    void erase_mask(const PixelSet<T> & pset, int flag)
    {
        for (auto [pt, _] : pset)
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                auto index = m_mask.index_at(pt.coordinate());
                if (m_mask[index] != flags::bad) m_mask[index] -= flag;
            }
        }
    }

    void insert_mask(const PixelSet<T> & pset, int flag)
    {
        for (auto [pt, _] : pset)
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                auto index = m_mask.index_at(pt.coordinate());
                if (m_mask[index] != flags::bad) m_mask[index] += flag;
            }
        }
    }

    template <typename U>
    T p_value(const PixelSet<T> & pset, const Line<U> & line, T xtol, T vmin, T p, int flag) const
    {
        size_t n = 0, k = 0;
        for (auto [pt, val] : pset)
        {
            if (m_mask.is_inbound(pt.coordinate()))
            {
                if (val > T() && m_mask.at(pt.coordinate()) == flag && line.distance(pt) < xtol)
                {
                    n++;
                    if (val > vmin) k++;
                }
            }
        }

        return detail::logbinom(n, k, p);
    }
};

struct StreakFinder
{
    Structure structure;
    unsigned min_size;
    unsigned lookahead;
    unsigned nfa;

    template <typename Struct, typename = std::enable_if_t<std::is_same_v<Structure, std::remove_cvref_t<Struct>>>>
    StreakFinder(Struct && s, unsigned ms, unsigned lad, unsigned nf) :
        structure(std::forward<Struct>(s)), min_size(ms), lookahead(lad), nfa(nf)
    {
        structure.sort();
    }

    template <typename T>
    StreakFinderResult<T> detect_streaks(StreakFinderResult<T> && result, const array<T> & data, Peaks peaks, T xtol, T vmin)
    {
        auto p = result.probability(data, vmin);
        auto log_eps = std::log(p) * min_size;

        // The key is a pair of values : number of pixels in the streaks and the streak id
        // The streaks will be sorted from small to large
        std::map<std::pair<size_t, int>, typename StreakFinderResult<T>::iterator> streaks;
        int streak_id = 0;

        auto indices = peaks.sort(data);
        while (indices.size())
        {
            auto piter = indices.front();
            auto seed = *piter;
            indices.pop_front(); peaks->erase(piter);

            auto streak = new_streak(seed, result, data, peaks, xtol);
            if (result.p_value(streak, xtol, vmin, p) < log_eps)
            {
                auto [riter, is_added] = result.insert(std::make_pair(streak_id, std::move(streak)));

                if (is_added)
                {
                    // Removing all the peaks that belong to the detected streak
                    for (auto iiter = indices.begin(); iiter != indices.end();)
                    {
                        if (!result.is_free(**iiter))
                        {
                            peaks->erase(*iiter);
                            iiter = indices.erase(iiter);
                        }
                        else ++iiter;
                    }

                    streaks.emplace(std::make_pair(riter->second.pixels().size(), streak_id), riter);
                    streak_id++;
                }
            }
        }

        for (auto [_, siter] : streaks)
        {
            if (result.p_value(siter, xtol, vmin, p) >= log_eps) result.erase(siter);
        }

        return std::move(result);
    }

    template <typename T>
    Streak<T> new_streak(const Point<long> & seed, const StreakFinderResult<T> & result, const array<T> & data, Peaks peaks, T xtol) const
    {
        auto streak = result.new_streak(data, structure, seed);

        Line<long> old_line = Line<long>{}, line = streak.central_line();
        while (old_line != line)
        {
            old_line = line;

            streak = grow_streak<T, false>(std::move(streak), result, data, old_line.pt0, peaks, xtol);
            streak = grow_streak<T, true>(std::move(streak), result, data, old_line.pt1, peaks, xtol);
            line = streak.central_line();
        }

        return streak;
    }

    std::string info() const
    {
        return "<StreakFinder, struct = " + structure.info() + ", min_size = " + std::to_string(min_size) +
               ", lookahead = " + std::to_string(lookahead) + ", nfa = " + std::to_string(nfa) + ">";
    }

private:
    template <typename T>
    std::pair<bool, Streak<T>> add_point_to_streak(Streak<T> && streak, const StreakFinderResult<T> & result, const array<T> & data, const Point<long> & pt, T xtol) const
    {
        auto new_streak = streak;
        new_streak.merge(result.new_streak(data, structure, pt));
        auto new_line = new_streak.line();

        auto is_unaligned = [&new_line, xtol](const Point<T> & pt)
        {
            return new_line.distance(pt) >= xtol;
        };
        auto num_unaligned = std::transform_reduce(new_streak.ends().begin(), new_streak.ends().end(), unsigned(), std::plus(), is_unaligned);

        if (num_unaligned <= nfa)
        {
            return std::make_pair(true, std::move(new_streak));
        }

        return std::make_pair(false, std::move(streak));
    }

    template <typename T, bool IsForward>
    Point<long> find_next_step(const Streak<T> & streak, const Point<long> & point, int n_steps) const
    {
        auto iter = BresenhamPlotter<T, 2, IsForward>{streak.line()}.begin(point);
        for (int i = 0; i < n_steps; i++) iter++;
        return *iter;
    }

    template <typename T, bool IsForward>
    Streak<T> grow_streak(Streak<T> && streak, const StreakFinderResult<T> & result, const array<T> & data, Point<long> point, const Peaks & peaks, T xtol) const
    {
        unsigned tries = 0;
        while (tries <= lookahead)
        {
            Point<long> pt = find_next_step<T, IsForward>(streak, point, structure.rank);

            // Find the closest peak in structure vicinity
            for (const auto & shift : structure)
            {
                auto iter = peaks->find(pt + shift);
                // Check if the point is not used already
                if (iter != peaks.end() && result.is_free(*iter) && *iter != point)
                {
                    pt = *iter; break;
                }
            }

            if (!result.is_bad(pt) && pt != point)
            {
                auto [is_add, new_streak] = add_point_to_streak(std::move(streak), result, data, pt, xtol);

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
