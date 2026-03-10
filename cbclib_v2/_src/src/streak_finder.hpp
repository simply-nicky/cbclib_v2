#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "label.hpp"

namespace cbclib {

namespace detail {

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

} // namespace detail

// Local maxima finder for N-dimensional arrays, using a given structure element to determine neighbourhood

template <typename T>
class MaximaND
{
public:
    MaximaND(array<T> arr, Structure str) : m_arr(std::move(arr)), m_str(std::move(str)) {}

    template <typename UnaryFunction, typename = std::enable_if_t<
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, long>
    >>
    UnaryFunction find(size_t first, size_t last, size_t index, UnaryFunction && unary_op)
    {
        // We slice the array along the last (fast) axis
        auto line = m_arr.slice_back(index, m_arr.ndim() - 1);
        return find_maxima(std::next(line.begin(), first), std::next(line.begin(), last), std::forward<UnaryFunction>(unary_op), index);
    }

    template <typename UnaryFunction, typename = std::enable_if_t<
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, long>
    >>
    UnaryFunction find(size_t index, UnaryFunction && unary_op)
    {
        // We slice the array along the last (fast) axis
        auto line = m_arr.slice_back(index, m_arr.ndim() - 1);
        return find_maxima(line.begin(), line.end(), std::forward<UnaryFunction>(unary_op), index);
    }

    const array<T> & data() const {return m_arr;}

private:
    array<T> m_arr;
    Structure m_str;

    size_t index_at(size_t rest, size_t last_index)
    {
        size_t stride = m_arr.shape(m_arr.ndim() - 1);
        size_t index = last_index;

        for (size_t n = m_arr.ndim() - 1; n > 0; --n)
        {
            auto coord = rest % m_arr.shape(n - 1);
            index += coord * stride;
            rest /= m_arr.shape(n - 1);
            stride *= m_arr.shape(n - 1);
        }

        return index;
    }

    template <typename InputIt, typename UnaryFunction, typename = std::enable_if_t<
        (std::is_base_of_v<typename array<T>::const_iterator, InputIt> ||
         std::is_base_of_v<typename array<T>::iterator, InputIt>) &&
        std::is_invocable_v<remove_cvref_t<UnaryFunction>, long>
    >>
    UnaryFunction find_maxima(InputIt first, InputIt last, UnaryFunction && unary_op, size_t index)
    {
        T last_val = std::numeric_limits<T>::lowest();
        for (auto iter = first; iter < last; ++iter)
        {
            long running_idx = index_at(index, iter.index());

            T val = m_arr[running_idx];

            // If the current value is less than the last value, it cannot be a local maximum, so skip checking its neighbours
            if (val < last_val)
            {
                last_val = val;
                continue;
            }
            else last_val = val;

            for (const auto & shift : m_str.shifts())
            {
                size_t rest = running_idx;
                long neighbour_idx = 0;
                size_t stride = 1;

                // Converting linear index to multi-dimensional coordinate and applying shift
                for (size_t n = m_arr.ndim(); n > 0; --n)
                {
                    auto coord = rest % m_arr.shape(n - 1) + shift[n - 1];
                    if (coord < 0 || coord >= m_arr.shape(n - 1))
                    {
                        neighbour_idx = -1; // Out of bounds
                        break;
                    }
                    neighbour_idx += coord * stride;
                    rest /= m_arr.shape(n - 1);
                    stride *= m_arr.shape(n - 1);
                }

                if (neighbour_idx < 0 || m_arr[neighbour_idx] > val)
                {
                    running_idx = -1;
                    break;
                }
            }

            if (running_idx >= 0)
            {
                std::forward<UnaryFunction>(unary_op)(running_idx);
            }
        }

        return std::forward<UnaryFunction>(unary_op);
    }
};

// Enum to select which member of a pair to extract
enum class PairMember { First, Second };

// General pair member iterator — select First or Second member of pairs via template parameter
template <PairMember Member, typename Iterator>
class PairRange
{
public:
    class iterator
    {
    public:
        using value_type = std::conditional_t<
            Member == PairMember::First,
            typename std::iterator_traits<Iterator>::value_type::first_type,
            typename std::iterator_traits<Iterator>::value_type::second_type
        >;
        using iterator_category = typename std::iterator_traits<Iterator>::iterator_category;
        using difference_type = iter_difference_t<Iterator>;
        using reference = const value_type &;
        using pointer = const value_type *;

        iterator() = default;
        iterator(Iterator && iter) : m_iter(std::move(iter)) {}
        iterator(const Iterator & iter) : m_iter(iter) {}

        template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
        iterator & operator++()
        {
            ++m_iter;
            return *this;
        }

        template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
        iterator operator++(int)
        {
            return iterator(m_iter++);
        }

        template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
        iterator & operator--()
        {
            --m_iter;
            return *this;
        }

        template <typename I = Iterator, typename = std::enable_if_t<bidirectional_iterator_v<I>>>
        iterator operator--(int)
        {
            return iterator(m_iter--);
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        iterator & operator+=(difference_type offset)
        {
            m_iter += offset;
            return *this;
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        iterator operator+(difference_type offset) const
        {
            return iterator(m_iter + offset);
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        iterator & operator-=(difference_type offset)
        {
            m_iter -= offset;
            return *this;
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        iterator operator-(difference_type offset) const
        {
            return iterator(m_iter - offset);
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        difference_type operator-(const iterator & rhs) const
        {
            return m_iter - rhs.m_iter;
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        reference operator[](difference_type offset) const
        {
            if constexpr (Member == PairMember::First)
                return (m_iter + offset)->first;
            else
                return (m_iter + offset)->second;
        }

        template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
        bool operator==(const iterator & rhs) const
        {
            return m_iter == rhs.m_iter;
        }

        template <typename I = Iterator, typename = std::enable_if_t<forward_iterator_v<I>>>
        bool operator!=(const iterator & rhs) const
        {
            return !(*this == rhs);
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        bool operator<(const iterator & rhs) const
        {
            return m_iter < rhs.m_iter;
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        bool operator>(const iterator & rhs) const
        {
            return m_iter > rhs.m_iter;
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        bool operator<=(const iterator & rhs) const
        {
            return !(*this > rhs);
        }

        template <typename I = Iterator, typename = std::enable_if_t<random_access_iterator_v<I>>>
        bool operator>=(const iterator & rhs) const
        {
            return !(*this < rhs);
        }

        reference operator*() const
        {
            if constexpr (Member == PairMember::First)
                return m_iter->first;
            else
                return m_iter->second;
        }

        pointer operator->() const
        {
            if constexpr (Member == PairMember::First)
                return &(m_iter->first);
            else
                return &(m_iter->second);
        }

        Iterator base() const { return m_iter; }

    private:
        Iterator m_iter;
    };

    PairRange() = default;
    PairRange(Iterator begin, Iterator end) : m_begin(std::move(begin)), m_end(std::move(end)) {}

    iterator begin() const { return m_begin; }
    iterator end() const { return m_end; }

private:
    Iterator m_begin, m_end;
};

// Sparse 2D peaks

struct Peaks
{
protected:
    std::map<long, long> m_ctr;
    std::array<size_t, 2> m_shape, m_nbins;
    long m_radius;

public:
    using container_type = std::map<long, long>;
    using value_type = long;
    using size_type = typename container_type::size_type;

    using iterator = typename PairRange<PairMember::Second, container_type::iterator>::iterator;
    using const_iterator = typename PairRange<PairMember::Second, container_type::const_iterator>::iterator;

    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    Peaks(const Container & shape, long radius) : m_ctr(), m_shape(), m_nbins(), m_radius(radius)
    {
        m_shape = {static_cast<size_t>(shape[shape.size() - 2]),
                   static_cast<size_t>(shape[shape.size() - 1])};
        m_nbins = {m_shape[0] / radius + (m_shape[0] % radius != 0),
                   m_shape[1] / radius + (m_shape[1] % radius != 0)};
    }

    void clear() {m_ctr.clear();}

    const_iterator find(const Point<long> & key) const
    {
        return m_ctr.find(to_index(key));
    }

    iterator find(const Point<long> & key)
    {
        return m_ctr.find(to_index(key));
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
                iter = choose(iter, m_ctr.find(to_index(x, y)), key);
            }
        }

        if (iter != m_ctr.end() && distance(*iter, key) < range * range) return iter;
        return m_ctr.end();
    }

    std::pair<iterator, bool> insert(long index)
    {
        auto point = make_point<2>(index, m_shape);
        auto [iter, is_inserted] = m_ctr.emplace(to_index(point), index);
        return std::make_pair(iterator(iter), is_inserted);
    }

    template <typename T>
    auto inserter(const array<T> & data, T vmin)
    {
        return [this, &data, vmin](long index)
        {
            if (data[index] > vmin)
            {
                auto point = make_point<2>(index, m_shape);
                auto bin_idx = to_index(point);

                auto iter = m_ctr.find(bin_idx);
                if (iter == m_ctr.end()) m_ctr.emplace(bin_idx, index);
                else if (data[index] > data[iter->second]) iter->second = index;
            }
        };
    }

    iterator erase(const_iterator pos)
    {
        return m_ctr.erase(pos.base());
    }

    iterator erase(iterator pos)
    {
        return m_ctr.erase(pos.base());
    }

    void merge(Peaks & source)
    {
        if (source.m_radius == m_radius) m_ctr.merge(source.m_ctr);
    }

    void merge(Peaks && source)
    {
        if (source.m_radius == m_radius) m_ctr.merge(std::move(source.m_ctr));
    }

    const_iterator begin() const {return m_ctr.begin();}
    iterator begin() {return m_ctr.begin();}
    const_iterator end() const {return m_ctr.end();}
    iterator end() {return m_ctr.end();}

    size_type size() const {return m_ctr.size();}

    long radius() const {return m_radius;}
    const std::array<size_t, 2> & shape() const {return m_shape;}

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

    long to_index(long x, long y) const
    {
        return y * m_nbins[1] + x;
    }

    long to_index(const Point<long> & key) const
    {
        return (key.y() / m_radius) * m_nbins[1] + (key.x() / m_radius);
    }

    long distance(long index, const Point<long> & point) const
    {
        long x = index % m_shape[1];
        index /= m_shape[1];
        long y = index % m_shape[0];

        return (x - point.x()) * (x - point.x()) + (y - point.y()) * (y - point.y());
    }
};

template <typename T>
struct FilterData
{
public:
    FilterData(array<T> data) : m_good(data.shape(), 0), m_data(std::move(data)) {}

    template <typename InputIt, typename = std::enable_if_t<
        std::is_base_of_v<typename Peaks::iterator, InputIt>
    >>
    void filter(InputIt first, InputIt last, std::vector<InputIt> & output, const Structure & srt, T vmin, size_t npts)
    {
        auto func = [this, vmin](long index)
        {
            return m_data[index] > vmin;
        };
        auto stop = [npts](const Region & support)
        {
            return support.size() < npts;
        };

        for (auto iter = first; iter != last; ++iter)
        {
            if (!m_good[*iter])
            {
                Region support;
                support.insert(*iter);
                support.dilate(func, srt, stop, m_data.shape());

                if (support.size() < npts) output.push_back(iter);
                else support.mask(m_good, u_char(1));
            }
        }
    }

protected:
    vector_array<unsigned char> m_good;
    array<T> m_data;
};

// Streak class

template <typename T>
class Streak
{
public:
    Streak(const Region & points, long seed, const array<T> & data) : m_pxls(points, data)
    {
        auto [pt0, pt1] = line().to_pair();
        m_ctrs.emplace_back(make_point<2>(seed, data.shape()));
        m_ends.emplace_back(std::move(pt0));
        m_ends.emplace_back(std::move(pt1));
        update_minmax();
    }

    Streak(Region && points, long seed, const array<T> & data) : m_pxls(std::move(points), data)
    {
        auto [pt0, pt1] = line().to_pair();
        m_ctrs.emplace_back(make_point<2>(seed, data.shape()));
        m_ends.emplace_back(std::move(pt0));
        m_ends.emplace_back(std::move(pt1));
        update_minmax();
    }

    void merge(Streak & streak, const array<T> & data)
    {
        m_pxls.merge(streak.m_pxls, data);
        m_ctrs.insert(m_ctrs.end(), streak.m_ctrs.begin(), streak.m_ctrs.end());
        m_ends.insert(m_ends.end(), streak.m_ends.begin(), streak.m_ends.end());
        update_minmax();
    }

    void merge(Streak && streak, const array<T> & data)
    {
        merge(streak, data);
    }

    const Point<long> & center() const {return m_ctrs.front();}
    long id() const {return center().x() + center().y();}

    Line<long> central_line() const {return Line<long>{m_ctrs[m_min], m_ctrs[m_max]};}
    Line<T> line() const {return m_pxls.line();}

    const Region & region() const {return m_pxls.region();}
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

class StreakWrapper
{
public:
    template <typename T>
    StreakWrapper(Streak<T> && streak) : m_streak(std::move(streak)) {}

    template <typename T>
    StreakWrapper(const Streak<T> & streak) : m_streak(streak) {}

    // Accessors that work uniformly across both types
    Point<long> center() const
    {
        return std::visit([](const auto & s) { return s.center(); }, m_streak);
    }

    long id() const
    {
        return std::visit([](const auto & s) { return s.id(); }, m_streak);
    }

    size_t size() const
    {
        return std::visit([](const auto & s) { return s.region().size(); }, m_streak);
    }

    Line<long> central_line() const
    {
        return std::visit([](const auto & s) { return s.central_line(); }, m_streak);
    }

    std::vector<std::array<long, 2>> centers() const
    {
        return std::visit([](const auto & s)
        {
            std::vector<std::array<long, 2>> result;
            for (const auto & pt : s.centers()) result.emplace_back(pt.to_array());
            return result;
        }, m_streak);
    }

    std::vector<std::array<double, 2>> ends() const
    {
        return std::visit([](const auto & s)
        {
            std::vector<std::array<double, 2>> result;
            for (const auto & pt : s.ends()) result.emplace_back(std::array<double, 2>{pt.x(), pt.y()});
            return result;
        }, m_streak);
    }

    const Region & region() const
    {
        return *std::visit([](const auto & s) { return &s.region(); }, m_streak);
    }

    template <typename T>
    Line<T> line_as() const
    {
        return std::visit([](const auto & s) { return Line<T>{s.line()}; }, m_streak);
    }

    template <typename T>
    const Streak<T> & as() const
    {
        return *std::get_if<Streak<T>>(&m_streak);
    }

private:
    std::variant<Streak<float>, Streak<double>> m_streak;
};

class StreakMask
{
protected:
    using ShapeContainer = AnyContainer<size_t>;

public:
    enum flags
    {
        bad = -1,
        not_used = 0
    };

    StreakMask() = default;
    StreakMask(ShapeContainer sh) : m_flags(std::move(sh), not_used) {}

    void remove(const Region & region, long id)
    {
        for (auto index : region) m_flags[index] -= id;
    }

    void add(const Region & region, long id)
    {
        for (auto index : region) m_flags[index] += id;
    }

    bool is_free(long index) const
    {
        return m_flags[index] == not_used;
    }

    template <typename T>
    T p_value(const Region & region, const Line<T> & line, const array<T> & data, T xtol, T vmin, T p, int flag) const
    {
        size_t n = 0, k = 0;
        for (auto index : region)
        {
            auto point = make_point<2>(index, m_flags.shape());
            if (m_flags[index] == flag)
            {
                if (data[index] > T() && line.distance(point) < xtol)
                {
                    n++;
                    if (data[index] > vmin) k++;
                }
            }
        }

        return detail::logbinom(n, k, p);
    }

    const vector_array<int> & flags() const {return m_flags;}
    int flags(size_t index) const {return m_flags[index];}

    void clear()
    {
        std::fill(m_flags.begin(), m_flags.end(), not_used);
    }

protected:
    vector_array<int> m_flags;
};

template <typename T>
struct StreakFinderInput
{
public:
    static constexpr size_t MAX_NUM_ITER = 1000;

    using const_iterator = std::vector<Point<long>>::const_iterator;
    using iterator = std::vector<Point<long>>::iterator;

    StreakFinderInput() = default;

    StreakFinderInput(Peaks peaks, array<T> data, const Structure & structure, unsigned lookahead, unsigned nfa) :
        m_peaks(std::move(peaks)), m_data(std::move(data)), m_structure(structure), m_lookahead(lookahead), m_nfa(nfa) {}

    const array<T> & data() const {return m_data;}
    const Peaks & peaks() const {return m_peaks;}

    Streak<T> get_streak(long seed) const
    {
        return Streak<T>{Region{seed, m_structure, m_data.shape()}, seed, m_data};
    }

    Streak<T> get_streak(long seed, T xtol) const
    {
        auto streak = get_streak(seed);

        Line<long> old_line = Line<long>{}, line = streak.central_line();
        size_t n_iter = 0;
        while (old_line != line && n_iter++ < MAX_NUM_ITER)
        {
            old_line = line;

            streak = grow_streak<false>(std::move(streak), old_line.pt0, xtol);
            streak = grow_streak<true>(std::move(streak), old_line.pt1, xtol);
            line = streak.central_line();
        }
        if (n_iter == MAX_NUM_ITER)
        {
            auto err_txt = "get_streak: Streak growth did not converge for seed index " +
                           std::to_string(seed) + " after " + std::to_string(MAX_NUM_ITER) + " iterations.";
            throw std::runtime_error(err_txt);
        }

        return streak;
    }

    std::vector<long> seeds(size_t index, size_t size) const
    {
        std::vector<long> points;
        auto first = std::next(m_peaks.begin(), index);
        auto last = std::next(first, size);
        for (auto iter = first; iter != last; iter++) points.push_back(*iter);

        auto compare = [this](long a, long b)
        {
            return m_data[a] > m_data[b];
        };
        std::sort(points.begin(), points.end(), compare);
        return points;
    }

protected:
    Peaks m_peaks;              // sparse 2D peaks
    array<T> m_data;            // 2D data array
    Structure m_structure;      // connectivity structure
    unsigned m_lookahead = 0;   // maximum number of lookahead steps
    unsigned m_nfa = 0;         // maximum number of unaligned points allowed

    template <bool IsForward>
    Point<long> find_next_step(const Streak<T> & streak, const Point<long> & point, int n_steps) const
    {
        auto line = streak.line();
        if (line.pt0 == line.pt1) return point; // zero-length line

        auto tau = line.tangent();
        tau /= amplitude(tau);

        Point<long> pt2;
        if constexpr (IsForward) pt2 = (point + tau * n_steps).round();
        else pt2 = (point - tau * n_steps).round();
        return pt2;
    }

    template <bool IsForward>
    Streak<T> grow_streak(Streak<T> && streak, Point<long> point, T xtol) const
    {
        unsigned tries = 0;
        while (tries <= m_lookahead)
        {
            Point<long> candidate = find_next_step<IsForward>(streak, point, m_structure.connectivity);
            auto candidate_idx = m_data.index_at(candidate.coordinate());

            // Find the closest peak in structure vicinity, if there is none, try the next point
            auto iter = m_peaks.find_range(candidate, m_structure.connectivity);
            if (iter != m_peaks.end())
            {
                auto peak = make_point<2>(*iter, m_data.shape());
                if (peak != point)
                {
                    candidate = peak;
                    candidate_idx = *iter;
                }
            }

            // Try to add the point to the streak
            auto new_streak = streak;
            new_streak.merge(get_streak(candidate_idx), m_data);

            // Check if the new streak is valid by counting the number of unaligned points
            auto new_line = new_streak.line();
            size_t num_unaligned = 0;
            for (const auto & end : new_streak.ends())
            {
                if (new_line.distance(end) >= xtol)
                {
                    if (++num_unaligned > m_nfa) break;
                }
            }

            // If the new streak is valid return it, otherwise try the next point
            if (num_unaligned <= m_nfa) return new_streak;
            else
            {
                tries++;
                point = candidate;
            }
        }

        return streak;
    }
};

}

#endif
