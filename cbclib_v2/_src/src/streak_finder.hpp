#ifndef STREAK_FINDER_
#define STREAK_FINDER_
#include "label.hpp"

namespace cbclib {

// Labeling index and peak type
using lint_t = long;

// Index and bin type
using bint_t = size_t;

namespace detail {

template <typename T, typename I, typename Func>
unsigned maximality(I index, const Structure & str, const array<T> & array, Func && func)
{
    T val = array[index];
    unsigned result = 0;
    for (const auto & shift : str.shifts())
    {
        auto neighbour_idx = cbclib::detail::shift_index(index, shift, array.shape());

        if (neighbour_idx >= 0)
        {
            if (array[neighbour_idx] <= val)
            {
                result++;
            }
            else
            {
                std::forward<Func>(func)(neighbour_idx);
            }
        }
    }

    return result;
}

template <typename T, typename I>
unsigned maximality(I index, const Structure & str, const array<T> & array)
{
    T val = array[index];
    unsigned result = 0;
    for (const auto & shift : str.shifts())
    {
        auto neighbour_idx = cbclib::detail::shift_index(index, shift, array.shape());
        if (neighbour_idx >= 0 && array[neighbour_idx] <= val) result++;
    }

    return result;
}

} // namespace detail

struct PointIndex
{
    bint_t index, bin;
};

// Indexing routines for converting between point indices and bin indices
class PeaksIndexer
{
public:
    template <typename Container, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
    PeaksIndexer(const Container & shape, size_t radius) : m_radius(radius)
    {
        auto n_frames = std::reduce(shape.begin(), std::prev(shape.end(), 2), size_t(1), std::multiplies<>());

        m_shape = std::array<size_t, 3>{n_frames, size_t(shape[shape.size() - 2]), size_t(shape[shape.size() - 1])};
        m_binned_shape = std::array<size_t, 3>{n_frames, (m_shape[1] + radius - 1) / radius, (m_shape[2] + radius - 1) / radius};
    }

    size_t radius() const {return m_radius;}

    const std::array<size_t, 3> & shape() const {return m_shape;}
    size_t shape(size_t dim) const {return m_shape[dim];}

    size_t size() const {return m_shape[0] * m_shape[1] * m_shape[2];}

    const std::array<size_t, 3> & binned_shape() const {return m_binned_shape;}
    size_t binned_shape(size_t dim) const {return m_binned_shape[dim];}

    size_t n_bins() const {return m_binned_shape[0] * m_binned_shape[1] * m_binned_shape[2];}

    bint_t index_at(bint_t frame, bint_t y, bint_t x) const
    {
        return frame * m_shape[1] * m_shape[2] + y * m_shape[2] + x;
    }

    bint_t index_at(const PointND<bint_t, 3> & point) const
    {
        return index_at(point.z(), point.y(), point.x());
    }

    bint_t bin_at(bint_t frame, bint_t y, bint_t x) const
    {
        return frame * m_binned_shape[1] * m_binned_shape[2] + (y / m_radius) * m_binned_shape[2] + (x / m_radius);
    }

    bint_t bin_at(const PointND<bint_t, 3> & point) const
    {
        return bin_at(point.z(), point.y(), point.x());
    }

    bint_t bin_at(bint_t index) const
    {
        auto key = point_at(index);
        return bin_at(key.z(), key.y(), key.x());
    }

    PointIndex point_id_at(const PointND<bint_t, 3> & point) const
    {
        return PointIndex{index_at(point), bin_at(point)};
    }

    PointND<bint_t, 3> point_at(bint_t index) const
    {
        return make_point<3>(index, m_shape);
    }

protected:
    size_t m_radius;                      // Grid sampling interval, in pixels
    std::array<size_t, 3> m_shape;        // Shape of the original array
    std::array<size_t, 3> m_binned_shape; // Shape of the binned grid
};

class Peaks
{
public:
    Peaks(array<lint_t> labels, array<lint_t> peaks, size_t & n_labels)
        : m_labels(std::move(labels)), m_peaks(std::move(peaks)), m_nlabels(n_labels) {}

    // Unsafe access to peak index by bin index, assumes the bin is a peak (label > 0)
    lint_t operator[](bint_t bin_idx) const { return m_peaks[m_labels[bin_idx] - 1];}
    lint_t & operator[](bint_t bin_idx) { return m_peaks[m_labels[bin_idx] - 1]; }

    template <typename T>
    lint_t insert(const PointIndex & candidate, const array<T> & data)
    {
        if (m_nlabels < n_good() && is_good(candidate.bin))
        {
            m_labels[candidate.bin] = m_nlabels + 1;
            m_peaks[m_nlabels] = candidate.index;
            return m_nlabels++;
        }
        if (is_peak(candidate.bin) && data[candidate.index] > data[m_peaks[m_labels[candidate.bin] - 1]])
        {
            m_peaks[m_labels[candidate.bin] - 1] = candidate.index;
            return m_labels[candidate.bin] - 1;
        }
        return -1;
    }

    lint_t index(bint_t bin_idx) const
    {
        return m_labels[bin_idx] - 1;
    }

    bool is_peak(bint_t bin_idx) const {return m_labels[bin_idx] > 0;}
    bool is_good(bint_t bin_idx) const {return m_labels[bin_idx] == 0;}
    bool is_bad(bint_t bin_idx) const {return m_labels[bin_idx] < 0;}

    size_t n_good() const {return m_peaks.size();}
    size_t n_labels() const {return m_nlabels;}

protected:
    array<long> m_labels;               // Array of peak labels, shape (N_frames, NY_bins, NX_bins)
    array<long> m_peaks;                // Flat array of peak indices, (N_good,) size (labels > 0)
    size_t & m_nlabels;                 // Reference to the global number of labels currently in use
};

template <typename T, size_t N = 2>
class Linelets
{
public:
    Linelets(array<T> linelets) : m_linelets(std::move(linelets)), n_lines(m_linelets.size() / (2 * N)) {}

    LineND<T, N> line(lint_t line_id) const
    {
        if (line_id >= 0 && line_id < n_lines)
        {
            return LineND<T, N>{to_point<N>(m_linelets, 2 * N * line_id), to_point<N>(m_linelets, 2 * N * line_id + N)};
        }
        return LineND<T, N>{};
    }

    PointND<T, N> point(lint_t point_id) const
    {
        if (point_id >= 0 && point_id < 2 * n_lines)
        {
            return to_point<N>(m_linelets, N * point_id);
        }
        return PointND<T, N>{};
    }

    bool insert(lint_t index, const LineND<T, N> & line)
    {
        if (index >= 0 && index < n_lines)
        {
            auto line_points = line.to_array();
            for (size_t j = 0; j < 2 * N; j++) m_linelets[2 * N * index + j] = line_points[j];
            return true;
        }
        return false;
    }

protected:
    array<T> m_linelets;    // linelet array, shape (N_linelets, 2 * N), where linelet = {x0, y0, ..., x1, y1, ...}
    lint_t n_lines;
};

template <typename T, size_t N = 2>
class LineFitter
{
public:
    LineFitter(array<T> data, Structure structure, T vmin, size_t radius) :
        m_data(std::move(data)), m_structure(std::move(structure)), m_indexer(m_data.shape(), radius), m_vmin(vmin)
    {
        for (size_t i = 0; i < 3; i++) m_ubound[i] = m_indexer.shape(2 - i) - 1;
    }

    const PeaksIndexer & indexer() const { return m_indexer; }

    bool neighbour_behind(bint_t bin_idx, const LineND<T, N> & line, const Peaks & peaks, PointIndex & neighbour) const
    {
        return locate_neighbour<false>(neighbour, line, bin_idx, peaks);
    }

    bool neighbour_ahead(bint_t bin_idx, const LineND<T, N> & line, const Peaks & peaks, PointIndex & neighbour) const
    {
        return locate_neighbour<true>(neighbour, line, bin_idx, peaks);
    }

    bool peak_behind(bint_t bin_idx, const LineND<T, N> & line, const Peaks & peaks, PointIndex & neighbour) const
    {
        return locate_peak<false>(neighbour, line, bin_idx, peaks);
    }

    bool peak_ahead(bint_t bin_idx, const LineND<T, N> & line, const Peaks & peaks, PointIndex & neighbour) const
    {
        return locate_peak<true>(neighbour, line, bin_idx, peaks);
    }

    LineND<T, N> fit_linelet(bint_t index) const
    {
        auto point = m_indexer.point_at(index);

        PointND<T, N> origin;
        for (size_t i = 0; i < N; i++) origin[i] = point[i];
        MomentsND<T, N> moments (origin);

        for (const auto & shift : m_structure)
        {
            auto neighbour_idx = cbclib::detail::shift_index(index, shift, m_data.shape());
            if (neighbour_idx >= 0) moments.insert(neighbour_idx, m_data);
        }

        return moments.central().line();
    }

protected:
    array<T> m_data;        // data array
    Structure m_structure;  // structure element for finding linelets
    PeaksIndexer m_indexer; // indexer for accessing peaks in the original array
    T m_vmin;               // minimum data value
    PointND<lint_t, 3> m_lbound {}, m_ubound {};

    // Intersect ray start + tau * t (t > 0) with the nearest x/y bin boundary.
    // Because we round floating coordinates to integer pixels, bin boundaries are
    // centered between integer pixels at k * radius - 0.5.
    T intersection(const PointND<T, N> & tau, const PointND<bint_t, 3> & start) const
    {
        T t_min = std::numeric_limits<T>::infinity();

        for (size_t i = 0; i < N; i++)
        {
            if (std::abs(tau[i]) <= std::numeric_limits<T>::epsilon()) continue;

            auto bin = start[i] / m_indexer.radius();
            T boundary;
            if (tau[i] > T())
            {
                // Move to the next boundary in the positive direction.
                boundary = (bin + 1) * m_indexer.radius() - 0.5;
            }
            else
            {
                // Move to the previous boundary in the negative direction.
                boundary = bin * m_indexer.radius() - 0.5;
            }

            T t = (boundary - start[i]) / tau[i];
            if (t > T() && t < t_min) t_min = t;
        }

        return std::min<T>(t_min, m_indexer.radius());
    }

    PointND<bint_t, 3> next_point(const PointND<T, N> & tangent, const PointND<bint_t, 3> & start) const
    {
        auto tau = tangent / amplitude(tangent);

        auto t = intersection(tau, start);
        PointND<T, 3> next = start;
        for (size_t i = 0; i < N; i++) next[i] += tau[i] * (t + 0.5);

        // We need to cast to signed label index to avoid stack overflow
        PointND<lint_t, 3> rounded = next.round();
        return rounded.clamp(m_lbound, m_ubound);
    }

    template <bool IsForward>
    PointIndex next_point_id(const LineND<T, N> & line, bint_t start_bin, const Peaks & peaks) const
    {
        auto peak_idx = peaks[start_bin];
        auto start = m_indexer.point_at(peak_idx);

        PointND<bint_t, 3> candidate;
        if constexpr (IsForward) candidate = next_point(line.tangent(), start);
        else candidate = next_point(-line.tangent(), start);

        return m_indexer.point_id_at(candidate);
    }

    template <bool PeaksOnly>
    bool search_neighbours(PointIndex & neighbour, bint_t index, bint_t start_bin, const Peaks & peaks) const
    {
        for (const auto & shift : m_structure.shifts())
        {
            auto neighbour_idx = cbclib::detail::shift_index(index, shift, m_indexer.shape());
            if (neighbour_idx < 0 || m_data[neighbour_idx] < m_vmin) continue;

            auto new_bin = m_indexer.bin_at(neighbour_idx);
            if (new_bin == start_bin) continue;

            if constexpr (PeaksOnly)
            {
                if (!peaks.is_peak(new_bin)) continue;
            }
            else
            {
                if (peaks.is_bad(new_bin)) continue;
            }

            neighbour = PointIndex{static_cast<bint_t>(neighbour_idx), new_bin};
            return true;
        }
        return false;
    }

    template <bool IsForward>
    bool locate_neighbour(PointIndex & neighbour, const LineND<T, N> & line, bint_t start_bin, const Peaks & peaks) const
    {
        if (!peaks.is_peak(start_bin)) return false;

        neighbour = next_point_id<IsForward>(line, start_bin, peaks);

        // Only if the neighbour was clipped from out of bounds to the start_bin
        if (neighbour.bin == start_bin) return false;

        if (peaks.is_bad(neighbour.bin) || m_data[neighbour.index] < m_vmin)
        {
            if (!search_neighbours<false>(neighbour, neighbour.index, start_bin, peaks)) return false;
        }
        return !peaks.is_bad(neighbour.bin) && m_data[neighbour.index] >= m_vmin;
    }

    template <bool IsForward>
    bool locate_peak(PointIndex & neighbour, const LineND<T, N> & line, bint_t start_bin, const Peaks & peaks) const
    {
        if (!peaks.is_peak(start_bin)) return false;

        neighbour = next_point_id<IsForward>(line, start_bin, peaks);

        // Only if the neighbour was clipped from out of bounds to the start_bin
        if (neighbour.bin == start_bin) return false;

        if (!peaks.is_peak(neighbour.bin))
        {
            if (!search_neighbours<true>(neighbour, neighbour.index, start_bin, peaks)) return false;
        }
        return peaks.is_peak(neighbour.bin);
    }
};

template <bool IsConst>
struct StreakTraits {};

template <>
struct StreakTraits<false>
{
    using index_type = bint_t;
    using index_reference = bint_t &;
    using label_type = lint_t;
    using label_reference = lint_t &;
};

template <>
struct StreakTraits<true>
{
    using index_type = const bint_t;
    using index_reference = const bint_t &;
    using label_type = const lint_t;
    using label_reference = const lint_t &;
};

struct EndValues
{
    lint_t start_id, end_id;
};

template <bool IsConst>
class StreakEnds
{
public:
    template <bool C = IsConst, typename = std::enable_if_t<!C>>
    StreakEnds<IsConst> & operator=(const EndValues & values)
    {
        m_start_id = values.start_id;
        m_end_id = values.end_id;
        return *this;
    }

    bool valid() const { return m_start_id >= 0 && m_end_id >= 0; }
    EndValues value() const { return EndValues{m_start_id, m_end_id}; }

    lint_t start_bin() const { return m_start_id / 2; }
    lint_t end_bin() const { return m_end_id / 2; }

    template <typename T, size_t N = 2>
    LineND<T, N> line(const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        return LineND<T, N>{linelets.point(2 * peaks.index(m_start_id / 2) + m_start_id % 2),
                            linelets.point(2 * peaks.index(m_end_id / 2) + m_end_id % 2)};
    }

    template <typename T, size_t N = 2>
    LineND<T, N> line(const Linelets<T, N> & linelets, const array<lint_t> & labels) const
    {
        return LineND<T, N>{linelets.point(2 * (labels[m_start_id / 2] - 1) + m_start_id % 2),
                            linelets.point(2 * (labels[m_end_id / 2] - 1) + m_end_id % 2)};
    }

    template <typename T, size_t N, bool C = IsConst, typename = std::enable_if_t<!C>>
    bool insert(bint_t bin_idx, const Linelets<T, N> & linelets, const Peaks & peaks)
    {
        if (!valid())
        {
            m_start_id = 2 * bin_idx;
            m_end_id = 2 * bin_idx + 1;
            return true;
        }

        auto old_line = line(linelets, peaks);
        auto tau = old_line.tangent();
        auto length = magnitude(tau);

        auto linelet = linelets.line(peaks.index(bin_idx));
        auto t0 = dot(tau, linelet.pt0 - old_line.pt0), t1 = dot(tau, linelet.pt1 - old_line.pt0);
        auto id0 = 2 * bin_idx, id1 = 2 * bin_idx + 1;

        if (t0 > t1)
        {
            std::swap(t0, t1);
            std::swap(id0, id1);
        }

        if (t0 < T())
        {
            m_start_id = id0;
            return true;
        }
        if (t1 > length)
        {
            m_end_id = id1;
            return true;
        }

        return false;
    }

protected:
    using reference = typename StreakTraits<IsConst>::label_reference;

    // Indices of line ends:
    // line_start = linelets[2 * N * m_indices[m_start_id / 2] + N * m_start_id % 2]
    // line_end   = linelets[2 * N * m_indices[m_end_id / 2]   + N * m_end_id % 2]
    reference m_start_id, m_end_id;

    StreakEnds(reference start_id, reference end_id) : m_start_id(start_id), m_end_id(end_id) {}

    friend class Streak;
};

class Streak
{
public:
    Streak() = default;
    Streak(std::vector<bint_t> && indices, lint_t start_id, lint_t end_id) : m_indices(std::move(indices)), m_start_id(start_id), m_end_id(end_id) {}
    Streak(const std::vector<bint_t> & indices, lint_t start_id, lint_t end_id) : m_indices(indices), m_start_id(start_id), m_end_id(end_id) {}

    StreakEnds<true> ends() const { return StreakEnds<true>{m_start_id, m_end_id}; }
    StreakEnds<false> ends() { return StreakEnds<false>{m_start_id, m_end_id}; }

    const std::vector<bint_t> & indices() const {return m_indices;}
    std::vector<bint_t> & indices() {return m_indices; }

    template <typename T, size_t N>
    bool insert(bint_t bin_idx, const Linelets<T, N> & linelets, const Peaks & peaks)
    {
        auto is_inserted = ends().insert(bin_idx, linelets, peaks);
        if (is_inserted) m_indices.push_back(bin_idx);
        return is_inserted;
    }

protected:
    std::vector<bint_t> m_indices;            // Indices of peaks in the original array that belong to this streak

    // Indices of line ends:
    // line_start = linelets[2 * N * m_indices[m_start_id / 2] + N * m_start_id % 2]
    // line_end   = linelets[2 * N * m_indices[m_end_id / 2]   + N * m_end_id % 2]
    lint_t m_start_id = -1, m_end_id = -1;
};

template <typename T, size_t N = 2>
class StreakFinder : public LineFitter<T, N>
{
public:
    using LineFitter<T, N>::peak_ahead;
    using LineFitter<T, N>::peak_behind;

    StreakFinder(array<T> data, Structure structure, T vmin, T xtol, unsigned nfa, size_t radius) :
        LineFitter<T, N>(std::move(data), structure, vmin, radius), m_xtol(xtol), m_nfa(nfa) {}

    bool detect(Streak & streak, const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        if (!streak.indices().size()) return false;

        bool grows_ahead = true, grows_behind = true;
        while (grows_ahead || grows_behind)
        {
            if (grows_ahead) grows_ahead = grow_ahead(streak, linelets, peaks);
            if (grows_behind) grows_behind = grow_behind(streak, linelets, peaks);
        }
        return true;
    }

protected:
    using LineFitter<T, N>::m_indexer;

    T m_xtol;               // maximum distance from the linelet to be considered part of the streak
    unsigned m_nfa = 0;     // maximum number of unaligned points allowed

    bool is_aligned(const PointIndex & candidate, const Streak & streak, const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        auto total_line = streak.ends().line(linelets, peaks);

        unsigned num_unaligned = 0;
        auto candidate_line = linelets.line(peaks.index(candidate.bin));
        if (total_line.distance(candidate_line.pt0) > m_xtol) num_unaligned++;
        if (total_line.distance(candidate_line.pt1) > m_xtol) num_unaligned++;

        for (auto bin_idx : streak.indices())
        {
            auto linelet = linelets.line(peaks.index(bin_idx));
            if (total_line.distance(linelet.pt0) > m_xtol) num_unaligned++;
            if (total_line.distance(linelet.pt1) > m_xtol) num_unaligned++;
            if (num_unaligned > m_nfa) break;
        }

        return num_unaligned <= m_nfa;
    }

    bool insert_candidate(Streak & streak, const PointIndex & candidate, const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        auto old_ends = streak.ends().value();
        auto is_inserted = streak.ends().insert(candidate.bin, linelets, peaks);

        if (!is_inserted) return false;

        if (is_aligned(candidate, streak, linelets, peaks))
        {
            streak.indices().push_back(candidate.bin);
            return true;
        }

        // Revert the insertion if the candidate is not aligned with the streak
        streak.ends() = old_ends;

        return false;
    }

    bool grow_ahead(Streak & streak, const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        if (streak.indices().size() == 0) return false;

        auto old_bin = streak.ends().end_bin();

        PointIndex candidate;
        auto is_found = peak_ahead(old_bin, streak.ends().line(linelets, peaks), peaks, candidate);

        if (!is_found) return false;

        // Check if the candidate point is aligned with the streak
        return insert_candidate(streak, candidate, linelets, peaks);
    }

    bool grow_behind(Streak & streak, const Linelets<T, N> & linelets, const Peaks & peaks) const
    {
        if (streak.indices().size() == 0) return false;

        auto old_bin = streak.ends().start_bin();

        PointIndex candidate;
        auto is_found = peak_behind(old_bin, streak.ends().line(linelets, peaks), peaks, candidate);

        if (!is_found) return false;

        // Check if the candidate point is aligned with the streak
        return insert_candidate(streak, candidate, linelets, peaks);
    }
};

} // namespace cbclib

#endif // STREAK_FINDER_
