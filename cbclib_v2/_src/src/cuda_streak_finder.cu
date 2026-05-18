#include <cub/cub.cuh>
#include "pybind11/warnings.h"
#include "label.hpp"
#include "cupy_array.hpp"

namespace cbclib::cuda {

// Labeling index type
using lint_t = int;

template <typename T>
HOST_DEVICE inline void swap(T & a, T & b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

struct PeakLabels
{
    array_t<lint_t> labels;
    py::ssize_t n_seeds, n_labels, n_good, radius;
};

struct PointIndex
{
    csize_t index, bin;
};

DeviceVector<lint_t> all_shifts(const Structure & structure)
{
    std::vector<lint_t> shifts;
    for (auto shift : structure)
    {
        shifts.insert(shifts.end(), shift.begin(), shift.end());
    }
    return DeviceVector<lint_t>::from_host(shifts.data(), shifts.size());
}

DeviceVector<lint_t> all_nonzero_shifts(const Structure & structure)
{
    std::vector<lint_t> shifts;
    for (auto shift : structure)
    {
        lint_t offset = 0;
        for (csize_t n = 0; n < structure.rank(); n++)
        {
            offset = offset * structure.shape(n) + shift[n];
        }
        if (offset != 0) shifts.insert(shifts.end(), shift.begin(), shift.end());
    }
    return DeviceVector<lint_t>::from_host(shifts.data(), shifts.size());
}

// Indexing routines for converting between point indices and bin indices
class PeaksIndexer
{
public:
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    HOST_DEVICE PeaksIndexer(const I * shape, size_t ndim, size_t radius) : m_radius(radius)
    {
        csize_t n_frames = 1;
        for (csize_t n = 0; n < ndim - 2; n++) n_frames *= shape[n];

        m_shape = {n_frames, shape[ndim - 2], shape[ndim - 1]};
        m_binned_shape = {n_frames, (shape[ndim - 2] + radius - 1) / radius, (shape[ndim - 1] + radius - 1) / radius};
    }

    HOST_DEVICE csize_t radius() const { return m_radius; }

    HOST_DEVICE const PointND<csize_t, 3> & shape() const { return m_shape; }
    HOST_DEVICE csize_t shape(csize_t n) const { return m_shape[n]; }

    HOST_DEVICE csize_t size() const { return m_shape[0] * m_shape[1] * m_shape[2]; }

    HOST_DEVICE const PointND<csize_t, 3> & binned_shape() const { return m_binned_shape; }
    HOST_DEVICE csize_t binned_shape(csize_t n) const { return m_binned_shape[n]; }

    HOST_DEVICE csize_t n_bins() const { return m_binned_shape[0] * m_binned_shape[1] * m_binned_shape[2]; }

    HOST_DEVICE csize_t index_at(csize_t frame, csize_t y, csize_t x) const
    {
        return (frame * m_shape[1] + y) * m_shape[2] + x;
    }

    HOST_DEVICE csize_t index_at(const PointND<csize_t, 3> & point) const
    {
        return index_at(point.z(), point.y(), point.x());
    }

    HOST_DEVICE csize_t bin_at(csize_t frame, csize_t y, csize_t x) const
    {
        return (frame * m_binned_shape[1] + y / m_radius) * m_binned_shape[2] + x / m_radius;
    }

    HOST_DEVICE csize_t bin_at(const PointND<csize_t, 3> & point) const
    {
        return bin_at(point.z(), point.y(), point.x());
    }

    HOST_DEVICE csize_t bin_at(long index) const
    {
        auto key = point_at(index);
        return bin_at(key.z(), key.y(), key.x());
    }

    HOST_DEVICE PointND<csize_t, 3> point_at(csize_t index) const
    {
        return make_point<3>(index, m_shape);
    }

    HOST_DEVICE PointIndex point_id_at(const PointND<csize_t, 3> & point) const
    {
        return PointIndex{index_at(point), bin_at(point)};
    }

protected:
    csize_t m_radius;                      // Grid sampling interval, in pixels
    PointND<csize_t, 3> m_shape;           // Shape of the original array
    PointND<csize_t, 3> m_binned_shape;    // Shape of the binned grid
};

class Counter
{
public:
    __device__ Counter(csize_t * counter) : m_counter(counter) {}
    __device__ Counter(DeviceRange<csize_t> counter) : m_counter(counter.data()) {}

    // Atomically increments the counter and returns the old value
    __device__ csize_t increment()
    {
        return atomicAdd(m_counter, 1);
    }

    __device__ csize_t value() const
    {
        return *m_counter;
    }

protected:
    csize_t * m_counter; // Pointer to a single counter in global memory
};

template <csize_t N>
class Peaks
{
public:
    HOST_DEVICE Peaks(ArrayViewND<lint_t, N> labels, ArrayViewND<lint_t, 1> peaks) : m_labels(labels), m_peaks(peaks) {}

    HOST_DEVICE lint_t operator[](csize_t bin_idx) const { return m_peaks[m_labels[bin_idx] - 1]; }

    HOST_DEVICE lint_t index(csize_t bin_idx) const { return m_labels[bin_idx] - 1; }

    HOST_DEVICE bool is_peak(csize_t bin_idx) const { return m_labels[bin_idx] > 0; }
    HOST_DEVICE bool is_good(csize_t bin_idx) const { return m_labels[bin_idx] == 0; }
    HOST_DEVICE bool is_bad(csize_t bin_idx) const { return m_labels[bin_idx] < 0; }

    HOST_DEVICE csize_t n_good() const { return m_peaks.size(); }

    template <typename T>
    __device__ lint_t insert(const PointIndex & candidate, const ArrayViewND<T, N> & data, Counter & n_labels)
    {
        if (is_good(candidate.bin))
        {
            // Atomically claim the bin by transitioning 0 (good) → -2 (in-progress).
            // Without this, two threads racing on the same good bin both pass the
            // is_good check, both call increment(), but only one m_labels write survives —
            // leaving a counter slot permanently claimed with no label pointing to it.
            lint_t old = atomicCAS(&m_labels[candidate.bin], lint_t(0), lint_t(-2));
            if (old != 0) return -1;

            auto new_index = n_labels.increment();
            m_peaks[new_index] = candidate.index;
            m_labels[candidate.bin] = new_index + 1;
            return new_index;
        }
        if (is_peak(candidate.bin) && data[candidate.index] > data[m_peaks[m_labels[candidate.bin] - 1]])
        {
            m_peaks[m_labels[candidate.bin] - 1] = candidate.index;
            return m_labels[candidate.bin] - 1;
        }
        return -1;
    }

protected:
    ArrayViewND<lint_t, N> m_labels;        // Bin labels array
    ArrayViewND<lint_t, 1> m_peaks;         // Bin peaks array
};

enum class Direction : unsigned
{
    None = 0u,
    Forward = 1u,
    Backward = 2u
};

HOST_DEVICE inline constexpr Direction operator|(Direction a, Direction b)
{
    return static_cast<Direction>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

HOST_DEVICE inline constexpr Direction operator&(Direction a, Direction b)
{
    return static_cast<Direction>(static_cast<unsigned>(a) & static_cast<unsigned>(b));
}

HOST_DEVICE inline constexpr bool is_none(Direction dir)
{
    return dir == Direction::None;
}

__device__ inline void add_flag(Direction * dst, Direction flag)
{
    auto ptr = reinterpret_cast<unsigned *>(dst);
    atomicOr(ptr, static_cast<unsigned>(flag));
}

class Candidates
{
public:
    class CandidatesView
    {
    public:
        template <typename T, csize_t N>
        __device__ csize_t insert(const PointIndex & candidate, Direction dir, const ArrayViewND<T, N> & data)
        {
            auto dir_ptr = reinterpret_cast<unsigned *>(m_directions.data(candidate.bin));
            atomicOr(dir_ptr, static_cast<unsigned>(dir));

            auto dst = m_indices.data(candidate.bin);
            csize_t current = atomicAdd(dst, csize_t(0)); // atomic read

            while (true)
            {
                if (current != m_null_index && data[current] >= data[candidate.index]) return current;

                csize_t observed = atomicCAS(dst, current, candidate.index);
                if (observed == current) return candidate.index;

                current = observed;
            }
        }

        HOST_DEVICE bool contains(csize_t bin) const { return m_indices[bin] != m_null_index; }

        HOST_DEVICE csize_t index(csize_t bin) const { return m_indices[bin]; }
        HOST_DEVICE Direction direction(csize_t bin) const { return m_directions[bin]; }

        HOST_DEVICE void clear(csize_t bin)
        {
            m_indices[bin] = m_null_index;
            m_directions[bin] = Direction::None;
        }

    protected:
        DeviceRange<csize_t> m_indices;
        DeviceRange<Direction> m_directions;
        csize_t m_null_index;

        HOST_DEVICE CandidatesView(DeviceRange<csize_t> indices, DeviceRange<Direction> directions, csize_t null_index) :
            m_indices(indices), m_directions(directions), m_null_index(null_index) {}

        friend class Candidates;
    };

    Candidates(csize_t n_bins, csize_t null_index) : m_indices(n_bins, null_index), m_directions(n_bins, Direction::None), m_null_index(null_index) {}

    CandidatesView view()
    {
        return CandidatesView{m_indices.view(), m_directions.view(), m_null_index};
    }

protected:
    DeviceVector<csize_t> m_indices;
    DeviceVector<Direction> m_directions;
    csize_t m_null_index;
};

template <typename T>
class Linelets
{
public:
    HOST_DEVICE Linelets(ArrayViewND<T, 2> linelets) : m_linelets(linelets) {}

    HOST_DEVICE LineND<T, 2> line(lint_t line_id) const
    {
        if (line_id >= 0 && line_id < m_linelets.shape(0))
        {
            return LineND<T, 2>{
                PointND<T, 2>{m_linelets[4 * line_id    ], m_linelets[4 * line_id + 1]},
                PointND<T, 2>{m_linelets[4 * line_id + 2], m_linelets[4 * line_id + 3]}
            };
        }
        return LineND<T, 2>{};
    }

    HOST_DEVICE PointND<T, 2> point(lint_t point_id) const
    {
        if (point_id >= 0 && point_id < 2 * m_linelets.shape(0))
        {
            return PointND<T, 2>{m_linelets[2 * point_id], m_linelets[2 * point_id + 1]};
        }
        return PointND<T, 2>{};
    }

    HOST_DEVICE bool insert(lint_t line_id, const LineND<T, 2> & line)
    {
        if (line_id < 0 || line_id >= m_linelets.shape(0)) return false;

        m_linelets[4 * line_id    ] = line.pt0[0];
        m_linelets[4 * line_id + 1] = line.pt0[1];
        m_linelets[4 * line_id + 2] = line.pt1[0];
        m_linelets[4 * line_id + 3] = line.pt1[1];
        return true;
    }

protected:
    ArrayViewND<T, 2> m_linelets;    // linelet array of shape (n_linelets, 4)
};

template <typename T, csize_t N>
class LineFitter
{
public:
    HOST_DEVICE LineFitter(ArrayViewND<T, N> data, DeviceRange<lint_t> shifts, csize_t n_shifts, T vmin, csize_t radius) :
        m_data(data), m_shifts(shifts), m_nshifts(n_shifts), m_indexer(data.shape(), data.ndim(), radius), m_vmin(vmin)
    {
        for (csize_t i = 0; i < 3; ++i) m_ubound[i] = m_indexer.shape(2 - i) - 1;
    }

    HOST_DEVICE const ArrayViewND<T, N> & data() const { return m_data; }
    HOST_DEVICE const PeaksIndexer & indexer() const { return m_indexer; }

    HOST_DEVICE bool neighbour_behind(csize_t bin_idx, const LineND<T, 2> & line, const Peaks<N> & peaks, PointIndex & neighbour) const
    {
        return locate_neighbour<false>(neighbour, line, bin_idx, peaks);
    }

    HOST_DEVICE bool neighbour_ahead(csize_t bin_idx, const LineND<T, 2> & line, const Peaks<N> & peaks, PointIndex & neighbour) const
    {
        return locate_neighbour<true>(neighbour, line, bin_idx, peaks);
    }

    HOST_DEVICE bool peak_behind(csize_t bin_idx, const LineND<T, 2> & line, const Peaks<N> & peaks, PointIndex & neighbour) const
    {
        return locate_peak<false>(neighbour, line, bin_idx, peaks);
    }

    HOST_DEVICE bool peak_ahead(csize_t bin_idx, const LineND<T, 2> & line, const Peaks<N> & peaks, PointIndex & neighbour) const
    {
        return locate_peak<true>(neighbour, line, bin_idx, peaks);
    }

    HOST_DEVICE LineND<T, 2> fit_linelet(csize_t index) const
    {
        auto point = m_indexer.point_at(index);

        PointND<T, 2> origin;
        for (size_t i = 0; i < 2; i++) origin[i] = point[i];

        T mu = m_data[index], mu_xy = T();
        PointND<T, 2> mu_x {}, mu_xx {};

        for (csize_t k = 0; k < m_nshifts; ++k)
        {
            auto neighbour_idx = detail::shift_index(index, m_data.shape(), N, m_shifts.data(k * N), N);
            if (neighbour_idx < 0) continue;

            auto neighbour = make_point<2, N>(neighbour_idx, m_data.shape());

            // We are clipping data values to non-negative
            T val = max(m_data[neighbour_idx], T());
            auto r = neighbour - origin;

            mu += val;
            mu_x += r * val;
            mu_xx += r * r * val;
            mu_xy += r[0] * r[1] * val;
        }

        if (mu <= T()) return LineND<T, 2>{origin, origin};

        auto M_X = mu_x / mu;
        auto M_XX = mu_xx / mu - M_X * M_X;
        auto M_XY = mu_xy / mu - M_X[0] * M_X[1];

        T theta = 0.5 * math_traits<T>::atan2(2 * M_XY, M_XX[0] - M_XX[1]);

        if (isnan(theta)) return LineND<T, 2>{origin, origin};

        PointND<T, 2> tau {math_traits<T>::cos(theta), math_traits<T>::sin(theta)};
        T delta = math_traits<T>::sqrt(4 * M_XY * M_XY + (M_XX[0] - M_XX[1]) * (M_XX[0] - M_XX[1]));
        T hw = math_traits<T>::sqrt(2 * numbers<T>::log2() * (M_XX[0] + M_XX[1] + delta));
        return LineND<T, 2>{M_X + origin + hw * tau, M_X + origin - hw * tau};
    }

protected:
    ArrayViewND<T, N> m_data;       // Data array
    DeviceRange<lint_t> m_shifts;   // Array of neighbor shifts for local maximum detection
    csize_t m_nshifts;              // Number of neighbor shifts
    PeaksIndexer m_indexer;         // Indexer for binned grid
    T m_vmin;                       // Minimum value for a pixel to be considered a peak
    PointND<lint_t, 3> m_lbound {}, m_ubound {};

    // Intersect ray start + tau * t (t > 0) with the nearest x/y bin boundary.
    // Because we round floating coordinates to integer pixels, bin boundaries are
    // centered between integer pixels at k * radius - 0.5.
    HOST_DEVICE T intersection(const PointND<T, 2> & tau, const PointND<csize_t, 3> & start) const
    {
        T t_min = numeric_limits<T>::infinity();

        for (size_t i = 0; i < 2; i++)
        {
            if (math_traits<T>::abs(tau[i]) <= numeric_limits<T>::epsilon()) continue;

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

        return math_traits<T>::min(t_min, m_indexer.radius());
    }

    HOST_DEVICE PointND<csize_t, 3> next_point(const PointND<T, 2> & tangent, const PointND<csize_t, 3> & start) const
    {
        auto tau = tangent / amplitude(tangent);
        auto t = intersection(tau, start);

        PointND<T, 3> next = start;
        for (size_t i = 0; i < 2; i++) next[i] += tau[i] * (t + 0.5);

        // We need to cast to signed label index to avoid stack overflow
        PointND<lint_t, 3> rounded = next.round();
        return rounded.clamp(m_lbound, m_ubound);
    }

    template <bool IsForward>
    HOST_DEVICE PointIndex next_point_id(const LineND<T, 2> & line, csize_t start_bin, const Peaks<N> & peaks) const
    {
        auto peak_idx = peaks[start_bin];
        auto start = m_indexer.point_at(peak_idx);

        PointND<csize_t, 3> candidate;
        if constexpr (IsForward) candidate = next_point(line.tangent(), start);
        else candidate = next_point(-line.tangent(), start);

        return m_indexer.point_id_at(candidate);
    }

    template <bool PeaksOnly>
    HOST_DEVICE bool search_neighbours(PointIndex & neighbour, csize_t index, csize_t start_bin, const Peaks<N> & peaks) const
    {
        for (csize_t k = 0; k < m_nshifts; ++k)
        {
            auto neighbour_idx = detail::shift_index(index, m_data.shape(), N, m_shifts.data(k * N), N);
            if (neighbour_idx < 0 || m_data[neighbour_idx] < m_vmin) continue;

            auto new_bin = m_indexer.bin_at(neighbour_idx);
            if (new_bin == start_bin) continue;

            if constexpr (PeaksOnly) if (!peaks.is_peak(new_bin)) continue;
            else if (peaks.is_bad(new_bin)) continue;

            neighbour = PointIndex{static_cast<csize_t>(neighbour_idx), new_bin};
            return true;
        }
        return false;
    }

    template <bool IsForward>
    HOST_DEVICE bool locate_neighbour(PointIndex & neighbour, const LineND<T, 2> & line, csize_t start_bin, const Peaks<N> & peaks) const
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
    HOST_DEVICE bool locate_peak(PointIndex & neighbour, const LineND<T, 2> & line, csize_t start_bin, const Peaks<N> & peaks) const
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

template <typename T, csize_t N>
__global__ void init_peaks_kernel(ArrayViewND<lint_t, N> peaks, ArrayViewND<lint_t, N> labels, PeaksIndexer indexer)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= indexer.n_bins()) return;

    // Find the coordinates of the 2D square neighbourhood
    // Points follow xyz convention and shape zyx
    auto start = make_point<3>(bin_idx, indexer.binned_shape());
    start[0] *= indexer.radius();
    start[1] *= indexer.radius();

    auto end = start;
    end[0] = min(end[0] + indexer.radius(), indexer.shape(2));
    end[1] = min(end[1] + indexer.radius(), indexer.shape(1));

    // Initialize the peak index as -1 if it's not inside the labelled region
    auto running = start;
    for (csize_t x = start[0]; x < end[0]; x++)
    {
        for (csize_t y = start[1]; y < end[1]; y++)
        {
            running[0] = x;
            running[1] = y;
            auto running_idx = indexer.index_at(running);

            if (labels[running_idx] > 0)
            {
                peaks[bin_idx] = indexer.size();
                return;
            }
        }
    }

    peaks[bin_idx] = -1;
}

template <typename T, csize_t N>
__global__ void detect_peaks_kernel(ArrayViewND<lint_t, N> peaks, ArrayViewND<T, N> data, DeviceRange<lint_t> shifts, csize_t n_shifts, PeaksIndexer indexer, T vmin)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= indexer.n_bins()) return;

    if (peaks[bin_idx] == -1) return; // Not a candidate for a peak

    // Find the coordinates of the 2D square neighbourhood
    // Points follow xyz convention and shape zyx convention
    auto start = make_point<3>(bin_idx, indexer.binned_shape());
    start[0] *= indexer.radius();
    start[1] *= indexer.radius();

    auto end = start;
    end[0] = min(end[0] + indexer.radius(), indexer.shape(2));
    end[1] = min(end[1] + indexer.radius(), indexer.shape(1));

    // Find local maximum in the 2D neighbourhood
    // A local maximum is a pixel whose value is higher than all its neighbors (defined by shifts)
    lint_t peak_idx = -1, running_idx = -1;
    T best_val = vmin;

    // Find the index of the (coord[0], ..., coord[N - 3], 0, 0) pixel
    auto running = start;
    running[0] = 0;
    running[1] = 0;
    auto zero_idx = indexer.index_at(running);

    for (csize_t x = start[0]; x < end[0]; ++x)
    {
        for (csize_t y = start[1]; y < end[1]; ++y)
        {
            running[0] = x;
            running[1] = y;
            running_idx = indexer.index_at(running);

            T val = data[running_idx];
            if (val < vmin) continue; // Skip if below threshold

            // Check if this is a local maximum by comparing to all neighbors
            for (csize_t k = 0; k < n_shifts; ++k)
            {
                auto neighbour_idx = detail::shift_index(running_idx, data.shape(), N, shifts.data((k + 1) * N - 2), 2);

                if (neighbour_idx < 0 || data[neighbour_idx + zero_idx] >= val)
                {
                    running_idx = -1; // Not a local maximum
                    break;
                }
            }

            // Keep track of the best local maximum found
            if (running_idx >= 0 && val > best_val)
            {
                best_val = val;
                peak_idx = running_idx;
            }
        }
    }

    if (peak_idx >= 0) peaks[bin_idx] = peak_idx; // Update only if a valid local maximum was found
}

template <typename T, csize_t N>
array_t<lint_t> detect_peaks_nd(array_t<lint_t> peaks, array_t<lint_t> labels, array_t<T> data, Structure structure, size_t radius, T vmin)
{
    auto peaks_view = cast_to_nd<lint_t, N>(peaks.view());
    auto labels_view = cast_to_nd<lint_t, N>(labels.view());
    auto data_view = cast_to_nd<T, N>(data.view());

    auto shifts = all_nonzero_shifts(structure);
    csize_t n_shifts = shifts.size() / structure.rank();

    PeaksIndexer indexer (data.shape(), data.ndim(), radius);

    // Initialize peaks
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (peaks_view.size() + block_size - 1) / block_size;
    init_peaks_kernel<T, N><<<n_blocks, block_size>>>(peaks_view, labels_view, indexer);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Detect peaks
    detect_peaks_kernel<T, N><<<n_blocks, block_size>>>(peaks_view, data_view, shifts.view(), n_shifts, indexer, vmin);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return peaks;
}

template <typename T>
array_t<lint_t> detect_peaks(array_t<lint_t> peaks, array_t<lint_t> labels, array_t<T> data, Structure structure, size_t radius, T vmin)
{
    if (structure.rank() != data.ndim())
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    check_equal("labels and data must have the same shape", labels.shape(), labels.shape() + labels.ndim(), data.shape(), data.shape() + data.ndim());
    if (peaks.ndim() != data.ndim())
    {
        throw std::invalid_argument("peaks array dimension (" + std::to_string(peaks.ndim()) + ") must match data array dimension (" + std::to_string(data.ndim()) + ")");
    }

    switch (peaks.ndim())
    {
        case 2: return detect_peaks_nd<T, 2>(peaks, labels, data, structure, radius, vmin);
        case 3: return detect_peaks_nd<T, 3>(peaks, labels, data, structure, radius, vmin);
        case 4: return detect_peaks_nd<T, 4>(peaks, labels, data, structure, radius, vmin);
        case 5: return detect_peaks_nd<T, 5>(peaks, labels, data, structure, radius, vmin);
        case 6: return detect_peaks_nd<T, 6>(peaks, labels, data, structure, radius, vmin);
        case 7: return detect_peaks_nd<T, 7>(peaks, labels, data, structure, radius, vmin);
        default: throw std::runtime_error("Unsupported number of dimensions: peaks.ndim = " + std::to_string(peaks.ndim()));
    }
}

template <typename T, csize_t N>
__global__ void init_linelets_kernel(LineFitter<T, N> fitter, Peaks<N> peaks, Linelets<T> linelets, DeviceRange<Direction> visited)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= fitter.indexer().n_bins()) return;

    if (!peaks.is_peak(bin_idx)) return;

    lint_t peak_idx = peaks[bin_idx];
    auto linelet = fitter.fit_linelet(peak_idx);
    auto is_inserted = linelets.insert(peaks.index(bin_idx), linelet);
    if (is_inserted) visited[bin_idx] = Direction::Forward | Direction::Backward;
}

template <typename T, csize_t N>
__global__ void fill_candidates_kernel(LineFitter<T, N> fitter, Peaks<N> peaks, Linelets<T> linelets,
                                       DeviceRange<Direction> visited, Candidates::CandidatesView candidates)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= fitter.indexer().n_bins()) return;

    if (is_none(visited[bin_idx])) return;

    PointIndex neighbour;
    auto line = linelets.line(peaks.index(bin_idx));

    if (line.pt0 == line.pt1) return;

    if (!is_none(visited[bin_idx] & Direction::Forward))
    {
        // Searches for a not bad bin with a data value above vmin in the forward direction
        if (fitter.neighbour_ahead(bin_idx, line, peaks, neighbour))
        {
            auto linelet = fitter.fit_linelet(neighbour.index);

            if (linelet.pt0 != linelet.pt1)
            {
                // We add a candidate with a direction to the frontier so that peaks stays immutable
                candidates.insert(neighbour, Direction::Forward, fitter.data());
            }
        }
    }

    if (!is_none(visited[bin_idx] & Direction::Backward))
    {
        // Searches for a not bad bin with a data value above vmin in the backward direction
        if (fitter.neighbour_behind(bin_idx, line, peaks, neighbour))
        {
            auto linelet = fitter.fit_linelet(neighbour.index);

            if (linelet.pt0 != linelet.pt1)
            {
                // We add a candidate with a direction to the frontier so that peaks stays immutable
                candidates.insert(neighbour, Direction::Backward, fitter.data());
            }
        }
    }

    visited[bin_idx] = Direction::None; // Mark the current bin as visited
}

template <typename T, csize_t N>
__global__ void commit_frontier_kernel(LineFitter<T, N> fitter, Peaks<N> peaks, Linelets<T> linelets,
                                       Candidates::CandidatesView candidates, DeviceRange<Direction> next_frontier,
                                       Counter n_labels)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= fitter.indexer().n_bins()) return;

    if (!candidates.contains(bin_idx)) return;

    PointIndex candidate {candidates.index(bin_idx), bin_idx};
    Direction dir = candidates.direction(bin_idx);

    auto linelet = fitter.fit_linelet(candidate.index);

    if (linelet.pt0 != linelet.pt1)
    {
        // Preserve your current semantics: mark for next iteration if linelet is valid.
        add_flag(&next_frontier[bin_idx], dir);

        // Same policy as Peaks::insert: create label if good, otherwise update if better.
        lint_t line_id = peaks.insert(candidate, fitter.data(), n_labels);
        bool is_inserted = linelets.insert(line_id, linelet);
    }

    // Clear slot for next gather pass.
    candidates.clear(bin_idx);
}

template <typename T, csize_t N>
array_t<T> line_fit_nd(array_t<T> out, PeakLabels & labels, array_t<lint_t> parray, array_t<T> data, Structure structure, T vmin)
{
    PeaksIndexer indexer (data.shape(), data.ndim(), labels.radius);
    auto shifts = all_nonzero_shifts(structure);
    csize_t n_shifts = shifts.size() / structure.rank();

    Peaks<N> peaks (cast_to_nd<lint_t, N>(labels.labels.view()), cast_to_nd<lint_t, 1>(parray.view()));
    Linelets<T> linelets (cast_to_nd<T, 2>(out.view()));
    LineFitter<T, N> fitter (cast_to_nd<T, N>(data.view()), shifts.view(), n_shifts, vmin, labels.radius);

    DeviceVector<Direction> visited (indexer.n_bins(), Direction::None);
    Candidates candidates (indexer.n_bins(), data.size()); // Use data.size() as a sentinel null index since it's out of bounds
    DeviceVector<csize_t> n_labels(1);

    handle_cuda_error(cudaMemcpy(n_labels.data(), &labels.n_labels, sizeof(csize_t), cudaMemcpyHostToDevice));

    // Initialize linelets at peaks
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (indexer.n_bins() + block_size - 1) / block_size;
    init_linelets_kernel<T, N><<<n_blocks, block_size>>>(fitter, peaks, linelets, visited.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    csize_t old_size = 0, new_size = labels.n_labels;

    while (old_size < new_size)
    {
        old_size = new_size;

        // Search for new candidates in the frontier and add them to the candidate list with a direction
        fill_candidates_kernel<T, N><<<n_blocks, block_size>>>(fitter, peaks, linelets, visited.view(), candidates.view());
        handle_cuda_error(cudaGetLastError());
        handle_cuda_error(cudaDeviceSynchronize());

        // Process candidates to update linelets and labels
        commit_frontier_kernel<T, N><<<n_blocks, block_size>>>(fitter, peaks, linelets, candidates.view(), visited.view(), n_labels.view());
        handle_cuda_error(cudaGetLastError());
        handle_cuda_error(cudaDeviceSynchronize());

        // Copy the updated number of labels back to host to check for convergence
        handle_cuda_error(cudaMemcpy(&new_size, n_labels.data(), sizeof(csize_t), cudaMemcpyDeviceToHost));
    }

    labels.n_labels = new_size;
    return out;
}

template <typename T>
array_t<T> line_fit(array_t<T> out, PeakLabels & labels, array_t<lint_t> parray, array_t<T> data, Structure structure, T vmin)
{
    if (structure.rank() != data.ndim())
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (data.ndim() != labels.labels.ndim())
    {
        throw std::invalid_argument("data and labels arrays must have the same number of dimensions");
    }

    switch (data.ndim())
    {
        case 2: return line_fit_nd<T, 2>(out, labels, parray, data, structure, vmin);
        case 3: return line_fit_nd<T, 3>(out, labels, parray, data, structure, vmin);
        case 4: return line_fit_nd<T, 4>(out, labels, parray, data, structure, vmin);
        case 5: return line_fit_nd<T, 5>(out, labels, parray, data, structure, vmin);
        case 6: return line_fit_nd<T, 6>(out, labels, parray, data, structure, vmin);
        case 7: return line_fit_nd<T, 7>(out, labels, parray, data, structure, vmin);
        default: throw std::runtime_error("Unsupported number of dimensions: data.ndim = " + std::to_string(data.ndim()));
    }
}

template <bool IsConst>
struct StreakTraits {};

template <>
struct StreakTraits<false>
{
    using index_type = csize_t;
    using index_reference = csize_t &;
    using label_type = lint_t;
    using label_reference = lint_t &;
};

template <>
struct StreakTraits<true>
{
    using index_type = const csize_t;
    using index_reference = const csize_t &;
    using label_type = const lint_t;
    using label_reference = const lint_t &;
};

struct EndValues
{
    lint_t start_id, end_id;
};

template <bool IsConst>
class Streak
{
protected:
    using index_type = typename StreakTraits<IsConst>::index_type;
    using index_reference = typename StreakTraits<IsConst>::index_reference;
    using label_reference = typename StreakTraits<IsConst>::label_reference;

public:
    class StreakEnds
    {
    protected:
        using reference = typename StreakTraits<IsConst>::label_reference;

    public:
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE StreakEnds & operator=(const EndValues & values)
        {
            m_start_id = values.start_id;
            m_end_id = values.end_id;
            return *this;
        }

        HOST_DEVICE bool valid() const { return m_start_id >= 0 && m_end_id >= 0; }
        HOST_DEVICE EndValues value() const { return EndValues{m_start_id, m_end_id}; }

        HOST_DEVICE lint_t start_bin() const { return m_start_id / 2; }
        HOST_DEVICE lint_t end_bin() const { return m_end_id / 2; }

        template <typename T, csize_t N>
        HOST_DEVICE LineND<T, 2> line(const Linelets<T> & linelets, const Peaks<N> & peaks) const
        {
            return LineND<T, 2>{linelets.point(2 * peaks.index(m_start_id / 2) + m_start_id % 2),
                                linelets.point(2 * peaks.index(m_end_id / 2) + m_end_id % 2)};
        }

        template <typename T, csize_t N>
        HOST_DEVICE LineND<T, 2> line(const Linelets<T> & linelets, const ArrayViewND<lint_t, N> labels) const
        {
            return LineND<T, 2>{linelets.point(2 * (labels[m_start_id / 2] - 1) + m_start_id % 2),
                                linelets.point(2 * (labels[m_end_id / 2] - 1) + m_end_id % 2)};
        }

        template <typename T, csize_t N, bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE bool insert(csize_t bin_idx, const Linelets<T> & linelets, const Peaks<N> & peaks)
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
                swap(t0, t1);
                swap(id0, id1);
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
        // Indices of line ends:
        // line_start = linelets[2 * N * m_indices[m_start_id / 2] + N * m_start_id % 2]
        // line_end   = linelets[2 * N * m_indices[m_end_id / 2]   + N * m_end_id % 2]
        reference m_start_id, m_end_id;

        HOST_DEVICE StreakEnds(reference start_id, reference end_id) : m_start_id(start_id), m_end_id(end_id) {}

        friend class Streak<IsConst>;
    };

    class StreakIndices
    {
    public:
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE void insert(csize_t bin_idx)
        {
            m_indices[m_length++ % m_max_length] = bin_idx;
        }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE csize_t & operator[](csize_t idx) { return m_indices[idx]; }
        HOST_DEVICE csize_t operator[](csize_t idx) const { return m_indices[idx]; }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE index_type * begin() { return m_indices.begin(); }
        HOST_DEVICE const index_type * begin() const { return m_indices.begin(); }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE index_type * end() { return m_indices.begin() + size(); }
        HOST_DEVICE const index_type * end() const { return m_indices.begin() + size(); }

        HOST_DEVICE csize_t size() const { return min(m_length, m_max_length); }
        HOST_DEVICE csize_t length() const { return m_length; }

        HOST_DEVICE bool overflowed() const { return m_length > m_max_length; }

    protected:
        DeviceRange<index_type> m_indices; // Bin indices of shape (length,)
        index_reference m_length;
        csize_t m_max_length;

        HOST_DEVICE StreakIndices(DeviceRange<index_type> indices, index_reference length, csize_t max_length) : m_indices(indices), m_length(length), m_max_length(max_length) {}

        friend class Streak<IsConst>;
    };

    HOST_DEVICE Streak(DeviceRange<index_type> indices, index_reference length, csize_t max_length, label_reference start_id, label_reference end_id)
        : m_indices(indices, length, max_length), m_ends(start_id, end_id) {}

    HOST_DEVICE const StreakEnds & ends() const { return m_ends; }
    HOST_DEVICE StreakEnds & ends() { return m_ends; }

    HOST_DEVICE const StreakIndices & indices() const { return m_indices; }
    HOST_DEVICE StreakIndices & indices() { return m_indices; }

    template <bool C = IsConst, typename = std::enable_if_t<!C>>
    HOST_DEVICE csize_t & indices(csize_t idx) { return m_indices[idx]; }
    HOST_DEVICE csize_t indices(csize_t idx) const { return m_indices[idx]; }

    template <typename T, csize_t N, bool C = IsConst, typename = std::enable_if_t<!C>>
    HOST_DEVICE bool insert(csize_t bin_idx, const Linelets<T> & linelets, const Peaks<N> & peaks)
    {
        auto old_ends = ends().value();
        auto is_inserted = ends().insert(bin_idx, linelets, peaks);

        if (!is_inserted)
        {
            ends() = old_ends; // Revert to old ends if insertion failed
            return false;
        }

        m_indices.insert(bin_idx);
        return true;
    }

protected:
    StreakIndices m_indices;
    StreakEnds m_ends;
};

class ShallowStreaks
{
public:
    template <bool IsConst>
    class ShallowStreaksView
    {
    protected:
        using index_type = typename StreakTraits<IsConst>::index_type;
        using label_type = typename StreakTraits<IsConst>::label_type;

    public:
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE Streak<false> operator[](csize_t idx)
        {
            auto begin = m_indices.data(idx * keep_last());
            auto end = m_indices.data((idx + 1) * keep_last());
            return Streak<false>(DeviceRange<index_type>(begin, end), m_lengths[idx], keep_last(), m_start_ids[idx], m_end_ids[idx]);
        }

        HOST_DEVICE Streak<true> operator[](csize_t idx) const
        {
            auto begin = m_indices.data(idx * keep_last());
            auto end = m_indices.data((idx + 1) * keep_last());
            return Streak<true>(DeviceRange<const index_type>(begin, end), m_lengths[idx], keep_last(), m_start_ids[idx], m_end_ids[idx]);
        }

        HOST_DEVICE csize_t size() const { return m_start_ids.size(); }

        HOST_DEVICE DeviceRange<index_type> lengths() const { return m_lengths; }

        HOST_DEVICE csize_t keep_last() const { return m_indices.size() / m_start_ids.size(); }

    protected:
        DeviceRange<label_type> m_start_ids, m_end_ids; // Start and end indices, shape (n_streaks,)
        DeviceRange<index_type> m_indices; // Last N indices of the streaks, shape (keep_last x n_streaks,)
        DeviceRange<index_type> m_lengths; // Lengths of the streaks, shape (n_streaks,)

        HOST_DEVICE ShallowStreaksView(DeviceRange<label_type> start_ids, DeviceRange<label_type> end_ids, DeviceRange<index_type> indices, DeviceRange<index_type> lengths)
            : m_start_ids(start_ids), m_end_ids(end_ids), m_indices(indices), m_lengths(lengths) {}

        friend class ShallowStreaks;
    };

    ShallowStreaks(csize_t n_streaks, csize_t keep_last) : m_start_ids(n_streaks, -1), m_end_ids(n_streaks, -1), m_indices(keep_last * n_streaks, 0), m_lengths(n_streaks, 0) {}

    ShallowStreaksView<false> view() { return ShallowStreaksView<false>(m_start_ids.view(), m_end_ids.view(), m_indices.view(), m_lengths.view()); }
    ShallowStreaksView<true> view() const { return ShallowStreaksView<true>(m_start_ids.view(), m_end_ids.view(), m_indices.view(), m_lengths.view()); }

protected:
    DeviceVector<lint_t> m_start_ids, m_end_ids; // Start and end indices
    DeviceVector<csize_t> m_indices; // Last N indices of the streaks, shape (keep_last x n_streaks,)
    DeviceVector<csize_t> m_lengths; // Lengths of the streaks, shape (n_streaks,)
};

class Streaks
{
public:
    template <bool IsConst>
    class StreaksView
    {
    protected:
        using index_type = typename StreakTraits<IsConst>::index_type;
        using label_type = typename StreakTraits<IsConst>::label_type;

    public:
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE Streak<false> operator[](csize_t idx)
        {
            auto start = m_indices.data(m_offsets[idx]);
            auto end = m_indices.data(m_offsets[idx + 1]);
            return Streak<false>(DeviceRange<index_type>(start, end), m_lengths[idx], end - start, m_start_ids[idx], m_end_ids[idx]);
        }

        HOST_DEVICE Streak<true> operator[](csize_t idx) const
        {
            auto start = m_indices.data(m_offsets[idx]);
            auto end = m_indices.data(m_offsets[idx + 1]);
            return Streak<true>(DeviceRange<const index_type>(start, end), m_lengths[idx], end - start, m_start_ids[idx], m_end_ids[idx]);
        }

        HOST_DEVICE csize_t size() const { return m_lengths.size(); }

    protected:
        DeviceRange<index_type> m_indices;
        DeviceRange<index_type> m_offsets; // CSR-like offsets for the indices, shape (n_streaks + 1,)
        DeviceRange<index_type> m_lengths;
        DeviceRange<label_type> m_start_ids, m_end_ids;

        HOST_DEVICE StreaksView(DeviceRange<index_type> indices, DeviceRange<index_type> offsets, DeviceRange<index_type> lengths, DeviceRange<label_type> start_ids, DeviceRange<label_type> end_ids) :
            m_indices(indices), m_offsets(offsets), m_lengths(lengths), m_start_ids(start_ids), m_end_ids(end_ids) {}

        friend class Streaks;
    };

    class HostStreak
    {
    public:
        const std::vector<csize_t> & indices() const { return m_indices; }

        template <typename T>
        std::array<T, 4> line(const array_t<T> & linelets, const array_t<lint_t> & labels) const
        {
            std::array<T, 4> result;
            lint_t label_id;
            handle_cuda_error(cudaMemcpy(&label_id, labels.data() + m_start_id / 2, sizeof(lint_t), cudaMemcpyDeviceToHost));
            if (label_id > 0) handle_cuda_error(cudaMemcpy(result.data(), linelets.data() + 2 * (2 * (label_id - 1) + m_start_id % 2), 2 * sizeof(T), cudaMemcpyDeviceToHost));

            handle_cuda_error(cudaMemcpy(&label_id, labels.data() + m_end_id / 2, sizeof(lint_t), cudaMemcpyDeviceToHost));
            if (label_id > 0) handle_cuda_error(cudaMemcpy(result.data() + 2, linelets.data() + 2 * (2 * (label_id - 1) + m_end_id % 2), 2 * sizeof(T), cudaMemcpyDeviceToHost));
            return result;
        }
    protected:
        std::vector<csize_t> m_indices;
        lint_t m_start_id, m_end_id;

        HostStreak(std::vector<csize_t> && indices, lint_t start_id, lint_t end_id) : m_indices(std::move(indices)), m_start_id(start_id), m_end_id(end_id) {}
        HostStreak(const std::vector<csize_t> & indices, lint_t start_id, lint_t end_id) : m_indices(indices), m_start_id(start_id), m_end_id(end_id) {}

        friend class Streaks;
    };

    Streaks(DeviceVector<csize_t> && offsets, csize_t total_size)
        : m_indices(total_size, 0), m_offsets(std::move(offsets))
    {
        if (m_offsets.size() == 0) throw std::invalid_argument("Offsets array cannot be empty.");

        m_lengths = DeviceVector<csize_t>(m_offsets.size() - 1, 0);
        m_start_ids = DeviceVector<lint_t>(m_offsets.size() - 1, -1);
        m_end_ids = DeviceVector<lint_t>(m_offsets.size() - 1, -1);
    }

    StreaksView<false> view()
    {
        return StreaksView<false>(m_indices.view(), m_offsets.view(), m_lengths.view(), m_start_ids.view(), m_end_ids.view());
    }

    StreaksView<true> view() const
    {
        return StreaksView<true>(m_indices.view(), m_offsets.view(), m_lengths.view(), m_start_ids.view(), m_end_ids.view());
    }

    csize_t size() const { return m_lengths.size(); }

    HostStreak to_host(csize_t i) const
    {
        if (i >= size()) throw std::out_of_range("Streak index out of range.");

        lint_t start_id, end_id;
        handle_cuda_error(cudaMemcpy(&start_id, m_start_ids.data(i), sizeof(lint_t),
                                     cudaMemcpyDeviceToHost));
        handle_cuda_error(cudaMemcpy(&end_id, m_end_ids.data(i), sizeof(lint_t),
                                     cudaMemcpyDeviceToHost));

        csize_t start, length;
        handle_cuda_error(cudaMemcpy(&start, m_offsets.data(i), sizeof(csize_t),
                                     cudaMemcpyDeviceToHost));
        handle_cuda_error(cudaMemcpy(&length, m_lengths.data(i), sizeof(csize_t),
                                     cudaMemcpyDeviceToHost));

        DeviceRange<const csize_t> indices (m_indices.data(start), m_indices.data(start + length));
        return HostStreak(indices.to_host(), start_id, end_id);
    }

protected:
    DeviceVector<csize_t> m_indices; // Bin indices of shape (total_size,)
    DeviceVector<csize_t> m_offsets; // CSR-like offsets for the indices, shape (n_streaks + 1,)
    DeviceVector<csize_t> m_lengths; // Lengths of the streaks, shape (n_streaks,)
    DeviceVector<lint_t> m_start_ids, m_end_ids; // Start and end indices
};

class StreaksIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using reference = Streaks::HostStreak;

    StreaksIterator(const Streaks & streaks, csize_t start = 0) : m_streaks(streaks), m_current(start) {}

    StreaksIterator & operator++()
    {
        ++m_current;
        return *this;
    }

    StreaksIterator operator++(int)
    {
        StreaksIterator temp = *this;
        operator++();
        return temp;
    }

    reference operator*() const { return m_streaks.to_host(m_current); }

    bool operator==(const StreaksIterator & other) const { return m_current == other.m_current; }
    bool operator!=(const StreaksIterator & other) const { return !(*this == other); }

private:
    const Streaks & m_streaks;
    csize_t m_current;
};

template <typename T, csize_t N>
class StreakFinder : public LineFitter<T, N>
{
public:
    using LineFitter<T, N>::peak_ahead;
    using LineFitter<T, N>::peak_behind;
    using Streak = Streak<false>;

    HOST_DEVICE StreakFinder(ArrayViewND<T, N> data, DeviceRange<lint_t> shifts, csize_t n_shifts, T vmin, T xtol, unsigned nfa, csize_t radius) :
        LineFitter<T, N>(data, shifts, n_shifts, vmin, radius), m_xtol(xtol), m_nfa(nfa) {}

    HOST_DEVICE bool detect(Streak & streak, const Linelets<T> & linelets, const Peaks<N> & peaks) const
    {
        if (!streak.ends().valid()) return false;

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

    HOST_DEVICE bool is_aligned(const PointIndex & candidate, const Streak & streak, const Linelets<T> & linelets, const Peaks<N> & peaks) const
    {
        auto total_line = streak.ends().line(linelets, peaks);

        csize_t num_unaligned = 0;
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

    HOST_DEVICE bool insert_candidate(Streak & streak, const PointIndex & candidate, const Linelets<T> & linelets, const Peaks<N> & peaks) const
    {
        auto old_ends = streak.ends().value();
        auto is_inserted = streak.ends().insert(candidate.bin, linelets, peaks);

        if (!is_inserted) return false;

        if (is_aligned(candidate, streak, linelets, peaks))
        {
            streak.indices().insert(candidate.bin);
            return true;
        }

        // Revert the insertion if the candidate is not aligned with the streak
        streak.ends() = old_ends;
        return false;
    }

    HOST_DEVICE bool grow_ahead(Streak & streak, const Linelets<T> & linelets, const Peaks<N> & peaks) const
    {
        if (!streak.ends().valid()) return false;

        auto old_bin = streak.ends().end_bin();

        PointIndex candidate;
        auto is_found = peak_ahead(old_bin, streak.ends().line(linelets, peaks), peaks, candidate);

        if (!is_found) return false;

        // Check if the candidate point is aligned with the streak
        return insert_candidate(streak, candidate, linelets, peaks);
    }

    HOST_DEVICE bool grow_behind(Streak & streak, const Linelets<T> & linelets, const Peaks<N> & peaks) const
    {
        if (!streak.ends().valid()) return false;

        auto old_bin = streak.ends().start_bin();

        PointIndex candidate;
        auto is_found = peak_behind(old_bin, streak.ends().line(linelets, peaks), peaks, candidate);

        if (!is_found) return false;

        // Check if the candidate point is aligned with the streak
        return insert_candidate(streak, candidate, linelets, peaks);
    }
};

template <typename T, csize_t N>
__global__ void count_streaks_kernel(StreakFinder<T, N> finder, Peaks<N> peaks, Linelets<T> linelets, ShallowStreaks::ShallowStreaksView<false> streaks)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= finder.indexer().n_bins()) return;

    if (!peaks.is_peak(bin_idx)) return;

    auto id = peaks.index(bin_idx);
    if (id >= streaks.size()) return;

    auto streak = streaks[id];
    auto is_inserted = streak.insert(bin_idx, linelets, peaks);

    if (!is_inserted) return;

    finder.detect(streak, linelets, peaks);

    auto ends = streak.ends().value();
    auto length = streak.indices().length();
}

__global__ void copy_shallow_indices(ShallowStreaks::ShallowStreaksView<false> shallow_streaks, Streaks::StreaksView<false> streaks)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= streaks.size()) return;

    auto shallow_streak = shallow_streaks[idx];
    auto streak = streaks[idx];

    // Copy short streaks
    if (!shallow_streak.indices().overflowed())
    {
        for (auto bin_idx : shallow_streak.indices()) streak.indices().insert(bin_idx);
        streak.ends() = shallow_streak.ends().value();
    }
}

template <typename T, csize_t N>
__global__ void detect_streaks_kernel(StreakFinder<T, N> finder, Peaks<N> peaks, Linelets<T> linelets, Streaks::StreaksView<false> streaks, DeviceRange<lint_t> overflowed_id)
{
    csize_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= finder.indexer().n_bins()) return;

    if (!peaks.is_peak(bin_idx)) return;

    auto id = peaks.index(bin_idx);
    if (id >= streaks.size()) return;

    auto streak = streaks[id];
    if (streak.ends().valid()) return; // Already processed in the shallow pass

    auto is_inserted = streak.insert(bin_idx, linelets, peaks);

    if (!is_inserted) return;

    finder.detect(streak, linelets, peaks);

    if (streak.indices().overflowed() && overflowed_id.size() > 0)
    {
        // Race-free first-writer-wins assignment: only set when current value is -1.
        atomicCAS(overflowed_id.data(), lint_t(-1), static_cast<lint_t>(id));
    }
}

template <typename T, csize_t N>
Streaks detect_streaks_nd(PeakLabels labels, array_t<lint_t> parray, array_t<T> larray, array_t<T> data, Structure structure, T vmin, T xtol, unsigned nfa, unsigned keep_last)
{
    auto shifts = all_nonzero_shifts(structure);
    csize_t n_shifts = shifts.size() / structure.rank();

    Peaks<N> peaks (cast_to_nd<lint_t, N>(labels.labels.view()), cast_to_nd<lint_t, 1>(parray.view()));
    Linelets<T> linelets (cast_to_nd<T, 2>(larray.view()));

    StreakFinder<T, N> finder (cast_to_nd<T, N>(data.view()), shifts.view(), n_shifts, vmin, xtol, nfa, labels.radius);

    ShallowStreaks shallow_streaks (labels.n_seeds, keep_last);

    csize_t block_size = BLOCK_SIZE;
    if (labels.n_seeds > 0)
    {
        csize_t cnt_blocks = (finder.indexer().n_bins() + block_size - 1) / block_size;
        count_streaks_kernel<T, N><<<cnt_blocks, block_size>>>(finder, peaks, linelets, shallow_streaks.view());
        handle_cuda_error(cudaGetLastError());
        handle_cuda_error(cudaDeviceSynchronize());
    }

    DeviceVector<csize_t> offsets (labels.n_seeds + 1, csize_t());

    // offsets[1:] = inclusive cumsum(lengths)
    if (labels.n_seeds > 0)
    {
        size_t temp_storage_bytes = 0;
        handle_cuda_error(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes,
                                                        shallow_streaks.view().lengths().data(), offsets.data() + 1,
                                                        shallow_streaks.view().lengths().size()));

        DeviceVector<char> temp_storage(temp_storage_bytes);
        handle_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage.data(), temp_storage_bytes,
                                                        shallow_streaks.view().lengths().data(), offsets.data() + 1,
                                                        shallow_streaks.view().lengths().size()));
    }

    csize_t total_size = 0;
    handle_cuda_error(cudaMemcpy(&total_size, offsets.data(labels.n_seeds), sizeof(csize_t), cudaMemcpyDeviceToHost));

    Streaks streaks (std::move(offsets), total_size);

    if (labels.n_seeds == 0) return streaks; // No seeds, return empty streaks

    DeviceVector<lint_t> overflowed (1, lint_t(-1));

    csize_t cp_blocks = (streaks.size() + block_size - 1) / block_size;
    copy_shallow_indices<<<cp_blocks, block_size>>>(shallow_streaks.view(), streaks.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    csize_t det_blocks = (finder.indexer().n_bins() + block_size - 1) / block_size;
    detect_streaks_kernel<T, N><<<det_blocks, block_size>>>(finder, peaks, linelets, streaks.view(), overflowed.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    lint_t overflowed_id;
    handle_cuda_error(cudaMemcpy(&overflowed_id, overflowed.data(), sizeof(lint_t), cudaMemcpyDeviceToHost));
    if (overflowed_id >= 0)
    {
        std::string msg = "A streak at index " + std::to_string(overflowed_id) + " exceeded the maximum length. Consider increasing keep_last to capture the full streak.";
        py::warnings::warn(msg.c_str(), PyExc_RuntimeWarning);
    }

    return streaks;
}

template <typename T>
Streaks detect_streaks(PeakLabels labels, array_t<lint_t> parray, array_t<T> larray, array_t<T> data, Structure structure, T vmin, T xtol, unsigned nfa, unsigned keep_last)
{
    if (structure.rank() != data.ndim())
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (data.ndim() != labels.labels.ndim())
    {
        throw std::invalid_argument("data and labels arrays must have the same number of dimensions");
    }

    switch (data.ndim())
    {
        case 2: return detect_streaks_nd<T, 2>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        case 3: return detect_streaks_nd<T, 3>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        case 4: return detect_streaks_nd<T, 4>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        case 5: return detect_streaks_nd<T, 5>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        case 6: return detect_streaks_nd<T, 6>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        case 7: return detect_streaks_nd<T, 7>(labels, parray, larray, data, structure, vmin, xtol, nfa, keep_last);
        default: throw std::runtime_error("Unsupported number of dimensions: data.ndim = " + std::to_string(data.ndim()));
    }
}

template <typename T, csize_t N>
__global__ void to_lines_kernel(Streaks::StreaksView<true> streaks, ArrayViewND<T, 2> out, ArrayViewND<lint_t, N> labels, Linelets<T> linelets)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= streaks.size()) return;

    auto streak = streaks[idx];
    if (!streak.ends().valid()) return;

    auto line = streak.ends().line(linelets, labels);
    out[4 * idx + 0] = line.pt0[0];
    out[4 * idx + 1] = line.pt0[1];
    out[4 * idx + 2] = line.pt1[0];
    out[4 * idx + 3] = line.pt1[1];
}

template <typename T, csize_t N>
array_t<T> to_lines_nd(const Streaks & streaks, array_t<T> out, PeakLabels labels, array_t<T> larray)
{
    auto out_view = cast_to_nd<T, 2>(out.view());
    auto labels_view = cast_to_nd<lint_t, N>(labels.labels.view());
    Linelets<T> linelets (cast_to_nd<T, 2>(larray.view()));

    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (streaks.size() + block_size - 1) / block_size;
    to_lines_kernel<T, N><<<n_blocks, block_size>>>(streaks.view(), out_view, labels_view, linelets);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T>
array_t<T> to_lines(const Streaks & streaks, array_t<T> out, PeakLabels labels, array_t<T> larray)
{
    if (out.ndim() != 2 || out.shape(1) != 4)
    {
        throw std::invalid_argument("Output array must have shape (n_streaks, 4)");
    }

    switch (labels.labels.ndim())
    {
        case 2: return to_lines_nd<T, 2>(streaks, out, labels, larray);
        case 3: return to_lines_nd<T, 3>(streaks, out, labels, larray);
        case 4: return to_lines_nd<T, 4>(streaks, out, labels, larray);
        case 5: return to_lines_nd<T, 5>(streaks, out, labels, larray);
        case 6: return to_lines_nd<T, 6>(streaks, out, labels, larray);
        case 7: return to_lines_nd<T, 7>(streaks, out, labels, larray);
        default: throw std::runtime_error("Unsupported number of dimensions: labels.labels.ndim = " + std::to_string(labels.labels.ndim()));
    }
}

struct StreakPixel
{
    csize_t streak_id;
    lint_t index;

    friend HOST_DEVICE bool operator<(const StreakPixel & lhs, const StreakPixel & rhs)
    {
        if (lhs.streak_id < rhs.streak_id) return true;
        if (lhs.streak_id > rhs.streak_id) return false;
        return lhs.index < rhs.index;
    }

    HOST_DEVICE bool operator==(const StreakPixel & rhs) const
    {
        return streak_id == rhs.streak_id && index == rhs.index;
    }
};

struct StreakPixelLess
{
    HOST_DEVICE bool operator()(const StreakPixel & lhs, const StreakPixel & rhs) const
    {
        return lhs < rhs;
    }
};

template <typename T>
__global__ void init_streak_lengths_kernel(DeviceRange<csize_t> lengths, Streaks::StreaksView<true> streaks, csize_t n_shifts)
{
    csize_t streak_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (streak_idx >= lengths.size()) return;

    auto streak = streaks[streak_idx];
    lengths[streak_idx] = streak.indices().size() * n_shifts;
}

template <typename T, csize_t N>
__global__ void init_footprint_keys_kernel(DeviceRange<StreakPixel> keys,
                                           DeviceRange<csize_t> entry_offsets, Streaks::StreaksView<true> streaks,
                                           Peaks<N> peaks, DeviceRange<lint_t> shifts, csize_t n_shifts, ArrayViewND<T, N> data)
{
    csize_t streak_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (streak_idx >= streaks.size()) return;

    auto streak = streaks[streak_idx];
    auto n_points = streak.indices().size();
    csize_t base = entry_offsets[streak_idx];

    for (csize_t i = 0; i < n_points; ++i)
    {
        auto bin_idx = streak.indices(i);
        auto peak_idx = peaks[bin_idx];

        for (csize_t k = 0; k < n_shifts; ++k)
        {
            auto neighbour_idx = detail::shift_index(peak_idx, data.shape(), N, shifts.data(k * N), N);
            keys[base + i * n_shifts + k] = StreakPixel{streak_idx, neighbour_idx};
        }
    }
}

template <typename T, csize_t N>
__global__ void count_signal_from_unique_kernel(ArrayViewND<csize_t, 1> counts,
                                                DeviceRange<StreakPixel> unique_keys,
                                                const csize_t * n_unique, ArrayViewND<T, N> data, T vmin)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *n_unique) return;

    auto key = unique_keys[idx];
    auto index = key.index;
    if (index < 0) return;

    if (data[index] >= vmin)
    {
        atomicAdd(counts.data(key.streak_id), csize_t(1));
    }
}

template <typename T, csize_t N>
array_t<csize_t> n_signal_nd(array_t<csize_t> out, const Streaks & streaks, PeakLabels labels, array_t<lint_t> parray, array_t<T> data, Structure structure, T vmin)
{
    auto shifts = all_shifts(structure);
    csize_t n_shifts = structure.size();

    out.fill(csize_t(0));

    auto out_view = cast_to_nd<csize_t, 1>(out.view());
    auto data_view = cast_to_nd<T, N>(data.view());
    Peaks<N> peaks (cast_to_nd<lint_t, N>(labels.labels.view()), cast_to_nd<lint_t, 1>(parray.view()));

    csize_t n_streaks = streaks.size();

    if (n_streaks == 0 || n_shifts == 0) return out;

    DeviceVector<csize_t> d_lengths (n_streaks, csize_t(0));
    DeviceVector<csize_t> d_point_offsets (n_streaks, csize_t(0));

    csize_t init_blocks = (n_streaks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_streak_lengths_kernel<T><<<init_blocks, BLOCK_SIZE>>>(d_lengths.view(), streaks.view(), n_shifts);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    size_t scan_temp_storage_bytes = 0;
    handle_cuda_error(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_storage_bytes,
                                                    d_lengths.data(), d_point_offsets.data(), n_streaks));
    DeviceVector<char> scan_temp_storage (scan_temp_storage_bytes);
    handle_cuda_error(cub::DeviceScan::ExclusiveSum(scan_temp_storage.data(), scan_temp_storage_bytes,
                                                    d_lengths.data(), d_point_offsets.data(), n_streaks));

    csize_t last_offset = 0;
    csize_t last_length = 0;
    handle_cuda_error(cudaMemcpy(&last_offset, d_point_offsets.data(n_streaks - 1), sizeof(csize_t), cudaMemcpyDeviceToHost));
    handle_cuda_error(cudaMemcpy(&last_length, d_lengths.data(n_streaks - 1), sizeof(csize_t), cudaMemcpyDeviceToHost));
    csize_t total_entries = last_offset + last_length;

    if (total_entries == 0) return out;

    DeviceVector<StreakPixel> footprint_keys (total_entries);

    init_footprint_keys_kernel<T, N><<<init_blocks, BLOCK_SIZE>>>(footprint_keys.view(), d_point_offsets.view(),
                                                                  streaks.view(), peaks, shifts.view(), n_shifts, data_view);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    size_t sort_temp_storage_bytes = 0;
    handle_cuda_error(cub::DeviceMergeSort::SortKeys(nullptr, sort_temp_storage_bytes,
                                                     footprint_keys.data(), total_entries,
                                                     StreakPixelLess{}, 0));
    DeviceVector<char> sort_temp_storage (sort_temp_storage_bytes);
    handle_cuda_error(cub::DeviceMergeSort::SortKeys(sort_temp_storage.data(), sort_temp_storage_bytes,
                                                     footprint_keys.data(), total_entries,
                                                     StreakPixelLess{}, 0));

    DeviceVector<StreakPixel> unique_keys (total_entries);

    size_t unique_temp_storage_bytes = 0;
    DeviceVector<csize_t> d_num_selected (1);
    handle_cuda_error(cub::DeviceSelect::Unique(nullptr, unique_temp_storage_bytes,
                                                footprint_keys.data(), unique_keys.data(),
                                                d_num_selected.data(), total_entries));
    DeviceVector<char> unique_temp_storage (unique_temp_storage_bytes);
    handle_cuda_error(cub::DeviceSelect::Unique(unique_temp_storage.data(), unique_temp_storage_bytes,
                                                footprint_keys.data(), unique_keys.data(),
                                                d_num_selected.data(), total_entries));

    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (total_entries + block_size - 1) / block_size;
    count_signal_from_unique_kernel<T, N><<<n_blocks, block_size>>>(out_view, unique_keys.view(),
                                                                    d_num_selected.data(), data_view, vmin);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T>
array_t<csize_t> n_signal(array_t<csize_t> out, const Streaks & streaks, PeakLabels labels, array_t<lint_t> parray, array_t<T> data, Structure structure, T vmin)
{
    if (structure.rank() != data.ndim())
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (data.ndim() != labels.labels.ndim())
    {
        throw std::invalid_argument("data and labels arrays must have the same number of dimensions");
    }

    switch (data.ndim())
    {
        case 2: return n_signal_nd<T, 2>(out, streaks, labels, parray, data, structure, vmin);
        case 3: return n_signal_nd<T, 3>(out, streaks, labels, parray, data, structure, vmin);
        case 4: return n_signal_nd<T, 4>(out, streaks, labels, parray, data, structure, vmin);
        case 5: return n_signal_nd<T, 5>(out, streaks, labels, parray, data, structure, vmin);
        case 6: return n_signal_nd<T, 6>(out, streaks, labels, parray, data, structure, vmin);
        case 7: return n_signal_nd<T, 7>(out, streaks, labels, parray, data, structure, vmin);
        default: throw std::runtime_error("Unsupported number of dimensions: data.ndim = " + std::to_string(data.ndim()));
    }
}

// Step 1: init output to INT32_MAX (fill kernel or out.fill())
// Step 2: streak_labels_kernel — one thread per streak
template <csize_t N>
__global__ void streak_labels_kernel(ArrayViewND<lint_t, N> out, Streaks::StreaksView<true> streaks, ArrayViewND<lint_t, 1> ranks, Peaks<N> peaks, DeviceRange<lint_t> shifts, csize_t n_shifts)
{
    csize_t streak_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (streak_idx >= streaks.size()) return;

    auto streak = streaks[streak_idx];
    lint_t candidate = ranks[streak_idx] + 1;

    for (csize_t i = 0; i < streak.indices().size(); i++)
    {
        auto bin_idx = streak.indices(i);
        auto peak_idx = peaks[bin_idx];

        for (csize_t k = 0; k < n_shifts; k++)
        {
            auto neighbour_idx = detail::shift_index(peak_idx, out.shape(), N, shifts.data(k * N), N);
            if (neighbour_idx >= 0) atomicMin(out.data(neighbour_idx), candidate);
        }
    }
}

// Step 3: finalize — replace INT32_MAX back to 0
template <csize_t N>
__global__ void finalize_streak_labels_kernel(ArrayViewND<lint_t, N> out, lint_t max_label)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;
    if (out[idx] == max_label) out[idx] = lint_t(0);
}

template <csize_t N>
array_t<lint_t> streak_labels_nd(array_t<lint_t> out, const Streaks & streaks, array_t<lint_t> ranks, PeakLabels labels, array_t<lint_t> parray, Structure structure)
{
    auto shifts = all_shifts(structure);
    csize_t n_shifts = shifts.size() / structure.rank();
    csize_t n_streaks = streaks.size();

    if (n_streaks == 0 || n_shifts == 0)
    {
        out.fill(lint_t(0));
        return out;
    }

    out.fill(std::numeric_limits<lint_t>::max());

    auto out_view = cast_to_nd<lint_t, N>(out.view());
    Peaks<N> peaks (cast_to_nd<lint_t, N>(labels.labels.view()), cast_to_nd<lint_t, 1>(parray.view()));

    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_streaks + block_size - 1) / block_size;
    streak_labels_kernel<<<n_blocks, block_size>>>(out_view, streaks.view(), cast_to_nd<lint_t, 1>(ranks.view()), peaks, shifts.view(), n_shifts);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    csize_t total_size = out.size();
    n_blocks = (total_size + block_size - 1) / block_size;
    finalize_streak_labels_kernel<<<n_blocks, block_size>>>(out_view, std::numeric_limits<lint_t>::max());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

array_t<lint_t> streak_labels(array_t<lint_t> out, const Streaks & streaks, array_t<lint_t> ranks, PeakLabels labels, array_t<lint_t> parray, Structure structure)
{
    if (structure.rank() != out.ndim())
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must match output array dimension (" + std::to_string(out.ndim()) + ")");
    }
    if (out.ndim() != labels.labels.ndim())
    {
        throw std::invalid_argument("Output and labels arrays must have the same number of dimensions");
    }

    switch (out.ndim())
    {
        case 2: return streak_labels_nd<2>(out, streaks, ranks, labels, parray, structure);
        case 3: return streak_labels_nd<3>(out, streaks, ranks, labels, parray, structure);
        case 4: return streak_labels_nd<4>(out, streaks, ranks, labels, parray, structure);
        case 5: return streak_labels_nd<5>(out, streaks, ranks, labels, parray, structure);
        case 6: return streak_labels_nd<6>(out, streaks, ranks, labels, parray, structure);
        case 7: return streak_labels_nd<7>(out, streaks, ranks, labels, parray, structure);
        default: throw std::runtime_error("Unsupported number of dimensions: out.ndim = " + std::to_string(out.ndim()));
    }
}

} // namespace cbclib::cuda

PYBIND11_MODULE(cuda_streak_finder, m)
{
    using namespace cbclib;
    namespace cu = cbclib::cuda;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    py::class_<cu::PeakLabels>(m, "PeakLabels")
        .def(py::init<cu::array_t<cu::lint_t>, py::ssize_t, py::ssize_t, py::ssize_t, py::ssize_t>(), py::arg("labels"), py::arg("n_seeds"), py::arg("n_labels"), py::arg("n_good"), py::arg("radius"))
        .def_readonly("labels", &cu::PeakLabels::labels)
        .def_readonly("n_seeds", &cu::PeakLabels::n_seeds)
        .def_readonly("n_labels", &cu::PeakLabels::n_labels)
        .def_readonly("n_good", &cu::PeakLabels::n_good)
        .def_readonly("radius", &cu::PeakLabels::radius)
        .def("keep_best", [](const cu::PeakLabels & labels, double quantile)
        {
            return cu::PeakLabels{labels.labels, py::ssize_t(labels.n_seeds * quantile), labels.n_labels, labels.n_good, labels.radius};
        }, py::arg("quantile") = 0.5);

    py::class_<cu::Streaks::HostStreak>(m, "HostStreak")
        .def_property_readonly("indices", [](const cu::Streaks::HostStreak & streak) { return streak.indices(); })
        .def("line", [](const cu::Streaks::HostStreak & streak, cu::PeakLabels labels, cu::array_t<float> lines) { return streak.line(lines, labels.labels); })
        .def("line", [](const cu::Streaks::HostStreak & streak, cu::PeakLabels labels, cu::array_t<double> lines) { return streak.line(lines, labels.labels); });

    py::class_<cu::Streaks>(m, "Streaks")
        .def("__len__", [](const cu::Streaks & streaks) { return streaks.size(); })
        .def("__iter__", [](const cu::Streaks & streaks)
        {
            return py::make_iterator(cu::StreaksIterator(streaks), cu::StreaksIterator(streaks, streaks.size()));
        })
        .def("__getitem__", [](const cu::Streaks & streaks, cu::csize_t idx) { return streaks.to_host(idx); }, py::arg("index"))
        .def("to_lines", [](const cu::Streaks & streaks, cu::array_t<float> out, cu::PeakLabels labels, cu::array_t<float> lines) { return cu::to_lines(streaks, out, labels, lines); })
        .def("to_lines", [](const cu::Streaks & streaks, cu::array_t<double> out, cu::PeakLabels labels, cu::array_t<double> lines) { return cu::to_lines(streaks, out, labels, lines); });

    m.def("detect_peaks", &cu::detect_peaks<float>, py::arg("peaks"), py::arg("labels"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"));
    m.def("detect_peaks", &cu::detect_peaks<double>, py::arg("peaks"), py::arg("labels"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"));

    m.def("line_fit", &cu::line_fit<float>, py::arg("out"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"));
    m.def("line_fit", &cu::line_fit<double>, py::arg("out"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"));

    m.def("detect_streaks", &cu::detect_streaks<float>, py::arg("labels"), py::arg("peaks"), py::arg("linelets"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("xtol"), py::arg("nfa"), py::arg("keep_last")=11);
    m.def("detect_streaks", &cu::detect_streaks<double>, py::arg("labels"), py::arg("peaks"), py::arg("linelets"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("xtol"), py::arg("nfa"), py::arg("keep_last")=11);

    m.def("n_signal", &cu::n_signal<float>, py::arg("out"), py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"));
    m.def("n_signal", &cu::n_signal<double>, py::arg("out"), py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"));

    m.def("streak_labels", &cu::streak_labels, py::arg("out"), py::arg("streaks"), py::arg("ranks"), py::arg("labels"), py::arg("parray"), py::arg("structure"));
}
