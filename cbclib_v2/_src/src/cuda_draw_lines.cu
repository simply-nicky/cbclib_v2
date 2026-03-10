#include <cub/cub.cuh>
#include "cupy_array.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace cbclib::cuda {

// Lightweight structure for calculating line distance
template <typename T, csize_t N>
struct LineData
{
    PointND<T, N> _M_tau;   // direction vector
    PointND<T, N> _M_ctr;   // center point
    T _M_width;             // line width

    HOST_DEVICE LineData() : _M_tau(), _M_ctr(), _M_width(T()) {}
    HOST_DEVICE LineData(StridedIterator<T> iter) : _M_tau(), _M_ctr(), _M_width(iter[2 * N])
    {
        for (csize_t n = 0; n < N; n++)
        {
            T a = iter[n], b = iter[N + n];
            _M_ctr[n] = T(0.5) * (a + b);
            _M_tau[n] = b - a;
        }
    }

    HOST_DEVICE T distance(const PointND<T, N> & point) const
    {
        auto mag = magnitude(_M_tau);

        if (mag > numeric_limits<T>::epsilon())
        {
            auto r = point - _M_ctr;
            auto r_tau = dot(r, _M_tau) / mag;
            return amplitude(math_traits<T>::clamp(r_tau, -0.5, 0.5) * _M_tau + _M_ctr - point);
        }
        return amplitude(_M_ctr - point);
    }

    HOST_DEVICE const PointND<T, N> & center() const { return _M_ctr; }
    HOST_DEVICE const PointND<T, N> & tangent() const { return _M_tau; }
    HOST_DEVICE T width() const { return _M_width; }
};

// Building line data structures on device
template <typename T, csize_t N>
__global__ void build_lines_kernel(ArrayViewND<T, 2> lines, csize_t n_lines, DeviceRange<LineData<T, N>> line_data)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    line_data[idx] = LineData<T, N>(lines.begin_at(idx * lines.strides(0), 1));
}

template <typename T, csize_t N>
DeviceVector<LineData<T, N>> build_lines(const array_view<T, py::ssize_t> & lines)
{
    DeviceVector<LineData<T, N>> line_data(lines.shape(0));

    int block_size = BLOCK_SIZE;
    int num_blocks = static_cast<int>((lines.shape(0) + block_size - 1) / block_size);
    build_lines_kernel<T, N><<<num_blocks, block_size>>>(cast_to_nd<T, 2>(lines), lines.shape(0), line_data.view());

    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return line_data;
}

// DrawIndices and AccumulateIndices classes for managing line indices on device
struct Counter
{
    csize_t * counts;

    __device__ Counter(csize_t * counts) : counts(counts) {}

    __device__ void operator()(csize_t bin_idx) const
    {
        atomicAdd(&counts[bin_idx], 1);
    }
};

class DrawIndices
{
public:
    struct Filler
    {
        const csize_t * _M_offsets;
        csize_t * _M_indices;
        csize_t * _M_counters;
        csize_t _M_index;

        __device__ Filler(const csize_t * offsets, csize_t * indices, csize_t * counters, csize_t idx)
            : _M_offsets(offsets), _M_indices(indices), _M_counters(counters), _M_index(idx) {}

        __device__ void operator()(csize_t bin_idx) const
        {
            csize_t pos = atomicAdd(&_M_counters[bin_idx], 1);
            _M_indices[_M_offsets[bin_idx] + pos] = _M_index;
        }
    };

    template <bool IsConst, typename Value = std::conditional_t<IsConst, const csize_t, csize_t>>
    struct DrawIndicesView
    {
        DeviceRange<Value> _M_indices;
        DeviceRange<Value> _M_offsets;

        HOST_DEVICE DeviceRange<Value> operator[](csize_t idx) const
        {
            csize_t start = _M_offsets[idx];
            csize_t end = _M_offsets[idx + 1];
            return DeviceRange<Value>(_M_indices.begin() + start, _M_indices.begin() + end);
        }

        HOST_DEVICE const DeviceRange<Value> & indices() const { return _M_indices; }
        HOST_DEVICE const DeviceRange<Value> & offsets() const { return _M_offsets; }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE DeviceRange<Value> & indices() {return _M_indices; }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        HOST_DEVICE DeviceRange<Value> & offsets() {return _M_offsets; }

        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        __device__ Filler filler(csize_t line_idx, csize_t * counters)
        {
            return Filler(_M_offsets.data(), _M_indices.data(), counters, line_idx);
        }

        // Overload with offset for processing multiple frames
        // Frame offset is applied to _M_offsets and counters only
        // Since indices is flat and the _M_indices position is stored in _M_offsets
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        __device__ Filler filler(csize_t offset, csize_t line_idx, csize_t * counters)
        {
            return Filler(_M_offsets.data() + offset, _M_indices.data(), counters + offset, line_idx);
        }
    };

    using size_type = csize_t;
    using view_type = DrawIndicesView<false>;
    using const_view_type = DrawIndicesView<true>;

    DrawIndices() = default;
    DrawIndices(DeviceVector<csize_t> indices, DeviceVector<csize_t> offsets)
        : m_indices(std::move(indices)), m_offsets(std::move(offsets)) {}

    view_type view()
    {
        return {m_indices.view(), m_offsets.view()};
    }

    const_view_type view() const
    {
        return {m_indices.view(), m_offsets.view()};
    }

    const DeviceVector<csize_t> & indices() const { return m_indices; }
    const DeviceVector<csize_t> & offsets() const { return m_offsets; }

protected:
    DeviceVector<csize_t> m_indices;
    DeviceVector<csize_t> m_offsets;
};

struct IndexPair
{
    csize_t line;
    csize_t term;
};

class AccumulateIndices
{
protected:
    struct IndexSentinel
    {
        const csize_t * idx_ptr;
    };

    struct IndexIterator
    {
        const csize_t * idx_ptr;
        const csize_t * term_ptr;

        HOST_DEVICE IndexIterator(const csize_t * iptr, const csize_t * tptr)
            : idx_ptr(iptr), term_ptr(tptr) {}

        HOST_DEVICE IndexIterator & operator++()
        {
            ++idx_ptr;
            ++term_ptr;
            return *this;
        }

        HOST_DEVICE friend bool operator==(const IndexIterator & lhs, const IndexSentinel & rhs)
        {
            return lhs.idx_ptr == rhs.idx_ptr;
        }
        HOST_DEVICE friend bool operator!=(const IndexIterator & lhs, const IndexSentinel & rhs)
        {
            return !(lhs == rhs);
        }

        HOST_DEVICE IndexPair operator*() const
        {
            return { *idx_ptr, *term_ptr};
        }
    };

public:
    struct AccumulateIndicesRange
    {
    public:
        using value_type = IndexPair;
        using size_type = csize_t;
        using iterator = IndexIterator;
        using const_iterator = IndexIterator;
        using sentinel = IndexSentinel;
        using const_sentinel = IndexSentinel;

        DeviceRange<const csize_t> _M_indices;
        DeviceRange<const csize_t> _M_terms;

        HOST_DEVICE size_type size() const { return _M_indices.size(); }

        HOST_DEVICE const_iterator begin() const { return {_M_indices.begin(), _M_terms.begin()}; }
        HOST_DEVICE const_sentinel end() const { return {_M_indices.end()}; }
    };

    struct Filler
    {
        const csize_t * _M_offsets;
        csize_t * _M_indices;
        csize_t * _M_terms;
        csize_t * _M_counters;
        IndexPair _M_pair;

        __device__ Filler(const csize_t * offsets, csize_t * indices, csize_t * terms, csize_t * counters, IndexPair pair)
            : _M_offsets(offsets), _M_indices(indices), _M_terms(terms), _M_counters(counters), _M_pair(pair) {}

        __device__ void operator()(csize_t bin_idx) const
        {
            csize_t pos = atomicAdd(&_M_counters[bin_idx], 1);
            _M_indices[_M_offsets[bin_idx] + pos] = _M_pair.line;
            _M_terms[_M_offsets[bin_idx] + pos] = _M_pair.term;
        }
    };

    template <bool IsConst, typename Value = std::conditional_t<IsConst, const csize_t, csize_t>>
    struct AccumulateIndicesView
    {
        DeviceRange<Value> _M_indices;
        DeviceRange<Value> _M_terms;
        DeviceRange<Value> _M_offsets;

        HOST_DEVICE AccumulateIndicesRange operator[](csize_t idx) const
        {
            csize_t start = _M_offsets[idx];
            csize_t end = _M_offsets[idx + 1];
            return {DeviceRange<const csize_t>(_M_indices.begin() + start, _M_indices.begin() + end),
                    DeviceRange<const csize_t>(_M_terms.begin() + start, _M_terms.begin() + end)};
        }

        // Offsets/counters are per-frame; indices/terms remain flat
        template <bool C = IsConst, typename = std::enable_if_t<!C>>
        __device__ Filler filler(csize_t offset, IndexPair pair, csize_t * counters)
        {
            return Filler(_M_offsets.data() + offset, _M_indices.data(), _M_terms.data(), counters + offset, pair);
        }

        HOST_DEVICE DeviceRange<Value> & indices() { return _M_indices; }
        HOST_DEVICE const DeviceRange<Value> & indices() const { return _M_indices; }

        HOST_DEVICE DeviceRange<Value> & terms() { return _M_terms; }
        HOST_DEVICE const DeviceRange<Value> & terms() const { return _M_terms; }

        HOST_DEVICE DeviceRange<Value> & offsets() { return _M_offsets; }
        HOST_DEVICE const DeviceRange<Value> & offsets() const { return _M_offsets; }
    };

    using view_type = AccumulateIndicesView<false>;
    using const_view_type = AccumulateIndicesView<true>;

    AccumulateIndices() = default;
    AccumulateIndices(DeviceVector<csize_t> indices, DeviceVector<csize_t> terms, DeviceVector<csize_t> offsets)
        : m_indices(std::move(indices)), m_terms(std::move(terms)), m_offsets(std::move(offsets)) {}

    view_type view()
    {
        return {m_indices.view(), m_terms.view(), m_offsets.view()};
    }

    const_view_type view() const
    {
        return {m_indices.view(), m_terms.view(), m_offsets.view()};
    }

    DeviceVector<csize_t> & indices() { return m_indices; }
    const DeviceVector<csize_t> & indices() const { return m_indices; }

    DeviceVector<csize_t> & terms() { return m_terms; }
    const DeviceVector<csize_t> & terms() const { return m_terms; }

    DeviceVector<csize_t> & offsets() { return m_offsets; }
    const DeviceVector<csize_t> & offsets() const { return m_offsets; }

protected:
    DeviceVector<csize_t> m_indices, m_terms, m_offsets;
};

// DrawContext class for managing drawing parameters and line indices on device

template <class Indices, typename T, csize_t N, typename IndicesView = typename Indices::const_view_type, typename IndicesRange = decltype(std::declval<IndicesView &>()[0])>
class DrawContext
{
protected:
    struct DrawContextView
    {
        IndicesView _M_indices;
        const ShapeND<N> _M_grid;
        const PointND<T, N> _M_bin;

        HOST_DEVICE const IndicesView & indices() const { return _M_indices; }
        HOST_DEVICE IndicesRange indices(csize_t idx) const { return _M_indices[idx]; }

        HOST_DEVICE const ShapeND<N> & grid() const { return _M_grid; }
        HOST_DEVICE csize_t grid(csize_t dim) const { return _M_grid.shape(dim); }

        HOST_DEVICE const PointND<T, N> & bin() const { return _M_bin; }
        HOST_DEVICE T bin(csize_t dim) const { return _M_bin[dim]; }

        template <typename Func>
        HOST_DEVICE void apply_to_line(StridedIterator<T> iter, Func func) const
        {
            T offset = math_traits<T>::ceil(iter[2 * N]) + 1;

            auto pt0 = iter;
            auto pt1 = iter + N;

            // Compute normalized direction vector tau
            T length = T();
            PointND<T, N> tau;
            for (csize_t n = 0; n < N; n++) {tau[n] = pt1[n] - pt0[n]; length += tau[n] * tau[n];}
            tau = tau / math_traits<T>::sqrt(length);

            // coord0 and coord1 follow zyx convention
            PointND<csize_t, N> coord0, coord1;
            for (csize_t zyx_n = 0; zyx_n < N; zyx_n++)
            {
                // Compute bounding box in voxel coordinates
                // Points are in xyz convention; grid/bin are in zyx convention
                // Start index can be greater than end index if line is drawn backwards along this axis
                // Start and end points can be outside of the volume

                csize_t xyz_n = N - zyx_n - 1;  // Convert zyx index to xyz index
                auto start = math_traits<T>::round(pt0[xyz_n] - math_traits<T>::abs(offset / tau[xyz_n]) * tau[xyz_n]);
                auto end = math_traits<T>::round(pt1[xyz_n] + math_traits<T>::abs(offset / tau[xyz_n]) * tau[xyz_n]);
                if (start > end) {auto temp = start; start = end; end = temp;}
                start = math_traits<T>::floor(start / _M_bin[zyx_n]);
                end = math_traits<T>::ceil(end / _M_bin[zyx_n]);
                coord0[zyx_n] = math_traits<T>::max(T(), start);
                coord1[zyx_n] = math_traits<T>::min(_M_grid.shape(zyx_n), end);
            }

            // Shape follows zyx convention
            ShapeND<N> shape = coord1 - coord0;

            // Coordinate also follows zyx convention
            PointND<csize_t, N> coord;
            for (csize_t offset = 0; offset < shape.size(); offset++)
            {
                shape.coord_at(coord, offset);
                coord += coord0;

                csize_t bin_idx = _M_grid.index_at(coord);
                func(bin_idx);
            }
        }
    };

public:
    using view_type = DrawContextView;
    using const_view_type = DrawContextView;

    DrawContext() = default;

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    DrawContext(const I * shape, const I * grid) : m_grid(grid)
    {
        for (csize_t i = 0; i < N; i++)
        {
            m_bin[i] = static_cast<T>(shape[i]) / static_cast<T>(grid[i]);
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    DrawContext(const I (&shape)[N], const I (&grid)[N]) : DrawContext(shape, grid) {}

    const ShapeND<N> & grid() const { return m_grid; }
    csize_t grid(csize_t dim) const { return m_grid.shape(dim); }

    const PointND<T, N> & bin() const { return m_bin; }
    const T & bin(csize_t dim) const { return m_bin[dim]; }

    const_view_type view() const
    {
        return {m_indices.view(), m_grid, m_bin};
    }

    void import_indices(Indices && indices)
    {
        m_indices = std::move(indices);
    }

protected:
    Indices m_indices; 		        // indices of lines in each bin on GPU
    ShapeND<N> m_grid;				// number of bins in each dimension
    PointND<T, N> m_bin;     	    // size of each bin in voxels
};

template <class Indices, typename T, csize_t N>
using DrawContextView = typename DrawContext<Indices, T, N>::const_view_type;
using DrawIndicesView = DrawIndices::view_type;
using AccumulateIndicesView = AccumulateIndices::view_type;

// CUDA kernel: point-parallel thick line drawing (2D/3D)
template <typename T, csize_t N, int Update, typename Kernel>
__global__ void draw_thick_lines_kernel(
    ArrayViewND<T, N + 1> volume, DeviceRange<LineData<T, N>> lines,
    DrawContextView<DrawIndices, T, N> context, T max_val, Kernel kernel)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= volume.size()) return;

    // Get voxel coordinate
    auto coord = volume.coord_at(idx);
    auto frame = coord[0];

    csize_t bin_idx = 0;
    PointND<T, N> point;
    for (csize_t i = 0; i < N; i++)
    {
        auto x = coord[1 + i] / context.bin(i);
        if (x >= context.grid(i)) x = context.grid(i) - 1;
        bin_idx = bin_idx * context.grid(i) + x;

        point[N - i - 1] = coord[1 + i];
    }

    for (auto index : context.indices(frame * context.grid().size() + bin_idx))
    {
        T dist = lines[index].distance(point);
        T width = lines[index].width();
        if (width <= T()) continue;

        T error = max_val * kernel(dist / width);
        if constexpr (Update & 1) volume[idx] = math_traits<T>::max(volume[idx], error);
        else volume[idx] += error;
    }
}

// CUDA kernel: point-parallel thick line accumulation (2D/3D)
template <typename T, csize_t N, int Update, typename Kernel>
__global__ void accumulate_thick_lines_kernel(
    ArrayViewND<T, N + 1> volume, DeviceRange<LineData<T, N>> lines,
    DrawContextView<AccumulateIndices, T, N> context, T max_val, Kernel kernel)
{
    constexpr int TermUpdate = Update & 1;
    constexpr int GlobalUpdate = (Update >> 1) & 1;

    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= volume.size()) return;

    // Get voxel coordinate
    auto coord = volume.coord_at(idx);
    auto frame = coord[0];
    csize_t frame_idx = 0;

    csize_t bin_idx = 0;
    PointND<T, N> point;
    for (csize_t i = 0; i < N; i++)
    {
        auto x = coord[1 + i] / context.bin(i);
        if (x >= context.grid(i)) x = context.grid(i) - 1;
        bin_idx = bin_idx * context.grid(i) + x;
        frame_idx = frame_idx * volume.shape(i + 1) + coord[1 + i];

        point[N - i - 1] = coord[1 + i];
    }

    csize_t previous_term = 0;
    T term_value = T();
    for (auto index : context.indices(frame * context.grid().size() + bin_idx))
    {
        T dist = lines[index.line].distance(point);
        T width = lines[index.line].width();
        if (width <= T()) continue;

        if (index.term != previous_term)
        {
            // Commit previous term value to global volume
            if constexpr (GlobalUpdate) volume[idx] = math_traits<T>::max(volume[idx], term_value);
            else volume[idx] += term_value;

            // Start new term
            previous_term = index.term;
            term_value = T();
        }

        // Accumulate within term
        if constexpr (TermUpdate) term_value = math_traits<T>::max(term_value, max_val * kernel(dist / width));
        else term_value += max_val * kernel(dist / width);
    }

    // Commit last term value to global volume
    if constexpr (GlobalUpdate) volume[idx] = math_traits<T>::max(volume[idx], term_value);
    else volume[idx] += term_value;
}

// GPU preprocessing: build spatial index (device side)
template <typename T, csize_t N>
__global__ void count_lines(ArrayViewND<T, 2> lines, DrawContextView<DrawIndices, T, N> context, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), Counter(counts.data()));
}

template <typename T, csize_t N>
__global__ void fill_indices(ArrayViewND<T, 2> lines, DrawContextView<DrawIndices, T, N> context, DrawIndicesView draw_indices,
                             DeviceRange<csize_t> counters)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), draw_indices.filler(idx, counters.data()));
}

template <typename T, csize_t N>
DrawContext<DrawIndices, T, N> build_context(const array_view<T, py::ssize_t> & lines, const py::ssize_t * shape, py::ssize_t * grid)
{
    constexpr csize_t L = 2 * N + 1;
    DrawContext<DrawIndices, T, N> context (shape, grid);
    DeviceVector<csize_t> max_counts (context.grid().size(), 0);

    // First pass: count lines per bin
    csize_t n_lines = lines.size() / L;
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_lines + block_size - 1) / block_size;
    count_lines<T, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), context.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());

    // Second pass: scan max_counts to get offsets
    DeviceVector<csize_t> offsets (context.grid().size() + 1, 0);
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, max_counts.data(), offsets.data() + 1, context.grid().size());

    // Allocate temp buffer and run inclusive prefix sum
    DeviceVector<char> temp_storage (temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(temp_storage.data(), temp_storage_bytes, max_counts.data(), offsets.data() + 1, context.grid().size());

    // Allocating line indices
    csize_t total_indices = 0;
    handle_cuda_error(cudaMemcpy(&total_indices, offsets.data() + context.grid().size(), sizeof(csize_t), cudaMemcpyDeviceToHost));
    DrawIndices indices (DeviceVector<csize_t>(total_indices), std::move(offsets));
    max_counts.set(0);  // Reset max_counts to use as counters during filling

    // Third pass: fill in line indices
    fill_indices<T, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), context.view(), indices.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());

    context.import_indices(std::move(indices));
    return context;
}

template <typename T, typename I, csize_t N>
__global__ void count_lines(ArrayViewND<T, 2> lines, ArrayViewND<I, 1> idxs, csize_t n_frames,
                            DrawContextView<DrawIndices, T, N> context, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    csize_t frame = static_cast<csize_t>(idxs[idx]);
    if (frame >= n_frames) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), Counter(counts.data() + frame * context.grid().size()));
}

template <typename T, typename I, csize_t N>
__global__ void fill_indices(ArrayViewND<T, 2> lines, ArrayViewND<I, 1> idxs, csize_t n_frames,
                             DrawContextView<DrawIndices, T, N> context, DrawIndicesView draw_indices,
                             DeviceRange<csize_t> counters)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    csize_t frame = static_cast<csize_t>(idxs[idx]);
    if (frame >= n_frames) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), draw_indices.filler(frame * context.grid().size(), idx, counters.data()));
}

template <typename T, typename I, csize_t N>
DrawContext<DrawIndices, T, N> build_context(const array_view<T, py::ssize_t> & lines, const array_view<I, py::ssize_t> & idxs, csize_t n_frames, const py::ssize_t * shape, py::ssize_t * grid)
{
    constexpr csize_t L = 2 * N + 1;
    DrawContext<DrawIndices, T, N> context (shape, grid);
    DeviceVector<csize_t> max_counts (n_frames * context.grid().size(), 0);

    // First pass: count lines per bin
    csize_t n_lines = lines.size() / L;
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_lines + block_size - 1) / block_size;
    count_lines<T, I, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), cast_to_nd<I, 1>(idxs), n_frames, context.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Second pass: scan max_counts to get offsets
    DeviceVector<csize_t> offsets (n_frames * context.grid().size() + 1, 0);
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, max_counts.data(), offsets.data() + 1, n_frames * context.grid().size());

    // Allocate temp buffer and run inclusive prefix sum
    DeviceVector<char> temp_storage (temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(temp_storage.data(), temp_storage_bytes, max_counts.data(), offsets.data() + 1, n_frames * context.grid().size());
    handle_cuda_error(cudaDeviceSynchronize());

    // Allocating line indices
    csize_t total_indices = 0;
    handle_cuda_error(cudaMemcpy(&total_indices, offsets.data() + n_frames * context.grid().size(), sizeof(csize_t), cudaMemcpyDeviceToHost));

    DrawIndices indices (DeviceVector<csize_t>(total_indices), std::move(offsets));
    max_counts.set(0);  // Reset max_counts to use as counters during filling

    // Third pass: fill in line indices
    fill_indices<T, I, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), cast_to_nd<I, 1>(idxs), n_frames, context.view(), indices.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    context.import_indices(std::move(indices));
    return context;
}

template <typename T, typename I, csize_t N>
__global__ void count_lines(ArrayViewND<T, 2> lines, ArrayViewND<I, 1> terms, csize_t n_terms, ArrayViewND<I, 1> frames,  csize_t n_frames,
                            DrawContextView<AccumulateIndices, T, N> context, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    csize_t term = static_cast<csize_t>(terms[idx]);
    if (term >= n_terms) return;
    csize_t frame = static_cast<csize_t>(frames[term]);
    if (frame >= n_frames) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), Counter(counts.data() + (frame * context.grid().size())));
}

template <typename T, typename I, csize_t N>
__global__ void fill_indices(ArrayViewND<T, 2> lines, ArrayViewND<I, 1> terms, csize_t n_terms, ArrayViewND<I, 1> frames,  csize_t n_frames,
                             typename DrawContext<AccumulateIndices, T, N>::const_view_type context, AccumulateIndicesView acc_indices,
                             DeviceRange<csize_t> counters)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lines.shape(0)) return;

    csize_t term = static_cast<csize_t>(terms[idx]);
    if (term >= n_terms) return;
    csize_t frame = static_cast<csize_t>(frames[term]);
    if (frame >= n_frames) return;

    context.apply_to_line(lines.begin_at(idx * lines.strides(0), 1), acc_indices.filler(frame * context.grid().size(), IndexPair{idx, term}, counters.data()));
}

template <typename T, typename I, csize_t N>
DrawContext<AccumulateIndices, T, N> build_context(const array_view<T, py::ssize_t> & lines, const array_view<I, py::ssize_t> & terms, const array_view<I, py::ssize_t> & frames,
                                                   csize_t n_frames, const py::ssize_t * shape, py::ssize_t * grid)
{
    constexpr csize_t L = 2 * N + 1;
    DrawContext<AccumulateIndices, T, N> context (shape, grid);
    DeviceVector<csize_t> max_counts (n_frames * context.grid().size(), 0);

    // First pass: count lines per bin
    csize_t n_lines = lines.size() / L;
    csize_t n_terms = frames.size();
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_lines + block_size - 1) / block_size;
    count_lines<T, I, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), cast_to_nd<I, 1>(terms), n_terms, cast_to_nd<I, 1>(frames), n_frames,
                                                   context.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());

    // Second pass: scan max_counts to get offsets
    DeviceVector<csize_t> offsets (n_frames * context.grid().size() + 1, 0);
    void * d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, max_counts.data(), offsets.data() + 1, n_frames * context.grid().size());

    // Allocate temp buffer and run inclusive prefix sum
    DeviceVector<char> temp_storage (temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(temp_storage.data(), temp_storage_bytes, max_counts.data(), offsets.data() + 1, n_frames * context.grid().size());

    // Allocating line indices
    csize_t total_indices = 0;
    handle_cuda_error(cudaMemcpy(&total_indices, offsets.data() + n_frames * context.grid().size(), sizeof(csize_t), cudaMemcpyDeviceToHost));
    AccumulateIndices indices (
        DeviceVector<csize_t>(total_indices),
        DeviceVector<csize_t>(total_indices),
        std::move(offsets)
    );
    max_counts.set(0);  // Reset max_counts to use as counters during filling

    // Third pass: fill in line indices
    fill_indices<T, I, N><<<n_blocks, block_size>>>(cast_to_nd<T, 2>(lines), cast_to_nd<I, 1>(terms), n_terms, cast_to_nd<I, 1>(frames), n_frames,
                                                    context.view(), indices.view(), max_counts.view());
    handle_cuda_error(cudaGetLastError());

    // Fourth pass: sort within each bin [offsets[i], offsets[i + 1]) by term (ascending)
    if (total_indices > 1)
    {
        DeviceVector<csize_t> sorted_terms(total_indices);
        DeviceVector<csize_t> sorted_indices(total_indices);

        // Determine temp storage size
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            indices.terms().data(), sorted_terms.data(),
            indices.indices().data(), sorted_indices.data(),
            static_cast<int>(total_indices),
            static_cast<int>(n_frames * context.grid().size()),
            indices.offsets().data(),
            indices.offsets().data() + 1
        );

        // Allocate temp storage and perform sort
        temp_storage.resize(temp_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortPairs(
            temp_storage.data(), temp_storage_bytes,
            indices.terms().data(), sorted_terms.data(),
            indices.indices().data(), sorted_indices.data(),
            static_cast<int>(total_indices),
            static_cast<int>(n_frames * context.grid().size()),
            indices.offsets().data(),
            indices.offsets().data() + 1
        );

        // Copy results back in place (device-to-device)
        sorted_terms.to_device(indices.terms());
        sorted_indices.to_device(indices.indices());
    }

    context.import_indices(std::move(indices));
    return context;
}

template <csize_t N, csize_t LinesPerBin = 10>
std::array<py::ssize_t, N> make_grid(const py::ssize_t * shape, csize_t n_lines)
{
    std::array<py::ssize_t, N> grid;

    if (n_lines == 0)
    {
        // No lines, use minimal grid
        for (csize_t i = 0; i < N; i++) grid[i] = 1;
    }
    else
    {
        csize_t total_bins = std::max<csize_t>(1, n_lines / LinesPerBin);

        // Compute product of output dimensions
        double volume = 1.0;
        for (csize_t i = 0; i < N; i++) volume *= shape[i];

        // Distribute bins proportionally across dimensions
        // k = (total_bins / volume)^(1/N) such that product(grid) ≈ total_bins
        double k = std::pow(total_bins / volume, 1.0 / N);

        // grid[i] = k * shape[i], clamped to [1, shape[i]]
        for (csize_t i = 0; i < N; i++)
        {
            double bins_dim = k * shape[i];
            grid[i] = std::max<py::ssize_t>(1, std::min<py::ssize_t>(shape[i] / 4, std::lround(bins_dim)));
        }
    }

    return grid;
}

// Main drawing function in 2D
template <typename T, csize_t N, int Update, kernels::type K>
array_t<T> draw_lines_nd_no_index(array_t<T> out, array_t<T> lines, T max_val, std::optional<std::array<py::ssize_t, N>> grid)
{
    constexpr csize_t L = 2 * N + 1;
    constexpr auto kernel = kernels_t<T, cuda::kernel_traits>::template select<K>();

    if (out.ndim() != N) throw std::invalid_argument("Output array has incorrect number of dimensions");
    if (lines.shape(lines.ndim() - 1) != L) throw std::invalid_argument("Line array has incorrect shape");

    auto shape = out.shape();
    csize_t n_lines = lines.size() / L;

    // Early return for empty output or no lines
    if (out.size() == 0 || n_lines == 0) return out;

    if (lines.ndim() != 2) lines = lines.reshape({n_lines, L});
    if (!grid) grid = make_grid<N>(shape, n_lines);

    auto line_data = build_lines<T, N>(lines.view());
    auto context = build_context<T, N>(lines.view(), shape, grid->data());

    int block_size = BLOCK_SIZE;
    int num_blocks = static_cast<int>((out.size() + block_size - 1) / block_size);
    draw_thick_lines_kernel<T, N, Update><<<num_blocks, block_size>>>(
        cast_to_nd<T, N + 1>(out.view()), line_data.view(), context.view(), max_val, kernel);

    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T, typename I, csize_t N, int Update, kernels::type K>
array_t<T> draw_lines_nd_with_index(array_t<T> out, array_t<T> lines, array_t<I> idxs, T max_val, std::optional<std::array<py::ssize_t, N>> grid)
{
    constexpr csize_t L = 2 * N + 1;
    constexpr auto kernel = kernels_t<T, cuda::kernel_traits>::template select<K>();
    if (out.ndim() < N) throw std::invalid_argument("Output array has insufficient number of dimensions");
    if (lines.shape(lines.ndim() - 1) != L) throw std::invalid_argument("Line array has incorrect shape");

    if (idxs.ndim() != 1) idxs = idxs.reshape({idxs.size()});

    auto shape = out.shape() + (out.ndim() - N);
    csize_t n_lines = lines.size() / L;
    csize_t n_frames = std::reduce(out.shape(), shape, csize_t(1), std::multiplies());

    // Early return for empty output
    if (out.size() == 0 || n_lines == 0) return out;

    std::vector<py::ssize_t> old_shape;
    if (out.ndim() != N + 1)
    {
        old_shape = std::vector<py::ssize_t> (out.shape(), out.shape() + out.ndim());
        std::vector<py::ssize_t> new_shape (out.shape() + out.ndim() - N, out.shape() + out.ndim());
        new_shape.insert(new_shape.begin(), n_frames);
        out = out.reshape(new_shape);
    }

    if (lines.ndim() != 2) lines = lines.reshape({n_lines, L});
    if (!grid)
    {
        csize_t lines_per_frame = n_lines / n_frames + ((n_lines % n_frames) ? 1 : 0);
        grid = make_grid<N>(shape, lines_per_frame);
    }

    auto line_data = build_lines<T, N>(lines.view());
    auto context = build_context<T, I, N>(lines.view(), idxs.view(), n_frames, shape, grid->data());

    int block_size = BLOCK_SIZE;
    int num_blocks = static_cast<int>((out.size() + block_size - 1) / block_size);

    draw_thick_lines_kernel<T, N, Update><<<num_blocks, block_size>>>(
        cast_to_nd<T, N + 1>(out.view()), line_data.view(), context.view(), max_val, kernel);

    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    if (old_shape.size()) out = out.reshape(old_shape);
    return out;
}

template <typename T, typename I, csize_t N, int Update, kernels::type K>
array_t<T> draw_lines_nd_impl(array_t<T> out, array_t<T> lines, std::optional<array_t<I>> idxs, T max_val, std::optional<std::array<py::ssize_t, N>> grid)
{
    if (idxs) return draw_lines_nd_with_index<T, I, N, Update, K>(out, lines, *idxs, max_val, grid);
    else return draw_lines_nd_no_index<T, N, Update, K>(out, lines, max_val, grid);
}

template <typename T, typename I, csize_t N, int Update>
array_t<T> draw_lines_nd(array_t<T> out, array_t<T> lines, std::optional<array_t<I>> idxs, T max_val, std::string kernel_name, std::optional<std::array<py::ssize_t, N>> grid)
{
    auto ktype = kernels::get_type(kernel_name);
    switch (ktype)
    {
        case kernels::biweight:
            return draw_lines_nd_impl<T, I, N, Update, kernels::biweight>(out, lines, idxs, max_val, grid);
        case kernels::gaussian:
            return draw_lines_nd_impl<T, I, N, Update, kernels::gaussian>(out, lines, idxs, max_val, grid);
        case kernels::parabolic:
            return draw_lines_nd_impl<T, I, N, Update, kernels::parabolic>(out, lines, idxs, max_val, grid);
        case kernels::rectangular:
            return draw_lines_nd_impl<T, I, N, Update, kernels::rectangular>(out, lines, idxs, max_val, grid);
        case kernels::triangular:
            return draw_lines_nd_impl<T, I, N, Update, kernels::triangular>(out, lines, idxs, max_val, grid);
        default:
            throw std::invalid_argument("Invalid kernel type");
    }
}

template <typename T, typename I, int Update>
array_t<T> draw_lines_2d_3d(array_t<T> out, array_t<T> lines, std::optional<array_t<I>> idxs, T max_val, std::string kernel, std::optional<std::vector<py::ssize_t>> grid)
{
    size_t L = lines.shape(lines.ndim() - 1);
    if (out.ndim() >= 2 && L == 5)
    {
        if (grid)
        {
            if (grid->size() != 2) throw std::invalid_argument("Grid size must match output array dimensions");
            return draw_lines_nd<T, I, 2, Update>(out, lines, idxs, max_val, kernel, std::array<py::ssize_t, 2>{(*grid)[0], (*grid)[1]});
        }
        return draw_lines_nd<T, I, 2, Update>(out, lines, idxs, max_val, kernel, std::nullopt);
    }
    else if (out.ndim() >= 3 && L == 7)
    {
        if (grid)
        {
            if (grid->size() != 3) throw std::invalid_argument("Grid size must match output array dimensions");
            return draw_lines_nd<T, I, 3, Update>(out, lines, idxs, max_val, kernel, std::array<py::ssize_t, 3>{(*grid)[0], (*grid)[1], (*grid)[2]});
        }
        return draw_lines_nd<T, I, 3, Update>(out, lines, idxs, max_val, kernel, std::nullopt);
    }
    else
    {
        throw std::invalid_argument("Output array dimensions (" + std::to_string(out.ndim()) + ") do not match the line size (" + std::to_string(L) + ")");
    }
}

template <typename T, typename I>
array_t<T> draw_lines(array_t<T> out, array_t<T> lines, std::optional<array_t<I>> idxs, T max_val, std::string kernel, std::string overlap, std::optional<std::vector<py::ssize_t>> grid)
{
    if (overlap == "sum") return draw_lines_2d_3d<T, I, 0>(out, lines, idxs, max_val, kernel, grid);
    else if (overlap == "max") return draw_lines_2d_3d<T, I, 1>(out, lines, idxs, max_val, kernel, grid);
    else throw std::invalid_argument("Invalid overlap keyword: " + overlap);
}

template <typename T, typename I, csize_t N, int Update, kernels::type K>
array_t<T> accumulate_lines_nd_impl(array_t<T> out, array_t<T> lines, array_t<I> terms, array_t<I> frames, T max_val, std::optional<std::array<py::ssize_t, N>> grid)
{
    constexpr csize_t L = 2 * N + 1;
    constexpr auto kernel = kernels_t<T, cuda::kernel_traits>::template select<K>();

    if (out.ndim() < N) throw std::invalid_argument("Output array has incorrect number of dimensions");
    if (lines.shape(lines.ndim() - 1) != L) throw std::invalid_argument("Line array has incorrect shape");

    if (frames.ndim() != 1) frames = frames.reshape({frames.size()});
    if (terms.ndim() != 1) terms = terms.reshape({terms.size()});

    auto shape = out.shape() + out.ndim() - N;
    csize_t n_lines = lines.size() / lines.shape(lines.ndim() - 1);
    csize_t n_frames = std::reduce(out.shape(), shape, csize_t(1), std::multiplies());

    // Early return for empty output
    if (out.size() == 0 || n_lines == 0) return out;

    if (terms.size() != n_lines) throw std::invalid_argument("Number of term indices does not match number of lines");
    std::vector<py::ssize_t> old_shape;
    if (out.ndim() != N + 1)
    {
        old_shape = std::vector<py::ssize_t> (out.shape(), out.shape() + out.ndim());
        std::vector<py::ssize_t> new_shape (out.shape() + out.ndim() - N, out.shape() + out.ndim());
        new_shape.insert(new_shape.begin(), n_frames);
        out = out.reshape(new_shape);
    }

    if (lines.ndim() != 2) lines = lines.reshape({n_lines, L});
    if (!grid)
    {
        csize_t lines_per_frame = n_lines / n_frames + ((n_lines % n_frames) ? 1 : 0);
        grid = make_grid<N>(shape, lines_per_frame);
    }

    auto line_data = build_lines<T, N>(lines.view());
    auto context = build_context<T, I, N>(lines.view(), terms.view(), frames.view(), n_frames, shape, grid->data());

    int block_size = BLOCK_SIZE;
    int num_blocks = static_cast<int>((out.size() + block_size - 1) / block_size);

    accumulate_thick_lines_kernel<T, N, Update><<<num_blocks, block_size>>>(
        cast_to_nd<T, N + 1>(out.view()), line_data.view(), context.view(), max_val, kernel);

    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    if (old_shape.size()) out = out.reshape(old_shape);
    return out;
}

template <typename T, typename I, csize_t N, int Update>
array_t<T> accumulate_lines_nd(array_t<T> out, array_t<T> larr, array_t<I> terms, array_t<I> frames, T max_val, std::string kernel_name, std::optional<std::array<py::ssize_t, N>> grid)
{
    auto ktype = kernels::get_type(kernel_name);
    switch (ktype)
    {
        case kernels::biweight:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::biweight>(out, larr, terms, frames, max_val, grid);
        case kernels::gaussian:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::gaussian>(out, larr, terms, frames, max_val, grid);
        case kernels::parabolic:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::parabolic>(out, larr, terms, frames, max_val, grid);
        case kernels::rectangular:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::rectangular>(out, larr, terms, frames, max_val, grid);
        case kernels::triangular:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::triangular>(out, larr, terms, frames, max_val, grid);
        default:
            throw std::invalid_argument("Invalid kernel type");
    }
}

template <typename T, typename I, int Update>
array_t<T> accumulate_lines_2d_3d(array_t<T> out, array_t<T> larr, array_t<I> terms, array_t<I> frames, T max_val, std::string kernel, std::optional<std::vector<py::ssize_t>> grid)
{
    size_t L = larr.shape(larr.ndim() - 1);
    if (out.ndim() >= 2 && L == 5)
    {
        if (grid)
        {
            if (grid->size() != 2) throw std::invalid_argument("Grid size must match output array dimensions");
            return accumulate_lines_nd<T, I, 2, Update>(out, larr, terms, frames, max_val, kernel, std::array<py::ssize_t, 2>{(*grid)[0], (*grid)[1]});
        }
        return accumulate_lines_nd<T, I, 2, Update>(out, larr, terms, frames, max_val, kernel, std::nullopt);
    }
    else if (out.ndim() >= 3 && L == 7)
    {
        if (grid)
        {
            if (grid->size() != 3) throw std::invalid_argument("Grid size must match output array dimensions");
            return accumulate_lines_nd<T, I, 3, Update>(out, larr, terms, frames, max_val, kernel, std::array<py::ssize_t, 3>{(*grid)[0], (*grid)[1], (*grid)[2]});
        }
        return accumulate_lines_nd<T, I, 3, Update>(out, larr, terms, frames, max_val, kernel, std::nullopt);
    }
    else
    {
        throw std::invalid_argument("Output array dimensions (" + std::to_string(out.ndim()) + ") do not match the line size (" + std::to_string(L) + ")");
    }
}

template <typename T, typename I>
array_t<T> accumulate_lines(array_t<T> out, array_t<T> larr, array_t<I> terms, array_t<I> frames, T max_val, std::string kernel, std::string in_overlap, std::string out_overlap, std::optional<std::vector<py::ssize_t>> grid)
{
    if (in_overlap =="sum" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 0>(out, larr, terms, frames, max_val, kernel, grid);
    if (in_overlap =="max" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 1>(out, larr, terms, frames, max_val, kernel, grid);
    if (in_overlap =="sum" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 2>(out, larr, terms, frames, max_val, kernel, grid);
    if (in_overlap =="max" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 3>(out, larr, terms, frames, max_val, kernel, grid);
    throw std::invalid_argument("Invalid overlap keyword: " + in_overlap + ", " + out_overlap);
}

#undef HOST_DEVICE

} // namespace cbclib

PYBIND11_MODULE(cuda_draw_lines, m)
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

    m.def("accumulate_lines", &cu::accumulate_lines<float, int>,
          py::arg("out"),
          py::arg("lines"),
          py::arg("terms"),
          py::arg("frames"),
          py::arg("max_val") = 1.0f,
          py::arg("kernel") = "rectangular",
          py::arg("in_overlap") = "sum",
          py::arg("out_overlap") = "sum",
          py::arg("grid") = nullptr);
    m.def("accumulate_lines", &cu::accumulate_lines<double, long>,
          py::arg("out"),
          py::arg("lines"),
          py::arg("terms"),
          py::arg("frames"),
          py::arg("max_val") = 1.0,
          py::arg("kernel") = "rectangular",
          py::arg("in_overlap") = "sum",
          py::arg("out_overlap") = "sum",
          py::arg("grid") = nullptr);

    m.def("draw_lines", &cu::draw_lines<float, int>,
          py::arg("out"),
          py::arg("lines"),
          py::arg("idxs") = nullptr,
          py::arg("max_val") = 1.0f,
          py::arg("kernel") = "rectangular",
          py::arg("overlap") = "sum",
          py::arg("grid") = nullptr);
    m.def("draw_lines", &cu::draw_lines<double, long>,
          py::arg("out"),
          py::arg("lines"),
          py::arg("idxs") = nullptr,
          py::arg("max_val") = 1.0,
          py::arg("kernel") = "rectangular",
          py::arg("overlap") = "sum",
          py::arg("grid") = nullptr);
}
