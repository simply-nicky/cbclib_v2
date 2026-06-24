#include "cupy_array.hpp"
#include "online_detector.hpp"

namespace cbclib::cuda {

template <typename R, typename I>
struct CudaPanelGeometry
{
    I offset = 0;
    PointND<I, 4> bounds;   // fs_min, fs_max, ss_min, ss_max
    PointND<I, 2> shape;    // ss_size, fs_size
    PointND<I, 2> stride;   // ss_stride, fs_stride
    PointND<R, 2> corner;   // x, y
    PointND<R, 3> ss;       // ss_x, ss_y, ss_z
    PointND<R, 3> fs;       // fs_x, fs_y, fs_z

    HOST_DEVICE I size() const
    {
        return shape[0] * shape[1];
    }

    HOST_DEVICE R lab_x(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return corner[0] + shift + ss_index * ss[0] + fs_index * fs[0];
    }

    HOST_DEVICE R lab_y(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return corner[1] + shift + ss_index * ss[1] + fs_index * fs[1];
    }

    HOST_DEVICE R lab_z(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return shift + ss_index * ss[2] + fs_index * fs[2];
    }
};

template <typename R, typename I>
struct CudaDetectorView
{
    DeviceRange<CudaPanelGeometry<R, I>> panels;
    DeviceRange<I> panel_offsets;
    PointND<I, 3> shape;
    PointND<R, 4> bounds;   // x_min, x_max, y_min, y_max
    I ndims = 2;
    bool half_pixel_shift = true;

    HOST_DEVICE I ndim() const
    {
        return ndims;
    }

    HOST_DEVICE I size() const
    {
        if (ndims == 2)
        {
            return shape[0] * shape[1];
        }
        return shape[0] * shape[1] * shape[2];
    }

    HOST_DEVICE I panel_size() const
    {
        return panel_offsets[panel_offsets.size() - 1];
    }

    HOST_DEVICE R radius(const CudaPanelGeometry<R, I> & panel,
                         const PointND<R, 2> & center, I ss_index, I fs_index) const
    {
        R dx = panel.lab_x(ss_index, fs_index, half_pixel_shift) - bounds[0] - center[0];
        R dy = panel.lab_y(ss_index, fs_index, half_pixel_shift) - bounds[2] - center[1];
        return math_traits<R>::sqrt(dx * dx + dy * dy);
    }
};

template <typename R, typename I>
class CudaDetectorGeometry
{
public:
    CudaDetectorGeometry(const cbclib::DetectorGeometry<R, I> & geometry,
                         bool half_pixel_shift)
    {
        std::vector<CudaPanelGeometry<R, I>> panels;
        panels.reserve(geometry.panels.size());
        for (const auto & panel : geometry.panels)
        {
            CudaPanelGeometry<R, I> cuda_panel;
            cuda_panel.offset = panel.offset;
            cuda_panel.bounds = PointND<I, 4>(panel.bounds[0], panel.bounds[1],
                                              panel.bounds[2], panel.bounds[3]);
            cuda_panel.shape = PointND<I, 2>(panel.shape[0], panel.shape[1]);
            cuda_panel.stride = PointND<I, 2>(panel.stride[0], panel.stride[1]);
            cuda_panel.corner = PointND<R, 2>(panel.corner[0], panel.corner[1]);
            cuda_panel.ss = PointND<R, 3>(panel.ss[0], panel.ss[1], panel.ss[2]);
            cuda_panel.fs = PointND<R, 3>(panel.fs[0], panel.fs[1], panel.fs[2]);
            panels.push_back(cuda_panel);
        }

        m_panels = DeviceVector<CudaPanelGeometry<R, I>>::from_host(panels.data(),
                                                                    panels.size());
        m_panel_offsets = DeviceVector<I>::from_host(geometry.panel_offsets.data(),
                                                     geometry.panel_offsets.size());

        m_ndims = static_cast<I>(geometry.ndim());
        if (geometry.ndim() == 2)
        {
            m_shape = PointND<I, 3>(geometry.shape[0], geometry.shape[1], I(1));
        }
        else
        {
            m_shape = PointND<I, 3>(geometry.shape[0], geometry.shape[1], geometry.shape[2]);
        }
        m_bounds = PointND<R, 4>(geometry.bounds[0], geometry.bounds[1],
                                 geometry.bounds[2], geometry.bounds[3]);
        m_half_pixel_shift = half_pixel_shift;
    }

    CudaDetectorView<R, I> view()
    {
        return {m_panels.view(), m_panel_offsets.view(), m_shape, m_bounds, m_ndims,
                m_half_pixel_shift};
    }

private:
    DeviceVector<CudaPanelGeometry<R, I>> m_panels;
    DeviceVector<I> m_panel_offsets;
    PointND<I, 3> m_shape;
    PointND<R, 4> m_bounds;
    I m_ndims = 2;
    bool m_half_pixel_shift = true;
};

template <typename R, typename I>
struct PanelPoint
{
    CudaPanelGeometry<R, I> panel;
    I ss;
    I fs;
    bool valid;
};

template <typename R, typename I>
__device__ PanelPoint<R, I> panel_point_at(const CudaDetectorView<R, I> & geometry,
                                           I frame_index)
{
    I module = 0;
    I y = 0;
    I x = 0;
    if (geometry.ndims == 2)
    {
        y = frame_index / geometry.shape[1];
        x = frame_index - y * geometry.shape[1];
    }
    else
    {
        I module_size = geometry.shape[1] * geometry.shape[2];
        module = frame_index / module_size;
        I module_index = frame_index - module * module_size;
        y = module_index / geometry.shape[2];
        x = module_index - y * geometry.shape[2];
    }

    for (I panel_index = 0; panel_index < geometry.panels.size(); ++panel_index)
    {
        const auto & panel = geometry.panels[panel_index];
        I panel_module = geometry.ndims == 3 ? panel.offset / (geometry.shape[1] *
                         geometry.shape[2]) : I();
        if ((geometry.ndims == 2 || module == panel_module) &&
            x >= panel.bounds[0] && x <= panel.bounds[1] &&
            y >= panel.bounds[2] && y <= panel.bounds[3])
        {
            return {panel, y - panel.bounds[2], x - panel.bounds[0], true};
        }
    }
    return {CudaPanelGeometry<R, I>(), I(), I(), false};
}

template <typename R, typename I, csize_t N>
__global__ void pixel_map_kernel(ArrayViewND<R, N> out, CudaDetectorView<R, I> geometry)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    I frame_size = geometry.size();
    I output_size = out.size();
    if (index >= output_size) return;

    I channel = index / frame_size;
    I frame_index = index - channel * frame_size;
    auto point = panel_point_at(geometry, frame_index);
    if (!point.valid) return;

    if (channel == 0)
        out[index] = point.panel.lab_x(point.ss, point.fs, geometry.half_pixel_shift);
    else if (channel == 1)
        out[index] = point.panel.lab_y(point.ss, point.fs, geometry.half_pixel_shift);
    else
        out[index] = point.panel.lab_z(point.ss, point.fs, geometry.half_pixel_shift);
}

template <typename R, typename I, csize_t N>
__global__ void radius_kernel(ArrayViewND<R, N> out, CudaDetectorView<R, I> geometry,
                              PointND<R, 2> center)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out.size()) return;

    auto point = panel_point_at(geometry, index);
    if (!point.valid) return;

    out[index] = geometry.radius(point.panel, center, point.ss, point.fs);
}

template <typename R, typename I, csize_t N>
__global__ void radial_index_kernel(ArrayViewND<I, N> out, CudaDetectorView<R, I> geometry,
                                    PointND<R, 2> center, R inv_radius_step, I n_bins)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= out.size()) return;

    auto point = panel_point_at(geometry, index);
    if (!point.valid) return;

    I bin = static_cast<I>(math_traits<R>::floor(
        geometry.radius(point.panel, center, point.ss, point.fs) * inv_radius_step + R(0.5)
    ));
    out[index] = bin >= 0 && bin < n_bins ? bin : I(-1);
}

template <typename I>
__device__ void atomic_add_count(I * count)
{
    if constexpr (std::is_same_v<I, int>)
    {
        atomicAdd(count, 1);
    }
    else if constexpr (std::is_same_v<I, unsigned int>)
    {
        atomicAdd(count, 1u);
    }
    else if constexpr (sizeof(I) == sizeof(unsigned long long))
    {
        atomicAdd(reinterpret_cast<unsigned long long *>(count), 1ull);
    }
}

template <typename T, typename R, typename I, csize_t N>
__global__ void accumulate_kernel(ArrayViewND<T, N> data, ArrayViewND<I, N> radial_index,
                                  DeviceRange<R> sum, DeviceRange<R> sumsq,
                                  DeviceRange<I> counts,
                                  ArrayViewND<R, 2> whitefield, ArrayViewND<R, 2> std,
                                  I frame_size, I n_bins, I interval, R clip_snr, R std_min,
                                  bool clip)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data.size() || index % interval != 0) return;

    I frame_index = index % frame_size;
    I bin = radial_index[frame_index];
    if (bin < 0 || bin >= n_bins) return;

    I frame = index / frame_size;
    R value = static_cast<R>(data[index]);

    if (clip)
    {
        R sigma = math_traits<R>::max(std.at(frame, bin), std_min);
        R threshold = whitefield.at(frame, bin) + clip_snr * sigma;
        if (value > threshold) return;
    }

    I profile_index = frame * n_bins + bin;
    atomicAdd(sum.data(profile_index), value);
    atomicAdd(sumsq.data(profile_index), value * value);
    atomic_add_count(counts.data(profile_index));
}

template <typename R, typename I>
__global__ void finalize_kernel(ArrayViewND<R, 2> whitefield, ArrayViewND<R, 2> std,
                                ArrayViewND<I, 2> out_counts, DeviceRange<R> sum,
                                DeviceRange<R> sumsq, DeviceRange<I> counts)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= whitefield.size()) return;

    I count = counts[index];
    out_counts[index] = count;
    if (count > 0)
    {
        R mean = sum[index] / static_cast<R>(count);
        R var = sumsq[index] / static_cast<R>(count) - mean * mean;
        whitefield[index] = mean;
        std[index] = math_traits<R>::sqrt(math_traits<R>::max(var, R()));
    }
    else
    {
        whitefield[index] = R();
        std[index] = R();
    }
}

template <typename T, typename R, typename I, csize_t N>
__global__ void is_signal_kernel(ArrayViewND<bool, N> out, ArrayViewND<T, N> data,
                                 ArrayViewND<R, 2> whitefield, ArrayViewND<R, 2> std,
                                 ArrayViewND<I, N> radial_index, I frame_size, I n_bins,
                                 R min_snr, R std_min)
{
    I index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data.size()) return;

    I frame_index = index % frame_size;
    I bin = radial_index[frame_index];
    if (bin < 0 || bin >= n_bins)
    {
        out[index] = false;
        return;
    }

    I frame = index / frame_size;
    R sigma = math_traits<R>::max(std.at(frame, bin), std_min);
    out[index] = static_cast<R>(data[index]) > whitefield.at(frame, bin) + min_snr * sigma;
}

template <typename T, typename R, typename I>
bool frame_shape_matches(const array_t<T> & out, const cbclib::DetectorGeometry<R, I> & geometry)
{
    if (out.ndim() != geometry.ndim())
    {
        return false;
    }
    for (I dim = 0; dim < geometry.ndim(); ++dim)
    {
        if (out.shape(dim) != geometry.shape[dim])
        {
            return false;
        }
    }
    return true;
}

template <typename R, typename I>
bool pixel_map_shape_matches(const array_t<R> & out,
                             const cbclib::DetectorGeometry<R, I> & geometry)
{
    if (out.ndim() != geometry.ndim() + 1 || out.shape(0) != 3)
    {
        return false;
    }
    for (I dim = 0; dim < geometry.ndim(); ++dim)
    {
        if (out.shape(dim + 1) != geometry.shape[dim])
        {
            return false;
        }
    }
    return true;
}

template <typename R, typename I, csize_t N>
array_t<R> pixel_map_nd(array_t<R> out, const CudaDetectorView<R, I> & geometry)
{
    I output_size = static_cast<I>(out.size());
    int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pixel_map_kernel<R, I, N><<<num_blocks, BLOCK_SIZE>>>(cast_to_nd<R, N>(out.view()),
                                                          geometry);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());
    return out;
}

template <typename R, typename I>
array_t<R> pixel_map(array_t<R> out, cbclib::PyDetectorGeometry py_geometry,
                     bool half_pixel_shift)
{
    auto geometry = cbclib::cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();
    if (!pixel_map_shape_matches(out, geometry))
    {
        throw std::invalid_argument("pixel_map output shape mismatch");
    }

    CudaDetectorGeometry<R, I> cuda_geometry(geometry, half_pixel_shift);
    if (geometry.ndim() == 2)
    {
        return pixel_map_nd<R, I, 3>(out, cuda_geometry.view());
    }
    return pixel_map_nd<R, I, 4>(out, cuda_geometry.view());
}

template <typename R, typename I, csize_t N>
array_t<R> radius_nd(array_t<R> out, const CudaDetectorView<R, I> & geometry,
                     PointND<R, 2> center)
{
    I output_size = static_cast<I>(out.size());
    int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    radius_kernel<R, I, N><<<num_blocks, BLOCK_SIZE>>>(cast_to_nd<R, N>(out.view()),
                                                       geometry, center);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());
    return out;
}

template <typename R, typename I>
array_t<R> radius(array_t<R> out, cbclib::PyDetectorGeometry py_geometry,
                  std::tuple<py::ssize_t, py::ssize_t> center, bool half_pixel_shift)
{
    auto geometry = cbclib::cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();
    if (!frame_shape_matches(out, geometry))
    {
        throw std::invalid_argument("radius output shape mismatch");
    }

    CudaDetectorGeometry<R, I> cuda_geometry(geometry, half_pixel_shift);
    PointND<R, 2> center_point(std::get<0>(center), std::get<1>(center));
    if (geometry.ndim() == 2)
    {
        return radius_nd<R, I, 2>(out, cuda_geometry.view(), center_point);
    }
    return radius_nd<R, I, 3>(out, cuda_geometry.view(), center_point);
}

template <typename R, typename I, csize_t N>
array_t<I> radial_index_nd(array_t<I> out, const CudaDetectorView<R, I> & geometry,
                           PointND<R, 2> center, R inv_radius_step, I n_bins)
{
    I output_size = static_cast<I>(out.size());
    int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    radial_index_kernel<R, I, N><<<num_blocks, BLOCK_SIZE>>>(cast_to_nd<I, N>(out.view()),
                                                             geometry, center,
                                                             inv_radius_step, n_bins);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());
    return out;
}

template <typename R, typename I>
array_t<I> radial_index(array_t<I> out, cbclib::PyDetectorGeometry py_geometry,
                        std::tuple<py::ssize_t, py::ssize_t> center, I n_bins,
                        bool half_pixel_shift)
{
    auto geometry = cbclib::cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();
    if (n_bins <= 1) throw std::invalid_argument("n_bins must be greater than 1");
    if (!frame_shape_matches(out, geometry))
    {
        throw std::invalid_argument("radial_index output shape mismatch");
    }

    R max_radius = geometry.max_radius(center);
    if (max_radius <= R()) throw std::invalid_argument("max radius must be positive");

    CudaDetectorGeometry<R, I> cuda_geometry(geometry, half_pixel_shift);
    PointND<R, 2> center_point(std::get<0>(center), std::get<1>(center));
    R inv_radius_step = (n_bins - 1) / max_radius;
    if (geometry.ndim() == 2)
    {
        return radial_index_nd<R, I, 2>(out, cuda_geometry.view(), center_point,
                                        inv_radius_step, n_bins);
    }
    return radial_index_nd<R, I, 3>(out, cuda_geometry.view(), center_point,
                                    inv_radius_step, n_bins);
}

template <typename T, typename R, typename I, csize_t N>
std::tuple<array_t<R>, array_t<R>, array_t<I>> radial_profiles_nd(
    array_t<R> whitefield, array_t<R> std, array_t<I> counts, array_t<T> data,
    array_t<I> radial_index, I n_bins, I interval, R clip_snr, I n_iter, R std_min
)
{
    I frame_size = static_cast<I>(radial_index.size());
    I n_frames = static_cast<I>(data.size()) / frame_size;
    I profile_size = n_frames * n_bins;

    DeviceVector<R> sum(profile_size);
    DeviceVector<R> sumsq(profile_size);
    DeviceVector<I> local_counts(profile_size);

    auto sum_view = sum.view();
    auto sumsq_view = sumsq.view();
    auto local_counts_view = local_counts.view();
    auto whitefield_view = cast_to_nd<R, 2>(whitefield.view());
    auto std_view = cast_to_nd<R, 2>(std.view());
    auto counts_view = cast_to_nd<I, 2>(counts.view());
    auto data_view = cast_to_nd<T, N>(data.view());
    auto radial_index_view = cast_to_nd<I, N>(radial_index.view());

    int data_blocks = (static_cast<I>(data.size()) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int profile_blocks = (profile_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto run_accumulation = [&](bool clip)
    {
        sum.fill(R());
        sumsq.fill(R());
        local_counts.fill(I());
        accumulate_kernel<T, R, I, N><<<data_blocks, BLOCK_SIZE>>>(
            data_view, radial_index_view, sum_view, sumsq_view, local_counts_view,
            whitefield_view, std_view, frame_size, n_bins, interval, clip_snr, std_min, clip
        );
        handle_cuda_error(cudaGetLastError());
        finalize_kernel<R, I><<<profile_blocks, BLOCK_SIZE>>>(
            whitefield_view, std_view, counts_view, sum_view, sumsq_view, local_counts_view
        );
        handle_cuda_error(cudaGetLastError());
    };

    run_accumulation(false);
    for (I iter = 0; iter < n_iter; ++iter)
    {
        run_accumulation(true);
    }

    handle_cuda_error(cudaDeviceSynchronize());
    return std::make_tuple(whitefield, std, counts);
}

template <typename T, typename R, typename I>
std::tuple<array_t<R>, array_t<R>, array_t<I>> radial_profiles(
    array_t<R> whitefield, array_t<R> std, array_t<I> counts, array_t<T> data,
    array_t<I> radial_index, py::ssize_t n_bins, py::ssize_t interval, double clip_snr, py::ssize_t n_iter, double std_min
)
{
    if (n_bins <= 1) throw std::invalid_argument("n_bins must be greater than 1");
    if (n_iter < 0) throw std::invalid_argument("n_iter must be non-negative");
    if (interval <= 0) throw std::invalid_argument("interval must be positive");
    if (data.ndim() < radial_index.ndim())
    {
        throw std::invalid_argument("data ndim must be at least radial_index ndim");
    }
    for (I dim = 0; dim < radial_index.ndim(); ++dim)
    {
        I data_dim = data.ndim() - radial_index.ndim() + dim;
        if (data.shape(data_dim) != radial_index.shape(dim))
            throw std::invalid_argument("data trailing shape must match radial_index shape");
    }
    if (data.size() % radial_index.size() != 0)
    {
        throw std::invalid_argument("data size must be divisible by radial_index size");
    }

    I frame_size = static_cast<I>(radial_index.size());
    I n_frames = static_cast<I>(data.size()) / frame_size;
    I profile_size = n_frames * n_bins;
    if (whitefield.ndim() != 2 || std.ndim() != 2 || counts.ndim() != 2 ||
        whitefield.size() != profile_size || std.size() != profile_size ||
        counts.size() != profile_size)
    {
        throw std::invalid_argument("profile output shape mismatch");
    }

    switch (data.ndim())
    {
        case 1: return radial_profiles_nd<T, R, I, 1>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 2: return radial_profiles_nd<T, R, I, 2>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 3: return radial_profiles_nd<T, R, I, 3>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 4: return radial_profiles_nd<T, R, I, 4>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 5: return radial_profiles_nd<T, R, I, 5>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 6: return radial_profiles_nd<T, R, I, 6>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        case 7: return radial_profiles_nd<T, R, I, 7>(whitefield, std, counts, data, radial_index, n_bins, interval, clip_snr, n_iter, std_min);
        default: throw std::runtime_error("Unsupported number of data dimensions: " + std::to_string(data.ndim()));
    }
}

template <typename T, typename R, typename I, csize_t N>
array_t<bool> is_signal_nd(array_t<bool> out, array_t<T> data, array_t<R> whitefield,
                           array_t<R> std, array_t<I> radial_index, R min_snr, R std_min)
{
    I frame_size = static_cast<I>(radial_index.size());
    I n_bins = static_cast<I>(whitefield.shape(whitefield.ndim() - 1));
    int num_blocks = (static_cast<I>(data.size()) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    is_signal_kernel<T, R, I, N><<<num_blocks, BLOCK_SIZE>>>(
        cast_to_nd<bool, N>(out.view()), cast_to_nd<T, N>(data.view()),
        cast_to_nd<R, 2>(whitefield.view()), cast_to_nd<R, 2>(std.view()),
        cast_to_nd<I, N>(radial_index.view()), frame_size, n_bins, min_snr, std_min
    );
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());
    return out;
}

template <typename T, typename R, typename I>
array_t<bool> is_signal(array_t<bool> out, array_t<T> data, array_t<R> whitefield,
                        array_t<R> std, array_t<I> radial_index, double min_snr, double std_min)
{
    if (whitefield.size() != std.size())
    {
        throw std::invalid_argument("whitefield and std size mismatch");
    }
    if (data.ndim() < radial_index.ndim())
    {
        throw std::invalid_argument("data ndim must be at least radial_index ndim");
    }
    for (I dim = 0; dim < radial_index.ndim(); ++dim)
    {
        I data_dim = data.ndim() - radial_index.ndim() + dim;
        if (data.shape(data_dim) != radial_index.shape(dim))
            throw std::invalid_argument("data trailing shape must match radial_index shape");
    }
    if (data.size() % radial_index.size() != 0)
    {
        throw std::invalid_argument("data size must be divisible by radial_index size");
    }
    if (out.size() != data.size())
    {
        throw std::invalid_argument("is_signal output shape mismatch");
    }

    switch (data.ndim())
    {
        case 1: return is_signal_nd<T, R, I, 1>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 2: return is_signal_nd<T, R, I, 2>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 3: return is_signal_nd<T, R, I, 3>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 4: return is_signal_nd<T, R, I, 4>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 5: return is_signal_nd<T, R, I, 5>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 6: return is_signal_nd<T, R, I, 6>(out, data, whitefield, std, radial_index, min_snr, std_min);
        case 7: return is_signal_nd<T, R, I, 7>(out, data, whitefield, std, radial_index, min_snr, std_min);
        default: throw std::runtime_error("Unsupported number of data dimensions: " + std::to_string(data.ndim()));
    }
}

} // namespace cbclib::cuda

PYBIND11_MODULE(cuda_online_detector, m)
{
    using namespace cbclib;
    namespace cu = cbclib::cuda;
    py::options options;
    options.disable_function_signatures();

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("pixel_map", &cu::pixel_map<double, py::ssize_t>, py::arg("out"),
          py::arg("geometry"), py::arg("half_pixel_shift") = true);
    m.def("pixel_map", &cu::pixel_map<float, int>, py::arg("out"),
          py::arg("geometry"), py::arg("half_pixel_shift") = true);

    m.def("radius", &cu::radius<double, py::ssize_t>, py::arg("out"), py::arg("geometry"),
          py::arg("center"), py::arg("half_pixel_shift") = true);
    m.def("radius", &cu::radius<float, int>, py::arg("out"), py::arg("geometry"),
          py::arg("center"), py::arg("half_pixel_shift") = true);

    m.def("radial_index", &cu::radial_index<double, py::ssize_t>, py::arg("out"),
          py::arg("geometry"), py::arg("center"), py::arg("n_bins"),
          py::arg("half_pixel_shift") = true);
    m.def("radial_index", &cu::radial_index<float, int>, py::arg("out"),
          py::arg("geometry"), py::arg("center"), py::arg("n_bins"),
          py::arg("half_pixel_shift") = true);

    m.def("radial_profiles", &cu::radial_profiles<double, double, py::ssize_t>,
          py::arg("whitefield"), py::arg("std"), py::arg("counts"), py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0);
    m.def("radial_profiles", &cu::radial_profiles<float, float, int>,
          py::arg("whitefield"), py::arg("std"), py::arg("counts"), py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0f, py::arg("n_iter") = 3, py::arg("std_min") = 0.0f);
    m.def("radial_profiles", &cu::radial_profiles<int, float, int>,
          py::arg("whitefield"), py::arg("std"), py::arg("counts"), py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0f, py::arg("n_iter") = 3, py::arg("std_min") = 0.0f);
    m.def("radial_profiles", &cu::radial_profiles<unsigned int, float, int>,
          py::arg("whitefield"), py::arg("std"), py::arg("counts"), py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0f, py::arg("n_iter") = 3, py::arg("std_min") = 0.0f);
    m.def("radial_profiles", &cu::radial_profiles<py::ssize_t, double, py::ssize_t>,
          py::arg("whitefield"), py::arg("std"), py::arg("counts"), py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0);

    m.def("is_signal", &cu::is_signal<double, double, py::ssize_t>, py::arg("out"),
          py::arg("data"), py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0);
    m.def("is_signal", &cu::is_signal<float, float, int>, py::arg("out"),
          py::arg("data"), py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0f, py::arg("std_min") = 0.0f);
    m.def("is_signal", &cu::is_signal<int, float, int>, py::arg("out"),
          py::arg("data"), py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0f, py::arg("std_min") = 0.0f);
    m.def("is_signal", &cu::is_signal<unsigned int, float, int>, py::arg("out"),
          py::arg("data"), py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0f, py::arg("std_min") = 0.0f);
    m.def("is_signal", &cu::is_signal<py::ssize_t, double, py::ssize_t>, py::arg("out"),
          py::arg("data"), py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0);
}
