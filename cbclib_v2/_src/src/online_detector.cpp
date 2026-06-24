#include "online_detector.hpp"
#include <algorithm>
#include <cmath>

namespace cbclib {

template <typename R, typename I>
using ProfileResult = std::tuple<py::array_t<R>, py::array_t<R>, py::array_t<I>>;

template <typename R, typename I>
struct Profiles
{
    array<R> whitefield;
    array<R> std;
    array<I> count;
};

template <typename R, typename I>
struct Buffers
{
    std::vector<R> sum;
    std::vector<R> sumsq;
    std::vector<I> count;

    Buffers(size_t size) : sum(size, R()), sumsq(size, R()), count(size, I()) {}

    void reset()
    {
        std::fill(sum.begin(), sum.end(), R());
        std::fill(sumsq.begin(), sumsq.end(), R());
        std::fill(count.begin(), count.end(), I());
    }

    void finalize(Profiles<R, I> & profiles)
    {
        for (I r = 0; r < sum.size(); ++r)
        {
            profiles.count[r] = count[r];
            if (count[r] > 0)
            {
                R mean = sum[r] / count[r];
                R var = sumsq[r] / count[r] - mean * mean;
                profiles.whitefield[r] = mean;
                profiles.std[r] = std::sqrt(std::max(var, R()));
            }
            else
            {
                profiles.whitefield[r] = R();
                profiles.std[r] = R();
            }
        }
    }
};

template <typename R, typename I>
struct ProfileParameters
{
    I n_bins;
    I interval;
    R clip_snr;
    R std_min;
};

template <typename R, typename I>
class PanelRangeState
{
public:
    PanelRangeState(const DetectorGeometry<R, I> & geometry, I index) :
        m_geometry(geometry)
    {
        reset(index);
    }

    void reset(I index)
    {
        m_index = index;
        auto panel_iter = std::upper_bound(m_geometry.panel_offsets.begin(),
                                           m_geometry.panel_offsets.end(), m_index);
        m_panel_index = static_cast<I>(panel_iter - m_geometry.panel_offsets.begin() - 1);
        m_panel_offset = m_geometry.panel_offsets[m_panel_index];

        I panel_index = m_index - m_panel_offset;
        m_ss = panel_index / panel().shape[1];
        m_fs = panel_index - m_ss * panel().shape[1];
    }

    void increment()
    {
        ++m_index;
        ++m_fs;

        if (m_fs == panel().shape[1])
        {
            m_fs = 0;
            ++m_ss;
        }

        if (m_index == m_geometry.panel_offsets[m_panel_index + 1] &&
            m_index < m_geometry.panel_size())
        {
            reset(m_index);
        }
    }

    I index() const
    {
        return m_index;
    }

    const PanelGeometry<R, I> & panel() const
    {
        return m_geometry.panels[m_panel_index];
    }

    I ss() const
    {
        return m_ss;
    }

    I fs() const
    {
        return m_fs;
    }

    I frame_index() const
    {
        const auto & current_panel = panel();
        return current_panel.offset + m_ss * current_panel.stride[0] +
               m_fs * current_panel.stride[1];
    }

private:
    const DetectorGeometry<R, I> & m_geometry;
    I m_index = 0;
    I m_panel_index = 0;
    I m_panel_offset = 0;
    I m_ss = 0;
    I m_fs = 0;
};

template <typename I>
class FrameRangeState
{
public:
    FrameRangeState(I index, I frame_size) : m_frame_size(frame_size)
    {
        reset(index);
    }

    void reset(I index)
    {
        m_index = index;
        m_frame = m_index / m_frame_size;
        m_frame_index = m_index - m_frame * m_frame_size;
    }

    void increment(I step = 1)
    {
        m_index += step;
        m_frame_index += step;

        while (m_frame_index >= m_frame_size)
        {
            m_frame_index -= m_frame_size;
            ++m_frame;
        }
    }

    I index() const
    {
        return m_index;
    }

    I frame() const
    {
        return m_frame;
    }

    I frame_index() const
    {
        return m_frame_index;
    }

private:
    I m_frame_size;
    I m_index = 0;
    I m_frame = 0;
    I m_frame_index = 0;
};

template <typename R, typename I, typename Func>
void for_panel_range(const DetectorGeometry<R, I> & geometry, I begin, I end, Func && func)
{
    PanelRangeState<R, I> state(geometry, begin);

    while (state.index() < end)
    {
        func(state.panel(), state.ss(), state.fs(), state.frame_index());
        state.increment();
    }
}

template <typename R, typename I>
bool frame_shape_matches(const py::buffer_info & buffer,
                         const DetectorGeometry<R, I> & geometry)
{
    if (buffer.ndim != geometry.ndim())
    {
        return false;
    }
    for (I dim = 0; dim < geometry.ndim(); ++dim)
    {
        if (buffer.shape[dim] != geometry.shape[dim])
        {
            return false;
        }
    }
    return true;
}

template <typename R, typename I>
bool pixel_map_shape_matches(const py::buffer_info & buffer,
                             const DetectorGeometry<R, I> & geometry)
{
    if (buffer.ndim != geometry.ndim() + 1 || buffer.shape[0] != 3)
    {
        return false;
    }
    for (I dim = 0; dim < geometry.ndim(); ++dim)
    {
        if (buffer.shape[dim + 1] != geometry.shape[dim])
        {
            return false;
        }
    }
    return true;
}

template <typename R, typename I>
py::array_t<R> pixel_map(py::array_t<R> result, PyDetectorGeometry py_geometry,
                         bool half_pixel_shift, unsigned threads)
{
    auto geometry = cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();

    auto buffer = result.request();
    if (!pixel_map_shape_matches(buffer, geometry))
    {
        throw std::invalid_argument("pixel_map output shape mismatch");
    }

    array<R> out {buffer};

    I frame_size = geometry.size();
    I panel_size = geometry.panel_size();
    threads = std::max<unsigned>(1, std::min<unsigned>(threads, panel_size));
    I work_chunk = std::max<I>(1, panel_size / threads);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (I index = 0; index < panel_size; index += work_chunk)
    {
        for_panel_range(geometry, index, std::min(index + work_chunk, panel_size),
                        [&](const PanelGeometry<R, I> & panel, I ss, I fs, I frame_index)
        {
            out[frame_index] = panel.lab_x(ss, fs, geometry.half_pixel_shift);
            out[frame_size + frame_index] = panel.lab_y(ss, fs, geometry.half_pixel_shift);
            out[2 * frame_size + frame_index] = panel.lab_z(ss, fs,
                                                            geometry.half_pixel_shift);
        });
    }

    py::gil_scoped_acquire acquire;
    return result;
}

template <typename R, typename I>
py::array_t<R> radius(py::array_t<R> result, PyDetectorGeometry py_geometry,
                      std::tuple<py::ssize_t, py::ssize_t> center, bool half_pixel_shift,
                      unsigned threads)
{
    auto geometry = cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();

    auto buffer = result.request();
    if (!frame_shape_matches(buffer, geometry))
    {
        throw std::invalid_argument("radius output shape mismatch");
    }

    array<R> out {buffer};

    I panel_size = geometry.panel_size();
    threads = std::max<unsigned>(1, std::min<unsigned>(threads, panel_size));
    I work_chunk = std::max<I>(1, panel_size / threads);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (I index = 0; index < panel_size; index += work_chunk)
    {
        for_panel_range(geometry, index, std::min(index + work_chunk, panel_size),
                        [&](const PanelGeometry<R, I> & panel, I ss, I fs, I frame_index)
        {
            out[frame_index] = geometry.radius(panel, center, ss, fs);
        });
    }

    py::gil_scoped_acquire acquire;
    return result;
}

template <typename R, typename I>
py::array_t<I> radial_index(py::array_t<I> result, PyDetectorGeometry py_geometry,
                            std::tuple<py::ssize_t, py::ssize_t> center, I n_bins,
                            bool half_pixel_shift, unsigned threads)
{
    auto geometry = cast_detector_geometry<R, I>(py_geometry);
    geometry.half_pixel_shift = half_pixel_shift;
    geometry.validate();
    if (n_bins <= 1) throw std::invalid_argument("n_bins must be greater than 1");

    auto buffer = result.request();
    if (!frame_shape_matches(buffer, geometry))
    {
        throw std::invalid_argument("radial_index output shape mismatch");
    }

    R max_radius = geometry.max_radius(center);
    if (max_radius <= R()) throw std::invalid_argument("max radius must be positive");
    R inv_radius_step = (n_bins - 1) / max_radius;

    array<I> out {buffer};

    I panel_size = geometry.panel_size();
    threads = std::max<unsigned>(1, std::min<unsigned>(threads, panel_size));
    I work_chunk = std::max<I>(1, panel_size / threads);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (I index = 0; index < panel_size; index += work_chunk)
    {
        for_panel_range(geometry, index, std::min(index + work_chunk, panel_size),
                        [&](const PanelGeometry<R, I> & panel, I ss, I fs, I frame_index)
        {
            out[frame_index] = std::floor(geometry.radius(panel, center, ss, fs) *
                                          inv_radius_step + R(0.5));
        });
    }

    py::gil_scoped_acquire acquire;
    return result;
}

template <typename T, typename R, typename I>
void accumulate_index(const array<T> & darr, const array<I> & rarr, I index, I frame,
                      I frame_index, ProfileParameters<R, I> & params,
                      Buffers<R, I> & buffers, Profiles<R, I> * out)
{
    I bin = rarr[frame_index];
    if (bin < 0 || bin >= params.n_bins) return;

    R value = darr[index];
    bool accepted = true;

    if (out)
    {
        R sigma = std::max((out->std)[frame * params.n_bins + bin], params.std_min);
        R threshold = (out->whitefield)[frame * params.n_bins + bin] +
                      params.clip_snr * sigma;
        accepted = value <= threshold;
    }

    if (accepted)
    {
        buffers.sum[frame * params.n_bins + bin] += value;
        buffers.sumsq[frame * params.n_bins + bin] += value * value;
        buffers.count[frame * params.n_bins + bin]++;
    }
}

template <typename T, typename R, typename I>
ProfileResult<R, I> radial_profiles(py::array_t<T> data, py::array_t<I> radial_index, py::ssize_t n_bins,
                                    py::ssize_t interval, double clip_snr, py::ssize_t n_iter, double std_min,
                                    unsigned threads)
{
    array<T> darr {data.request()};
    array<I> rarr {radial_index.request()};

    ProfileParameters<R, I> params {static_cast<I>(n_bins), static_cast<I>(interval),
                                    static_cast<R>(clip_snr), static_cast<R>(std_min)};

    if (params.n_bins <= 1) throw std::invalid_argument("n_bins must be greater than 1");
    if (n_iter < 0) throw std::invalid_argument("n_iter must be non-negative");
    if (params.interval <= 0) throw std::invalid_argument("interval must be positive");

    I frame_size = rarr.size();
    if (darr.ndim() < rarr.ndim())
    {
        throw std::invalid_argument("data ndim must be at least radial_index ndim");
    }
    for (I dim = 0; dim < rarr.ndim(); ++dim)
    {
        I data_dim = darr.ndim() - rarr.ndim() + dim;
        if (darr.shape(data_dim) != rarr.shape(dim))
            throw std::invalid_argument("data trailing shape must match radial_index shape");
    }
    if (darr.size() % frame_size != 0)
    {
        throw std::invalid_argument("data size must be divisible by radial_index size");
    }

    threads = std::max<unsigned>(1, std::min<unsigned>(threads, data.size()));

    I n_frames = darr.size() / frame_size;
    I work_size = darr.size();
    I n_chunks = threads / n_frames + (threads % n_frames > 0);
    I work_chunk = std::max<I>(1, frame_size / n_chunks);

    py::array_t<R> whitefield {{n_frames, params.n_bins}};
    py::array_t<R> std {{n_frames, params.n_bins}};
    py::array_t<I> counts {{n_frames, params.n_bins}};

    Profiles<R, I> profiles {whitefield.request(), std.request(), counts.request()};

    fill_array(counts, I());

    py::gil_scoped_release release;

    Buffers<R, I> buffers (n_frames * params.n_bins);

    #pragma omp parallel num_threads(threads)
    {
        Buffers<R, I> local_buffers (n_frames * params.n_bins);

        #pragma omp for nowait
        for (I index = 0; index < work_size; index += work_chunk)
        {
            I sub_index = index;
            I remainder = sub_index % params.interval;
            if (remainder) sub_index += params.interval - remainder;
            FrameRangeState<I> state(sub_index, frame_size);

            while (state.index() < std::min(index + work_chunk, work_size))
            {
                accumulate_index(darr, rarr, state.index(), state.frame(), state.frame_index(),
                                 params, local_buffers,
                                 static_cast<Profiles<R, I> *>(nullptr));
                state.increment(params.interval);
            }
        }

        #pragma omp critical
        {
            for (I r = 0; r < n_frames * params.n_bins; ++r)
            {
                buffers.sum[r] += local_buffers.sum[r];
                buffers.sumsq[r] += local_buffers.sumsq[r];
                buffers.count[r] += local_buffers.count[r];
            }
        }

        #pragma omp barrier

        #pragma omp single
        {
            buffers.finalize(profiles);
        }

        #pragma omp barrier

        for (I iter = 0; iter < n_iter; ++iter)
        {
            #pragma omp single
            {
                buffers.reset();
            }

            #pragma omp barrier

            local_buffers.reset();

            #pragma omp for nowait
            for (I index = 0; index < work_size; index += work_chunk)
            {
                I sub_index = index;
                I remainder = sub_index % params.interval;
                if (remainder) sub_index += params.interval - remainder;
                FrameRangeState<I> state(sub_index, frame_size);

                while (state.index() < std::min(index + work_chunk, work_size))
                {
                    accumulate_index(darr, rarr, state.index(), state.frame(),
                                     state.frame_index(), params, local_buffers, &profiles);
                    state.increment(params.interval);
                }
            }

            #pragma omp critical
            {
                for (I r = 0; r < n_frames * params.n_bins; ++r)
                {
                    buffers.sum[r] += local_buffers.sum[r];
                    buffers.sumsq[r] += local_buffers.sumsq[r];
                    buffers.count[r] += local_buffers.count[r];
                }
            }

            #pragma omp barrier

            #pragma omp single
            {
                buffers.finalize(profiles);
            }

            #pragma omp barrier
        }
    }

    py::gil_scoped_acquire acquire;

    return std::make_tuple(std::move(whitefield), std::move(std), std::move(counts));
}

template <typename T, typename R, typename I>
py::array_t<bool> is_signal(py::array_t<T> data, py::array_t<R> whitefield, py::array_t<R> std,
                            py::array_t<I> radial_index, double min_snr, double std_min, unsigned threads)
{
    array<T> darr {data.request()};
    array<R> warr {whitefield.request()};
    array<R> sarr {std.request()};
    array<I> rarr {radial_index.request()};

    if (warr.size() != sarr.size())
    {
        throw std::invalid_argument("whitefield and std size mismatch");
    }
    if (darr.size() % rarr.size() != 0)
    {
        throw std::invalid_argument("data size must be divisible by radial_index size");
    }

    size_t n_bins = warr.shape(warr.ndim() - 1);
    py::array_t<bool> result {darr.shape()};
    array<bool> out {result.request()};

    threads = std::max<unsigned>(1, std::min<unsigned>(threads, data.size()));

    size_t frame_size = rarr.size();
    size_t n_frames = darr.size() / frame_size;
    size_t n_chunks = threads / n_frames + (threads % n_frames > 0);
    size_t work_chunk = std::max<size_t>(1, frame_size / n_chunks);

    #pragma omp parallel for num_threads(threads)
    for (size_t index = 0; index < darr.size(); index += work_chunk)
    {
        FrameRangeState<size_t> state(index, frame_size);
        size_t end = std::min(index + work_chunk, darr.size());

        while (state.index() < end)
        {
            auto bin = rarr[state.frame_index()];
            if (bin < 0 || bin >= n_bins)
            {
                out[state.index()] = false;
            }
            else
            {
                auto sigma = std::max<R>(sarr[state.frame() * n_bins + bin], std_min);
                out[state.index()] = darr[state.index()] >
                                     warr[state.frame() * n_bins + bin] + min_snr * sigma;
            }
            state.increment();
        }
    }

    return result;
}

}

PYBIND11_MODULE(online_detector, m)
{
    using namespace cbclib;
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

    m.def("pixel_map", &pixel_map<double, py::ssize_t>, py::arg("out"),
          py::arg("geometry"), py::arg("half_pixel_shift") = true,
          py::arg("num_threads") = 1);
    m.def("pixel_map", &pixel_map<float, int>, py::arg("out"),
          py::arg("geometry"), py::arg("half_pixel_shift") = true,
          py::arg("num_threads") = 1);
    m.def("radius", &radius<double, py::ssize_t>, py::arg("out"),
          py::arg("geometry"), py::arg("center"), py::arg("half_pixel_shift") = true,
          py::arg("num_threads") = 1);
    m.def("radius", &radius<float, int>, py::arg("out"), py::arg("geometry"),
          py::arg("center"), py::arg("half_pixel_shift") = true,
          py::arg("num_threads") = 1);
    m.def("radial_index", &radial_index<double, py::ssize_t>, py::arg("out"),
          py::arg("geometry"), py::arg("center"), py::arg("n_bins"),
          py::arg("half_pixel_shift") = true, py::arg("num_threads") = 1);
    m.def("radial_index", &radial_index<float, int>, py::arg("out"),
          py::arg("geometry"), py::arg("center"), py::arg("n_bins"),
          py::arg("half_pixel_shift") = true, py::arg("num_threads") = 1);

    m.def("radial_profiles", &radial_profiles<double, double, py::ssize_t>, py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0, py::arg("num_threads") = 1);
    m.def("radial_profiles", &radial_profiles<float, float, int>, py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0, py::arg("num_threads") = 1);
    m.def("radial_profiles", &radial_profiles<int, float, int>, py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0, py::arg("num_threads") = 1);
    m.def("radial_profiles", &radial_profiles<unsigned int, float, int>, py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0, py::arg("num_threads") = 1);
    m.def("radial_profiles", &radial_profiles<py::ssize_t, double, py::ssize_t>, py::arg("data"),
          py::arg("radial_index"), py::arg("n_bins"), py::arg("interval") = 1,
          py::arg("clip_snr") = 3.0, py::arg("n_iter") = 3, py::arg("std_min") = 0.0, py::arg("num_threads") = 1);

    m.def("is_signal", &is_signal<double, double, py::ssize_t>, py::arg("data"),
          py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0,
          py::arg("num_threads") = 1);
    m.def("is_signal", &is_signal<float, float, int>, py::arg("data"),
          py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0,
          py::arg("num_threads") = 1);
    m.def("is_signal", &is_signal<int, float, int>, py::arg("data"),
          py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0,
          py::arg("num_threads") = 1);
    m.def("is_signal", &is_signal<unsigned int, double, py::ssize_t>, py::arg("data"),
          py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0,
          py::arg("num_threads") = 1);
    m.def("is_signal", &is_signal<py::ssize_t, double, py::ssize_t>, py::arg("data"),
          py::arg("whitefield"), py::arg("std"), py::arg("radial_index"),
          py::arg("min_snr") = 3.0, py::arg("std_min") = 0.0,
          py::arg("num_threads") = 1);
}
