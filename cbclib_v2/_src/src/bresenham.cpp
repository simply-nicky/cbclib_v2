#include "bresenham.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

template <typename I, int ExtraFlags>
void fill_indices(std::string name, size_t xsize, size_t isize, std::optional<py::array_t<I, ExtraFlags>> & idxs)
{
    idxs = py::array_t<I, ExtraFlags>(isize);

    if (isize)
    {
        if (xsize == 1) fill_array(idxs.value(), I());
        else if (xsize == isize)
        {
            for (size_t i = 0; i < isize; i++) idxs.value().mutable_data()[i] = i;
        }
        else throw std::invalid_argument(name + " has an icnompatible size (" + std::to_string(isize) + " != " +
                                         std::to_string(xsize) + ")");
    }
}

template <typename I, int ExtraFlags>
void check_indices(std::string name, size_t imax, size_t isize, const py::array_t<I, ExtraFlags> & idxs)
{
    if (idxs.size())
    {
        if (static_cast<size_t>(idxs.size()) != isize)
            throw std::invalid_argument(name + " has an invalid size (" + std::to_string(idxs.size()) +
                                        " != " + std::to_string(isize) + ")");

        auto [min, max] = std::minmax_element(idxs.data(), idxs.data() + idxs.size());
        if (*max >= static_cast<I>(imax) || *min < I())
            throw std::out_of_range(name + " range (" + std::to_string(*min) + ", " + std::to_string(*max) +
                                    ") is outside of (0, " + std::to_string(imax) + ")");
    }
}

template <typename T, typename I, size_t N, int Update>
py::array_t<T> draw_lines_nd(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                             std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    array<T> oarr {out.request()};
    array<T> larr (lines.request());

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimension("lines", larr.ndim() - 1, larr.shape().begin(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, lsize, idxs);
    else check_indices("idxs", n_frames, lsize, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, T> buffer (shape);

        auto write = [&oarr, size = buffer.size()](const std::tuple<size_t, size_t, T> & values)
        {
            auto [idx, frame, value] = values;
            size_t index = idx + size * frame;

            if constexpr (Update) oarr[index] = std::max(oarr[index], value);
            else oarr[index] = oarr[index] + value;
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, frame = iarr[i]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, frame, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), write);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I, int Update>
py::array_t<T> draw_lines_2d_3d(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                                std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<T, I, 2, Update>(out, lines, idxs, max_val, kernel, threads);
        case 7:
            return draw_lines_nd<T, I, 3, Update>(out, lines, idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I>
py::array_t<T> draw_lines(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                          std::string kernel, std::string overlap, unsigned threads)
{
    if (overlap == "sum") return draw_lines_2d_3d<T, I, 0>(out, lines, idxs, max_val, kernel, threads);
    if (overlap == "max") return draw_lines_2d_3d<T, I, 1>(out, lines, idxs, max_val, kernel, threads);
    throw std::invalid_argument("Invalid overlap keyword: " + overlap);
}

template <typename T, typename I, size_t N, int Update>
auto accumulate_lines_nd(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> counts, py::array_t<I> frames,
                         T max_val, std::string kernel, unsigned threads)
{
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    array<T> oarr {out.request()};
    array<T> larr {lines.request()};
    array<I> carr {counts.request()};

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimension("lines", larr.ndim() - 1, larr.shape().begin(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    check_indices("frames", n_frames, carr.size(), frames);

    array<I> farr {frames.request()};
    std::vector<size_t> lasts;
    std::partial_sum(carr.begin(), carr.end(), std::back_inserter(lasts));

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        vector_array<T> buffer {shape};
        std::vector<size_t> indices;

        #pragma omp for nowait
        for (size_t frame = 0; frame < carr.size(); frame++)
        {
            size_t shift = farr[frame] * buffer.size();
            size_t last = lasts[frame];
            size_t first = last - carr[frame];
            T value;

            if (last <= lsize)
            {
                for (size_t i = first; i < last; i++)
                {
                    e.run([&]()
                    {
                        auto draw_pixel = [&buffer, &indices, &krn, max_val](const PointND<long, N> & pt, T error)
                        {
                            if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                            {
                                size_t index = buffer.index_at(pt.rbegin(), pt.rend());
                                if (buffer[index] == T()) indices.push_back(index);
                                if constexpr (Update & 1) buffer[index] = std::max(buffer[index], max_val * krn(std::sqrt(error)));
                                else buffer[index] = buffer[index] + max_val * krn(std::sqrt(error));
                            }
                        };

                        draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)},
                                     larr[L * i + 2 * N], draw_pixel);
                    });
                }

                for (auto index : indices)
                {
                    if constexpr (Update >> 1)
                    {
                        #pragma omp atomic read
                        value = oarr[index + shift];

                        #pragma omp atomic write
                        oarr[index + shift] = std::max(value, buffer[index]);
                    }
                    else
                    {
                        #pragma omp atomic update
                        oarr[index + shift] = buffer[index] + oarr[index + shift];
                    }

                    buffer[index] = T();
                }

                indices.clear();
            }
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I, int Update>
py::array_t<T> accumulate_lines_2d_3d(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> in_idxs, py::array_t<I> out_idxs,
                                      T max_val, std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return accumulate_lines_nd<T, I, 2, Update>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
        case 7:
            return accumulate_lines_nd<T, I, 3, Update>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I>
py::array_t<T> accumulate_lines(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> in_idxs, py::array_t<I> out_idxs,
                                T max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
{
    if (in_overlap == "sum" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 0>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
    if (in_overlap == "max" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 1>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
    if (in_overlap == "sum" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 2>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
    if (in_overlap == "max" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 3>(out, lines, in_idxs, out_idxs, max_val, kernel, threads);
    throw std::invalid_argument("Invalid in_overlap and out_overlap keywords: " + in_overlap + " and " + out_overlap);
}

template <typename T, typename I, size_t N>
auto write_lines_nd(py::array_t<T> lines, std::vector<size_t> shape, std::optional<py::array_t<I>> idxs,
                    T max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);

    auto n_frames = std::reduce(shape.begin(), std::prev(shape.end(), N), size_t(1), std::multiplies());
    std::vector<size_t> fshape {std::prev(shape.end(), N), shape.end()};

    auto larr = array<T>(lines.request());
    check_dimension("lines", larr.ndim() - 1, larr.shape().begin(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, lsize, idxs);
    else check_indices("idxs", n_frames, lsize, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    std::vector<I> out_idxs, lidxs;
    std::vector<T> values;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, size_t, T> buffer {fshape};

        auto merge = [&out_idxs, &lidxs, &values, size = buffer.size()](const std::tuple<size_t, size_t, size_t, T> & value)
        {
            out_idxs.push_back(std::get<0>(value) + size * std::get<1>(value));
            lidxs.push_back(std::get<2>(value));
            values.push_back(std::get<3>(value));
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, frame = iarr[i], id = i](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, frame, id, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), merge);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(as_pyarray(std::move(out_idxs)), as_pyarray(std::move(lidxs)), as_pyarray(std::move(values)));
}

template <typename T, typename I>
auto write_lines(py::array_t<T> lines, std::vector<size_t> shape, std::optional<py::array_t<I>> idxs,
                 T max_val, std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return write_lines_nd<T, I, 2>(lines, shape, idxs, max_val, kernel, threads);
        case 7:
            return write_lines_nd<T, I, 3>(lines, shape, idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

}

PYBIND11_MODULE(bresenham, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("accumulate_lines",
        [](py::array_t<double> lines, std::vector<size_t> shape, py::array_t<size_t> counts, py::array_t<size_t> frames, double max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return accumulate_lines(out, lines, counts, frames, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("counts"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py::array_t<float> lines, std::vector<size_t> shape, py::array_t<size_t> counts, py::array_t<size_t> frames, float max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return accumulate_lines(out, lines, counts, frames, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("counts"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py::array_t<double> lines, std::vector<size_t> shape, py::array_t<long> counts, py::array_t<long> frames, double max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return accumulate_lines(out, lines, counts, frames, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("counts"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py::array_t<float> lines, std::vector<size_t> shape, py::array_t<long> counts, py::array_t<long> frames, float max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return accumulate_lines(out, lines, counts, frames, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("counts"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);

    m.def("draw_lines",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);

    m.def("write_lines", &write_lines<float, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<float, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
