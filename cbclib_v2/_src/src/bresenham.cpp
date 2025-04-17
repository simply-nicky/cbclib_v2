#include "bresenham.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

template <typename T>
using py_array_t = typename py::array_t<T, py::array::c_style | py::array::forcecast>;

namespace detail {

template <typename T>
T sum(const T & a, const T & b)
{
    return a + b;
}

template <typename T>
T max(const T & a, const T & b)
{
    return std::max(a, b);
}

}

template <typename T>
struct combiners
{
    enum combiner_type
    {
        sum = 0,
        max = 1
    };

    using combiner = T(*)(const T &, const T &);

    static inline std::map<std::string, combiner_type> combiner_names =
    {
        {"sum", combiner_type::sum},
        {"max", combiner_type::max}
    };

    static inline std::map<combiner_type, combiner> registered_combiners =
    {
        {combiner_type::sum, detail::sum},
        {combiner_type::max, detail::max}
    };

    static combiner get_combiner(combiner_type c, bool throw_if_missing = true)
    {
        auto it = registered_combiners.find(c);
        if (it != registered_combiners.end()) return it->second;
        if (throw_if_missing)
            throw std::invalid_argument("combiner is missing for " + std::to_string(c));
        return nullptr;
    }

    static combiner get_combiner(std::string name, bool throw_if_missing = true)
    {
        auto it = combiner_names.find(name);
        if (it != combiner_names.end()) return get_combiner(it->second, throw_if_missing);
        if (throw_if_missing)
            throw std::invalid_argument("combiner is missing for " + name);
        return nullptr;
    }
};

template <typename I, int ExtraFlags>
void fill_indices(std::string name, size_t xsize, size_t isize, std::optional<py::array_t<I, ExtraFlags>> & idxs)
{
    if (xsize == 1)
    {
        idxs = py::array_t<I, ExtraFlags>(isize);
        fill_array(idxs.value(), I());
    }
    else if (xsize == isize)
    {
        idxs = py::array_t<I, ExtraFlags>(isize);
        auto ptr = static_cast<I *>(idxs.value().request().ptr);
        for (size_t i = 0; i < isize; i++) ptr[i] = i;
    }
    else throw std::invalid_argument(name + " is not defined");
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

template <typename T>
using combiner_t = typename combiners<T>::combiner;

template <typename T, typename I, size_t N>
py::array_t<T> draw_lines_nd(py_array_t<T> out, py_array_t<T> lines, std::optional<py_array_t<I>> idxs, T max_val,
                             std::string kernel, std::string overlap, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    auto cbn = combiners<T>::get_combiner(overlap);
    array<T> oarr {out.request()};
    array<T> larr (lines.request());

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimensions("lines", larr.ndim() - 1, larr.shape(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, lsize, idxs);
    else check_indices("idxs", n_frames, lsize, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, T> buffer (shape);

        auto write = [&oarr, &cbn, size = buffer.size()](const std::tuple<size_t, size_t, T> & value)
        {
            size_t index = std::get<0>(value) + size * std::get<1>(value);
            oarr[index] = cbn(oarr[index], std::get<2>(value));
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

template <typename T, typename I>
py::array_t<T> draw_lines(py_array_t<T> out, py_array_t<T> lines, std::optional<py_array_t<I>> idxs, T max_val,
                          std::string kernel, std::string overlap, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<T, I, 2>(out, lines, idxs, max_val, kernel, overlap, threads);
        case 7:
            return draw_lines_nd<T, I, 3>(out, lines, idxs, max_val, kernel, overlap, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I, size_t N>
auto accumulate_lines_nd(py_array_t<T> out, py_array_t<T> lines, py_array_t<I> in_idxs, py_array_t<I> out_idxs,
                         T max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
{
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    auto in_cbn = combiners<T>::get_combiner(in_overlap);
    auto out_cbn = combiners<T>::get_combiner(out_overlap);
    array<T> oarr {out.request()};
    array<T> larr (lines.request());

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimensions("lines", larr.ndim() - 1, larr.shape(), L);
    auto lsize = larr.size() / larr.shape(larr.ndim() - 1);

    if (static_cast<size_t>(in_idxs.size()) != lsize)
        throw std::invalid_argument("in_idxs has an invalid size (" + std::to_string(in_idxs.size()) +
                                    " != " + std::to_string(lsize) + ")");
    check_indices("out_idxs", n_frames, lsize, out_idxs);

    array<I> in_iarr (in_idxs.request());
    array<I> out_iarr (out_idxs.request());

    std::vector<size_t> idxs (lsize);
    std::iota(idxs.begin(), idxs.end(), size_t());
    std::sort(idxs.begin(), idxs.end(), [&in_iarr](size_t i0, size_t i1){return in_iarr[i0] < in_iarr[i1];});

    vector_array<std::tuple<size_t, size_t, T>> obuffer {shape};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, size_t, T> buffer (shape);

        auto write = [&obuffer, &in_cbn, &out_cbn, &oarr](const std::tuple<size_t, size_t, size_t, T> & value)
        {
            auto [idx, new_id, new_frame, new_value] = value;
            auto & [old_frame, old_id, current] = obuffer[idx];
            if (old_frame == new_frame && old_id == new_id) current = in_cbn(current, new_value);
            else
            {
                size_t index = idx + obuffer.size() * old_frame;
                oarr[index] = out_cbn(oarr[index], current);
                old_frame = new_frame; old_id = new_id; current = new_value;
            }
        };

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, in_idx = in_iarr[idxs[i]], out_idx = out_iarr[idxs[i]]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, in_idx, out_idx, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * idxs[i]), to_point<N>(larr, L * idxs[i] + N)}, larr[L * idxs[i] + 2 * N], draw_pixel);
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            std::for_each(buffer.begin(), buffer.end(), write);
        }

        #pragma omp for
        for (size_t i = 0; i < buffer.size(); i++)
        {
            auto [frame, _, value] = obuffer[i];
            size_t index = i + obuffer.size() * frame;
            oarr[index] = out_cbn(oarr[index], value);
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I>
py::array_t<T> accumulate_lines(py_array_t<T> out, py_array_t<T> lines, py_array_t<I> in_idxs, py_array_t<I> out_idxs,
                                T max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return accumulate_lines_nd<T, I, 2>(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        case 7:
            return accumulate_lines_nd<T, I, 3>(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I, size_t N>
auto write_lines_nd(py_array_t<T> lines, std::vector<size_t> shape, std::optional<py_array_t<I>> idxs,
                    T max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);

    auto n_frames = std::reduce(shape.begin(), std::prev(shape.end(), N), size_t(1), std::multiplies());
    std::vector<size_t> fshape {std::prev(shape.end(), N), shape.end()};

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim() - 1, larr.shape(), L);
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
auto write_lines(py_array_t<T> lines, std::vector<size_t> shape, std::optional<py_array_t<I>> idxs,
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
        [](py_array_t<double> lines, std::vector<size_t> shape, py_array_t<size_t> in_idxs, py_array_t<size_t> out_idxs, double max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return accumulate_lines(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("in_idxs"), py::arg("out_idxs"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, py_array_t<size_t> in_idxs, py_array_t<size_t> out_idxs, float max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return accumulate_lines(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("in_idxs"), py::arg("out_idxs"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py_array_t<double> lines, std::vector<size_t> shape, py_array_t<long> in_idxs, py_array_t<long> out_idxs, double max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return accumulate_lines(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("in_idxs"), py::arg("out_idxs"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, py_array_t<long> in_idxs, py_array_t<long> out_idxs, float max_val, std::string kernel, std::string in_overlap, std::string out_overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return accumulate_lines(out, lines, in_idxs, out_idxs, max_val, kernel, in_overlap, out_overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("in_idxs"), py::arg("out_idxs"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);

    m.def("draw_lines",
        [](py_array_t<double> lines, std::vector<size_t> shape, std::optional<py_array_t<size_t>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, std::optional<py_array_t<size_t>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<double> lines, std::vector<size_t> shape, std::optional<py_array_t<long>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines",
        [](py_array_t<float> lines, std::vector<size_t> shape, std::optional<py_array_t<long>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py_array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);

    m.def("write_lines", &write_lines<float, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<float, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("write_lines", &write_lines<double, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
