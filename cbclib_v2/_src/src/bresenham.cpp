#include "bresenham.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

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
void check_indices(std::string name, size_t xsize, size_t isize, std::optional<py::array_t<I, ExtraFlags>> & idxs)
{
    if (idxs && idxs.value().size())
    {
        if (static_cast<size_t>(idxs.value().size()) != isize)
            throw std::invalid_argument(name + " has an invalid size (" + std::to_string(idxs.value().size()) +
                                        " != " + std::to_string(isize) + ")");
        auto begin = idxs.value().data();
        auto end = begin + idxs.value().size();
        auto [min, max] = std::minmax_element(begin, end);
        if (*max >= static_cast<I>(xsize) || *min < I())
            throw std::out_of_range(name + " is out of range");
    }
}

template <size_t N>
std::array<size_t, N + 1> normalise_shape(const std::vector<size_t> & shape)
{
    if (shape.size() < N)
        fail_container_check("wrong number of dimensions (" + std::to_string(shape.size()) +
                             " < " + std::to_string(N) + ")", shape);
    std::array<size_t, N + 1> res {std::reduce(shape.begin(), std::prev(shape.end(), N), size_t(1), std::multiplies())};
    for (size_t i = 0; i < N; i++) res[i + 1] = shape[shape.size() - N + i];
    return res;
}

template <size_t N>
PointND<long, N> get_bound(const std::array<size_t, N + 1> & shape)
{
    PointND<long, N> out;
    for (size_t i = 0; i < N; i++) out[i] = (shape[N - i]) ? shape[N - i] - 1 : 0;
    return out;
}

template <typename Out, typename T, typename I, size_t N>
py::array_t<Out> draw_lines_nd(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
                               std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    auto oarr = array<Out>(out.request());

    auto shape = normalise_shape<N>(oarr.shape);
    auto bound = get_bound<N>(shape);

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim - 1, larr.shape, L);
    auto lsize = larr.size / larr.shape[larr.ndim - 1];

    if (!idxs) fill_indices("idxs", shape[0], lsize, idxs);
    else check_indices("idxs", shape[0], lsize, idxs);
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<std::pair<I, T>> buffer (shape);

        auto write = [&buffer, &oarr](const std::pair<I, T> & value)
        {
            oarr[value.first] += value.second;
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &krn, max_val, frame = iarr[i]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0)
                    {
                        std::array<long, N + 1> coord {long(frame)};
                        for (size_t i = 0; i < N; i++) coord[i + 1] = pt[N - i - 1];
                        if (buffer.is_inbound(coord)) buffer.emplace_back(coord, max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(bound, LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), write);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename Out, typename T, typename I>
py::array_t<Out> draw_lines(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
                            std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<Out, T, I, 2>(out, lines, idxs, max_val, kernel, threads);
        case 7:
            return draw_lines_nd<Out, T, I, 3>(out, lines, idxs, max_val, kernel, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I, size_t N>
auto draw_lines_table_nd(py::array_t<T> lines, std::vector<size_t> shape, std::optional<py::array_t<I>> idxs,
                         T max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);

    auto new_shape = normalise_shape<N>(shape);
    auto bound = get_bound<N>(new_shape);

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim - 1, larr.shape, L);
    auto lsize = larr.size / larr.shape[larr.ndim - 1];

    if (!idxs) fill_indices("idxs", new_shape[0], lsize, idxs);
    else check_indices("idxs", new_shape[0], lsize, idxs);
    auto iarr = array<I>(idxs.value().request());

    table_t<T> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        table_t<T> buffer;
        detail::shape_handler handler (new_shape);

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &handler, &krn, max_val, index = i + 1, frame = iarr[i]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0)
                    {
                        std::array<long, N + 1> coord {long(frame)};
                        for (size_t i = 0; i < N; i++) coord[i + 1] = pt[N - i - 1];
                        buffer.emplace(std::make_pair(index, handler.ravel_index(coord)), max_val * krn(std::sqrt(error)));
                    }
                };

                draw_line_nd(bound, LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        result.merge(buffer);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T, typename I>
auto draw_lines_table(py::array_t<T> lines, std::vector<size_t> shape, std::optional<py::array_t<I>> idxs,
                      T max_val, std::string kernel, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_table_nd<T, I, 2>(lines, shape, idxs, max_val, kernel, threads);
        case 7:
            return draw_lines_table_nd<T, I, 3>(lines, shape, idxs, max_val, kernel, threads);
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

    m.def("draw_line_mask",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);

    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, double max_val, std::string kernel, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, float max_val, std::string kernel, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, double max_val, std::string kernel, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, float max_val, std::string kernel, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
