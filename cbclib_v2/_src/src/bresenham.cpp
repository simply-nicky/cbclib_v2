#include "bresenham.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

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

};

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

template <typename T>
using combiner_t = typename combiners<T>::combiner;

template <typename Out, typename T, typename I, size_t N>
py::array_t<Out> draw_lines_nd(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
                               std::string kernel, std::string overlap, unsigned threads)
{
    assert(PyArray_API);
    constexpr static size_t L = 2 * N + 1;

    auto krn = kernels<T>::get_kernel(kernel);
    auto cbn = combiners<T>::get_combiner(overlap);
    auto oarr = array<Out>(out.request());

    auto shape = normalise_shape<N>(oarr.shape);

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

        auto write = [&buffer, &oarr, &cbn](const std::pair<I, T> & value)
        {
            oarr[value.first] = cbn(oarr[value.first], value.second);
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

template <typename Out, typename T, typename I>
py::array_t<Out> draw_lines(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
                            std::string kernel, std::string overlap, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<Out, T, I, 2>(out, lines, idxs, max_val, kernel, overlap, threads);
        case 7:
            return draw_lines_nd<Out, T, I, 3>(out, lines, idxs, max_val, kernel, overlap, threads);
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
                        if (handler.is_inbound(coord))
                        {
                            buffer.emplace(std::make_pair(index, handler.ravel_index(coord)), max_val * krn(std::sqrt(error)));
                        }
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
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
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);

    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, double max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, float max_val, std::string kernel, std::string overlap, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_lines(out, lines, idxs, max_val, kernel, overlap, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);

    m.def("draw_line_table", &draw_lines_table<float, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_lines_table<double, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_lines_table<float, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_lines_table<double, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
