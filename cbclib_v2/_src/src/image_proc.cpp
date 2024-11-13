#include "image_proc.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

template <typename T>
using grad_t = typename kernels<T>::kernel;

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

template <typename Out, typename T>
Out line_value(BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter, T width, Out max_val, kernel_t<T> krn)
{
    T length = amplitude(liter.tau);
    auto r1 = liter.error / length, r2 = eiter.error / length;

    if (r2 < T())
    {
        return max_val * krn(std::sqrt(r1 * r1 + r2 * r2) / width);
    }
    else if (r2 > length)
    {
        return max_val * krn(std::sqrt(r1 * r1 + (r2 - length) * (r2 - length)) / width);
    }
    else
    {
        return max_val * krn(r1 / width);
    }
}

std::array<size_t, 3> normalise_shape(const std::vector<size_t> & shape)
{
    if (shape.size() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(shape.size()) + " < 2)", shape);
    return {std::reduce(shape.begin(), std::prev(shape.end(), 2), size_t(1), std::multiplies()),
            shape[shape.size() - 2], shape[shape.size() - 1]};
}

template <typename Out, typename T, typename I, class Func>
void draw_bresenham(const point_t & ubound, I frame, I index, const Line<T> & line, T width, Out max_val, kernel_t<T> krn, Func && draw_pixel)
{
    width = std::clamp(width, T(), std::numeric_limits<T>::max());

    auto get_val = [&krn, width, max_val](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        return line_value(liter, eiter, width, max_val, krn);
    };

    auto draw = [&get_val, &draw_pixel, frame, index](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        std::forward<Func>(draw_pixel)(liter.point, frame, index, get_val(liter, eiter));
    };

    draw_bresenham_func(ubound, line, width, draw);
}

template <typename Out, typename T, typename I>
py::array_t<Out> draw_line_sum(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
                               std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);
    auto oarr = array<Out>(out.request());

    auto n_shape = normalise_shape(oarr.shape);
    auto ubound = get_ubound(oarr.shape);

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim - 1, larr.shape, 5);
    auto lsize = larr.size / larr.shape[larr.ndim - 1];

    if (!idxs) fill_indices("idxs", n_shape[0], lsize, idxs);
    else check_indices("idxs", n_shape[0], lsize, idxs);
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<std::pair<I, T>> buffer (n_shape);
        auto draw_pixel = [&buffer](const point_t & pt, I frame, I index, T val)
        {
            detail::draw_pixel(buffer, pt, frame, val);
        };
        auto write = [&buffer, &oarr](const std::pair<I, T> & value)
        {
            oarr[value.first] += value.second;
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto line = std::make_from_tuple<Line<T>>(to_tuple<4>(larr, 5 * i));
                draw_bresenham(ubound, iarr[i], static_cast<I>(i + 1), line, larr[5 * i + 4], max_val, krn, draw_pixel);
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
auto draw_line_table(py::array_t<T> lines, std::vector<size_t> shape, std::optional<py::array_t<I>> idxs,
                     T max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_kernel(kernel);

    auto n_shape = normalise_shape(shape);
    auto ubound = get_ubound(shape);

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim - 1, larr.shape, 5);
    auto lsize = larr.size / larr.shape[larr.ndim - 1];

    if (!idxs) fill_indices("idxs", n_shape[0], lsize, idxs);
    else check_indices("idxs", n_shape[0], lsize, idxs);
    auto iarr = array<I>(idxs.value().request());

    table_t<T> result;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        table_t<T> buffer;
        detail::shape_handler handler (n_shape);
        auto draw_pixel = [&buffer, &handler](const point_t & pt, I frame, I index, T val)
        {
            if (val) buffer.emplace(std::make_pair(index, handler.ravel_index(frame, pt.y(), pt.x())), val);
        };

        #pragma omp for nowait
        for (size_t i = 0; i < lsize; i++)
        {
            e.run([&]()
            {
                auto line = std::make_from_tuple<Line<T>>(to_tuple<4>(larr, 5 * i));
                draw_bresenham(ubound, iarr[i], static_cast<I>(i + 1), line, larr[5 * i + 4], max_val, krn, draw_pixel);
            });
        }

        #pragma omp critical
        result.merge(buffer);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

}

PYBIND11_MODULE(image_proc, m)
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

    m.def("draw_line_mask",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_mask",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, uint32_t max_val, std::string kernel, unsigned threads)
        {
            py::array_t<uint32_t> out {shape};
            fill_array(out, uint32_t());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);

    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, double max_val, std::string kernel, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<size_t>> idxs, float max_val, std::string kernel, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<double> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, double max_val, std::string kernel, unsigned threads)
        {
            py::array_t<double> out {shape};
            fill_array(out, double());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_image",
        [](py::array_t<float> lines, std::vector<size_t> shape, std::optional<py::array_t<long>> idxs, float max_val, std::string kernel, unsigned threads)
        {
            py::array_t<float> out {shape};
            fill_array(out, float());
            return draw_line_sum(out, lines, idxs, max_val, kernel, threads);
        },
        py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);

    m.def("draw_line_table", &draw_line_table<float, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table<double, size_t>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table<float, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
    m.def("draw_line_table", &draw_line_table<double, long>, py::arg("lines"), py::arg("shape"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "rectangular", py::arg("num_threads") = 1);
}
