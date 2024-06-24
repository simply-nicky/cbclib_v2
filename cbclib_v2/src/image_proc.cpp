#include "image_proc.hpp"

namespace cbclib {

template <typename T>
using kernel_t = typename kernels<T>::kernel;

template <typename T>
using grad_t = typename kernels<T>::kernel;

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

template <typename Out, typename T>
void line_value_vjp(std::array<T, 5> & ct_line, const Line<T> & line, Out ct_pt, BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter, T width, Out max_val, grad_t<T> grad)
{
    T length = amplitude(line.tau);
    auto mag = length * length;
    auto r1 = liter.error / length, r2 = eiter.error / length;
    auto v0 = liter.point - line.pt0, v1 = liter.point - line.pt1;

    T ct_r1, ct_r2, val;
    if (r2 < T())
    {
        val = std::sqrt(r1 * r1 + r2 * r2);
        ct_r1 = max_val * grad(val / width) * ct_pt * r1 / (val * width);
        ct_r2 = max_val * grad(val / width) * ct_pt * r2 / (val * width);
    }
    else if (r2 > length)
    {
        val = std::sqrt(r1 * r1 + (r2 - length) * (r2 - length));
        ct_r1 = max_val * grad(val / width) * ct_pt * r1 / (val * width);
        ct_r2 = max_val * grad(val / width) * ct_pt * (r2 - length) / (val * width);

        ct_line[0] += ct_r2 * line.tau.x() / length;
        ct_line[1] += ct_r2 * line.tau.y() / length;
        ct_line[2] -= ct_r2 * line.tau.x() / length;
        ct_line[3] -= ct_r2 * line.tau.y() / length;
    }
    else
    {
        val = r1;
        ct_r1 = max_val * grad(val / width) * ct_pt / width;
        ct_r2 = T();
    }

    ct_line[0] += ct_r1 * (line.tau.x() * r1 / mag + v1.y() / length) +
                  ct_r2 * (line.tau.x() * r2 / mag - (line.tau.x() + v0.x()) / length);
    ct_line[1] += ct_r1 * (line.tau.y() * r1 / mag - v1.x() / length) +
                  ct_r2 * (line.tau.y() * r2 / mag - (line.tau.y() + v0.y()) / length);
    ct_line[2] += ct_r1 * (-line.tau.x() * r1 / mag - v0.y() / length) +
                  ct_r2 * (-line.tau.x() * r2 / mag + v0.x() / length);
    ct_line[3] += ct_r1 * (-line.tau.y() * r1 / mag + v0.x() / length) +
                  ct_r2 * (-line.tau.y() * r2 / mag + v0.y() / length);
    ct_line[4] -= ct_pt * max_val * grad(val / width) * val / (width * width);
}

std::array<size_t, 3> normalise_shape(const std::vector<size_t> & shape)
{
    if (shape.size() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(shape.size()) + " < 2)", shape);
    return {std::reduce(shape.begin(), std::prev(shape.end(), 2), size_t(1), std::multiplies()),
            shape[shape.size() - 2], shape[shape.size() - 1]};
}

point_t get_ubound(const std::vector<size_t> & shape)
{
    using I = typename point_t::value_type;
    return point_t{static_cast<I>((shape[shape.size() - 1]) ? shape[shape.size() - 1] - 1 : 0),
                   static_cast<I>((shape[shape.size() - 2]) ? shape[shape.size() - 2] - 1 : 0)};
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

template <typename Out, typename T, typename I, class Func>
auto draw_bresenham_vjp(const point_t & ubound, I frame, I index, const Line<T> & line, T width, Out max_val, grad_t<T> grad, Func && get_pixel)
{
    width = std::clamp(width, T(), std::numeric_limits<T>::max());
    std::array<T, 5> ct_line {};

    auto propagate = [&ct_line, &line, &grad, width, max_val](Out ct_pt, BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        line_value_vjp(ct_line, line, ct_pt, liter, eiter, width, max_val, grad);
    };

    auto get_cotangent = [&propagate, &get_pixel, frame, index](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        auto [ct_pt, flag] = std::forward<Func>(get_pixel)(liter.point, frame, index);
        if (flag) propagate(ct_pt, liter, eiter);
    };

    draw_bresenham_func(ubound, line, width, get_cotangent);

    return to_tuple(ct_line);
}

template <typename I>
std::map<I, std::vector<I>> sort_indices(array<I> idxs)
{
    std::map<I, std::vector<I>> sorted_idxs;
    for (auto iter = idxs.begin(); iter != idxs.end(); ++iter)
    {
        auto it = sorted_idxs.find(*iter);
        if (it != sorted_idxs.end()) it->second.push_back(std::distance(idxs.begin(), iter));
        else sorted_idxs.emplace(*iter, std::vector<I>{static_cast<I>(std::distance(idxs.begin(), iter))});
    }
    return sorted_idxs;
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

template <typename Out, typename T, typename I>
auto draw_line_max(py::array_t<Out> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, Out max_val,
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

    auto out_idxs = py::array_t<I>(oarr.shape);
    auto oiarr = array<I>(out_idxs.request());

    fill_array(out_idxs, I());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<std::tuple<I, I, T>> buffer (n_shape);
        auto draw_pixel = [&buffer](const point_t & pt, I frame, I index, T val)
        {
            detail::draw_pixel(buffer, pt, frame, index, val);
        };
        auto write = [&buffer, &oarr, &oiarr](const std::tuple<I, I, T> & value)
        {
            if (std::get<2>(value) > oarr[std::get<0>(value)])
            {
                oarr[std::get<0>(value)] = std::get<2>(value);
                oiarr[std::get<0>(value)] = std::get<1>(value);
            }
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

    return std::make_tuple(out, out_idxs);
}

template <typename Out, typename T, typename I>
py::array_t<T> draw_line_max_vjp(py::array_t<Out> cotangents, py::array_t<I> out_idxs, py::array_t<T> lines, std::optional<py::array_t<I>> idxs,
                                 Out max_val, std::string kernel, unsigned threads)
{
    assert(PyArray_API);

    auto krn = kernels<T>::get_grad(kernel);
    auto ctarr = array<Out>(cotangents.request());
    auto oiarr = array<I>(out_idxs.request());

    auto n_shape = normalise_shape(ctarr.shape);
    auto ubound = get_ubound(ctarr.shape);

    ctarr = ctarr.reshape(n_shape);
    oiarr = oiarr.reshape(n_shape);

    auto larr = array<T>(lines.request());
    check_dimensions("lines", larr.ndim - 1, larr.shape, 5);
    auto lsize = larr.size / larr.shape[larr.ndim - 1];

    if (!idxs) fill_indices("idxs", n_shape[0], lsize, idxs);
    else check_indices("idxs", n_shape[0], lsize, idxs);
    auto iarr = array<I>(idxs.value().request());

    std::vector<py::ssize_t> out_shape (larr.shape.begin(), std::prev(larr.shape.end(), 1));
    out_shape.push_back(5);
    auto ct_l = py::array_t<T>(out_shape);
    auto ct_larr = array<T>(ct_l.request());

    auto get_pixel = [&ctarr, &oiarr](const point_t & pt, I frame, I index)
    {
        using integer_type = typename point_t::value_type;
        std::array<integer_type, 3> coord {static_cast<integer_type>(frame), pt.y(), pt.x()};
        if (ctarr.is_inbound(coord)) return std::make_pair(ctarr.at(coord), oiarr.at(coord) == index);
        return std::make_pair(Out(), false);
    };

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < lsize; i++)
    {
        e.run([&]()
        {
            auto line = std::make_from_tuple<Line<T>>(to_tuple<4>(larr, 5 * i));
            to_tie<5>(ct_larr, 5 * i) = draw_bresenham_vjp(ubound, iarr[i], static_cast<I>(i + 1), line, larr[5 * i + 4], max_val, krn, get_pixel);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return ct_l;
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

    m.def("fill_indices",
        [](size_t xsize, size_t isize, std::optional<py::array_t<size_t>> idxs)
        {
            if (!idxs) fill_indices("idxs", xsize, isize, idxs);
            else check_indices("idxs", xsize, isize, idxs);
            return idxs.value();
        }, py::arg("xsize"), py::arg("isize"), py::arg("idxs") = nullptr);
    m.def("fill_indices",
        [](size_t xsize, size_t isize, std::optional<py::array_t<long>> idxs)
        {
            if (!idxs) fill_indices("idxs", xsize, isize, idxs);
            else check_indices("idxs", xsize, isize, idxs);
            return idxs.value();
        }, py::arg("xsize"), py::arg("isize"), py::arg("idxs") = nullptr);

    m.def("draw_line", &draw_line_max<float, float, size_t>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line", &draw_line_max<double, double, size_t>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line", &draw_line_max<float, float, long>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line", &draw_line_max<double, double, long>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);

    m.def("draw_line_vjp", &draw_line_max_vjp<float, float, size_t>, py::arg("cotangents"), py::arg("out_idxs"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line_vjp", &draw_line_max_vjp<double, double, size_t>, py::arg("cotangents"), py::arg("out_idxs"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line_vjp", &draw_line_max_vjp<float, float, long>, py::arg("cotangents"), py::arg("out_idxs"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);
    m.def("draw_line_vjp", &draw_line_max_vjp<double, double, long>, py::arg("cotangents"), py::arg("out_idxs"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1.0, py::arg("kernel") = "biweight", py::arg("num_threads") = 1);

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
