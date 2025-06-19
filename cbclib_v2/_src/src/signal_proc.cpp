#include "signal_proc.hpp"

namespace cbclib {

template <typename T, typename U>
py::array_t<T> binterpolate(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                            std::vector<py::array_t<U, py::array::c_style | py::array::forcecast>> grid,
                            py::array_t<U, py::array::c_style | py::array::forcecast> coords, unsigned threads)
{
    auto ndim = grid.size();
    auto ibuf = inp.request();
    if (ndim != static_cast<size_t>(ibuf.ndim))
        throw std::invalid_argument("data number of dimensions (" + std::to_string(ibuf.ndim) + ")" +
                                    " isn't equal to the number of grid arrays (" + std::to_string(ndim) + ")");

    auto carr = array<U>(coords.request());
    auto npts = carr.size() / ndim;
    check_dimensions("coords", carr.ndim() - 1, carr.shape(), ndim);

    std::vector<array<U>> gvec;
    for (size_t n = 0; n < ndim; n++)
    {
        auto & arr = gvec.emplace_back(grid[n].request());
        check_dimensions("grid coordinates", arr.ndim() - 1, arr.shape(), ibuf.shape[ndim - 1 - n]);
    }

    auto iarr = array<T>(ibuf);
    auto out = py::array_t<T>(std::vector(carr.shape().begin(), std::prev(carr.shape().end())));
    auto oarr = array<T>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > npts) ? npts : threads;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < npts; i++)
    {
        e.run([&]
        {
            oarr[i] = bilinear(iarr, gvec, carr.slice(i, carr.ndim() - 1));
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

/*----------------------------------------------------------------------------*/
/*---------------------------- Kernel regression -----------------------------*/
/*----------------------------------------------------------------------------*/

template <typename T>
py::array_t<T> kr_predict(py::array_t<T, py::array::c_style | py::array::forcecast> y,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma, std::string kernel,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads)
{
    check_optional("w", y.shape(), y.shape() + y.ndim(), w, T(1));

    auto krn = kernels<T>::get_kernel(kernel);

    auto ybuf = y.request(), xbuf = x.request(), xhbuf = x_hat.request();
    size_t ndim = xhbuf.shape[xhbuf.ndim - 1], npts = ybuf.size;
    check_dimensions("x", xbuf.ndim - 1, xbuf.shape, ndim);
    check_equal("y and x have incompatible shapes",
                ybuf.shape.begin(), ybuf.shape.end(),
                xbuf.shape.begin(), std::prev(xbuf.shape.end()));

    auto xarr = array<T>(xbuf);
    auto yarr = array<T>(ybuf);
    auto warr = array<T>(w.value().request());
    auto xharr = array<T>(xhbuf);

    auto out_shape = std::vector<py::ssize_t>(xharr.shape().begin(), std::prev(xharr.shape().end()));
    auto out = py::array_t<T>(out_shape);

    auto oarr = array<T>(out.request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > oarr.size()) ? oarr.size() : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> idxs (npts);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&xarr, ndim](size_t i1, size_t i2){return xarr[i1 * ndim] < xarr[i2 * ndim];});

        #pragma omp for
        for (size_t i = 0; i < oarr.size(); i++)
        {
            e.run([&]
            {
                auto xhline = xharr.slice(i, xharr.ndim() - 1);
                auto window = idxs;

                for (size_t axis = 0; axis < ndim; axis++)
                {
                    auto comp_lb = [&xarr, axis, ndim](size_t index, T val){return xarr[index * ndim + axis] < val;};
                    auto comp_ub = [&xarr, axis, ndim](T val, size_t index){return val < xarr[index * ndim + axis];};

                    // begin is LESS OR EQUAL than val - sigma
                    auto begin = std::upper_bound(window.begin(), window.end(), xhline[axis] - sigma, comp_ub);
                    if (begin != window.begin()) begin = std::prev(begin);

                    // end - 1 is GREATER OR EQUAL than val + sigma
                    auto end = std::lower_bound(window.begin(), window.end(), xhline[axis] + sigma, comp_lb);
                    if (end != window.end()) end = std::next(end);

                    if (begin >= end)
                    {
                        window.clear(); break;
                    }
                    else
                    {
                        window = std::vector<size_t>(begin, end);
                        if (axis + 1 < ndim)
                        {
                            auto less = [&xarr, axis, ndim](size_t i1, size_t i2){return xarr[i1 * ndim + axis + 1] < xarr[i2 * ndim + axis + 1];};
                            std::sort(window.begin(), window.end(), less);
                        }
                    }
                }

                if (window.size())
                {
                    T Y = T(), W = T();
                    for (auto index : window)
                    {
                        T dist = T();
                        for (size_t axis = 0; axis < ndim; axis++) dist += std::pow(xarr[index * ndim + axis] - xhline[axis], 2);
                        T rbf = krn(std::sqrt(dist) / sigma);
                        Y += yarr[index] * warr[index] * rbf;
                        W += warr[index] * warr[index] * rbf;
                    }
                    oarr[i] = (W > T()) ? Y / W : T();
                }
                else oarr[i] = T();
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T>
auto kr_grid(py::array_t<T, py::array::c_style | py::array::forcecast> y, py::array_t<T, py::array::c_style | py::array::forcecast> x,
             std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> grid, T sigma, std::string kernel,
             std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads)
{
    auto krn = kernels<T>::get_kernel(kernel);

    auto xbuf = x.request();
    auto ybuf = y.request();

    size_t ndim = grid.size(), npts = xbuf.size / ndim, nf = ybuf.size / npts;
    check_dimensions("x", xbuf.ndim - 1, xbuf.shape, ndim);
    check_equal("x and y have incompatible shapes",
                xbuf.shape.begin(), std::prev(xbuf.shape.end()),
                std::prev(ybuf.shape.end(), xbuf.ndim - 1), ybuf.shape.end());
    check_optional("w", std::prev(ybuf.shape.end(), xbuf.ndim - 1), ybuf.shape.end(), w, T(1));

    auto xarr = array<T>(xbuf).reshape({npts, ndim});
    auto yarr = array<T>(ybuf).reshape({nf, npts});
    auto warr = array<T>(w.value().request()).reshape({npts});

    std::vector<array<T>> grid_arrs;
    for (auto coords : grid) grid_arrs.emplace_back(coords.request());

    std::vector<size_t> roi, wshape;
    size_t wsize = 1;
    for (size_t n = 0; n < ndim; ++n)
    {
        auto xline = xarr.slice(ndim - 1 - n, 0);
        auto carr = grid_arrs[ndim - 1 - n];
        auto [xmin, xmax] = std::minmax_element(xline.begin(), xline.end());

        auto begin = std::upper_bound(carr.begin(), carr.end(), *xmin);
        if (begin != carr.begin()) begin = std::prev(begin);
        auto min = roi.emplace_back(std::distance(carr.begin(), begin));

        auto end = std::lower_bound(carr.begin(), carr.end(), *xmax);
        if (end != carr.end()) end = std::next(end);
        auto max = roi.emplace_back(std::distance(carr.begin(), end));

        wshape.push_back(max - min);
        wsize *= max - min;
    }

    vector_array<T> Wsum (wshape);

    auto oshape = wshape;
    oshape.insert(oshape.begin(), nf);

    py::array_t<T> y_hat (oshape);
    array<T> yharr (y_hat.request());

    fill_array(y_hat, T());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        vector_array<T> Y (oshape), W (wshape);
        std::vector<size_t> coord (ndim), origin (ndim), shape (ndim);

        #pragma omp for nowait
        for (size_t i = 0; i < npts; i++)
        {
            e.run([&]
            {
                for (size_t n = 0; n < ndim; n++)
                {
                    auto carr = grid_arrs[ndim - 1 - n];

                    auto begin = std::upper_bound(carr.begin(), carr.end(), xarr.at(i, ndim - 1 - n) - sigma);
                    if (begin != carr.begin()) begin = std::prev(begin);
                    auto origin_index = std::clamp<size_t>(std::distance(carr.begin(), begin), roi[2 * n], roi[2 * n + 1]);

                    auto end = std::lower_bound(carr.begin(), carr.end(), xarr.at(i, ndim - 1 - n) + sigma);
                    if (end != carr.end()) end = std::next(end);

                    shape[n] = std::clamp<size_t>(std::distance(carr.begin(), end), roi[2 * n], roi[2 * n + 1]) - origin_index;
                    origin[n] = origin_index - roi[2 * n];
                }

                for (const auto & point : rectangle_range(shape))
                {
                    std::transform(origin.begin(), origin.end(), point.begin(), coord.begin(), std::plus<long>());

                    T dist = T();
                    for (size_t n = 0; n < ndim; n++)
                    {
                        dist += std::pow(grid_arrs[ndim - 1 - n][coord[n] + roi[2 * n]] - xarr.at(i, ndim - 1 - n), 2);
                    }
                    T rbf = krn(std::sqrt(dist) / sigma);

                    size_t index = W.index_at(coord);

                    for (size_t j = 0; j < nf; j++) Y[index + j * wsize] += yarr[i + j * npts] * warr[i] * rbf;
                    W[index] += warr[i] * warr[i] * rbf;
                }
            });
        }

        #pragma omp critical
        std::transform(W.begin(), W.end(), Wsum.begin(), Wsum.begin(), std::plus());

        #pragma omp barrier
        #pragma omp critical
        {
            for (size_t i = 0; i < wsize; i++)
            {
                if (W[i]) for (size_t j = 0; j < nf; j++) yharr[i + j * wsize] += Y[i + j * wsize] / Wsum[i];
            }
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return std::make_tuple(y_hat, roi);
}

template <typename T, typename U>
py::array_t<size_t> local_maxima(py::array_t<T, py::array::c_style | py::array::forcecast> inp, U axis, unsigned threads)
{
    array<T> iarr (inp.request());

    Sequence<long> seq (axis);
    seq.unwrap(iarr.ndim());

    for (auto ax : seq)
    {
        if (iarr.shape(ax) < 3)
            throw std::invalid_argument("The shape along axis " + std::to_string(ax) + "is below 3 (" +
                                        std::to_string(iarr.shape(ax)) + ")");
    }
    size_t repeats = iarr.size() / iarr.shape(seq[0]);

    std::vector<size_t> peaks;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> buffer;
        auto add_peak = [&buffer, &iarr](size_t index){iarr.unravel_index(std::back_inserter(buffer), index);};

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                auto iline = iarr.slice(i, seq[0]);
                maxima_nd(iline.begin(), iline.end(), add_peak, iarr, seq, seq.size());
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            peaks.insert(peaks.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    if (peaks.size() % iarr.ndim())
        throw std::runtime_error("peaks have invalid size of " + std::to_string(peaks.size()));

    std::array<size_t, 2> out_shape = {peaks.size() / iarr.ndim(), iarr.ndim()};
    return as_pyarray(std::move(peaks)).reshape(out_shape);
}

PYBIND11_MODULE(signal_proc, m)
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

    m.def("binterpolate", &binterpolate<float, float>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);
    m.def("binterpolate", &binterpolate<float, long>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);
    m.def("binterpolate", &binterpolate<double, double>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);
    m.def("binterpolate", &binterpolate<double, long>, py::arg("inp"), py::arg("grid"), py::arg("coords"), py::arg("num_threads") = 1);

    m.def("kr_predict", &kr_predict<float>, py::arg("y"), py::arg("x"), py::arg("x_hat"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);
    m.def("kr_predict", &kr_predict<double>, py::arg("y"), py::arg("x"), py::arg("x_hat"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);

    m.def("kr_grid", &kr_grid<float>, py::arg("y"), py::arg("x"), py::arg("grid"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);
    m.def("kr_grid", &kr_grid<double>, py::arg("y"), py::arg("x"), py::arg("grid"), py::arg("sigma"), py::arg("kernel") = "gaussian", py::arg("w") = nullptr, py::arg("num_threads") = 1);

    m.def("local_maxima", &local_maxima<int, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<int, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, int>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, std::vector<int>>, py::arg("inp"), py::arg("axis"), py::arg("num_threads") = 1);
}

}
