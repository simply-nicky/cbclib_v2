#include "label.hpp"
#include "zip.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::PointSetND<2>>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::PointSetND<3>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<cbclib::PointSetND<2>>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<cbclib::PointSetND<3>>>)

namespace cbclib {

template <size_t N>
auto dilate(py::array_t<bool> input, StructureND<N> structure, py::none seeds, size_t iterations,
            std::optional<py::array_t<bool>> m, std::optional<std::array<long, N>> ax, unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(output.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(output.ndim() - n);

    py::array_t<bool> mask;
    if (m) mask = m.value();
    else
    {
        mask.resize(std::vector<py::ssize_t>(output.shape(), output.shape() + output.ndim()), false);
        fill_array(mask, true);
    }

    output = axes.swap_back(output);
    mask = axes.swap_back(mask);
    array<bool> out {output.request()};
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < repeats; i++)
    {
        auto frame = out.slice_back(i, N);

        PointND<size_t, N> shape;
        for (size_t n = 0; n < N; n++) shape[n] = frame.shape(n);

        PointSetND<N> pixels;
        size_t index = 0;
        for (auto && pt : rectangle_range<PointND<long, N>, true>{std::move(shape)})
        {
            if (frame[index++]) pixels->emplace_hint(pixels.end(), std::forward<decltype(pt)>(pt));
        }

        auto func = [mask = marr.slice_back(i, N)](const PointND<long, N> & pt)
        {
            return mask.is_inbound(pt.coordinate()) && mask.at(pt.coordinate());
        };
        pixels.dilate(func, structure, iterations);
        pixels.mask(frame, true);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return axes.swap_from_back(output);
}

template <size_t N>
auto dilate_seeded(py::array_t<bool> input, StructureND<N> structure, PointSetND<N> seeds, size_t iterations,
                   std::optional<py::array_t<bool>> m, py::none ax, unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};

    py::array_t<bool> mask;
    if (m) mask = m.value();
    else
    {
        mask.resize(std::vector<py::ssize_t>(output.shape(), output.shape() + output.ndim()), false);
        fill_array(mask, true);
    }

    array<bool> out {output.request()};
    array<bool> marr {mask.request()};

    if (marr.ndim() != N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    py::gil_scoped_release release;

    auto func = [&marr](const PointND<long, N> & pt)
    {
        return marr.is_inbound(pt.coordinate()) && marr.at(pt.coordinate());
    };
    seeds.dilate(func, structure, iterations);
    seeds.mask(out, true);

    py::gil_scoped_acquire acquire;

    return output;
}

template <size_t N>
auto dilate_seeded_vec(py::array_t<bool> input, StructureND<N> structure, std::vector<PointSetND<N>> seeds, size_t iterations,
                       std::optional<py::array_t<bool>> m, std::optional<std::array<long, N>> ax, unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(output.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(output.ndim() - n);

    py::array_t<bool> mask;
    if (m) mask = m.value();
    else
    {
        mask.resize(std::vector<py::ssize_t>(output.shape(), output.shape() + output.ndim()), false);
        fill_array(mask, true);
    }

    output = axes.swap_back(output);
    mask = axes.swap_back(mask);
    array<bool> out {output.request()};
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());
    if (seeds.size() != repeats)
        throw std::invalid_argument("seeds length (" + std::to_string(seeds.size()) + ") is incompatible with mask shape");

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < repeats; i++)
    {
        auto func = [mask = marr.slice_back(i, N)](const PointND<long, N> & pt)
        {
            return mask.is_inbound(pt.coordinate()) && mask.at(pt.coordinate());
        };
        seeds[i].dilate(func, structure, iterations);
        seeds[i].mask(out.slice_back(i, N), true);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return axes.swap_from_back(output);
}

template <size_t N>
auto label(py::array_t<bool> mask, StructureND<N> structure, py::none seeds, size_t npts, std::optional<std::array<long, N>> ax, unsigned threads)
{
    // Deep copy of mask array
    mask = py::array_t<bool>{mask.request()};
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(mask.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(mask.ndim() - n);

    mask = axes.swap_back(mask);
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());

    std::vector<std::vector<PointSetND<N>>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<PointSetND<N>>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            auto frame = marr.slice_back(i, axes.size());
            auto func = [&frame](const PointND<long, N> & pt)
            {
                return frame.is_inbound(pt.coordinate()) && frame.at(pt.coordinate());
            };

            auto & regions = buffer.emplace_back();

            PointND<size_t, N> shape;
            for (size_t n = 0; n < N; n++) shape[n] = frame.shape(n);

            size_t index = 0;
            for (auto pt : rectangle_range<PointND<long, N>, true>{std::move(shape)})
            {
                if (frame[index++])
                {
                    PointSetND<N> points;
                    points->insert(pt);
                    points.dilate(func, structure);
                    points.mask(frame, false);
                    if (points.size() >= npts) regions.emplace_back(std::move(points));
                }
            }
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <size_t N>
auto label_seeded(py::array_t<bool> mask, StructureND<N> structure, PointSetND<N> seeds, size_t npts, py::none ax, unsigned threads)
{
    // Deep copy of mask array
    mask = py::array_t<bool>{mask.request()};
    array<bool> marr {mask.request()};

    if (marr.ndim() != N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    std::vector<std::vector<PointSetND<N>>> result;
    auto & regions = result.emplace_back();

    py::gil_scoped_release release;

    auto func = [&marr](const PointND<long, N> & pt)
    {
        return marr.is_inbound(pt.coordinate()) && marr.at(pt.coordinate());
    };

    for (auto pt : seeds)
    {
        size_t index = marr.index_at(pt.coordinate());
        if (marr[index])
        {
            PointSetND<N> points;
            points->insert(pt);
            points.dilate(func, structure);
            points.mask(marr, false);
            if (points.size() >= npts) regions.emplace_back(std::move(points));
        }
    }

    py::gil_scoped_acquire acquire;

    return result;
}

template <size_t N>
auto label_seeded_vec(py::array_t<bool> mask, StructureND<N> structure, std::vector<PointSetND<N>> seeds, size_t npts,
                      std::optional<std::array<long, N>> ax, unsigned threads)
{
    // Deep copy of mask array
    mask = py::array_t<bool>{mask.request()};
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(mask.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(mask.ndim() - n);

    mask = axes.swap_back(mask);
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());
    if (seeds.size() != repeats)
        throw std::invalid_argument("seeds length (" + std::to_string(seeds.size()) + ") is incompatible with mask shape");

    std::vector<std::vector<PointSetND<N>>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<PointSetND<N>>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            auto frame = marr.slice_back(i, axes.size());
            auto func = [&frame](const PointND<long, N> & pt)
            {
                return frame.is_inbound(pt.coordinate()) && frame.at(pt.coordinate());
            };

            auto & regions = buffer.emplace_back();

            for (auto pt : seeds[i])
            {
                size_t index = frame.index_at(pt.coordinate());
                if (frame[index])
                {
                    PointSetND<N> points;
                    points->insert(pt);
                    points.dilate(func, structure);
                    points.mask(frame, false);
                    if (points.size() >= npts) regions.emplace_back(std::move(points));
                }
            }
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            result.insert(result.end(), std::make_move_iterator(buffer.begin()), std::make_move_iterator(buffer.end()));
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T, size_t N, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, PixelsND<T, N>> && is_all_integral_v<Ix ...>
>>
py::array_t<T> apply(const std::vector<std::vector<PointSetND<N>>> & list, py::array_t<T> data, Func && func, std::optional<std::array<long, N>> ax, Ix... sizes)
{
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(data.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(data.ndim() - n);

    data = axes.swap_back(data);
    auto shape = normalise_shape<N>(std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});
    check_dimension("data", 0, shape.begin(), list.size());

    data = data.reshape(shape);
    array<T> darr {data.request()};

    size_t size = 0;
    std::vector<T> results;
    for (size_t i = 0; i < list.size(); i++)
    {
        auto frame = darr.slice_back(i, N);
        for (const auto & region : list[i])
        {
            auto result = std::forward<Func>(func)(PixelsND<T, N>{region, frame});
            results.insert(results.end(), result.begin(), result.end());
        }
        size += list[i].size();
    }

    return as_pyarray(std::move(results), std::array<size_t, 1 + sizeof...(Ix)>{size, static_cast<size_t>(sizes)...});
}

template <typename T, size_t N, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, const PixelsND<T, N> &> && is_all_integral_v<Ix...>
>>
void declare_region_func(py::module & m, Func && func, const std::string & funcstr, Ix... sizes)
{
    m.def(funcstr.c_str(), [f = std::forward<Func>(func), sizes...](std::vector<std::vector<PointSetND<N>>> regions, py::array_t<T> data, std::optional<std::array<long, N>> ax)
    {
        return apply(regions, std::move(data), f, ax, sizes...);
    }, py::arg("regions"), py::arg("data"), py::arg("axes")=std::nullopt);
}

template <typename T>
void declare_pixels(py::module & m, const std::string & typestr)
{
    py::class_<PixelsND<T, 2>>(m, (std::string("Pixels2D") + typestr).c_str())
        .def(py::init([](std::vector<long> x, std::vector<long> y, std::vector<T> values)
        {
            PixelSetND<T, 2> result;
            for (auto [x, y, val] : zip::zip(x, y, values)) result.insert(make_pixel(val, x, y));
            return PixelsND<T, 2>{std::move(result)};
        }), py::arg("x") = std::vector<long>{}, py::arg("y") = std::vector<long>{}, py::arg("value") = std::vector<T>{})
        .def(py::init([](py::array_t<long> x, py::array_t<long> y, py::array_t<T> values)
        {
            PixelSetND<T, 2> result;
            for (auto [x, y, val] : zip::zip(array<long>{x.request()}, array<long>{y.request()}, array<T>{values.request()}))
            {
                result.insert(make_pixel(val, x, y));
            }
            return PixelsND<T, 2>{std::move(result)};
        }), py::arg("x") = py::array_t<long>{}, py::arg("y") = py::array_t<long>{}, py::arg("value") = py::array_t<T>{})
        .def_property("x", [](const PixelsND<T, 2> & pixels)
        {
            std::vector<long> xvec;
            for (const auto & [pt, _]: pixels.pixels()) xvec.push_back(pt.x());
            return xvec;
        }, nullptr)
        .def_property("y", [](const PixelsND<T, 2> & pixels)
        {
            std::vector<long> yvec;
            for (const auto & [pt, _]: pixels.pixels()) yvec.push_back(pt.y());
            return yvec;
        }, nullptr)
        .def_property("value", [](const PixelsND<T, 2> & pixels)
        {
            std::vector<T> values;
            for (const auto & [_, val]: pixels.pixels()) values.push_back(val);
            return values;
        }, nullptr)
        .def("merge", [](PixelsND<T, 2> & pixels, PixelsND<T, 2> source) -> PixelsND<T, 2> &
        {
            pixels.merge(source);
            return pixels;
        }, py::arg("source"), py::return_value_policy::reference_internal)
        .def("total_mass", [](const PixelsND<T, 2> & pixels){return pixels.moments().zeroth();})
        .def("mean", [](const PixelsND<T, 2> & pixels){return pixels.moments().first();})
        .def("center_of_mass", [](const PixelsND<T, 2> & pixels){return pixels.moments().central().first();})
        .def("moment_of_inertia", [](const PixelsND<T, 2> & pixels){return pixels.moments().second();})
        .def("covariance_matrix", [](const PixelsND<T, 2> & pixels){return pixels.moments().central().second();})
        .def("__repr__", [typestr](const PixelsND<T, 2> & pixels)
        {
            return "<Pixels2D" + typestr + ", size = " + std::to_string(pixels.pixels().size()) + ">";
        });
}

}

PYBIND11_MODULE(label, m)
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

    py::class_<PointSetND<2>>(m, "PointSet2D")
        .def(py::init([](long x, long y)
        {
            std::set<PointND<long, 2>> points;
            points.insert(PointND<long, 2>{x, y});
            return PointSetND<2>(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def(py::init([](std::vector<long> xvec, std::vector<long> yvec)
        {
            std::set<PointND<long, 2>> points;
            for (auto [x, y] : zip::zip(xvec, yvec)) points.insert(PointND<long, 2>{x, y});
            return PointSetND<2>(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def_property("x", [](const PointSetND<2> & points){return detail::get_x(points, 0);}, nullptr)
        .def_property("y", [](const PointSetND<2> & points){return detail::get_x(points, 1);}, nullptr)
        .def("__contains__", [](const PointSetND<2> & points, std::array<long, 2> point)
        {
            if (points->find(PointND<long, 2>{point}) != points.end()) return true;
            return false;
        })
        .def("__iter__", [](const PointSetND<2> & points)
        {
            return py::make_iterator(make_python_iterator(points.begin()), make_python_iterator(points.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const PointSetND<2> & points){return points.size();})
        .def("__repr__", &PointSetND<2>::info);

    py::class_<PointSetND<3>>(m, "PointSet3D")
        .def(py::init([](long x, long y, long z)
        {
            std::set<PointND<long, 3>> points;
            points.insert(PointND<long, 3>{x, y, z});
            return PointSetND<3>(std::move(points));
        }), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(py::init([](std::vector<long> xvec, std::vector<long> yvec, std::vector<long> zvec)
        {
            std::set<PointND<long, 3>> points;
            for (auto [x, y, z] : zip::zip(xvec, yvec, zvec)) points.insert(PointND<long, 3>{x, y, z});
            return PointSetND<3>(std::move(points));
        }), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_property("x", [](const PointSetND<3> & points){return detail::get_x(points, 0);}, nullptr)
        .def_property("y", [](const PointSetND<3> & points){return detail::get_x(points, 1);}, nullptr)
        .def_property("z", [](const PointSetND<3> & points){return detail::get_x(points, 2);}, nullptr)
        .def("__contains__", [](const PointSetND<3> & points, std::array<long, 3> point)
        {
            if (points->find(PointND<long, 3>{point}) != points.end()) return true;
            return false;
        })
        .def("__iter__", [](const PointSetND<3> & points)
        {
            return py::make_iterator(make_python_iterator(points.begin()), make_python_iterator(points.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const PointSetND<3> & points){return points.size();})
        .def("__repr__", &PointSetND<3>::info);

    py::class_<StructureND<2>>(m, "Structure2D")
        .def(py::init<int, int>(), py::arg("radius"), py::arg("rank"))
        .def_readonly("radius", &StructureND<2>::radius)
        .def_readonly("rank", &StructureND<2>::rank)
        .def_property("x", [](const StructureND<2> & srt){return detail::get_x(srt, 0);}, nullptr)
        .def_property("y", [](const StructureND<2> & srt){return detail::get_x(srt, 1);}, nullptr)
        .def("__iter__", [](const StructureND<2> & srt)
        {
            return py::make_iterator(make_python_iterator(srt.begin()), make_python_iterator(srt.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const StructureND<2> & srt){return srt.size();})
        .def("__repr__", &StructureND<2>::info);

    py::class_<StructureND<3>>(m, "Structure3D")
        .def(py::init<int, int>(), py::arg("radius"), py::arg("rank"))
        .def_readonly("radius", &StructureND<3>::radius)
        .def_readonly("rank", &StructureND<3>::rank)
        .def_property("x", [](const StructureND<3> & srt){return detail::get_x(srt, 0);}, nullptr)
        .def_property("y", [](const StructureND<3> & srt){return detail::get_x(srt, 1);}, nullptr)
        .def_property("z", [](const StructureND<3> & srt){return detail::get_x(srt, 2);}, nullptr)
        .def("__iter__", [](const StructureND<3> & srt)
        {
            return py::make_iterator(make_python_iterator(srt.begin()), make_python_iterator(srt.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const StructureND<3> & srt){return srt.size();})
        .def("__repr__", &StructureND<3>::info);

    py::class_<std::vector<PointSetND<2>>> regions_2d (m, "Regions2D");
    declare_list(regions_2d, "Regions2D");
    regions_2d.def_property("x", [](const std::vector<PointSetND<2>> & regions)
        {
            std::vector<long> x;
            for (auto region : regions)
            {
                auto x_vec = detail::get_x(region, 0);
                x.insert(x.end(), x_vec.begin(), x_vec.end());
            }
            return x;
        }, nullptr)
        .def_property("y", [](const std::vector<PointSetND<2>> & regions)
        {
            std::vector<long> y;
            for (auto region : regions)
            {
                auto y_vec = detail::get_x(region, 1);
                y.insert(y.end(), y_vec.begin(), y_vec.end());
            }
            return y;
        }, nullptr);

    py::class_<std::vector<PointSetND<3>>> regions_3d (m, "Regions3D");
    declare_list(regions_3d, "Regions3D");
    regions_3d.def_property("x", [](const std::vector<PointSetND<3>> & regions)
        {
            std::vector<long> x;
            for (auto region : regions)
            {
                auto x_vec = detail::get_x(region, 0);
                x.insert(x.end(), x_vec.begin(), x_vec.end());
            }
            return x;
        }, nullptr)
        .def_property("y", [](const std::vector<PointSetND<3>> & regions)
        {
            std::vector<long> y;
            for (auto region : regions)
            {
                auto y_vec = detail::get_x(region, 1);
                y.insert(y.end(), y_vec.begin(), y_vec.end());
            }
            return y;
        }, nullptr)
        .def_property("z", [](const std::vector<PointSetND<3>> & regions)
        {
            std::vector<long> z;
            for (auto region : regions)
            {
                auto z_vec = detail::get_x(region, 2);
                z.insert(z.end(), z_vec.begin(), z_vec.end());
            }
            return z;
        }, nullptr);

    py::class_<std::vector<std::vector<PointSetND<2>>>> regions_list_2d (m, "RegionsList2D");
    declare_list(regions_list_2d, "RegionsList2D");
    regions_list_2d.def("frames", [](const std::vector<std::vector<PointSetND<2>>> & list)
        {
            std::vector<py::ssize_t> indices;
            for (size_t index = 0; index < list.size(); index++)
            {
                for (size_t i = 0; i < list[index].size(); i++) indices.push_back(index);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("index", [](const std::vector<std::vector<PointSetND<2>>> & list)
        {
            std::vector<py::ssize_t> indices;
            auto index = 0;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (size_t i = 0; i < region.size(); i++) indices.push_back(index);
                }
                index++;
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("x", [](const std::vector<std::vector<PointSetND<2>>> & list)
        {
            std::vector<long> x;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (const auto & point : region) x.push_back(point.x());
                }
            }
            return as_pyarray(std::move(x), std::array<size_t, 1>{x.size()});
        })
        .def("y", [](const std::vector<std::vector<PointSetND<2>>> & list)
        {
            std::vector<long> y;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (const auto & point : region) y.push_back(point.y());
                }
            }
            return as_pyarray(std::move(y), std::array<size_t, 1>{y.size()});
        });

    py::class_<std::vector<std::vector<PointSetND<3>>>> regions_list_3d (m, "RegionsList3D");
    declare_list(regions_list_3d, "RegionsList3D");
    regions_list_3d.def("frames", [](const std::vector<std::vector<PointSetND<3>>> & list)
        {
            std::vector<py::ssize_t> indices;
            for (size_t index = 0; index < list.size(); index++)
            {
                for (size_t i = 0; i < list[index].size(); i++) indices.push_back(index);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("index", [](const std::vector<std::vector<PointSetND<3>>> & list)
        {
            std::vector<py::ssize_t> indices;
            auto index = 0;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (size_t i = 0; i < region.size(); i++) indices.push_back(index);
                }
                index++;
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("x", [](const std::vector<std::vector<PointSetND<3>>> & list)
        {
            std::vector<long> x;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (const auto & point : region) x.push_back(point.x());
                }
            }
            return as_pyarray(std::move(x), std::array<size_t, 1>{x.size()});
        })
        .def("y", [](const std::vector<std::vector<PointSetND<3>>> & list)
        {
            std::vector<long> y;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (const auto & point : region) y.push_back(point.y());
                }
            }
            return as_pyarray(std::move(y), std::array<size_t, 1>{y.size()});
        })
        .def("z", [](const std::vector<std::vector<PointSetND<3>>> & list)
        {
            std::vector<long> z;
            for (const auto & regions : list)
            {
                for (const auto & region : regions)
                {
                    for (const auto & point : region) z.push_back(point.z());
                }
            }
            return as_pyarray(std::move(z), std::array<size_t, 1>{z.size()});
        });

    m.def("binary_dilation", &dilate<2>, py::arg("input"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_seeded<2>, py::arg("input"), py::arg("structure"), py::arg("seeds"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_seeded_vec<2>, py::arg("input"), py::arg("structure"), py::arg("seeds"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate<3>, py::arg("input"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_seeded<3>, py::arg("input"), py::arg("structure"), py::arg("seeds"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_seeded_vec<3>, py::arg("input"), py::arg("structure"), py::arg("seeds"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);

    m.def("label", &label<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_seeded<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_seeded_vec<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_seeded<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_seeded_vec<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);

    declare_pixels<float>(m, "Float");
    declare_pixels<double>(m, "Double");

    auto total_mass = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return std::array<T, 1>{region.moments().zeroth()};
    };

    declare_region_func<double, 2>(m, total_mass, "total_mass");
    declare_region_func<float, 2>(m, total_mass, "total_mass");
    declare_region_func<double, 3>(m, total_mass, "total_mass");
    declare_region_func<float, 3>(m, total_mass, "total_mass");

    auto mean = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().first();
    };

    declare_region_func<double, 2>(m, mean, "mean", 2);
    declare_region_func<float, 2>(m, mean, "mean", 2);
    declare_region_func<double, 3>(m, mean, "mean", 3);
    declare_region_func<float, 3>(m, mean, "mean", 3);

    auto center_of_mass = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().central().first();
    };

    declare_region_func<double, 2>(m, center_of_mass, "center_of_mass", 2);
    declare_region_func<float, 2>(m, center_of_mass, "center_of_mass", 2);
    declare_region_func<double, 3>(m, center_of_mass, "center_of_mass", 3);
    declare_region_func<float, 3>(m, center_of_mass, "center_of_mass", 3);

    auto moment_of_inertia = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().second();
    };

    declare_region_func<double, 2>(m, moment_of_inertia, "moment_of_inertia", 2, 2);
    declare_region_func<float, 2>(m, moment_of_inertia, "moment_of_inertia", 2, 2);
    declare_region_func<double, 3>(m, moment_of_inertia, "moment_of_inertia", 3, 3);
    declare_region_func<float, 3>(m, moment_of_inertia, "moment_of_inertia", 3, 3);

    auto covariance_matrix = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().central().second();
    };

    declare_region_func<double, 2>(m, covariance_matrix, "covariance_matrix", 2, 2);
    declare_region_func<float, 2>(m, covariance_matrix, "covariance_matrix", 2, 2);
    declare_region_func<double, 3>(m, covariance_matrix, "covariance_matrix", 3, 3);
    declare_region_func<float, 3>(m, covariance_matrix, "covariance_matrix", 3, 3);
}
