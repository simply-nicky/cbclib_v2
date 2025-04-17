#include "label.hpp"
#include "zip.hpp"

namespace cbclib {

template <size_t N>
auto binary_dilation(py::array_t<bool> input, StructureND<N> structure, size_t iterations,
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

    output = axes.swap_axes(output);
    mask = axes.swap_axes(mask);
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
        for (size_t index = 0; auto && pt : rectangle_range<PointND<long, N>, true>{std::move(shape)})
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

    return axes.swap_axes_back(output);
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

    mask = axes.swap_axes(mask);
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());

    std::vector<RegionsND<N>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<RegionsND<N>> buffer;

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
            for (size_t index = 0; auto pt : rectangle_range<PointND<long, N>, true>{std::move(shape)})
            {
                if (frame[index++])
                {
                    PointSetND<N> points;
                    points->insert(pt);
                    points.dilate(func, structure);
                    points.mask(frame, false);
                    if (points.size() >= npts) regions->emplace_back(std::move(points));
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
auto label_with_seeds(py::array_t<bool> mask, StructureND<N> structure, PointSetND<N> seeds, size_t npts, py::none ax, unsigned threads)
{
    // Deep copy of mask array
    mask = py::array_t<bool>{mask.request()};
    array<bool> marr {mask.request()};

    if (marr.ndim() != N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    RegionsND<N> result;

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
            if (points.size() >= npts) result->emplace_back(std::move(points));
        }
    }

    py::gil_scoped_acquire acquire;

    return result;
}

template <size_t N>
auto label_with_seeds_vec(py::array_t<bool> mask, StructureND<N> structure, std::vector<PointSetND<N>> seeds, size_t npts,
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

    mask = axes.swap_axes(mask);
    array<bool> marr {mask.request()};

    if (marr.ndim() < N)
        throw std::invalid_argument("mask array has wrong number of dimensions (" + std::to_string(marr.ndim()) + " < " + std::to_string(N) + ")");

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - N), 1, std::multiplies());
    if (seeds.size() != repeats)
        throw std::invalid_argument("seeds length (" + std::to_string(seeds.size()) + ") is incompatible with mask shape");

    std::vector<RegionsND<N>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<RegionsND<N>> buffer;

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
                    if (points.size() >= npts) regions->emplace_back(std::move(points));
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
    std::is_invocable_v<std::remove_cvref_t<Func>, PixelsND<T, N>>
>> requires is_all_integral<Ix...>
py::array_t<T> apply(const RegionsND<N> & regions, const array<T> & data, Func && func, Ix... sizes)
{
    std::vector<T> results;
    for (const auto & region : regions)
    {
        auto result = std::forward<Func>(func)(PixelsND<T, N>{region, data});
        results.insert(results.end(), result.begin(), result.end());
    }

    return as_pyarray(std::move(results), std::array<size_t, 1 + sizeof...(Ix)>{regions.size(), static_cast<size_t>(sizes)...});
}

template <typename T, size_t N, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, PixelsND<T, N>>
>> requires is_all_integral<Ix...>
std::vector<py::array_t<T>> apply_and_vectorise(const std::vector<RegionsND<N>> & stack, py::array_t<T> data, Func && func, std::optional<std::array<long, N>> ax, Ix... sizes)
{
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(data.ndim());
    }
    else for (long n = N; n > 0; n--) axes->push_back(data.ndim() - n);

    data = axes.swap_axes(data);
    auto dbuf = data.request();
    auto shape = normalise_shape<N>(dbuf.shape);
    check_dimensions("data", 0, shape, stack.size());

    array<T> darr {shape, static_cast<T *>(dbuf.ptr)};
    std::vector<py::array_t<T>> results;

    for (size_t i = 0; i < stack.size(); i++)
    {
        results.emplace_back(apply(stack[i], darr.slice_back(i, N), std::forward<Func>(func), sizes...));
    }

    return results;
}

template <typename T, size_t N, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, const PixelsND<T, N> &>
>> requires is_all_integral<Ix...>
void declare_region_func(py::module & m, Func && func, const std::string & funcstr, Ix... sizes)
{
    m.def(funcstr.c_str(), [f = std::forward<Func>(func), sizes...](RegionsND<N> regions, py::array_t<T> data, std::optional<std::array<long, N>> ax)
    {
        return apply(regions, array<T>{data.request()}, f, sizes...);
    }, py::arg("regions"), py::arg("data"), py::arg("axes")=std::nullopt);
    m.def(funcstr.c_str(), [f = std::forward<Func>(func), sizes...](std::vector<RegionsND<N>> regions, py::array_t<T> data, std::optional<std::array<long, N>> ax)
    {
        return apply_and_vectorise(regions, data, f, ax, sizes...);
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
        .def("merge", [](PixelsND<T, 2> & pixels, PixelsND<T, 2> source) -> PixelsND<T, 2>
        {
            pixels.merge(source);
            return pixels;
        }, py::arg("source"))
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
            return points->contains(PointND<long, 2>{point});
        })
        .def("__iter__", [](const PointSetND<2> & points)
        {
            return py::make_iterator(make_python_iterator(points.begin()), make_python_iterator(points.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const PointSetND<2> & points){return points.size();})
        .def("__repr__", &PointSetND<2>::info);

    py::class_<PointSetND<3>>(m, "PointSet3D")
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
            return points->contains(PointND<long, 3>{point});
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

    py::class_<RegionsND<2>>(m, "Regions2D")
        .def(py::init([](std::vector<PointSetND<2>> regions)
        {
            return RegionsND<2>(std::move(regions));
        }), py::arg("regions") = std::vector<PointSetND<2>>{}, py::keep_alive<1, 2>())
        .def_property("x", [](const RegionsND<2> & regions)
        {
            std::vector<long> x;
            for (auto region : regions)
            {
                auto x_vec = detail::get_x(region, 0);
                x.insert(x.end(), x_vec.begin(), x_vec.end());
            }
            return x;
        }, nullptr)
        .def_property("y", [](const RegionsND<2> & regions)
        {
            std::vector<long> y;
            for (auto region : regions)
            {
                auto y_vec = detail::get_x(region, 1);
                y.insert(y.end(), y_vec.begin(), y_vec.end());
            }
            return y;
        }, nullptr)
        .def("__delitem__", [](RegionsND<2> & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            regions->erase(std::next(regions.begin(), i));
        })
        .def("__getitem__", [](const RegionsND<2> & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            return (*regions)[i];
        })
        .def("__setitem__", [](RegionsND<2> & regions, size_t i, PointSetND<2> region)
        {
            if (i >= regions.size()) throw py::index_error();
            (*regions)[i] = std::move(region);
        }, py::keep_alive<1, 3>())
        .def("__delitem__", [](RegionsND<2> & regions, const py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                regions->erase(std::next(regions.begin(), start));
                start += step;
            }
        })
        .def("__getitem__", [](const RegionsND<2> & regions, const py::slice & slice) -> RegionsND<2>
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            RegionsND<2> new_regions {};
            for (size_t i = 0; i < slicelength; ++i)
            {
                new_regions->push_back((*regions)[start]);
                start += step;
            }
            return new_regions;
        })
        .def("__setitem__", [](RegionsND<2> & regions, const py::slice & slice, const RegionsND<2> & value)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                (*regions)[start] = (*value)[i];
                start += step;
            }
        }, py::keep_alive<1, 3>())
        .def("__iter__", [](RegionsND<2> & regions)
        {
            return py::make_iterator(regions.begin(), regions.end());
        })
        .def("__len__", [](RegionsND<2> & regions){return regions.size();})
        .def("__repr__", &RegionsND<2>::info)
        .def("append", [](RegionsND<2> & regions, PointSetND<2> region)
        {
            regions->emplace_back(std::move(region));
        }, py::keep_alive<1, 2>(), py::arg("region"), py::keep_alive<1, 2>());

    py::class_<RegionsND<3>>(m, "Regions3D")
        .def(py::init([](std::vector<PointSetND<3>> regions)
        {
            return RegionsND<3>(std::move(regions));
        }), py::arg("regions") = std::vector<PointSetND<3>>{}, py::keep_alive<1, 2>())
        .def_property("x", [](const RegionsND<3> & regions)
        {
            std::vector<long> x;
            for (auto region : regions)
            {
                auto x_vec = detail::get_x(region, 0);
                x.insert(x.end(), x_vec.begin(), x_vec.end());
            }
            return x;
        }, nullptr)
        .def_property("y", [](const RegionsND<3> & regions)
        {
            std::vector<long> y;
            for (auto region : regions)
            {
                auto y_vec = detail::get_x(region, 1);
                y.insert(y.end(), y_vec.begin(), y_vec.end());
            }
            return y;
        }, nullptr)
        .def_property("z", [](const RegionsND<3> & regions)
        {
            std::vector<long> z;
            for (auto region : regions)
            {
                auto z_vec = detail::get_x(region, 2);
                z.insert(z.end(), z_vec.begin(), z_vec.end());
            }
            return z;
        }, nullptr)
        .def("__delitem__", [](RegionsND<3> & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            regions->erase(std::next(regions.begin(), i));
        })
        .def("__getitem__", [](const RegionsND<3> & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            return (*regions)[i];
        })
        .def("__setitem__", [](RegionsND<3> & regions, size_t i, PointSetND<3> region)
        {
            if (i >= regions.size()) throw py::index_error();
            (*regions)[i] = std::move(region);
        }, py::keep_alive<1, 3>())
        .def("__delitem__", [](RegionsND<3> & regions, const py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                regions->erase(std::next(regions.begin(), start));
                start += step;
            }
        })
        .def("__getitem__", [](const RegionsND<3> & regions, const py::slice & slice) -> RegionsND<3>
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            RegionsND<3> new_regions {};
            for (size_t i = 0; i < slicelength; ++i)
            {
                new_regions->push_back((*regions)[start]);
                start += step;
            }
            return new_regions;
        })
        .def("__setitem__", [](RegionsND<3> & regions, const py::slice & slice, const RegionsND<3> & value)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i)
            {
                (*regions)[start] = (*value)[i];
                start += step;
            }
        }, py::keep_alive<1, 3>())
        .def("__iter__", [](RegionsND<3> & regions)
        {
            return py::make_iterator(regions.begin(), regions.end());
        })
        .def("__len__", [](RegionsND<3> & regions){return regions.size();})
        .def("__repr__", &RegionsND<3>::info)
        .def("append", [](RegionsND<3> & regions, PointSetND<3> region)
        {
            regions->emplace_back(std::move(region));
        }, py::keep_alive<1, 2>(), py::arg("region"), py::keep_alive<1, 2>());

    m.def("binary_dilation", &binary_dilation<2>, py::arg("input"), py::arg("structure") = StructureND<2>{1, 1}, py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &binary_dilation<3>, py::arg("input"), py::arg("structure") = StructureND<3>{1, 1}, py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);

    m.def("label", &label<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_with_seeds<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("threads") = 1);
    m.def("label", &label_with_seeds_vec<2>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds") = std::nullopt, py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label_with_seeds<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("threads") = 1);
    m.def("label", &label_with_seeds_vec<3>, py::arg("mask"), py::arg("structure"), py::arg("seeds"), py::arg("npts") = 1, py::arg("axes") = std::nullopt, py::arg("threads") = 1);

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
