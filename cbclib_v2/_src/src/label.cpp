#include "label.hpp"
#include "zip.hpp"

namespace cbclib {

template <typename InputIt>
Regions labelise(InputIt first, InputIt last, array<bool> && mask, const Structure & structure, size_t npts)
{
    std::vector<PointsSet> regions;

    for (; first != last; ++first)
    {
        size_t index = mask.index_at(first->coordinate());
        if (mask[index])
        {
            PointsSet points {*first, mask, structure};
            points.mask(mask, false);
            if (points.size() >= npts) regions.emplace_back(std::move(points));
        }
    }

    return Regions{std::move(regions)};
}

template <typename T>
auto label(py::array_t<bool> mask, Structure structure, size_t npts, std::optional<std::vector<PointsSet>> seeds,
           std::optional<std::array<long, 2>> ax, unsigned threads)
{
    // Deep copy of mask array
    mask = py::array_t<bool>{mask.request()};
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(mask.ndim());
    }
    else axes = {mask.ndim() - 2, mask.ndim() - 1};

    mask = axes.swap_axes(mask);
    array<bool> marr {mask.request()};

    if (marr.ndim() < 2)
        fail_container_check("wrong number of dimensions(" + std::to_string(marr.ndim()) + " < 2)", marr.shape());

    size_t repeats = std::reduce(marr.shape().begin(), std::next(marr.shape().begin(), marr.ndim() - 2), 1, std::multiplies());
    if (seeds && seeds.value().size() != repeats)
        throw std::invalid_argument("seeds length (" + std::to_string(seeds.value().size()) + ") is incompatible with mask shape");

    std::vector<Regions> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Regions> buffer;

        if (seeds)
        {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < repeats; i++)
            {
                buffer.emplace_back(labelise(seeds.value()[i].begin(), seeds.value()[i].end(), marr.slice_back(i, axes.size()), structure, npts));
            }
        }
        else
        {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < repeats; i++)
            {
                rectangle_range<Point<long>> range {Point<size_t>{marr.shape(axes[0]), marr.shape(axes[1])}};
                buffer.emplace_back(labelise(range.begin(), range.end(), marr.slice_back(i, axes.size()), structure, npts));
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

template <typename T, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, Pixels<T>>
>> requires is_all_integral<Ix...>
py::array_t<T> apply(const Regions & regions, const array<T> & data, Func && func, Ix... sizes)
{
    std::vector<T> results;
    for (const auto & region : regions)
    {
        auto result = std::forward<Func>(func)(Pixels<T>{region, data});
        results.insert(results.end(), result.begin(), result.end());
    }

    return as_pyarray(std::move(results), std::array<size_t, 1 + sizeof...(Ix)>{regions.size(), static_cast<size_t>(sizes)...});
}

template <typename T, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, Pixels<T>>
>> requires is_all_integral<Ix...>
std::vector<py::array_t<T>> apply_and_vectorise(const std::vector<Regions> & stack, py::array_t<T> data, Func && func, std::optional<std::array<long, 2>> ax, Ix... sizes)
{
    Sequence<long> axes;
    if (ax)
    {
        axes = ax.value();
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    auto dbuf = data.request();
    auto shape = normalise_shape<2>(dbuf.shape);
    check_dimensions("data", 0, shape, stack.size());

    array<T> darr {shape, static_cast<T *>(dbuf.ptr)};
    std::vector<py::array_t<T>> results;

    for (size_t i = 0; i < stack.size(); i++)
    {
        results.emplace_back(apply(stack[i], darr.slice_back(i, axes.size()), std::forward<Func>(func), sizes...));
    }

    return results;
}

template <typename T, typename Func, typename... Ix, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, const Pixels<T> &>
>> requires is_all_integral<Ix...>
void declare_region_func(py::module & m, Func && func, const std::string & funcstr, Ix... sizes)
{
    m.def(funcstr.c_str(), [f = std::forward<Func>(func), sizes...](Regions regions, py::array_t<T> data, std::optional<std::array<long, 2>> ax)
    {
        return apply(regions, array<T>{data.request()}, f, sizes...);
    }, py::arg("regions"), py::arg("data"), py::arg("axes")=std::nullopt);
    m.def(funcstr.c_str(), [f = std::forward<Func>(func), sizes...](std::vector<Regions> regions, py::array_t<T> data, std::optional<std::array<long, 2>> ax)
    {
        return apply_and_vectorise(regions, data, f, ax, sizes...);
    }, py::arg("regions"), py::arg("data"), py::arg("axes")=std::nullopt);
}

template <typename T>
void declare_pixels(py::module & m, const std::string & typestr)
{
    py::class_<Pixels<T>>(m, (std::string("Pixels") + typestr).c_str())
        .def(py::init([](std::vector<long> x, std::vector<long> y, std::vector<T> values)
        {
            PixelSet<T> result;
            for (auto [x, y, val] : zip::zip(x, y, values)) result.insert(make_pixel(x, y, val));
            return Pixels<T>{std::move(result)};
        }), py::arg("x") = std::vector<long>{}, py::arg("y") = std::vector<long>{}, py::arg("value") = std::vector<T>{})
        .def(py::init([](py::array_t<long> x, py::array_t<long> y, py::array_t<T> values)
        {
            PixelSet<T> result;
            for (auto [x, y, val] : zip::zip(array<long>{x.request()}, array<long>{y.request()}, array<T>{values.request()}))
            {
                result.insert(make_pixel(x, y, val));
            }
            return Pixels<T>{std::move(result)};
        }), py::arg("x") = py::array_t<long>{}, py::arg("y") = py::array_t<long>{}, py::arg("value") = py::array_t<T>{})
        .def_property("x", [](const Pixels<T> & pixels)
        {
            std::vector<long> xvec;
            for (const auto & [pt, _]: pixels.pixels()) xvec.push_back(pt.x());
            return xvec;
        }, nullptr)
        .def_property("y", [](const Pixels<T> & pixels)
        {
            std::vector<long> yvec;
            for (const auto & [pt, _]: pixels.pixels()) yvec.push_back(pt.y());
            return yvec;
        }, nullptr)
        .def_property("value", [](const Pixels<T> & pixels)
        {
            std::vector<T> values;
            for (const auto & [_, val]: pixels.pixels()) values.push_back(val);
            return values;
        }, nullptr)
        .def("merge", [](Pixels<T> & pixels, Pixels<T> source) -> Pixels<T>
        {
            pixels.merge(source);
            return pixels;
        }, py::arg("source"))
        .def("total_mass", [](const Pixels<T> & pixels){return pixels.moments().zeroth();})
        .def("mean", [](const Pixels<T> & pixels){return pixels.moments().first();})
        .def("center_of_mass", [](const Pixels<T> & pixels){return pixels.moments().central().first();})
        .def("moment_of_inertia", [](const Pixels<T> & pixels){return pixels.moments().second();})
        .def("covariance_matrix", [](const Pixels<T> & pixels){return pixels.moments().central().second();})
        .def("__repr__", [typestr](const Pixels<T> & pixels)
        {
            return "<Pixels" + typestr + ", size = " + std::to_string(pixels.pixels().size()) + ">";
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

    py::class_<PointsSet>(m, "PointsSet")
        .def(py::init([](std::vector<long> xvec, std::vector<long> yvec)
        {
            std::set<Point<long>> points;
            for (auto [x, y] : zip::zip(xvec, yvec)) points.insert(Point<long>{x, y});
            return PointsSet(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def(py::init([](py::array_t<long> xarr, py::array_t<long> yarr)
        {
            std::set<Point<long>> points;
            for (auto [x, y] : zip::zip(array<long>{xarr.request()}, array<long>{yarr.request()})) points.insert(Point<long>{x, y});
            return PointsSet(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def_property("x", [](const PointsSet & points){return detail::get_x(points);}, nullptr)
        .def_property("y", [](const PointsSet & points){return detail::get_y(points);}, nullptr)
        .def("__repr__", &PointsSet::info);

    py::class_<Structure>(m, "Structure")
        .def(py::init<int, int>(), py::arg("radius"), py::arg("rank"))
        .def_readonly("radius", &Structure::radius)
        .def_readonly("rank", &Structure::rank)
        .def_property("x", [](const Structure & srt){return detail::get_x(srt);}, nullptr)
        .def_property("y", [](const Structure & srt){return detail::get_y(srt);}, nullptr)
        .def("__repr__", &Structure::info);

    py::class_<Regions>(m, "Regions")
        .def(py::init([](std::vector<PointsSet> regions)
        {
            return Regions(std::move(regions));
        }), py::arg("regions") = std::vector<PointsSet>{}, py::keep_alive<1, 2>())
        .def_property("x", [](const Regions & regions)
        {
            std::vector<long> x;
            for (auto region : regions)
            {
                auto x_vec = detail::get_x(region);
                x.insert(x.end(), x_vec.begin(), x_vec.end());
            }
            return x;
        }, nullptr)
        .def_property("y", [](const Regions & regions)
        {
            std::vector<long> y;
            for (auto region : regions)
            {
                auto y_vec = detail::get_y(region);
                y.insert(y.end(), y_vec.begin(), y_vec.end());
            }
            return y;
        }, nullptr)
        .def("__delitem__", [](Regions & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            regions->erase(std::next(regions.begin(), i));
        })
        .def("__getitem__", [](const Regions & regions, size_t i)
        {
            if (i >= regions.size()) throw py::index_error();
            return (*regions)[i];
        })
        .def("__setitem__", [](Regions & regions, size_t i, PointsSet region)
        {
            if (i >= regions.size()) throw py::index_error();
            (*regions)[i] = std::move(region);
        }, py::keep_alive<1, 3>())
        .def("__delitem__", [](Regions & regions, const py::slice & slice)
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
        .def("__getitem__", [](const Regions & regions, const py::slice & slice) -> Regions
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(regions.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            Regions new_regions {};
            for (size_t i = 0; i < slicelength; ++i)
            {
                new_regions->push_back((*regions)[start]);
                start += step;
            }
            return new_regions;
        })
        .def("__setitem__", [](Regions & regions, const py::slice & slice, const Regions & value)
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
        .def("__iter__", [](Regions & regions)
        {
            return py::make_iterator(regions.begin(), regions.end());
        })
        .def("__len__", [](Regions & regions){return regions.size();})
        .def("__repr__", &Regions::info)
        .def("append", [](Regions & regions, PointsSet region){regions->emplace_back(std::move(region));}, py::keep_alive<1, 2>(), py::arg("region"), py::keep_alive<1, 2>());

    declare_pixels<float>(m, "Float");
    declare_pixels<double>(m, "Double");

    m.def("label", &label<float>, py::arg("mask"), py::arg("structure"), py::arg("npts") = 1, py::arg("seeds") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);
    m.def("label", &label<double>, py::arg("mask"), py::arg("structure"), py::arg("npts") = 1, py::arg("seeds") = std::nullopt, py::arg("axes") = std::nullopt, py::arg("num_threads") = 1);

    auto total_mass = []<typename T>(const Pixels<T> & region)
    {
        return std::array<T, 1>{region.moments().zeroth()};
    };

    declare_region_func<double>(m, total_mass, "total_mass");
    declare_region_func<float>(m, total_mass, "total_mass");

    auto mean = []<typename T>(const Pixels<T> & region)
    {
        return region.moments().first();
    };

    declare_region_func<double>(m, mean, "mean", 2);
    declare_region_func<float>(m, mean, "mean", 2);

    auto center_of_mass = []<typename T>(const Pixels<T> & region)
    {
        return region.moments().central().first();
    };

    declare_region_func<double>(m, center_of_mass, "center_of_mass", 2);
    declare_region_func<float>(m, center_of_mass, "center_of_mass", 2);

    auto moment_of_inertia = []<typename T>(const Pixels<T> & region)
    {
        return region.moments().second();
    };

    declare_region_func<double>(m, moment_of_inertia, "moment_of_inertia", 2, 2);
    declare_region_func<float>(m, moment_of_inertia, "moment_of_inertia", 2, 2);

    auto covariance_matrix = []<typename T>(const Pixels<T> & region)
    {
        return region.moments().central().second();
    };

    declare_region_func<double>(m, covariance_matrix, "covariance_matrix", 2, 2);
    declare_region_func<float>(m, covariance_matrix, "covariance_matrix", 2, 2);

}
