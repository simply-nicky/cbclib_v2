#include "label.hpp"
#include "zip.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Region>)

namespace cbclib {

auto dilate(py::array_t<bool> input, Structure structure, size_t iterations, py::none mask,
            unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};

    array<bool> inp {input.request()};
    array<bool> out {output.request()};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        auto func = [](long index){ return true; };

        // Finding the chunk for each thread
        int thread_id = omp_get_thread_num();
        size_t chunk = (out.size() + threads - 1) / threads;

        size_t thread_start = thread_id * chunk;
        size_t thread_end = std::min((thread_id + 1) * chunk, out.size());

        // Each thread processes its own chunk
        for (size_t index = thread_start; index < thread_end; index++)
        {
            if (inp[index])
            {
                Region region;
                region.insert(region.end(), index);
                region.dilate(func, structure, iterations, out.shape());

                for (auto index : region)
                {
                    if (index >= thread_start && index < thread_end) out[index] = true;
                }
            }
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return output;
}

auto dilate_with_mask(py::array_t<bool> input, Structure structure, size_t iterations, py::array_t<bool> mask,
                      unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};

    check_equal("input and mask must have the same shape",
                output.shape(), output.shape() + output.ndim(),
                mask.shape(), mask.shape() + mask.ndim());

    array<bool> out {output.request()};
    array<bool> marr {mask.request()};
    array<bool> inp {input.request()};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        auto func = [&marr](long index){ return true; };

        // Finding the chunk for each thread
        int thread_id = omp_get_thread_num();
        size_t chunk = (out.size() + threads - 1) / threads;

        size_t thread_start = thread_id * chunk;
        size_t thread_end = std::min((thread_id + 1) * chunk, out.size());

        // Each thread processes its own chunk
        for (size_t index = thread_start; index < thread_end; index++)
        {
            if (inp[index])
            {
                Region region;
                region.insert(region.end(), index);
                region.dilate(func, structure, iterations, out.shape());

                for (auto index : region)
                {
                    if (index >= thread_start && index < thread_end) out[index] = true;
                }
            }
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return output;
}

LabelResult label(py::array_t<bool> input, Structure structure, size_t npts, unsigned threads)
{
    // Deep copy of input array
    auto output = py::array_t<bool>{input.request()};
    array<bool> out {output.request()};
    array<bool> inp {input.request()};

    if (out.ndim() != structure.rank())
    {
        throw std::invalid_argument("input array dimension (" + std::to_string(out.ndim()) +
                                    ") does not match structure rank (" + std::to_string(structure.rank()) + ")");
    }

    std::vector<Region> result;
    std::vector<std::vector<Region>> thread_buffers(threads);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        auto func = [&inp](long index){ return inp[index]; };
        std::vector<Region> buffer;

        // Finding the chunk for each thread
        int thread_id = omp_get_thread_num();
        size_t chunk = (out.size() + threads - 1) / threads;

        size_t thread_start = thread_id * chunk;
        size_t thread_end = std::min((thread_id + 1) * chunk, out.size());

        // Each thread processes its own chunk
        for (size_t index = thread_start; index < thread_end; index++)
        {
            if (out[index])
            {
                Region region;
                region.insert(region.end(), index);
                region.dilate(func, structure, out.shape());

                for (auto index : region)
                {
                    if (index >= thread_start && index < thread_end) out[index] = false;
                }

                auto min_index = *region.begin();
                if (region.size() >= npts && min_index >= thread_start && min_index < thread_end)
                {
                    buffer.emplace_back(std::move(region));
                }
            }
        }

        thread_buffers[thread_id] = std::move(buffer);
    }

    // I need to keep the order of regions in the output the same as in the input
    for (auto & buffer : thread_buffers)
    {
        result.insert(result.end(), std::make_move_iterator(buffer.begin()),
                      std::make_move_iterator(buffer.end()));
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return LabelResult(std::vector<py::ssize_t>(output.shape(), output.shape() + output.ndim()), std::move(result));
}

template <typename T, size_t N, typename Func, typename = std::enable_if_t<
    std::is_invocable_v<remove_cvref_t<Func>, PixelsND<T, N>>
>>
py::array_t<T> apply_impl(const std::vector<Region> & regions, py::array_t<T> data, Func && func)
{
    array<T> darr {data.request()};

    std::vector<T> results;
    for (const auto & region : regions)
    {
        auto result = std::forward<Func>(func)(PixelsND<T, N>{region, darr});

        results.insert(results.end(), result.begin(), result.end());
    }

    if (results.size())
    {
        auto item_size = results.size() / regions.size();
        std::vector<size_t> shape {regions.size(), item_size};
        return as_pyarray(std::move(results), shape);
    }
    return py::array_t<T>{};
}

template <typename T, typename Func>
py::array_t<T> apply(const std::vector<Region> & regions, py::array_t<T> data, Func && func)
{
    switch(data.ndim())
    {
        case 2: return apply_impl<T, 2>(regions, data, std::forward<Func>(func));
        case 3: return apply_impl<T, 3>(regions, data, std::forward<Func>(func));
        case 4: return apply_impl<T, 4>(regions, data, std::forward<Func>(func));
        case 5: return apply_impl<T, 5>(regions, data, std::forward<Func>(func));
        case 6: return apply_impl<T, 6>(regions, data, std::forward<Func>(func));
        case 7: return apply_impl<T, 7>(regions, data, std::forward<Func>(func));
        default:
            throw std::invalid_argument("Unsupported number of dimensions: " + std::to_string(data.ndim()));
    }
}

template <typename T, typename Func>
void declare_region_func(py::module & m, Func && func, const std::string & funcstr)
{
    m.def(funcstr.c_str(), [f = std::forward<Func>(func)](LabelResult labels, py::array_t<T> data)
    {
        return apply(labels.regions(), std::move(data), f);
    }, py::arg("labels"), py::arg("data"));
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

    py::class_<Region>(m, "Region")
        .def(py::init())
        .def(py::init([](py::ssize_t index, const Structure & structure, std::vector<py::ssize_t> shape)
        {
            return Region(index, structure, shape);
        }), py::arg("index"), py::arg("structure"), py::arg("shape"))
        .def("__iter__", [](const Region & region)
        {
            return py::make_iterator(region.begin(), region.end());
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Region & region){return region.size();})
        .def("__repr__", &Region::info);

    py::class_<Structure>(m, "Structure")
        .def(py::init<const std::vector<py::ssize_t> &, int>(), py::arg("radii"), py::arg("connectivity"))
        .def_readonly("connectivity", &Structure::connectivity)
        .def_property_readonly("rank", [](const Structure & srt){ return srt.rank(); })
        .def_property_readonly("shape", [](const Structure & srt){ return srt.shape(); })
        .def("__iter__", [](const Structure & srt)
        {
            auto func = [](const typename Structure::const_reference & chunk)
            {
                return std::vector<long>(chunk.begin(), chunk.end());
            };
            return py::make_iterator(make_transform_iterator(srt.begin(), func), make_transform_iterator(srt.end(), func));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Structure & srt){return srt.size();})
        .def("__repr__", &Structure::info)
        .def("squeeze", [](const Structure & srt)
        {
            std::vector<py::ssize_t> new_shape;
            for (auto dim : srt.shape()) if (dim > 1) new_shape.push_back(static_cast<py::ssize_t>(dim / 2));
            return Structure{new_shape, srt.connectivity};
        })
        .def("expand_dims", [](const Structure & srt, size_t axis)
        {
            std::vector<py::ssize_t> new_shape;
            for (auto dim : srt.shape()) new_shape.push_back(static_cast<py::ssize_t>(dim / 2));

            axis = compute_index(axis, new_shape.size() + 1, "axis out of bounds for expand_dims");
            new_shape.insert(new_shape.begin() + axis, 0);

            return Structure{new_shape, srt.connectivity};
        }, py::arg("axis") = 0)
        .def("expand_dims", [](const Structure & srt, std::vector<py::ssize_t> axes)
        {
            std::vector<py::ssize_t> new_shape;
            for (auto dim : srt.shape()) new_shape.push_back(static_cast<py::ssize_t>(dim / 2));

            for (size_t i = 0; i < axes.size(); ++i)
            {
                axes[i] = compute_index(axes[i], new_shape.size() + axes.size(), "axis out of bounds for expand_dims");
            }
            std::sort(axes.begin(), axes.end());
            for (auto axis : axes) new_shape.insert(new_shape.begin() + axis, 0);

            return Structure{new_shape, srt.connectivity};
        }, py::arg("axes"))
        .def("to_array", [](const Structure & srt, py::none out)
        {
            py::array_t<bool> result (srt.shape());
            fill_array(result, false);
            array<bool> rarr {result.request()};

            std::vector<py::ssize_t> center;
            for (size_t n = 0; n < srt.rank(); ++n) center.push_back(static_cast<py::ssize_t>(srt.shape(n)) / 2);

            std::vector<py::ssize_t> coord (srt.rank());
            for (const auto & shift : srt)
            {
                for (size_t n = 0; n < srt.rank(); ++n) coord[n] = shift[n] + center[n];
                rarr.at(coord) = true;
            }
            return result;
        }, py::arg("out") = py::none())
        .def("to_array", [](const Structure & srt, py::array_t<bool> out) -> py::array_t<bool>
        {
            if (out.ndim() != static_cast<py::ssize_t>(srt.rank()))
            {
                throw std::invalid_argument("output array dimension (" + std::to_string(out.ndim()) +
                                            ") does not match structure rank (" + std::to_string(srt.rank()) + ")");
            }
            for (size_t n = 0; n < out.ndim(); ++n)
            {
                if (out.shape(n) < static_cast<py::ssize_t>(srt.shape(n)))
                {
                    throw std::invalid_argument("output array shape is smaller than structure shape "
                                                "at dimension " + std::to_string(n));
                }
            }

            array<bool> oarr {out.request()};

            std::vector<py::ssize_t> center;
            for (size_t n = 0; n < srt.rank(); ++n) center.push_back(static_cast<py::ssize_t>(oarr.shape(n)) / 2);

            std::vector<py::ssize_t> coord (srt.rank());
            for (const auto & shift : srt)
            {
                for (size_t n = 0; n < srt.rank(); ++n) coord[n] = shift[n] + center[n];
                oarr.at(coord) = true;
            }
            return out;
        }, py::arg("out"));

    py::class_<std::vector<Region>> regions (m, "Regions");
    declare_list(regions, "Regions");

    py::class_<LabelResult>(m, "LabelResult")
        .def_property("regions", [](const LabelResult & labels){ return labels.regions(); }, [](LabelResult & labels, std::vector<Region> regions){ labels.regions() = std::move(regions); })
        .def_property_readonly("shape", [](const LabelResult & labels){ return labels.shape(); })
        .def("to_mask", [](const LabelResult & labels, py::array_t<py::ssize_t> index, py::none out) -> py::array_t<py::ssize_t>
        {
            if (index.size() != static_cast<py::ssize_t>(labels.regions().size()))
                throw std::invalid_argument("Index array size does not match number of regions");

            py::array_t<py::ssize_t> result {labels.shape()};
            fill_array(result, py::ssize_t(0));
            array<py::ssize_t> oarr {result.request()};
            array<py::ssize_t> iarr {index.request()};

            size_t counter = 0;
            for (const auto & region : labels.regions())
            {
                region.mask(oarr, iarr[counter++]);
            }
            return result;
        }, py::arg("index"), py::arg("out") = py::none())
        .def("to_mask", [](const LabelResult & labels, py::array_t<int> index, py::array_t<int> out) -> py::array_t<int>
        {
            if (index.size() != static_cast<py::ssize_t>(labels.regions().size()))
                throw std::invalid_argument("Index array size does not match number of regions");
            check_equal("out array shape does not match region shape",
                        out.shape(), out.shape() + out.ndim(),
                        labels.shape().begin(), labels.shape().end());

            array<int> oarr {out.request()};
            array<int> iarr {index.request()};

            size_t counter = 0;
            for (const auto & region : labels.regions())
            {
                region.mask(oarr, iarr[counter++]);
            }
            return out;
        }, py::arg("index"), py::arg("out"));

    py::class_<PixelsND<double, 2>>(m, "Pixels2D")
        .def(py::init())
        .def(py::init([](Region region, py::array_t<double> data)
        {
            return PixelsND<double, 2>{std::move(region), array<double>{data.request()}};
        }), py::arg("region"), py::arg("data"))
        .def_property_readonly("region", [](const PixelsND<double, 2> & pixels){ return pixels.region(); })
        .def("merge", [](PixelsND<double, 2> & pixels, PixelsND<double, 2> other, py::array_t<double> data)
        {
            pixels.merge(other, array<double>{data.request()});
        }, py::arg("other"), py::arg("data"))
        .def("total_mass", [](const PixelsND<double, 2> & pixels)
        {
            return pixels.moments().zeroth();
        })
        .def("mean", [](const PixelsND<double, 2> & pixels)
        {
            return pixels.moments().first();
        })
        .def("center_of_mass", [](const PixelsND<double, 2> & pixels)
        {
            return pixels.moments().central().first();
        })
        .def("moment_of_inertia", [](const PixelsND<double, 2> & pixels)
        {
            return pixels.moments().second();
        })
        .def("covariance_matrix", [](const PixelsND<double, 2> & pixels)
        {
            return pixels.moments().central().second();
        });

    m.def("binary_dilation", &dilate, py::arg("inp"), py::arg("structure"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_with_mask, py::arg("inp"), py::arg("structure"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("num_threads") = 1);

    m.def("label", &label, py::arg("inp"), py::arg("structure"), py::arg("npts") = 1, py::arg("num_threads") = 1);

    auto total_mass = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return std::array<T, 1>{region.moments().zeroth()};
    };

    declare_region_func<double>(m, total_mass, "total_mass");
    declare_region_func<float>(m, total_mass, "total_mass");

    auto mean = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().first();
    };

    declare_region_func<double>(m, mean, "mean");
    declare_region_func<float>(m, mean, "mean");

    auto center_of_mass = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().central().first();
    };

    declare_region_func<double>(m, center_of_mass, "center_of_mass");
    declare_region_func<float>(m, center_of_mass, "center_of_mass");

    auto moment_of_inertia = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().second();
    };

    declare_region_func<double>(m, moment_of_inertia, "moment_of_inertia");
    declare_region_func<float>(m, moment_of_inertia, "moment_of_inertia");

    auto covariance_matrix = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().central().second();
    };

    declare_region_func<double>(m, covariance_matrix, "covariance_matrix");
    declare_region_func<float>(m, covariance_matrix, "covariance_matrix");

    auto line_fit = []<typename T, size_t N>(const PixelsND<T, N> & region)
    {
        return region.moments().central().line().to_array();
    };

    declare_region_func<double>(m, line_fit, "line_fit");
    declare_region_func<float>(m, line_fit, "line_fit");
}
