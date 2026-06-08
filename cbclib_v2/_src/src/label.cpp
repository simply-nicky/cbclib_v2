#include "label.hpp"
#include "zip.hpp"

namespace cbclib {

auto dilate_impl(py::array_t<bool> input, Structure structure, py::ssize_t iterations,
                 std::optional<py::array_t<bool>> mask, unsigned threads)
{
    array<bool> inp {input.request()};
    if (iterations < 0) throw std::invalid_argument("iterations must be non-negative");
    if (input.ndim() != structure.rank())
    {
        throw std::invalid_argument("input array dimension (" + std::to_string(input.ndim()) +
                                    ") does not match structure rank (" + std::to_string(structure.rank()) + ")");
    }

    py::array_t<bool> output {std::vector<py::ssize_t>(input.shape(), input.shape() + input.ndim())};
    array<bool> out {output.request()};

    std::optional<array<bool>> marr;
    if (mask)
    {
        marr.emplace(mask->request());
        check_equal("input and mask must have the same shape",
                    inp.shape().begin(), inp.shape().end(),
                    marr->shape().begin(), marr->shape().end());
    }

    std::vector<unsigned char> current(inp.size());
    std::vector<unsigned char> next(inp.size());

    threads = std::max<unsigned>(1, std::min<unsigned>(threads, std::max<size_t>(inp.size(), 1)));
    auto shape = inp.shape();
    auto shifts = detail::shift_offsets(structure, shape, true);

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (long index = 0; index < static_cast<long>(inp.size()); ++index)
    {
        current[index] = inp[index];
    }

    for (py::ssize_t iter = 0; iter < iterations; ++iter)
    {
        const bool has_mask = marr.has_value();

        #pragma omp parallel num_threads(threads)
        {
            std::vector<size_t> coord(shape.size());

            #pragma omp for
            for (long index = 0; index < static_cast<long>(current.size()); ++index)
            {
                if (current[index])
                {
                    next[index] = 1;
                    continue;
                }

                if (has_mask && !(*marr)[index])
                {
                    next[index] = 0;
                    continue;
                }

                inp.coord_at(coord.begin(), index);
                unsigned char value = 0;
                for (const auto & shift : shifts)
                {
                    if (detail::is_inbound_shift(coord, shift, shape) && current[index + shift.offset])
                    {
                        value = 1;
                        break;
                    }
                }
                next[index] = value;
            }
        }

        current.swap(next);
    }

    #pragma omp parallel for num_threads(threads)
    for (long index = 0; index < static_cast<long>(current.size()); ++index)
    {
        out[index] = static_cast<bool>(current[index]);
    }

    py::gil_scoped_acquire acquire;

    return output;
}

auto dilate(py::array_t<bool> input, Structure structure, py::ssize_t iterations, py::none mask,
            unsigned threads)
{
    return dilate_impl(std::move(input), std::move(structure), iterations, std::nullopt, threads);
}

auto dilate_with_mask(py::array_t<bool> input, Structure structure, py::ssize_t iterations, py::array_t<bool> mask,
                      unsigned threads)
{
    return dilate_impl(std::move(input), std::move(structure), iterations, std::move(mask), threads);
}

template <typename I>
LabelResult label(py::array_t<I> input, Structure structure, size_t npts, unsigned threads)
{
    array<I> inp {input.request()};

    if (input.ndim() != static_cast<py::ssize_t>(structure.rank()))
    {
        throw std::invalid_argument("input array dimension (" + std::to_string(input.ndim()) +
                                    ") does not match structure rank (" + std::to_string(structure.rank()) + ")");
    }

    py::array_t<long> labels {std::vector<py::ssize_t>(input.shape(), input.shape() + input.ndim())};
    array<long> out {labels.request()};

    if (inp.size() == 0)
    {
        return std::make_tuple(std::move(labels), py::array_t<long>{std::vector<py::ssize_t>{0}});
    }

    threads = std::max<unsigned>(1, std::min<unsigned>(threads, inp.size()));

    std::vector<long> parent (inp.size(), -1);
    std::vector<std::vector<std::pair<long, long>>> boundary_edges (threads);

    auto shape = inp.shape();
    auto shifts = detail::shift_offsets(structure, shape, false, true);

    auto find_root = [](std::vector<long> & parent, long index)
    {
        long root = index;
        while (parent[root] != root) root = parent[root];

        while (parent[index] != index)
        {
            long next = parent[index];
            parent[index] = root;
            index = next;
        }
        return root;
    };

    auto merge_roots = [&find_root](std::vector<long> & parent, long lhs, long rhs)
    {
        lhs = find_root(parent, lhs);
        rhs = find_root(parent, rhs);
        if (lhs == rhs) return;

        if (lhs < rhs) parent[rhs] = lhs;
        else parent[lhs] = rhs;
    };

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        int thread_id = omp_get_thread_num();
        long chunk = (inp.size() + threads - 1) / threads;

        long thread_start = thread_id * chunk;
        long thread_end = std::min<long>((thread_id + 1) * chunk, inp.size());

        for (long index = thread_start; index < thread_end; ++index)
        {
            if (inp[index]) parent[index] = index;
        }

        #pragma omp barrier

        std::vector<size_t> coord (shape.size());
        for (long index = thread_start; index < thread_end; ++index)
        {
            if (parent[index] < 0) continue;

            inp.coord_at(coord.begin(), index);
            for (const auto & shift : shifts)
            {
                if (!detail::is_inbound_shift(coord, shift, shape)) continue;

                long neighbour = index + shift.offset;
                if (parent[neighbour] < 0 || inp[neighbour] != inp[index]) continue;

                if (neighbour >= thread_start && neighbour < thread_end)
                {
                    merge_roots(parent, index, neighbour);
                }
                else
                {
                    boundary_edges[thread_id].emplace_back(index, neighbour);
                }
            }
        }
    }

    for (auto & edges : boundary_edges)
    {
        for (auto [lhs, rhs] : edges) merge_roots(parent, lhs, rhs);
    }

    std::vector<long> label_map (parent.size(), 0);
    long n_labels = 0;

    if (npts <= 1)
    {
        for (long index = 0; index < static_cast<long>(parent.size()); ++index)
        {
            if (parent[index] < 0) continue;

            parent[index] = find_root(parent, index);
            if (parent[index] == index) label_map[index] = ++n_labels;
        }
    }
    else
    {
        std::vector<size_t> label_sizes (parent.size(), 0);
        for (long index = 0; index < static_cast<long>(parent.size()); ++index)
        {
            if (parent[index] < 0) continue;

            parent[index] = find_root(parent, index);
            label_sizes[parent[index]]++;
        }

        for (size_t index = 0; index < parent.size(); ++index)
        {
            if (parent[index] == static_cast<long>(index) && label_sizes[index] >= npts)
            {
                label_map[index] = ++n_labels;
            }
        }
    }

    #pragma omp parallel for num_threads(threads)
    for (long index = 0; index < static_cast<long>(parent.size()); ++index)
    {
        out[index] = (parent[index] >= 0) ? label_map[parent[index]] : 0;
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    py::array_t<long> index {std::vector<py::ssize_t>{n_labels}};
    array<long> iarr {index.request()};
    for (long i = 0; i < n_labels; ++i) iarr[i] = i + 1;

    return std::make_tuple(std::move(labels), std::move(index));
}

template <typename T, size_t N>
std::vector<MomentsND<T, N>> moments_from_labels(const LabelResult & labels, py::array_t<T> data, unsigned threads)
{
    array<long> larr {std::get<0>(labels).request()};
    array<long> iarr {std::get<1>(labels).request()};
    array<T> darr {data.request()};

    check_equal("labels and data must have the same shape",
                larr.shape().begin(), larr.shape().end(),
                darr.shape().begin(), darr.shape().end());

    py::ssize_t max_label = 0;
    for (auto label_id : iarr) if (label_id > max_label) max_label = label_id;

    std::vector<long> label_to_slot (max_label + 1, -1);
    for (size_t i = 0; i < iarr.size(); ++i)
    {
        if (iarr[i] > 0) label_to_slot[iarr[i]] = i;
    }

    std::vector<long> first_index (iarr.size(), -1);
    for (size_t i = 0; i < larr.size(); ++i)
    {
        auto label_id = larr[i];
        if (label_id <= 0 || label_id > max_label) continue;

        auto slot = label_to_slot[label_id];
        if (slot >= 0 && first_index[slot] < 0) first_index[slot] = i;
    }

    std::vector<MomentsND<T, N>> moments (iarr.size());
    for (size_t i = 0; i < first_index.size(); ++i)
    {
        if (first_index[i] >= 0)
        {
            auto origin = make_point<N>(first_index[i], darr.shape());
            PointND<T, N> point;
            for (size_t n = 0; n < N; ++n) point[n] = static_cast<T>(origin[n]);
            moments[i] = MomentsND<T, N>(std::move(point));
        }
    }

    threads = std::max(1u, threads);

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        auto local_moments = moments;

        #pragma omp for
        for (long i = 0; i < static_cast<long>(larr.size()); ++i)
        {
            auto label_id = larr[i];
            if (label_id <= 0 || label_id > max_label) continue;

            auto slot = label_to_slot[label_id];
            if (slot >= 0) local_moments[slot].insert(i, darr);
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < moments.size(); ++i)
            {
                moments[i] += local_moments[i];
            }
        }
    }

    py::gil_scoped_acquire acquire;

    return moments;
}

template <typename T, size_t N, typename Func, typename Ret = std::invoke_result_t<remove_cvref_t<Func>, MomentsND<T, N>>, size_t M = std::tuple_size_v<Ret>>
py::array_t<T> apply_impl(const LabelResult & labels, py::array_t<T> data, unsigned threads, Func && func)
{
    auto moments = moments_from_labels<T, N>(labels, data, threads);

    std::vector<T> results;
    for (const auto & moment : moments)
    {
        auto result = std::forward<Func>(func)(moment);

        results.insert(results.end(), result.begin(), result.end());
    }

    std::vector<size_t> shape {moments.size(), M};

    if (results.size()) return as_pyarray(std::move(results), shape);
    return py::array_t<T>{shape};
}

template <typename T, typename Func>
py::array_t<T> apply(const LabelResult & labels, py::array_t<T> data, unsigned threads, Func && func)
{
    switch(data.ndim())
    {
        case 2: return apply_impl<T, 2>(labels, data, threads, std::forward<Func>(func));
        case 3: return apply_impl<T, 3>(labels, data, threads, std::forward<Func>(func));
        case 4: return apply_impl<T, 4>(labels, data, threads, std::forward<Func>(func));
        case 5: return apply_impl<T, 5>(labels, data, threads, std::forward<Func>(func));
        case 6: return apply_impl<T, 6>(labels, data, threads, std::forward<Func>(func));
        case 7: return apply_impl<T, 7>(labels, data, threads, std::forward<Func>(func));
        default:
            throw std::invalid_argument("Unsupported number of dimensions: " + std::to_string(data.ndim()));
    }
}

template <typename T, typename Func>
void declare_label_func(py::module & m, Func && func, const std::string & funcstr)
{
    m.def(funcstr.c_str(), [f = std::forward<Func>(func)](const LabelResult & labels, py::array_t<T> data, unsigned threads)
    {
        return apply(labels, std::move(data), threads, f);
    }, py::arg("labels"), py::arg("data"), py::arg("num_threads") = 1);
}

template <typename T, size_t N>
py::array_t<T> p_values_nd(const LabelResult & labels, py::array_t<T> larray, py::array_t<T> data, T p0, T vmin, T xtol,
                           unsigned threads)
{
    array<long> labels_array {std::get<0>(labels).request()};
    array<long> index_array {std::get<1>(labels).request()};

    py::array_t<T> result (std::vector<py::ssize_t>{py::ssize_t(index_array.size())});
    array<T> out {result.request()};
    array<T> lines {larray.request()};
    array<T> darr {data.request()};

    check_equal("labels and data must have the same shape",
                labels_array.shape().begin(), labels_array.shape().end(),
                darr.shape().begin(), darr.shape().end());

    py::ssize_t max_label = 0;
    for (auto label_id : index_array) if (label_id > max_label) max_label = label_id;

    std::vector<long> label_to_slot (max_label + 1, -1);
    for (size_t i = 0; i < index_array.size(); ++i)
    {
        if (index_array[i] > 0) label_to_slot[index_array[i]] = i;
    }

    threads = std::max(1u, threads);
    std::vector<size_t> n_counts (index_array.size(), 0);
    std::vector<size_t> k_counts (index_array.size(), 0);

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> local_n_counts (index_array.size(), 0);
        std::vector<size_t> local_k_counts (index_array.size(), 0);

        #pragma omp for
        for (long i = 0; i < static_cast<long>(labels_array.size()); ++i)
        {
            auto label_id = labels_array[i];
            if (label_id <= 0 || label_id > max_label) continue;

            auto slot = label_to_slot[label_id];
            if (slot < 0) continue;

            LineND<T, N> line {to_point<N>(lines, 2 * slot * N), to_point<N>(lines, 2 * slot * N + N)};
            auto point = make_point<N>(i, darr.shape());
            if (line.distance(point) < xtol)
            {
                local_n_counts[slot]++;
                if (darr[i] >= vmin) local_k_counts[slot]++;
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < index_array.size(); ++i)
            {
                n_counts[i] += local_n_counts[i];
                k_counts[i] += local_k_counts[i];
            }
        }
    }

    py::gil_scoped_acquire acquire;

    for (size_t i = 0; i < index_array.size(); ++i)
    {
        out[i] = detail::logbinom(n_counts[i], k_counts[i], p0);
    }

    return result;
}

template <typename T>
py::array_t<T> p_values(const LabelResult & labels, py::array_t<T> larray, py::array_t<T> data, T p0, T vmin, T xtol,
                        unsigned threads)
{
    if (larray.ndim() != 2 || larray.shape(1) != data.ndim() * 2)
    {
        throw std::invalid_argument("lines array must have shape (n_lines, data.ndim() * 2)");
    }

    switch (data.ndim())
    {
        case 2: return p_values_nd<T, 2>(labels, larray, data, p0, vmin, xtol, threads);
        case 3: return p_values_nd<T, 3>(labels, larray, data, p0, vmin, xtol, threads);
        case 4: return p_values_nd<T, 4>(labels, larray, data, p0, vmin, xtol, threads);
        case 5: return p_values_nd<T, 5>(labels, larray, data, p0, vmin, xtol, threads);
        case 6: return p_values_nd<T, 6>(labels, larray, data, p0, vmin, xtol, threads);
        case 7: return p_values_nd<T, 7>(labels, larray, data, p0, vmin, xtol, threads);
        default:
            throw std::invalid_argument("Unsupported number of dimensions: " + std::to_string(data.ndim()));
    }
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

    py::class_<Structure>(m, "Structure", py::module_local(false))
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
            for (py::ssize_t n = 0; n < out.ndim(); ++n)
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

    m.def("binary_dilation", &dilate, py::arg("inp"), py::arg("structure"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("num_threads") = 1);
    m.def("binary_dilation", &dilate_with_mask, py::arg("inp"), py::arg("structure"), py::arg("iterations") = 1, py::arg("mask") = std::nullopt, py::arg("num_threads") = 1);

    m.def("label", &label<bool>, py::arg("inp"), py::arg("structure"), py::arg("npts") = 1, py::arg("num_threads") = 1);
    m.def("label", &label<int>, py::arg("inp"), py::arg("structure"), py::arg("npts") = 1, py::arg("num_threads") = 1);
    m.def("label", &label<py::ssize_t>, py::arg("inp"), py::arg("structure"), py::arg("npts") = 1, py::arg("num_threads") = 1);

    auto total_mass = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return std::array<T, 1>{moments.zeroth()};
    };

    declare_label_func<double>(m, total_mass, "total_mass");
    declare_label_func<float>(m, total_mass, "total_mass");

    auto mean = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return moments.first();
    };

    declare_label_func<double>(m, mean, "mean");
    declare_label_func<float>(m, mean, "mean");

    auto center_of_mass = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return moments.central().first();
    };

    declare_label_func<double>(m, center_of_mass, "center_of_mass");
    declare_label_func<float>(m, center_of_mass, "center_of_mass");

    auto moment_of_inertia = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return moments.second();
    };

    declare_label_func<double>(m, moment_of_inertia, "moment_of_inertia");
    declare_label_func<float>(m, moment_of_inertia, "moment_of_inertia");

    auto covariance_matrix = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return moments.central().second();
    };

    declare_label_func<double>(m, covariance_matrix, "covariance_matrix");
    declare_label_func<float>(m, covariance_matrix, "covariance_matrix");

    auto line_fit = []<typename T, size_t N>(const MomentsND<T, N> & moments)
    {
        return moments.central().line().to_array();
    };

    declare_label_func<double>(m, line_fit, "line_fit");
    declare_label_func<float>(m, line_fit, "line_fit");

    m.def("p_values", &p_values<double>, py::arg("labels"), py::arg("lines"), py::arg("data"), py::arg("p0"),
          py::arg("vmin"), py::arg("xtol"), py::arg("num_threads") = 1);
    m.def("p_values", &p_values<float>, py::arg("labels"), py::arg("lines"), py::arg("data"), py::arg("p0"),
          py::arg("vmin"), py::arg("xtol"), py::arg("num_threads") = 1);
}
