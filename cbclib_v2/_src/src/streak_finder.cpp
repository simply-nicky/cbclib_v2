#include "streak_finder.hpp"
#include "zip.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Peaks>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Streak<double>>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Streak<float>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<cbclib::Streak<double>>>)
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<cbclib::Streak<float>>>)

namespace cbclib {

#pragma omp declare reduction(                                                                      \
    vector_plus :                                                                                   \
    std::vector<size_t> :                                                                           \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus()))   \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

template <typename T>
std::vector<Peaks> detect_peaks(py::array_t<T> data, py::array_t<bool> mask, size_t radius, T vmin, std::optional<std::vector<long>> ax, unsigned threads)
{
    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});
    check_equal("data and mask have incompatible shapes",
                data.shape() + data.ndim() - mask.ndim(), data.shape() + data.ndim(),
                mask.shape(), mask.shape() + mask.ndim());

    Sequence<long> axes;
    if (ax)
    {
        if (ax.value().size() != mask.ndim())
        {
            auto err_txt = "axes size (" + std::to_string(ax.value().size()) +  ") must be equal to the mask number of dimensions (" +
                           std::to_string(mask.ndim()) + ")";
            throw std::invalid_argument(err_txt);
        }
        axes = Sequence<long>{ax.value()}.unwrap(data.ndim());
    }
    else
    {
        for (long axis = data.ndim() - mask.ndim(); axis < data.ndim(); axis++) axes->push_back(axis);
    }

    data = axes.swap_back(data);

    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    size_t module_size = darr.shape(darr.ndim() - 1) * darr.shape(darr.ndim() - 2);
    size_t n_modules = marr.size() / module_size;
    size_t repeats = darr.size() / module_size;
    size_t n_chunks = threads / repeats + (threads % repeats > 0);
    size_t y_size = darr.shape(darr.ndim() - 2) / radius;
    size_t chunk_size = y_size / n_chunks;

    std::vector<Peaks> results;
    std::vector<PeaksData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++)
    {
        results.emplace_back(radius);
        peak_data.emplace_back(darr.slice_back(i, 2), marr.slice_back(i % n_modules, 2));
    }

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Peaks> buffer;
        for (size_t i = 0; i < repeats; i++) buffer.emplace_back(radius);

        #pragma omp for nowait
        for (size_t i = 0; i < n_chunks * repeats; i++)
        {
            e.run([&]
            {
                size_t index = i / n_chunks, remainder = i - index * n_chunks;
                size_t y_min = remainder * chunk_size;
                size_t y_max = (remainder == n_chunks - 1) ? y_size : y_min + chunk_size;

                for (size_t y = y_min * radius + radius / 2; y < y_max * radius; y += radius)
                {
                    auto line = peak_data[index].data().slice(y, 1);
                    peak_data[index].insert(line.begin(), line.end(), buffer[index], vmin, 1);
                }

                for (size_t x = radius / 2; x < peak_data[index].data().shape(1); x += radius)
                {
                    auto line = peak_data[index].data().slice(x, 0);
                    auto first = std::next(line.begin(), y_min * radius - (y_min > 0));
                    auto last = std::next(line.begin(), y_max * radius + (y_max < y_size));
                    peak_data[index].insert(first, last, buffer[index], vmin, 1);
                }
            });
        }

        if (n_chunks > 1)
        {
            #pragma omp critical
            for (size_t i = 0; i < repeats; i++) results[i].merge(std::move(buffer[i]));
        }
        else
        {
            #pragma omp critical
            for (size_t i = 0; i < repeats; i++) if (buffer[i].size()) results[i] = std::move(buffer[i]);
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return results;
}

template <typename T>
void filter_peaks(std::vector<Peaks> & peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                  std::optional<std::vector<long>> ax, unsigned threads)
{
    using PeakIterators = std::vector<Peaks::iterator>;

    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});
    check_equal("data and mask have incompatible shapes",
                data.shape() + data.ndim() - mask.ndim(), data.shape() + data.ndim(),
                mask.shape(), mask.shape() + mask.ndim());

    Sequence<long> axes;
    if (ax)
    {
        if (ax.value().size() != mask.ndim())
        {
            auto err_txt = "axes size (" + std::to_string(ax.value().size()) +  ") must be equal to the mask number of dimensions (" +
                           std::to_string(mask.ndim()) + ")";
            throw std::invalid_argument(err_txt);
        }
        axes = Sequence<long>{ax.value()}.unwrap(data.ndim());
    }
    else
    {
        for (long axis = data.ndim() - mask.ndim(); axis < data.ndim(); axis++) axes->push_back(axis);
    }

    data = axes.swap_back(data);

    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    size_t module_size = darr.shape(darr.ndim() - 1) * darr.shape(darr.ndim() - 2);
    size_t n_modules = marr.size() / module_size;
    size_t repeats = darr.size() / module_size;
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<FilterData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++) peak_data.emplace_back(darr.slice_back(i, 2), marr.slice_back(i % n_modules, 2));

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<PeakIterators> buffers (repeats);

        #pragma omp for
        for (size_t i = 0; i < repeats * n_chunks; i++)
        {
            e.run([&]
            {
                size_t index = i / n_chunks, remainder = i - index * n_chunks;
                size_t chunk_size = peaks[index].size() / n_chunks;
                auto first = std::next(peaks[index].begin(), remainder * chunk_size);
                auto last = (remainder == n_chunks - 1) ? peaks[index].end() : std::next(first, chunk_size);
                peak_data[index].filter(first, last, buffers[index], structure, vmin, npts);

                if (n_chunks == 1)
                {
                    for (auto iter : buffers[index]) peaks[i].erase(iter);
                }
            });
        }

        if (n_chunks > 1)
        {
            #pragma omp critical
            for (size_t i = 0; i < repeats; i++) for (auto iter : buffers[i]) peaks[i].erase(iter);
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();
}

template <typename T>
auto detect_streaks(const std::vector<Peaks> & peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T xtol, T vmin, unsigned min_size,
                    unsigned lookahead, unsigned nfa, std::optional<std::vector<long>> ax, unsigned threads)
{
    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});
    check_equal("data and mask have incompatible shapes",
                data.shape() + data.ndim() - mask.ndim(), data.shape() + data.ndim(),
                mask.shape(), mask.shape() + mask.ndim());

    Sequence<long> axes;
    if (ax)
    {
        if (ax.value().size() != mask.ndim())
        {
            auto err_txt = "axes size (" + std::to_string(ax.value().size()) +  ") must be equal to the mask number of dimensions (" +
                           std::to_string(mask.ndim()) + ")";
            throw std::invalid_argument(err_txt);
        }
        axes = Sequence<long>{ax.value()}.unwrap(data.ndim());
    }
    else
    {
        for (long axis = data.ndim() - mask.ndim(); axis < data.ndim(); axis++) axes->push_back(axis);
    }

    data = axes.swap_back(data);

    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    size_t module_size = darr.shape(darr.ndim() - 1) * darr.shape(darr.ndim() - 2);
    size_t n_modules = marr.size() / module_size;
    size_t repeats = darr.size() / module_size;
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<std::vector<Streak<T>>> results (repeats);
    std::vector<StreakFinderInput<T>> inputs;
    for (size_t i = 0; i < repeats; i++)
    {
        inputs.emplace_back(peaks[i], darr.slice_back(i, 2), structure, lookahead, nfa);
    }
    std::vector<size_t> totals (repeats, 0);
    std::vector<size_t> lesses (repeats, 0);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<Streak<T>>> locals (repeats);
        StreakMask buffer (module_size);
        auto compare = [](const Streak<T> & a, const Streak<T> & b){return a.pixels().size() < b.pixels().size();};

        // Initialisation phase
        #pragma omp for reduction(vector_plus : totals, lesses)
        for (size_t i = 0; i < darr.size(); i++)
        {
            size_t index = i / module_size, remainder = i % marr.size();
            if (marr[remainder])
            {
                totals[index]++;
                if (darr[i] < vmin) lesses[index]++;
            }
        }

        // Streak detection
        #pragma omp for
        for (size_t i = 0; i < repeats * n_chunks; i++)
        {
            e.run([&]
            {
                size_t index = i / n_chunks, remainder = i - index * n_chunks;
                size_t chunk_size = inputs[index].peaks().size() / n_chunks;
                T p = T(1.0) - T(lesses[index]) / totals[index];
                T log_eps = std::log(p) * min_size;

                auto first = remainder * chunk_size;
                auto last = (remainder == n_chunks - 1) ? inputs[index].peaks().size() : first + chunk_size;
                buffer.mask() = marr.slice_back(index % n_modules, 2);

                for (const auto & seed : inputs[index].points(first, last - first))
                {
                    if (buffer.is_free(seed))
                    {
                        auto streak = inputs[index].get_streak(seed, buffer, xtol);
                        if (buffer.p_value(streak, xtol, vmin, p, StreakMask::not_used) < log_eps)
                        {
                            buffer.add(streak);
                            locals[index].push_back(streak);
                        }
                    }
                }
                buffer.clear();
            });
        }

        // Streak assembly
        #pragma omp critical
        for (size_t i = 0; i < repeats; i++) results[i].insert(results[i].end(), locals[i].begin(), locals[i].end());

        // Streak filtering
        #pragma omp barrier
        #pragma omp for
        for (size_t i = 0; i < repeats; i++)
        {
            std::sort(results[i].begin(), results[i].end(), compare);

            buffer.mask() = marr.slice_back(i % n_modules, 2);
            for (const auto & streak : results[i]) buffer.add(streak);

            T p = T(1.0) - T(lesses[i]) / totals[i];
            T log_eps = std::log(p) * min_size;
            for (auto iter = results[i].begin(); iter != results[i].end();)
            {
                if (buffer.p_value(*iter, xtol, vmin, p, iter->id()) < log_eps) ++iter;
                else
                {
                    buffer.remove(*iter);
                    iter = results[i].erase(iter);
                }
            }
            buffer.clear();
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return results;
}

template <typename T>
std::tuple<py::array_t<T>, T> p_value(const std::vector<Streak<T>> & streaks, py::array_t<T> data, py::array_t<bool> mask, T xtol, T vmin)
{
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};
    py::array_t<T> p_values (streaks.size());

    check_equal("data and mask have incompatible shapes",
                darr.shape().begin(), darr.shape().end(), marr.shape().begin(), marr.shape().end());
    if (darr.ndim() != 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t total = 0, less = 0;

    for (size_t i = 0; i < darr.size(); i++)
    {
        if (marr[i])
        {
            total++;
            if (darr[i] < vmin) less++;
        }
    }

    T p = T(1.0) - T(less) / total;

    StreakMask buffer (marr);
    for (const auto & streak: streaks) buffer.add(streak);

    for (size_t i = 0; const auto & streak : streaks)
    {
        p_values.mutable_at(i++) = buffer.p_value(streak, xtol, vmin, p, streak.id());
    }

    return std::make_tuple(p_values, p);
}

template <typename Element>
void declare_list(py::class_<std::vector<Element>> & cls, const std::string & str)
{
    cls.def(py::init<>())
        .def("__delitem__", [str](std::vector<Element> & list, py::ssize_t index)
        {
            list.erase(std::next(list.begin(), compute_index(index, list.size(), str)));
        }, py::arg("index"))
        .def("__delitem__", [](std::vector<Element> & list, py::slice & slice)
        {
            auto range = slice_range(slice, list.size());
            auto iter = std::next(list.begin(), range.start());
            for (size_t i = 0; i < range.size(); ++i, iter += range.step() - 1) iter = list.erase(iter);
        }, py::arg("index"))
        .def("__getitem__", [str](const std::vector<Element> & list, py::ssize_t index)
        {
            return list[compute_index(index, list.size(), str)];
        }, py::arg("index"))
        .def("__getitem__", [](const std::vector<Element> & list, py::slice & slice)
        {
            std::vector<Element> sliced;
            for (auto [_, py_index] : slice_range(slice, list.size())) sliced.push_back(list[py_index]);
            return sliced;
        }, py::arg("index"))
        .def("__setitem__", [str](std::vector<Element> & list, py::ssize_t index, Element elem)
        {
            list[compute_index(index, list.size(), str)] = std::move(elem);
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__setitem__", [](std::vector<Element> & list, py::slice & slice, std::vector<Element> & elems)
        {
            for (auto [index, py_index] : slice_range(slice, list.size())) list[py_index] = elems[index];
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__iter__", [](const std::vector<Element> & list){return py::make_iterator(list.begin(), list.end());}, py::keep_alive<0, 1>())
        .def("__len__", [](const std::vector<Element> & list){return list.size();})
        .def("__repr__", [str](const std::vector<Element> & list)
        {
            return "<" + str + ", size = " + std::to_string(list.size()) + ">";
        })
        .def("append", [](std::vector<Element> & list, Element elem){list.emplace_back(std::move(elem));}, py::arg("value"), py::keep_alive<1, 2>())
        .def("extend", [](std::vector<Element> & list, const std::vector<Element> & elems)
        {
            for (const auto & elem : elems) list.push_back(elem);
        }, py::arg("values"), py::keep_alive<1, 2>());
}

template <typename T>
void declare_streak(py::module & m, const std::string & typestr)
{
    auto cls = "Streak" + typestr;
    py::class_<Streak<T>>(m, cls.c_str())
        .def(py::init([](long x, long y, Structure structure, py::array_t<T> data)
        {
            array<T> darr {data.request()};
            PixelSet<T> pset;
            for (auto shift : structure)
            {
                Point<long> pt {x + shift.x(), y + shift.y()};
                pset.emplace(make_pixel(std::move(pt), darr));
            }
            return Streak<T>{std::move(pset), Point<long>{x, y}};
        }), py::arg("x"), py::arg("y"), py::arg("structure"), py::arg("data"))
        .def_property("centers", [](const Streak<T> & streak)
        {
            std::vector<std::array<long, 2>> centers;
            for (auto ctr : streak.centers()) centers.emplace_back(ctr.to_array());
            return centers;
        }, nullptr)
        .def_property("ends", [](const Streak<T> & streak)
        {
            std::vector<std::array<T, 2>> ends;
            for (auto ctr : streak.ends()) ends.emplace_back(ctr.to_array());
            return ends;
        }, nullptr)
        .def_property("x", [](const Streak<T> & streak)
        {
            std::vector<long> xvec;
            for (const auto & [pt, _]: streak.pixels()) xvec.push_back(pt.x());
            return xvec;
        }, nullptr)
        .def_property("y", [](const Streak<T> & streak)
        {
            std::vector<long> yvec;
            for (const auto & [pt, _]: streak.pixels()) yvec.push_back(pt.y());
            return yvec;
        }, nullptr)
        .def_property("value", [](const Streak<T> & streak)
        {
            std::vector<T> values;
            for (const auto & [_, val]: streak.pixels()) values.push_back(val);
            return values;
        }, nullptr)
        .def_property("id", [](Streak<T> & streak){return streak.id();}, nullptr)
        .def("merge", [](Streak<T> & streak, Streak<T> & source)
        {
            streak.merge(source);
            return streak;
        }, py::arg("source"))
        .def("center", [](Streak<T> & streak){return streak.center().to_array();})
        .def("central_line", [](Streak<T> & streak){return streak.central_line().to_array();})
        .def("line", [](Streak<T> & streak){return streak.line().to_array();})
        .def("total_mass", [](Streak<T> & streak){return streak.moments().zeroth();})
        .def("mean", [](Streak<T> & streak){return streak.moments().first();})
        .def("center_of_mass", [](Streak<T> & streak){return streak.moments().central().first();})
        .def("moment_of_inertia", [](Streak<T> & streak){return streak.moments().second();})
        .def("covariance_matrix", [](Streak<T> & streak){return streak.moments().central().second();})
        .def("__repr__", [cls](Streak<T> & streak)
        {
            return "<" + cls + ", size = " + std::to_string(streak.pixels().size()) +
                   ", centers = <List[List[float]], size = " + std::to_string(streak.centers().size()) + ">>";
        });
}

template <typename T>
std::array<size_t, 2> push_to_lines(std::vector<T> & lines, const std::vector<Streak<T>> & streaks, const std::optional<T> & width)
{
    size_t n_pushed = 0;

    if (width)
    {
        for (const auto & streak : streaks)
        {
            for (auto x : streak.line().to_array()) lines.push_back(x);
            lines.push_back(width.value());
            n_pushed++;
        }

        return {n_pushed, 5};
    }

    for (const auto & streak : streaks)
    {
        for (auto x : streak.line().to_array()) lines.push_back(x);
        n_pushed++;
    }

    return {n_pushed, 4};
}

template <typename T>
std::array<size_t, 2> push_to_lines(std::vector<T> & lines, const std::vector<Streak<T>> & streaks, const std::optional<std::vector<T>> & width, size_t index = 0)
{
    size_t n_pushed = 0;

    if (width)
    {
        if (width.value().size() != streaks.size())
        {
            auto err_txt = "size of width sequence (" + std::to_string(width.value().size()) +
                           ") is incompatible with streaks size (" + std::to_string(streaks.size()) + ")";
            throw std::invalid_argument(err_txt);
        }

        for (const auto & streak : streaks)
        {
            for (auto x : streak.line().to_array()) lines.push_back(x);
            lines.push_back(width.value()[index++]);
            n_pushed++;
        }

        return {n_pushed, 5};
    }

    for (const auto & streak : streaks)
    {
        for (auto x : streak.line().to_array()) lines.push_back(x);
        n_pushed++;
    }

    return {n_pushed, 4};
}

template <typename T>
std::array<size_t, 2> push_to_lines(std::vector<T> & lines, const std::vector<Streak<T>> & streaks, const std::optional<py::array_t<T>> & width, size_t index = 0)
{
    size_t n_pushed;

    if (width)
    {
        if (width.value().ndim() != 1 || width.value().size() != streaks.size())
        {
            std::ostringstream oss;
            std::copy(width.value().shape(), width.value().shape() + width.value().ndim(), std::experimental::make_ostream_joiner(oss, ", "));
            auto err_txt = "shape of width sequence (" + oss.str() +
                           ") is incompatible with streaks size (" + std::to_string(streaks.size()) + ",)";
            throw std::invalid_argument(err_txt);
        }

        for (const auto & streak : streaks)
        {
            for (auto x : streak.line().to_array()) lines.push_back(x);
            lines.push_back(width.value().at(index++));
            n_pushed++;
        }

        return {n_pushed, 5};
    }

    for (const auto & streak : streaks)
    {
        for (auto x : streak.line().to_array()) lines.push_back(x);
        n_pushed++;
    }

    return {n_pushed, 4};
}

template <typename T>
void declare_pattern(py::module & m, const std::string & typestr)
{
    auto str = "Pattern" + typestr;
    py::class_<std::vector<Streak<T>>> pattern_cls (m, str.c_str());
    declare_list(pattern_cls, str);

    pattern_cls.def("to_lines", [](const std::vector<Streak<T>> & streaks, std::optional<T> width)
        {
            std::vector<T> lines;
            auto shape = push_to_lines(lines, streaks, width);
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const std::vector<Streak<T>> & streaks, std::optional<std::vector<T>> width)
        {
            std::vector<T> lines;
            auto shape = push_to_lines(lines, streaks, width);
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const std::vector<Streak<T>> & streaks, std::optional<py::array_t<T>> width)
        {
            std::vector<T> lines;
            auto shape = push_to_lines(lines, streaks, width);
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt)
        .def("to_regions", [](const std::vector<Streak<T>> & streaks)
        {
            RegionsND<2> regions;
            for (const auto & streak : streaks)
            {
                PointSet points;
                for (auto && [point, _] : streak.pixels()) points->emplace_hint(points.end(), std::forward<decltype(point)>(point));
                regions->emplace_back(std::move(points));
            }
            return regions;
        });
}

template <typename T>
void declare_pattern_list(py::module & m, const std::string & typestr)
{
    auto str = "Pattern" + typestr + "List";
    py::class_<std::vector<std::vector<Streak<T>>>> list_cls (m, str.c_str());
    declare_list(list_cls, str);

    list_cls.def("index", [](const std::vector<std::vector<Streak<T>>> & list)
        {
            std::vector<py::ssize_t> indices;
            for (size_t index = 0; index < list.size(); index++)
            {
                for (size_t i = 0; i < list[index].size(); i++) indices.push_back(index);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("to_lines", [](const std::vector<std::vector<Streak<T>>> & list, std::optional<T> width)
        {
            std::array<size_t, 2> shape {0, 0};
            std::vector<T> lines;
            for (const auto & pattern : list)
            {
                auto pushed_shape = push_to_lines(lines, pattern, width);
                shape[0] += pushed_shape[0]; shape[1] = pushed_shape[1];
            }
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const std::vector<std::vector<Streak<T>>> & list, std::optional<std::vector<T>> width)
        {
            std::array<size_t, 2> shape {0, 0};
            std::vector<T> lines;
            for (const auto & pattern : list)
            {
                auto pushed_shape = push_to_lines(lines, pattern, width, shape[0]);
                shape[0] += pushed_shape[0]; shape[1] = pushed_shape[1];
            }
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const std::vector<std::vector<Streak<T>>> & list, std::optional<py::array_t<T>> width)
        {
            std::array<size_t, 2> shape {0, 0};
            std::vector<T> lines;
            for (const auto & pattern : list)
            {
                auto pushed_shape = push_to_lines(lines, pattern, width, shape[0]);
                shape[0] += pushed_shape[0]; shape[1] = pushed_shape[1];
            }
            return as_pyarray(std::move(lines), shape);
        }, py::arg("width") = std::nullopt);
}

}

PYBIND11_MODULE(streak_finder, m)
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

    py::class_<Peaks>(m, "Peaks")
        .def(py::init<long>(), py::arg("radius"))
        .def_property("radius", [](const Peaks & peaks){return peaks.radius();}, nullptr)
        .def_property("x", [](const Peaks & peaks){return detail::get_x(peaks, 0);}, nullptr)
        .def_property("y", [](const Peaks & peaks){return detail::get_x(peaks, 1);}, nullptr)
        .def("__iter__", [](const Peaks & peaks)
        {
            return py::make_iterator(make_python_iterator(peaks.begin()), make_python_iterator(peaks.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Peaks & peaks){return peaks.size();})
        .def("__repr__", &Peaks::info)
        .def("find_range", [](const Peaks & peaks, long x, long y, long range)
        {
            auto iter = peaks.find_range(Point<long>{x, y}, range);
            if (iter != peaks.end()) return std::vector<long>{iter->x(), iter->y()};
            return std::vector<long>{};
        }, py::arg("x"), py::arg("y"), py::arg("range"))
        .def("append", [](Peaks & peaks, long x, long y){peaks.insert(Point<long>{x, y});}, py::arg("x"), py::arg("y"))
        .def("clear", [](Peaks & peaks){peaks.clear();})
        .def("extend", [](Peaks & peaks, std::vector<long> xvec, std::vector<long> yvec)
        {
            for (auto [x, y] : zip::zip(xvec, yvec)) peaks.insert(Point<long>{x, y});
        }, py::arg("xs"), py::arg("ys"))
        .def("remove", [](Peaks & peaks, long x, long y)
        {
            auto iter = peaks.find(Point<long>{x, y});
            if (iter == peaks.end()) throw std::invalid_argument("Peaks.remove(x, y): {x, y} not in peaks");
            peaks.erase(iter);
        }, py::arg("x"), py::arg("y"));

    py::class_<std::vector<Peaks>> list_cls (m, "PeaksList");
    declare_list(list_cls, "PeaksList");

    list_cls.def("index", [](const std::vector<Peaks> & list)
        {
            std::vector<py::ssize_t> indices;
            for (auto index = 0; index < list.size(); index++)
            {
                for (size_t i = 0; i < list[index].size(); i++) indices.push_back(index);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("x", [](const std::vector<Peaks> & list)
        {
            std::vector<long> x;
            for (auto index = 0; index < list.size(); index++)
            {
                for (const auto & point : list[index]) x.push_back(point.x());
            }
            return as_pyarray(std::move(x), std::array<size_t, 1>{x.size()});
        })
        .def("y", [](const std::vector<Peaks> & list)
        {
            std::vector<long> y;
            for (auto index = 0; index < list.size(); index++)
            {
                for (const auto & point : list[index]) y.push_back(point.y());
            }
            return as_pyarray(std::move(y), std::array<size_t, 1>{y.size()});
        });

    declare_streak<double>(m, "Double");
    declare_streak<float>(m, "Float");

    declare_pattern<double>(m, "Double");
    declare_pattern<float>(m, "Float");

    declare_pattern_list<double>(m, "Double");
    declare_pattern_list<float>(m, "Float");

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("p_value", &p_value<double>, py::arg("streaks"), py::arg("data"), py::arg("mask"), py::arg("xtol"), py::arg("vmin"));
    m.def("p_value", &p_value<float>, py::arg("streaks"), py::arg("data"), py::arg("mask"), py::arg("xtol"), py::arg("vmin"));
}
