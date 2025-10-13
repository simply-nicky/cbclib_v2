#include "streak_finder.hpp"
#include "zip.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Peaks>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Streak<double>>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Streak<float>>)

namespace cbclib {

#pragma omp declare reduction(                                                                      \
    vector_plus :                                                                                   \
    std::vector<size_t> :                                                                           \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus()))   \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

template <typename T>
std::vector<Peaks> detect_peaks(py::array_t<T> data, py::array_t<bool> mask, size_t radius, T vmin, std::optional<std::tuple<long, long>> ax, unsigned threads)
{
    Sequence<long> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_back(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    if (darr.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t repeats = std::reduce(darr.shape().begin(), std::prev(darr.shape().end(), 2), 1, std::multiplies());
    size_t n_chunks = threads / repeats + (threads % repeats > 0);
    size_t y_size = darr.shape(data.ndim() - 2) / radius;
    size_t chunk_size = y_size / n_chunks;

    std::vector<Peaks> results;
    std::vector<PeaksData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++)
    {
        results.emplace_back(radius);
        peak_data.emplace_back(darr.slice_back(i, axes.size()), marr);
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
                  std::optional<std::tuple<long, long>> ax, unsigned threads)
{
    using PeakIterators = std::vector<Peaks::iterator>;

    Sequence<long> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_back(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    check_equal("data and mask have incompatible shapes",
                std::prev(darr.shape().end(), 2), darr.shape().end(), marr.shape().begin(), marr.shape().end());
    if (darr.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t repeats = std::reduce(darr.shape().begin(), std::prev(darr.shape().end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<FilterData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++) peak_data.emplace_back(darr.slice_back(i, axes.size()), marr);

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
                    unsigned lookahead, unsigned nfa, std::optional<std::tuple<long, long>> ax, unsigned threads)
{
    Sequence<long> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_back(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    if (darr.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t repeats = std::reduce(darr.shape().begin(), std::prev(darr.shape().end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<std::vector<Streak<T>>> results (repeats);
    std::vector<StreakFinderInput<T>> inputs;
    for (size_t i = 0; i < repeats; i++)
    {
        inputs.emplace_back(peaks[i], darr.slice_back(i, axes.size()), structure, lookahead, nfa);
    }
    std::vector<size_t> totals (repeats, 0);
    std::vector<size_t> lesses (repeats, 0);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<Streak<T>>> locals (repeats);
        StreakMask buffer (marr);
        auto compare = [](const Streak<T> & a, const Streak<T> & b){return a.pixels().size() < b.pixels().size();};

        // Initialisation phase
        #pragma omp for reduction(vector_plus : totals, lesses)
        for (size_t i = 0; i < darr.size(); i++)
        {
            size_t index = i / marr.size(), remainder = i - index * marr.size();
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

template <typename T>
void declare_streak(py::module & m, const std::string & typestr)
{
    py::class_<Streak<T>>(m, (std::string("Streak") + typestr).c_str())
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
        .def("__repr__", [typestr](Streak<T> & streak)
        {
            return "<Streak" + typestr + ", size = " + std::to_string(streak.pixels().size()) +
                   ", centers = <List[List[float]], size = " + std::to_string(streak.centers().size()) + ">>";
        });
}

template <typename T, typename Width>
py::array_t<T> result_to_lines(const std::vector<Streak<T>> & streaks, std::optional<Width> width)
{
    std::vector<T> lines;

    if (width)
    {
        Sequence<float> wseq {width.value(), streaks.size()};
        for (auto witer = wseq.begin(); const auto & streak : streaks)
        {
            for (auto x : streak.line().to_array()) lines.push_back(x);
            lines.push_back(*(witer++));
        }
        return as_pyarray(std::move(lines), std::array<size_t, 2>{streaks.size(), 5});
    }

    for (const auto & streak : streaks)
    {
        for (auto x : streak.line().to_array()) lines.push_back(x);
    }
    return as_pyarray(std::move(lines), std::array<size_t, 2>{streaks.size(), 4});
}

template <typename T>
void declare_streak_list(py::module & m, const std::string & typestr)
{
    py::class_<std::vector<Streak<T>>>(m, (std::string("Streak") + typestr + std::string("List")).c_str())
        .def(py::init<>())
        .def("__delitem__", [typestr](std::vector<Streak<T>> & streaks, size_t index)
        {
            if (index >= streaks.size()) throw std::out_of_range("Streak" + typestr + "List index out of range");
            streaks.erase(std::next(streaks.begin(), index));
        }, py::arg("index"))
        .def("__delitem__", [](std::vector<Streak<T>> & streaks, py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(streaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            auto iter = std::next(streaks.begin(), start);
            for (size_t i = 0; i < slicelength; ++i, iter += step - 1) iter = streaks.erase(iter);
        }, py::arg("index"))
        .def("__getitem__", [typestr](std::vector<Streak<T>> & streaks, size_t index)
        {
            if (index >= streaks.size()) throw std::out_of_range("Streak" + typestr + "List index out of range");
            return streaks[index];
        }, py::arg("index"))
        .def("__getitem__", [](std::vector<Streak<T>> & streaks, py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(streaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            std::vector<Streak<T>> sliced;
            for (size_t i = 0; i < slicelength; ++i, start += step) sliced.push_back(streaks[start]);
            return sliced;
        }, py::arg("index"))
        .def("__setitem__", [typestr](std::vector<Streak<T>> & streaks, size_t index, Streak<T> elem)
        {
            if (index >= streaks.size()) throw std::out_of_range("Streak" + typestr + "List index out of range");
            streaks[index] = std::move(elem);
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__setitem__", [](std::vector<Streak<T>> & streaks, py::slice & slice, std::vector<Streak<T>> & elems)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(streaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i, start += step) streaks[start] = elems[i];
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__iter__", [](std::vector<Streak<T>> & streaks){return py::make_iterator(streaks.begin(), streaks.end());}, py::keep_alive<0, 1>())
        .def("__len__", [](std::vector<Streak<T>> & streaks){return streaks.size();})
        .def("__repr__", [typestr](const std::vector<Streak<T>> & streaks)
        {
            return "<Streak" + typestr + "List, size = " + std::to_string(streaks.size()) + ">";
        })
        .def("append", [](std::vector<Streak<T>> & streaks, Streak<T> elem){streaks.emplace_back(std::move(elem));}, py::arg("value"), py::keep_alive<1, 2>())
        .def("extend", [](std::vector<Streak<T>> & streaks, const std::vector<Streak<T>> & elems)
        {
            for (const auto & elem : elems) streaks.push_back(elem);
        }, py::arg("values"), py::keep_alive<1, 2>())
        .def("to_lines", [](const std::vector<Streak<T>> & streaks, std::optional<T> width)
        {
            return result_to_lines<T, T>(streaks, width);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const std::vector<Streak<T>> & streaks, std::optional<std::vector<T>> width)
        {
            return result_to_lines<T, std::vector<T>>(streaks, width);
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

    py::class_<std::vector<Peaks>>(m, "PeaksList")
        .def(py::init<>())
        .def("__delitem__", [](std::vector<Peaks> & peaks, size_t index)
        {
            if (index >= peaks.size()) throw std::out_of_range("PeaksList index out of range");
            peaks.erase(std::next(peaks.begin(), index));
        }, py::arg("index"))
        .def("__delitem__", [](std::vector<Peaks> & peaks, py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(peaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            auto iter = std::next(peaks.begin(), start);
            for (size_t i = 0; i < slicelength; ++i, iter += step - 1) iter = peaks.erase(iter);
        }, py::arg("index"))
        .def("__getitem__", [](std::vector<Peaks> & peaks, size_t index)
        {
            if (index >= peaks.size()) throw std::out_of_range("PeaksList index out of range");
            return peaks[index];
        }, py::arg("index"))
        .def("__getitem__", [](std::vector<Peaks> & peaks, py::slice & slice)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(peaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            std::vector<Peaks> sliced;
            for (size_t i = 0; i < slicelength; ++i, start += step) sliced.push_back(peaks[start]);
            return sliced;
        }, py::arg("index"))
        .def("__setitem__", [](std::vector<Peaks> & peaks, size_t index, Peaks elem)
        {
            if (index >= peaks.size()) throw std::out_of_range("PeaksList index out of range");
            peaks[index] = std::move(elem);
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__setitem__", [](std::vector<Peaks> & peaks, py::slice & slice, std::vector<Peaks> & elems)
        {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(peaks.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            for (size_t i = 0; i < slicelength; ++i, start += step) peaks[start] = elems[i];
        }, py::arg("index"), py::arg("value"), py::keep_alive<1, 3>())
        .def("__iter__", [](std::vector<Peaks> & peaks){return py::make_iterator(peaks.begin(), peaks.end());}, py::keep_alive<0, 1>())
        .def("__len__", [](std::vector<Peaks> & peaks){return peaks.size();})
        .def("__repr__", [](std::vector<Peaks> & peaks){return "<PeaksList, size = " + std::to_string(peaks.size()) + ">";})
        .def("append", [](std::vector<Peaks> & peaks, Peaks elem){peaks.emplace_back(std::move(elem));}, py::arg("value"), py::keep_alive<1, 2>())
        .def("extend", [](std::vector<Peaks> & peaks, const std::vector<Peaks> & elems)
        {
            for (const auto & elem : elems) peaks.push_back(elem);
        }, py::arg("values"), py::keep_alive<1, 2>());

    declare_streak<double>(m, "Double");
    declare_streak<float>(m, "Float");

    declare_streak_list<double>(m, "Double");
    declare_streak_list<float>(m, "Float");

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("p_value", &p_value<double>, py::arg("streaks"), py::arg("data"), py::arg("mask"), py::arg("xtol"), py::arg("vmin"));
    m.def("p_value", &p_value<float>, py::arg("streaks"), py::arg("data"), py::arg("mask"), py::arg("xtol"), py::arg("vmin"));
}
