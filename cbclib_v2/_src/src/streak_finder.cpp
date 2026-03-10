#include "log.hpp"
#include "streak_finder.hpp"
#include "zip.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Peaks>)
PYBIND11_MAKE_OPAQUE(std::vector<cbclib::StreakWrapper>)
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<cbclib::StreakWrapper>>)

namespace cbclib {

#pragma omp declare reduction(                                                                      \
    vector_plus :                                                                                   \
    std::vector<size_t> :                                                                           \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus()))   \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

template <typename T, typename U>
py::array_t<py::ssize_t> local_maxima(py::array_t<T> inp, Structure structure, unsigned threads)
{
    if (structure.rank() != inp.ndim())
        throw std::invalid_argument("structure rank does not match input dimensions");

    array<T> iarr (inp.request());
    size_t repeats = iarr.size() / iarr.shape(iarr.ndim() - 1);

    std::vector<py::ssize_t> maxima;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<py::ssize_t> buffer;
        MaximaND<T> finder (iarr, structure);

        auto inserter = [&buffer](long index) { buffer.push_back(index); };

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                finder.find(i, inserter);
            });
        }

        #pragma omp for schedule(static) ordered
        for (unsigned i = 0; i < threads; i++)
        {
            #pragma omp ordered
            maxima.insert(maxima.end(), buffer.begin(), buffer.end());
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return as_pyarray(std::move(maxima));
}

Sequence<long> init_axes(std::optional<std::array<py::ssize_t, 2>> ax, size_t ndim)
{
    Sequence<long> axes;
    if (ax)
    {
        if (static_cast<ssize_t>(ax.value().size()) != 2)
        {
            auto err_txt = "axes size (" + std::to_string(ax.value().size()) +  ") must be equal to 2";
            throw std::invalid_argument(err_txt);
        }
        axes = Sequence<long>{ax.value()}.unwrap(ndim);
    }
    else
    {
        for (long axis = ndim - 2; axis < ndim; axis++) axes->push_back(axis);
    }
    return axes;
}

template <typename T>
std::vector<Peaks> detect_peaks(py::array_t<T> data, Structure structure, size_t radius, T vmin, std::optional<std::array<py::ssize_t, 2>> ax, unsigned threads)
{
    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});

    Sequence<long> axes = init_axes(ax, data.ndim());

    data = axes.swap_back(data);
    array<T> darr {data.request()};

    size_t module_size = darr.shape(darr.ndim() - 1) * darr.shape(darr.ndim() - 2);
    size_t repeats = darr.size() / module_size;
    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    size_t y_size = darr.shape(darr.ndim() - 2);        // Frame size along y-axis
    size_t chunk_size = y_size / n_chunks;              // Number of rows each thread will process

    std::vector<Peaks> results;
    std::vector<MaximaND<T>> finders;
    for (size_t i = 0; i < repeats; i++)
    {
        results.emplace_back(darr.shape(), radius);
        finders.emplace_back(darr.slice_back(i, 2), structure);
    }

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Peaks> buffer;
        for (size_t i = 0; i < repeats; i++) buffer.emplace_back(darr.shape(), radius);

        #pragma omp for nowait
        for (size_t i = 0; i < n_chunks * repeats; i++)
        {
            e.run([&]
            {
                size_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;

                // Calculate the y-axis range for this thread
                size_t y_min = chunk_id * chunk_size;
                size_t y_max = (chunk_id == n_chunks - 1) ? y_size : y_min + chunk_size;

                // Scanning through each row inside the chunk
                for (size_t y = y_min; y < y_max; ++y)
                {
                    finders[frame].find(y, buffer[frame].inserter(finders[frame].data(), vmin));
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
void filter_peaks(std::vector<Peaks> & peaks, py::array_t<T> data, Structure structure, T vmin, size_t npts,
                  std::optional<std::array<py::ssize_t, 2>> ax, unsigned threads)
{
    using PeakIterators = std::vector<Peaks::iterator>;
    if (structure.rank() != 2) throw std::invalid_argument("Structure must have rank 2");

    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});

    Sequence<long> axes = init_axes(ax, data.ndim());

    data = axes.swap_back(data);

    array<T> darr {data.request()};

    size_t module_size = darr.shape(darr.ndim() - 1) * darr.shape(darr.ndim() - 2);
    size_t repeats = darr.size() / module_size;
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<FilterData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++) peak_data.emplace_back(darr.slice_back(i, 2));

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
                size_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;
                size_t chunk_size = peaks[frame].size() / n_chunks;
                auto first = std::next(peaks[frame].begin(), chunk_id * chunk_size);
                auto last = (chunk_id == n_chunks - 1) ? peaks[frame].end() : std::next(first, chunk_size);
                peak_data[frame].filter(first, last, buffers[frame], structure, vmin, npts);

                if (n_chunks == 1)
                {
                    for (auto iter : buffers[frame]) peaks[i].erase(iter);
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

template  <typename T>
auto p0_values(py::array_t<T> data, T vmin, std::optional<std::array<py::ssize_t, 2>> ax, unsigned threads)
{
    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});

    Sequence<long> axes = init_axes(ax, data.ndim());

    data = axes.swap_back(data);

    array<T> darr {data.request()};

    std::vector<size_t> shape {darr.shape(darr.ndim() - 2), darr.shape(darr.ndim() - 1)};
    size_t module_size = shape[0] * shape[1];
    size_t repeats = darr.size() / module_size;

    py::array_t<T> p0 {py::ssize_t(repeats)};
    fill_array(p0, T());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> n_signal (repeats, 0);

        #pragma omp for
        for (size_t i = 0; i < darr.size(); i++)
        {
            size_t index = i / module_size;
            if (darr[i] > vmin) n_signal[index]++;
        }

        #pragma omp critical
        for (size_t i = 0; i < repeats; i++) p0.mutable_at(i) += T(n_signal[i]) / module_size;
    }

    return p0;
}

template <typename T>
auto detect_streaks(const std::vector<Peaks> & peaks, py::array_t<T> p0, py::array_t<T> data, Structure structure, T xtol, T vmin, T min_size,
                    unsigned lookahead, unsigned nfa, std::optional<std::array<py::ssize_t, 2>> ax, unsigned threads)
{
    if (structure.rank() != 2) throw std::invalid_argument("Structure must have rank 2");
    if (data.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(data.ndim()) + " < 2)",
                             std::vector<py::ssize_t>{data.shape(), data.shape() + data.ndim()});

    Sequence<long> axes = init_axes(ax, data.ndim());

    data = axes.swap_back(data);

    array<T> darr {data.request()};
    array<T> p0_arr {p0.request()};

    std::vector<size_t> shape {darr.shape(darr.ndim() - 2), darr.shape(darr.ndim() - 1)};
    size_t module_size = shape[0] * shape[1];
    size_t repeats = darr.size() / module_size;

    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") must be equal to the number of data modules (" +
                                     std::to_string(repeats) + ")");
    if (repeats != p0.size())
        throw std::invalid_argument("P0 array size (" + std::to_string(p0.size()) + ") must be equal to the number of data modules (" +
                                     std::to_string(repeats) + ")");

    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    std::vector<std::vector<StreakWrapper>> results (repeats);
    std::vector<StreakFinderInput<T>> inputs;
    for (size_t i = 0; i < repeats; i++)
    {
        inputs.emplace_back(peaks[i], darr.slice_back(i, 2), structure, lookahead, nfa);
    }

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::vector<StreakWrapper>> locals (repeats);
        StreakMask buffer (shape);
        auto compare = [](const StreakWrapper & a, const StreakWrapper & b){return a.size() < b.size();};

        // Streak detection
        #pragma omp for
        for (size_t i = 0; i < repeats * n_chunks; i++)
        {
            e.run([&]
            {
                size_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;
                size_t chunk_size = inputs[frame].peaks().size() / n_chunks;
                T log_eps = std::log(p0_arr[frame]) * min_size;

                LOG(DEBUG) << "Processing frame " << frame << ", chunk " << chunk_id <<
                              "/" << n_chunks - 1 << ": log_eps = " << log_eps;

                auto first = chunk_id * chunk_size;
                auto last = (chunk_id == n_chunks - 1) ? inputs[frame].peaks().size() : first + chunk_size;

                for (auto seed : inputs[frame].seeds(first, last - first))
                {
                    if (buffer.is_free(seed))
                    {
                        auto streak = inputs[frame].get_streak(seed, xtol);

                        LOG(DEBUG) << "Found streak with " << streak.pixels().size() << " points for seed point " <<
                                       seed << " and line = " << streak.line();

                        auto p_val = buffer.p_value(streak.region(), streak.line(), inputs[frame].data(), xtol, vmin, p0_arr[frame], StreakMask::not_used);
                        if (p_val < log_eps)
                        {
                            buffer.add(streak.region(), streak.id());
                            locals[frame].emplace_back(std::move(streak));

                            LOG(DEBUG) << "Accepted streak for seed point " << seed << " with p_value = " << p_val;
                        }
                    }
                }
                buffer.clear();
            });
        }

        // Streak assembly
        #pragma omp critical
        for (size_t i = 0; i < repeats; i++)
        {
            results[i].insert(results[i].end(),
                std::make_move_iterator(locals[i].begin()),
                std::make_move_iterator(locals[i].end())
            );
        }

        // Streak filtering
        #pragma omp barrier
        #pragma omp for
        for (size_t i = 0; i < repeats; i++)
        {
            std::sort(results[i].begin(), results[i].end(), compare);

            for (const auto & streak : results[i]) buffer.add(streak.region(), streak.id());

            T log_eps = std::log(p0_arr[i]) * min_size;
            for (auto iter = results[i].begin(); iter != results[i].end();)
            {
            if (buffer.p_value(iter->region(), iter->line_as<T>(), inputs[i].data(), xtol, vmin, p0_arr[i], iter->id()) < log_eps) ++iter;
                else
                {
                    buffer.remove(iter->region(), iter->id());
                    iter = results[i].erase(iter);

                    LOG(DEBUG) << "Rejected streak during filtering with seed " << iter->center()
                               << " and line " << iter->line_as<T>();
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
py::array_t<T> p_value(const std::vector<StreakWrapper> & streaks, py::array_t<T> data, T p0, T xtol, T vmin)
{
    array<T> darr {data.request()};
    py::array_t<T> p_values (streaks.size());

    if (darr.ndim() != 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    StreakMask buffer (darr.shape());
    for (const auto & streak: streaks) buffer.add(streak.region(), streak.id());

    size_t i = 0;
    for (const auto & streak : streaks)
    {
        p_values.mutable_at(i++) = buffer.p_value(streak.region(), streak.line_as<T>(), darr, xtol, vmin, p0, streak.id());
    }

    return p_values;
}

template <typename T>
void write_to_lines(py::array_t<T> & lines, const std::vector<StreakWrapper> & pattern)
{
    if (lines.shape(lines.ndim() - 1) != 4)
    {
        auto err_txt = "last dimension of lines array must be 4 when width is not provided, but it is " + std::to_string(lines.shape(lines.ndim() - 1));
        throw std::invalid_argument(err_txt);
    }
    if (lines.size() / 4 != pattern.size())
    {
        auto err_txt = "lines array has incompatible size of " + std::to_string(lines.size()) +
                       " for " + std::to_string(pattern.size()) + " streaks";
        throw std::invalid_argument(err_txt);
    }

    array<T> arr {lines.request()};

    for (size_t i = 0; i < pattern.size(); i++)
    {
        auto line = pattern[i].line_as<T>();
        arr[4 * i + 0] = line.pt0.x(); arr[4 * i + 1] = line.pt0.y();
        arr[4 * i + 2] = line.pt1.x(); arr[4 * i + 3] = line.pt1.y();
    }
}

template <typename T>
void write_to_lines(py::array_t<T> & lines, const std::vector<std::vector<StreakWrapper>> & list)
{
    if (lines.shape(lines.ndim() - 1) != 4)
    {
        auto err_txt = "last dimension of lines array must be 4 when width is not provided, but it is " + std::to_string(lines.shape(lines.ndim() - 1));
        throw std::invalid_argument(err_txt);
    }

    size_t total = 0;
    for (const auto & pattern : list) total += pattern.size();

    if (lines.size() / 4 != total)
    {
        auto err_txt = "lines array has incompatible size of " + std::to_string(lines.size()) +
                       " for " + std::to_string(total) + " streaks";
        throw std::invalid_argument(err_txt);
    }

    array<T> arr {lines.request()};

    size_t offset = 0;
    for (const auto & pattern : list)
    {
        for (const auto & streak : pattern)
        {
            auto line = streak.line_as<T>();
            arr[4 * offset + 0] = line.pt0.x(); arr[4 * offset + 1] = line.pt0.y();
            arr[4 * offset + 2] = line.pt1.x(); arr[4 * offset + 3] = line.pt1.y();
            ++offset;
        }
    }
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

    m.def("local_maxima", &local_maxima<int, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<int, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<long, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<unsigned, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<size_t, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<float, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, int>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);
    m.def("local_maxima", &local_maxima<double, std::vector<int>>, py::arg("inp"), py::arg("structure"), py::arg("num_threads") = 1);

    py::class_<Peaks>(m, "Peaks")
        .def(py::init<std::vector<py::ssize_t>, long>(), py::arg("shape"), py::arg("radius"))
        .def(py::init([](py::list indices, std::array<py::ssize_t, 2> shape, long radius)
        {
            Peaks peaks(shape, radius);
            for (size_t i = 0; i < indices.size(); i++) peaks.insert(indices[i].cast<long>());
            return peaks;
        }), py::arg("indices"), py::arg("shape"), py::arg("radius"))
       .def(py::init([](py::array_t<py::ssize_t> indices, std::array<py::ssize_t, 2> shape, long radius)
        {
            array<py::ssize_t> iarr {indices.request()};
            Peaks peaks(shape, radius);
            for (size_t i = 0; i < iarr.size(); i++) peaks.insert(iarr[i]);
            return peaks;
        }), py::arg("indices"), py::arg("shape"), py::arg("radius"))
        .def_property("radius", [](const Peaks & peaks){return peaks.radius();}, nullptr)
        .def_property("shape", [](const Peaks & peaks)
        {
            auto shape = peaks.shape();
            return py::make_tuple(shape[0], shape[1]);
        }, nullptr)
        .def("__iter__", [](const Peaks & peaks)
        {
            return py::make_iterator(peaks.begin(), peaks.end());
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Peaks & peaks){return peaks.size();})
        .def("__repr__", &Peaks::info)
        .def("find_range", [](const Peaks & peaks, long index, long range)
        {
            auto point = make_point<2>(index, peaks.shape());
            auto iter = peaks.find_range(point, range);
            if (iter != peaks.end()) return *iter;
            return long(-1);
        }, py::arg("index"), py::arg("range"))
        .def("append", [](Peaks & peaks, long index){peaks.insert(index);}, py::arg("index"), py::keep_alive<1, 2>())
        .def("clear", [](Peaks & peaks){peaks.clear();})
        .def("extend", [](Peaks & peaks, py::list indices)
        {
            for (size_t i = 0; i < indices.size(); i++)
            {
                peaks.insert(indices[i].cast<long>());
            }
        }, py::arg("indices"), py::keep_alive<1, 2>())
        .def("extend", [](Peaks & peaks, py::array_t<long> indices)
        {
            array<long> iarr {indices.request()};
            for (size_t i = 0; i < iarr.size(); i++) peaks.insert(iarr[i]);
        }, py::arg("indices"), py::keep_alive<1, 2>())
        .def("remove", [](Peaks & peaks, long index)
        {
            auto point = make_point<2>(index, peaks.shape());
            auto iter = peaks.find(point);
            if (iter == peaks.end()) throw std::invalid_argument("Peaks.remove(index): index not in peaks");
            peaks.erase(iter);
        }, py::arg("index"));

    py::class_<std::vector<Peaks>> list_cls (m, "PeaksList");
    declare_list(list_cls, "PeaksList");

    list_cls.def("index", [](const std::vector<Peaks> & list)
        {
            std::vector<py::ssize_t> indices;
            for (size_t i = 0; i < list.size(); i++)
            {
                for (size_t j = 0; j < list[i].size(); j++) indices.push_back(i);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
    .def("to_array", [](const std::vector<Peaks> & list)
        {
            std::vector<long> indices;
            for (const auto & peaks : list) for (auto index : peaks) indices.push_back(index);
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        });

    py::class_<StreakWrapper>(m, "Streak")
        .def(py::init([](py::ssize_t seed, Structure structure, py::array_t<float> data)
        {
            if (structure.rank() != 2) throw std::invalid_argument("Structure must have rank 2");

            array<float> darr {data.request()};
            return StreakWrapper{Streak<float>{Region(seed, structure, darr.shape()), seed, darr}};
        }), py::arg("seed"), py::arg("structure"), py::arg("data"))
        .def(py::init([](py::ssize_t seed, Structure structure, py::array_t<double> data)
        {
            if (structure.rank() != 2) throw std::invalid_argument("Structure must have rank 2");

            array<double> darr {data.request()};
            return StreakWrapper{Streak<double>{Region(seed, structure, darr.shape()), seed, darr}};
        }), py::arg("seed"), py::arg("structure"), py::arg("data"))
        .def_property("centers", [](const StreakWrapper & streak){ return streak.centers(); }, nullptr)
        .def_property("ends", [](const StreakWrapper & streak){ return streak.ends(); }, nullptr)
        .def_property("region", [](const StreakWrapper & streak){ return streak.region(); }, nullptr)
        .def_property("id", [](const StreakWrapper & streak){return streak.id();}, nullptr)
        .def("center", [](const StreakWrapper & streak){return streak.center().to_array();})
        .def("central_line", [](const StreakWrapper & streak){return streak.central_line().to_array();})
        .def("line", [](const StreakWrapper & streak){return streak.line_as<double>().to_array();});

    py::class_<std::vector<StreakWrapper>> pattern_cls (m, "Pattern");
    declare_list(pattern_cls, "Pattern");

    pattern_cls.def("to_lines", [](const std::vector<StreakWrapper> & streaks, py::array_t<float> out)
        {
            write_to_lines(out, streaks);
            return out;
        }, py::arg("out"))
        .def("to_lines", [](const std::vector<StreakWrapper> & streaks, py::array_t<double> out)
        {
            write_to_lines(out, streaks);
            return out;
        }, py::arg("out"))
        .def("to_regions", [](const std::vector<StreakWrapper> & streaks)
        {
            std::vector<Region> regions;
            for (const auto & streak : streaks)
            {
                regions.push_back(streak.region());
            }
            return regions;
        });

    py::class_<std::vector<std::vector<StreakWrapper>>> patterns_cls (m, "PatternList");
    declare_list(patterns_cls, "PatternList");

    patterns_cls.def("total", [](const std::vector<std::vector<StreakWrapper>> & list)
        {
            size_t total = 0;
            for (const auto & pattern : list) total += pattern.size();
            return total;
        })
        .def("index", [](const std::vector<std::vector<StreakWrapper>> & list)
        {
            std::vector<py::ssize_t> indices;
            for (size_t index = 0; index < list.size(); index++)
            {
                for (size_t i = 0; i < list[index].size(); i++) indices.push_back(index);
            }
            return as_pyarray(std::move(indices), std::array<size_t, 1>{indices.size()});
        })
        .def("to_lines", [](const std::vector<std::vector<StreakWrapper>> & list, py::array_t<float> out)
        {
            write_to_lines(out, list);
            return out;
        }, py::arg("out"))
        .def("to_lines", [](const std::vector<std::vector<StreakWrapper>> & list, py::array_t<double> out)
        {
            write_to_lines(out, list);
            return out;
        }, py::arg("out"));

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("p0_values", &p0_values<double>, py::arg("data"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("p0_values", &p0_values<float>, py::arg("data"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("p0"), py::arg("data"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("p0"), py::arg("data"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("p_value", &p_value<double>, py::arg("streaks"), py::arg("data"), py::arg("p0"), py::arg("xtol"), py::arg("vmin"));
    m.def("p_value", &p_value<float>, py::arg("streaks"), py::arg("data"), py::arg("p0"), py::arg("xtol"), py::arg("vmin"));
}
