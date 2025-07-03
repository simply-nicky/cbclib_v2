#include "streak_finder.hpp"
#include "zip.hpp"

namespace cbclib {

template <typename T>
std::vector<Peaks> detect_peaks(py::array_t<T> data, py::array_t<bool> mask, size_t radius, T vmin, std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    Sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    if (darr.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t repeats = std::reduce(darr.shape().begin(), std::prev(darr.shape().end(), 2), 1, std::multiplies());
    size_t n_chunks = threads / repeats + (threads % repeats > 0);
    size_t y_size = darr.shape(data.ndim() - 2) / radius;
    size_t chunk_size = y_size / n_chunks;

    std::vector<Peaks> result;
    std::vector<PeaksData<T>> peak_data;
    for (size_t i = 0; i < repeats; i++)
    {
        result.emplace_back(radius);
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

        #pragma omp critical
        for (size_t i = 0; i < repeats; i++) result[i].merge(buffer[i]);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T>
auto filter_peaks(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                  std::optional<std::tuple<size_t, size_t>> ax, unsigned threads)
{
    Sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
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
        std::vector<std::vector<Peaks::iterator>> buffers (repeats);

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
            });
        }

        #pragma omp critical
        for (size_t i = 0; i < repeats; i++) for (auto iter : buffers[i]) peaks[i].erase(iter);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return peaks;
}

template <typename T>
auto filter_peak(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                 py::none ax, unsigned threads)
{
    return filter_peaks(std::vector<Peaks>{peaks}, data, mask, structure, vmin, npts, std::make_tuple(data.ndim() - 2, data.ndim() - 1), threads)[0];
}

template <typename T>
auto detect_streaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T xtol, T vmin,
                    unsigned min_size, unsigned lookahead, unsigned nfa, py::none ax, unsigned threads)
{
    if (xtol >= structure.rank + 0.5)
        throw std::invalid_argument("xtol (" + std::to_string(xtol) + ") must be lower than the rank of the structure (" +
                                    std::to_string(structure.rank) + ")");

    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    if (darr.ndim() != 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " != 2)", darr.shape());

    py::gil_scoped_release release;

    StreakFinder finder {structure, min_size, lookahead, nfa};
    StreakFinderResult<T> result = finder.detect_streaks(StreakFinderResult(darr, marr), darr, peaks, xtol, vmin);

    py::gil_scoped_acquire acquire;

    return result;
}

template <typename T>
auto detect_streaks_vec(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T xtol, T vmin,
                        unsigned min_size, unsigned lookahead, unsigned nfa, std::optional<std::tuple<size_t, size_t>> ax,
                        unsigned threads)
{
    if (xtol >= structure.rank + 0.5)
        throw std::invalid_argument("xtol (" + std::to_string(xtol) + ") must be lower than the rank of the structure (" +
                                    std::to_string(structure.rank) + ")");

    Sequence<size_t> axes;
    if (ax)
    {
        axes = {std::get<0>(ax.value()), std::get<1>(ax.value())};
        axes = axes.unwrap(data.ndim());
    }
    else axes = {data.ndim() - 2, data.ndim() - 1};

    data = axes.swap_axes(data);
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    if (darr.ndim() < 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " < 2)", darr.shape());

    size_t repeats = std::reduce(darr.shape().begin(), std::prev(darr.shape().end(), 2), 1, std::multiplies());
    if (repeats != peaks.size())
        throw std::invalid_argument("Size of peaks list (" + std::to_string(peaks.size()) + ") is incompatible with data");

    std::vector<StreakFinderResult<T>> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        StreakFinder finder {structure, min_size, lookahead, nfa};
        std::vector<StreakFinderResult<T>> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                buffer.emplace_back(finder.detect_streaks(StreakFinderResult(darr.slice_back(i, axes.size()), marr), darr.slice_back(i, axes.size()), std::move(peaks[i]), xtol, vmin));
            });
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
py::array_t<T> result_to_lines(const StreakFinderResult<T> & result, std::optional<Width> width)
{
    std::vector<T> lines;

    if (width)
    {
        Sequence<float> wseq {width.value(), result.size()};
        for (auto witer = wseq.begin(); const auto & [_, streak] : result)
        {
            for (auto x : streak.line().to_array()) lines.push_back(x);
            lines.push_back(*(witer++));
        }
        return as_pyarray(std::move(lines), std::array<size_t, 2>{result.size(), 5});
    }

    for (const auto & [_, streak] : result)
    {
        for (auto x : streak.line().to_array()) lines.push_back(x);
    }
    return as_pyarray(std::move(lines), std::array<size_t, 2>{result.size(), 4});
}

template <typename T>
void declare_streak_finder_result(py::module & m, const std::string & typestr)
{
    py::class_<StreakFinderResult<T>>(m, (std::string("StreakFinderResult") + typestr).c_str())
        .def(py::init([](py::array_t<T> data, py::array_t<bool> mask)
        {
            return StreakFinderResult<T>(array<T>{data.request()}, array<bool>{mask.request()});
        }), py::arg("data"), py::arg("mask"))
        .def("probability", [](const StreakFinderResult<T> & result, py::array_t<T> data, T vmin)
        {
            return result.probability(array<T>{data.request()}, vmin);
        }, py::arg("data"), py::arg("vmin"))
        .def("p_value", [](const StreakFinderResult<T> & result, int index, T xtol, T vmin, T p)
        {
            auto iter = result.find(index);
            if (iter == result.end())
                throw std::out_of_range("No streak with index " + std::to_string(index) + "in the result");
            return result.p_value(iter, xtol, vmin, p);
        }, py::arg("index"), py::arg("xtol"), py::arg("vmin"), py::arg("probability"))
        .def("to_lines", [](const StreakFinderResult<T> & result, std::optional<T> width)
        {
            return result_to_lines<T, T>(result, width);
        }, py::arg("width") = std::nullopt)
        .def("to_lines", [](const StreakFinderResult<T> & result, std::optional<std::vector<T>> width)
        {
            return result_to_lines<T, std::vector<T>>(result, width);
        }, py::arg("width") = std::nullopt)
        .def("to_regions", [](const StreakFinderResult<T> & result)
        {
            RegionsND<2> regions;
            for (const auto & [_, streak] : result)
            {
                PointSet points;
                for (auto && [point, _] : streak.pixels()) points->emplace_hint(points.end(), std::forward<decltype(point)>(point));
                regions->emplace_back(std::move(points));
            }
            return regions;
        })
        .def_property("mask", [](const StreakFinderResult<T> & result){return to_pyarray(result.mask(), result.mask().shape());}, nullptr)
        .def_property("streaks", [](const StreakFinderResult<T> & result)
        {
            std::map<int, Streak<T>> streaks;
            for (auto pair : result) streaks.emplace_hint(streaks.end(), pair);
            return streaks;
        }, nullptr)
        .def("__repr__", [typestr](const StreakFinderResult<T> & result)
        {
            return "<StreakFinderResult" + typestr + ", mask = <array, shape = {" + std::to_string(result.mask().shape(0)) + ", " +
                   std::to_string(result.mask().shape(1)) + "}>, streaks = <Dict[int, Streak" + typestr + "], size = " +
                   std::to_string(result.size()) + ">>";
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

    declare_streak<double>(m, "Double");
    declare_streak<float>(m, "Float");

    declare_streak_finder_result<double>(m, "Double");
    declare_streak_finder_result<float>(m, "Float");

    py::class_<StreakFinder>(m, "StreakFinder")
        .def(py::init<Structure, unsigned, unsigned, unsigned>(), py::arg("structure"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::keep_alive<1, 2>())
        .def("detect_streaks", [](StreakFinder & finder, py::array_t<double> data, py::array_t<bool> mask, Peaks peaks, double xtol, double vmin)
            {
                array<double> darr {data.request()};
                array<bool> marr {mask.request()};
                return finder.detect_streaks(StreakFinderResult(darr, marr), darr, peaks, xtol, vmin);
            }, py::arg("data"), py::arg("mask"), py::arg("peaks"), py::arg("xtol"), py::arg("vmin"))
        .def("detect_streaks", [](StreakFinder & finder, py::array_t<float> data, py::array_t<bool> mask, Peaks peaks, float xtol, float vmin)
            {
                array<float> darr {data.request()};
                array<bool> marr {mask.request()};
                return finder.detect_streaks(StreakFinderResult(darr, marr), darr, peaks, xtol, vmin);
            }, py::arg("data"), py::arg("mask"), py::arg("peaks"), py::arg("xtol"), py::arg("vmin"))
        .def_readwrite("lookahead", &StreakFinder::lookahead)
        .def_readwrite("nfa", &StreakFinder::nfa)
        .def_readwrite("structure", &StreakFinder::structure);

    m.def("detect_peaks", &detect_peaks<double>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("data"), py::arg("mask"), py::arg("radius"), py::arg("vmin"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("filter_peaks", &filter_peak<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peak<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
}
