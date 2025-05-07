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

    std::vector<Peaks> result;

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<Peaks> buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < repeats; i++)
        {
            e.run([&]
            {
                buffer.emplace_back(darr.slice_back(i, axes.size()), marr, radius, vmin);
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
auto filter_peaks(Peaks peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
                  py::none ax, unsigned threads)
{
    array<T> darr {data.request()};
    array<bool> marr {mask.request()};

    check_equal("data and mask have incompatible shapes",
                std::prev(darr.shape().end(), 2), darr.shape().end(), marr.shape().begin(), marr.shape().end());
    if (darr.ndim() != 2)
        fail_container_check("wrong number of dimensions (" + std::to_string(darr.ndim()) + " != 2)", darr.shape());

    py::gil_scoped_release release;

    Peaks result = peaks;
    result.filter(darr, marr, structure, vmin, npts);

    py::gil_scoped_acquire acquire;

    return result;
}

template <typename T>
auto filter_peaks_vec(std::vector<Peaks> peaks, py::array_t<T> data, py::array_t<bool> mask, Structure structure, T vmin, size_t npts,
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

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > repeats) ? repeats : threads;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < repeats; i++)
    {
        e.run([&]
        {
            peaks[i].filter(darr.slice_back(i, axes.size()), marr, structure, vmin, npts);
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return peaks;
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
        .def(py::init([](std::vector<long> xvec, std::vector<long> yvec)
        {
            std::set<Point<long>> points;
            for (auto [x, y] : zip::zip(xvec, yvec)) points.emplace(Point<long>{x, y});
            return Peaks(std::move(points));
        }), py::arg("x"), py::arg("y"))
        .def_property("x", [](const Peaks & peaks){return detail::get_x(peaks, 0);}, nullptr)
        .def_property("y", [](const Peaks & peaks){return detail::get_x(peaks, 1);}, nullptr)
        .def("__iter__", [](const Peaks & peaks)
        {
            return py::make_iterator(make_python_iterator(peaks.begin()), make_python_iterator(peaks.end()));
        }, py::keep_alive<0, 1>())
        .def("__len__", [](const Peaks & peaks){return peaks.size();})
        .def("filter", [](Peaks & peaks, py::array_t<float> data, py::array_t<bool> mask, Structure s, float vmin, size_t npts)
        {
            array<float> darr {data.request()};
            array<bool> marr {mask.request()};
            check_equal("data and good have incompatible shapes",
                        darr.shape().begin(), darr.shape().end(), marr.shape().begin(), marr.shape().end());
            auto result = peaks;
            result.filter(darr, marr, s, vmin, npts);
            return result;
        }, py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("filter", [](Peaks & peaks, py::array_t<double> data, py::array_t<bool> mask, Structure s, double vmin, size_t npts)
        {
            array<double> darr {data.request()};
            array<bool> marr {mask.request()};
            check_equal("data and good have incompatible shapes",
                        darr.shape().begin(), darr.shape().end(), marr.shape().begin(), marr.shape().end());
            auto result = peaks;
            result.filter(darr, marr, s, vmin, npts);
            return result;
        }, py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"))
        .def("__repr__", &Peaks::info);

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

    m.def("filter_peaks", &filter_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks_vec<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("filter_peaks", &filter_peaks_vec<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("vmin"), py::arg("npts"), py::arg("axes")=std::nullopt, py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<double>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks_vec<float>, py::arg("peaks"), py::arg("data"), py::arg("mask"), py::arg("structure"), py::arg("xtol"), py::arg("vmin"), py::arg("min_size"), py::arg("lookahead")=0, py::arg("nfa")=0, py::arg("axes")=std::nullopt, py::arg("num_threads")=1);
}
