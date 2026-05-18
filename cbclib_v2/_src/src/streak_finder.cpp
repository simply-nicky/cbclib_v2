#include "streak_finder.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<cbclib::Streak>)

static constexpr size_t L = 4; // Linelet size in 2D, should be 2 * N for N-dimensional data

namespace cbclib {

struct PeakLabels
{
    py::array_t<long> labels;
    size_t n_seeds, n_labels, n_good, radius;
};

template <typename T>
py::array_t<lint_t> detect_peaks(py::array_t<lint_t> labels, py::array_t<T> data, Structure structure, size_t radius, T vmin, unsigned threads)
{
    if (structure.rank() != data.ndim()) throw std::invalid_argument("Structure must have rank " + std::to_string(data.ndim()) + " to match data dimensions");
    check_equal("labels and data must have the same shape", labels.shape(), labels.shape() + labels.ndim(), data.shape(), data.shape() + data.ndim());

    array<lint_t> larr {labels.request()};
    array<T> darr {data.request()};

    PeaksIndexer indexer (darr.shape(), radius);
    std::vector<py::ssize_t> shape {data.shape(), data.shape() + data.ndim() - 2};
    shape.push_back(indexer.binned_shape(1));
    shape.push_back(indexer.binned_shape(2));
    py::array_t<lint_t> indices {shape};

    array<lint_t> result {indices.request()};

    size_t module_size = indexer.binned_shape(1) * indexer.binned_shape(2);
    size_t repeats = indexer.binned_shape(0);
    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    size_t chunk_size = module_size / n_chunks;

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < n_chunks * repeats; i++)
    {
        e.run([&]
        {
            bint_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;

            bint_t first = frame * module_size + chunk_id * chunk_size;
            bint_t last = (chunk_id == n_chunks - 1) ? (frame + 1) * module_size : first + chunk_size;

            for (bint_t bin_idx = first; bin_idx < last; bin_idx++)
            {
                auto start = make_point<3>(bin_idx, indexer.binned_shape());
                start[0] *= radius; start[1] *= radius;

                auto end = start;
                end[0] = std::min(end[0] + radius, indexer.shape(2));
                end[1] = std::min(end[1] + radius, indexer.shape(1));

                bool is_good = false;
                for (auto x = start[0]; x < end[0]; x++)
                {
                    for (auto y = start[1]; y < end[1]; y++)
                    {
                        auto running = indexer.index_at(frame, y, x);
                        if (larr[running] != 0)
                        {
                            is_good = true;
                            break;
                        }
                    }
                }

                if (!is_good) result[bin_idx] = -1; // Mark as not a peak and not usable for streak detection
                else
                {
                    // unsigned peak_maximality = 0;
                    lint_t peak_index = -1;
                    T best_val = vmin;

                    for (auto x = start[0]; x < end[0]; x++)
                    {
                        for (auto y = start[1]; y < end[1]; y++)
                        {
                            auto running = indexer.index_at(frame, y, x);

                            T val = darr[running];
                            if (val < vmin) continue;

                            auto maximality = detail::maximality(running, structure, darr);
                            if (maximality == structure.shifts().size() && val > best_val)
                            {
                                best_val = val;
                                peak_index = running;
                            }
                        }
                    }

                    if (peak_index != -1) result[bin_idx] = peak_index;
                    else result[bin_idx] = darr.size(); // Mark as can be used for streak detection, but not a peak itself
                }
            }
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return indices;
}

enum class Direction
{
    None = 0u,
    Forward = 1u,
    Backward = 2u
};

inline constexpr Direction operator|(Direction lhs, Direction rhs)
{
    return static_cast<Direction>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

inline constexpr Direction operator&(Direction lhs, Direction rhs)
{
    return static_cast<Direction>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

inline constexpr bool is_none(Direction dir)
{
    return dir == Direction::None;
}

template <typename T>
py::array_t<T> line_fit(PeakLabels & labels, py::array_t<lint_t> parray, py::array_t<T> data, Structure structure, T vmin, unsigned threads)
{
    if (structure.rank() != data.ndim()) throw std::invalid_argument("Structure must have rank " + std::to_string(data.ndim()) + " to match data dimensions");

    Peaks peaks {labels.labels.request(), parray.request(), labels.n_labels};
    array<T> darr {data.request()};
    PeaksIndexer indexer (darr.shape(), labels.radius);
    LineFitter<T> fitter (darr, structure, vmin, labels.radius);

    py::array_t<T> result (std::vector<py::ssize_t>{py::ssize_t(labels.n_good), py::ssize_t(L)});
    fill_array<T>(result, T());

    Linelets<T> linelets (result.request());

    size_t module_size = indexer.binned_shape(1) * indexer.binned_shape(2);
    size_t repeats = indexer.binned_shape(0);
    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    size_t chunk_size = module_size / n_chunks;

    std::vector<Direction> frontier (indexer.n_bins(), Direction::None);   // Current list of bins to process
    std::vector<std::pair<PointIndex, Direction>> candidates;
    size_t old_size = 0;

    py::gil_scoped_release release;

    thread_exception e;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<PointIndex, Direction>> local_candidates; // Candidate points to write down to linelets buffer

        #pragma omp for
        for (size_t i = 0; i < n_chunks * repeats; i++)
        {
            bint_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;

            bint_t first = frame * module_size + chunk_id * chunk_size;
            bint_t last = (chunk_id == n_chunks - 1) ? (frame + 1) * module_size : first + chunk_size;

            for (bint_t bin_idx = first; bin_idx < last; bin_idx++)
            {
                if (peaks.is_peak(bin_idx))
                {
                    lint_t peak_index = peaks[bin_idx];
                    auto linelet = fitter.fit_linelet(peak_index);
                    auto is_inserted = linelets.insert(peaks.index(bin_idx), linelet);
                    if (is_inserted) frontier[bin_idx] = Direction::Forward | Direction::Backward;
                }
            }
        }

        while (old_size < labels.n_labels)
        {
            local_candidates.clear();

            #pragma omp for
            for (size_t i = 0; i < n_chunks * repeats; i++)
            {
                bint_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;

                bint_t first = frame * module_size + chunk_id * chunk_size;
                bint_t last = (chunk_id == n_chunks - 1) ? (frame + 1) * module_size : first + chunk_size;

                for (bint_t bin_idx = first; bin_idx < last; bin_idx++)
                {
                    PointIndex neighbour {};
                    auto direction = frontier[bin_idx];

                    if (!is_none(direction & Direction::Backward))
                    {
                        auto is_good = fitter.neighbour_behind(bin_idx, linelets.line(peaks.index(bin_idx)), peaks, neighbour);
                        if (is_good) local_candidates.push_back({neighbour, Direction::Backward});
                    }

                    if (!is_none(direction & Direction::Forward))
                    {
                        auto is_good = fitter.neighbour_ahead(bin_idx, linelets.line(peaks.index(bin_idx)), peaks, neighbour);
                        if (is_good) local_candidates.push_back({neighbour, Direction::Forward});
                    }

                    frontier[bin_idx] = Direction::None; // Mark as processed
                }
            }

            #pragma omp single
            {
                old_size = labels.n_labels;
                candidates.clear();
            }

            #pragma omp critical
            candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());

            #pragma omp barrier
            #pragma omp single
            {
                for (const auto & [candidate, direction] : candidates)
                {
                    auto linelet = fitter.fit_linelet(candidate.index);
                    if (linelet.pt0 == linelet.pt1) continue;

                    frontier[candidate.bin] = frontier[candidate.bin] | direction; // Mark for processing in the next iteration if not marked as bad

                    auto line_id = peaks.insert(candidate, darr);
                    linelets.insert(line_id, linelet);
                }
            }
        }
    }

    e.rethrow();

    py::gil_scoped_acquire acquire;

    return result;
}

template <typename T>
std::vector<Streak> detect_streaks(PeakLabels labels, py::array_t<lint_t> parray, py::array_t<T> larray, py::array_t<T> data, Structure structure, T vmin, T xtol, unsigned nfa, unsigned threads)
{
    if (structure.rank() != data.ndim()) throw std::invalid_argument("Structure must have rank " + std::to_string(data.ndim()) + " to match data dimensions");

    Peaks peaks {labels.labels.request(), parray.request(), labels.n_labels};
    Linelets<T> linelets (larray.request());

    StreakFinder<T> finder (data.request(), structure, vmin, xtol, nfa, labels.radius);

    size_t module_size = finder.indexer().binned_shape(1) * finder.indexer().binned_shape(2);
    size_t repeats = finder.indexer().binned_shape(0);
    size_t n_chunks = threads / repeats + (threads % repeats > 0);

    size_t chunk_size = module_size / n_chunks;

    std::vector<Streak> results (labels.n_seeds);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for nowait
        for (size_t i = 0; i < n_chunks * repeats; i++)
        {
            e.run([&]
            {
                bint_t frame = i / n_chunks, chunk_id = i - frame * n_chunks;

                bint_t first = frame * module_size + chunk_id * chunk_size;
                bint_t last = (chunk_id == n_chunks - 1) ? (frame + 1) * module_size : first + chunk_size;

                for (bint_t bin_idx = first; bin_idx < last; bin_idx++)
                {
                    if (peaks.is_peak(bin_idx))
                    {
                        auto id = peaks.index(bin_idx);
                        if (id < static_cast<lint_t>(labels.n_seeds))
                        {
                            results[id].insert(bin_idx, linelets, peaks);
                            auto is_good = finder.detect(results[id], linelets, peaks);
                            if (!is_good) results[id] = Streak();
                        }
                    }
                }
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return results;
}

template <class InputIt, typename T>
T p_value(InputIt first, InputIt last, const array<T> & data, T p0, T vmin, T xtol)
{
    MomentsND<T, 2> moments;
    for (auto iter = first; iter != last; ++iter)
    {
        moments.insert(*iter, data);
    }
    auto line = moments.central().line();

    size_t n = 0, k = 0;
    for (auto iter = first; iter != last; ++iter)
    {
        auto point = make_point<2>(*iter, data.shape());
        if (line.distance(point) < xtol)
        {
            n++;
            if (data[*iter] >= vmin) k++;
        }
    }

    return cbclib::detail::logbinom(n, k, p0);
}

template <typename T>
py::array_t<T> p_values(const std::vector<Streak> & streaks, PeakLabels labels, py::array_t<lint_t> parray, py::array_t<T> data, Structure structure, T p0, T vmin, T xtol, unsigned threads)
{
    if (structure.rank() != data.ndim()) throw std::invalid_argument("Structure must have rank " + std::to_string(data.ndim()) + " to match data dimensions");

    Peaks peaks {labels.labels.request(), parray.request(), labels.n_labels};
    array<T> darr {data.request()};

    py::array_t<T> result (std::vector<py::ssize_t>{py::ssize_t(streaks.size())});
    array<T> p_vals {result.request()};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < streaks.size(); i++)
    {
        e.run([&]
        {
            const auto & streak = streaks[i];
            std::vector<long> footprint;
            footprint.reserve(streak.indices().size() * structure.size());

            for (auto bin_idx : streak.indices())
            {
                long peak_idx = peaks[bin_idx];
                for (const auto & shift : structure)
                {
                    auto neighbour_idx = cbclib::detail::shift_index(peak_idx, shift, darr.shape());
                    if (neighbour_idx >= 0) footprint.push_back(neighbour_idx);
                }
            }

            if (footprint.empty())
            {
                p_vals[i] = T();
            }
            else
            {
                std::sort(footprint.begin(), footprint.end());
                auto end_unique = std::unique(footprint.begin(), footprint.end());

                p_vals[i] = p_value(footprint.begin(), end_unique, darr, p0, vmin, xtol);
            }
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

template <typename T>
py::array_t<py::ssize_t> n_signal(const std::vector<Streak> & streaks, PeakLabels labels, py::array_t<lint_t> parray, py::array_t<T> data, Structure structure, T vmin, unsigned threads)
{
    if (structure.rank() != data.ndim()) throw std::invalid_argument("Structure must have rank " + std::to_string(data.ndim()) + " to match data dimensions");

    Peaks peaks {labels.labels.request(), parray.request(), labels.n_labels};
    array<T> darr {data.request()};

    py::array_t<py::ssize_t> result (std::vector<py::ssize_t>{py::ssize_t(streaks.size())});
    array<py::ssize_t> counts {result.request()};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < streaks.size(); i++)
    {
        e.run([&]
        {
            const auto & streak = streaks[i];
            std::vector<long> footprint;
            footprint.reserve(streak.indices().size() * structure.size());

            for (auto bin_idx : streak.indices())
            {
                long peak_idx = peaks[bin_idx];
                for (const auto & shift : structure)
                {
                    auto neighbour_idx = cbclib::detail::shift_index(peak_idx, shift, darr.shape());
                    if (neighbour_idx >= 0) footprint.push_back(neighbour_idx);
                }
            }

            if (footprint.empty())
            {
                counts[i] = 0;
            }
            else
            {
                std::sort(footprint.begin(), footprint.end());
                auto end_unique = std::unique(footprint.begin(), footprint.end());

                counts[i] = std::count_if(footprint.begin(), end_unique, [&darr, vmin](long idx)
                {
                    return darr[idx] >= vmin;
                });
            }
        });
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return result;
}

py::array_t<lint_t> streak_labels(py::array_t<lint_t> out, const std::vector<Streak> & streaks, py::array_t<py::ssize_t> ranking, PeakLabels labels, py::array_t<lint_t> parray, Structure structure, unsigned threads)
{
    if (static_cast<py::ssize_t>(structure.rank()) != out.ndim())
    {
        throw std::invalid_argument("Structure must have rank " + std::to_string(out.ndim()) + " to match labels dimensions");
    }
    if (static_cast<py::ssize_t>(streaks.size()) != ranking.size())
    {
        throw std::invalid_argument("Size of ranking array must match number of streaks");
    }

    Peaks peaks {labels.labels.request(), parray.request(), labels.n_labels};
    array<py::ssize_t> ranks {ranking.request()};
    array<lint_t> result {out.request()};
    fill_array<lint_t>(out, 0);

    PeaksIndexer indexer {result.shape(), labels.radius};

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for
        for (size_t i = 0; i < streaks.size(); i++)
        {
            e.run([&]
            {
                const auto & streak = streaks[i];
                for (auto bin_idx : streak.indices())
                {
                    lint_t peak_idx = peaks[bin_idx];
                    for (const auto & shift : structure)
                    {
                        auto neighbour_idx = cbclib::detail::shift_index(peak_idx, shift, result.shape());
                        if (neighbour_idx >= 0)
                        {
                            // Keep 0 as background; among streak labels, the smallest sorted index wins.
                            auto candidate = ranks[i] + 1;

                            #pragma omp atomic compare
                            if (result[neighbour_idx] == 0)
                            {
                                result[neighbour_idx] = candidate;
                            }

                            #pragma omp atomic compare
                            if (candidate < result[neighbour_idx])
                            {
                                result[neighbour_idx] = candidate;
                            }
                        }
                    }
                }
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

} // namespace cbclib

PYBIND11_MODULE(streak_finder, m)
{
    using namespace cbclib;
    namespace py = cbclib::py;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    py::class_<PeakLabels>(m, "PeakLabels")
        .def(py::init<py::array_t<long>, size_t, size_t, size_t, size_t>(), py::arg("labels"), py::arg("n_seeds"), py::arg("n_labels"), py::arg("n_good"), py::arg("radius"))
        .def_readonly("labels", &PeakLabels::labels)
        .def_readonly("n_seeds", &PeakLabels::n_seeds)
        .def_readonly("n_labels", &PeakLabels::n_labels)
        .def_readonly("n_good", &PeakLabels::n_good)
        .def_readonly("radius", &PeakLabels::radius)
        .def("keep_best", [](const PeakLabels & labels, double quantile)
        {
            return PeakLabels{labels.labels, size_t(labels.n_seeds * quantile), labels.n_labels, labels.n_good, labels.radius};
        }, py::arg("quantile")=0.5);

    py::class_<Streak>(m, "Streak")
        .def_property_readonly("indices", py::overload_cast<>(&Streak::indices, py::const_))
        .def("line", [](const Streak & streak, PeakLabels labels, py::array_t<double> lines)
        {
            array<lint_t> lbarr {labels.labels.request()};
            Linelets<double> linelets {lines.request()};
            return streak.ends().line(linelets, lbarr).to_array();
        }, py::arg("labels"), py::arg("linelets"))
        .def("line", [](const Streak & streak, PeakLabels labels, py::array_t<float> lines)
        {
            array<lint_t> lbarr {labels.labels.request()};
            Linelets<float> linelets {lines.request()};
            return streak.ends().line(linelets, lbarr).to_array();
        }, py::arg("labels"), py::arg("linelets"));

    py::class_<std::vector<Streak>> streaks_cls (m, "Streaks");
    declare_list(streaks_cls, "Streaks");

    streaks_cls.def("to_lines", [](const std::vector<Streak> & pattern, PeakLabels labels, py::array_t<double> lines)
        {
            array<lint_t> lbarr {labels.labels.request()};
            Linelets<double> linelets {lines.request()};
            std::vector<double> result;

            for (const auto & streak : pattern)
            {
                auto line = streak.ends().line(linelets, lbarr).to_array();
                result.insert(result.end(), line.begin(), line.end());
            }
            return as_pyarray(std::move(result), std::vector<py::ssize_t>{py::ssize_t(pattern.size()), L});
        }, py::arg("labels"), py::arg("lines"))
        .def("to_lines", [](const std::vector<Streak> & pattern, PeakLabels labels, py::array_t<float> lines)
        {
            array<lint_t> lbarr {labels.labels.request()};
            Linelets<float> linelets {lines.request()};
            std::vector<float> result;

            for (const auto & streak : pattern)
            {
                auto line = streak.ends().line(linelets, lbarr).to_array();
                result.insert(result.end(), line.begin(), line.end());
            }
            return as_pyarray(std::move(result), std::vector<py::ssize_t>{py::ssize_t(pattern.size()), L});
        }, py::arg("labels"), py::arg("lines"));

    m.def("detect_peaks", &detect_peaks<double>, py::arg("labels"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"), py::arg("num_threads")=1);
    m.def("detect_peaks", &detect_peaks<float>, py::arg("labels"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"), py::arg("num_threads")=1);

    m.def("line_fit", &line_fit<double>, py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("num_threads")=1);
    m.def("line_fit", &line_fit<float>, py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("num_threads")=1);

    m.def("detect_streaks", &detect_streaks<double>, py::arg("labels"), py::arg("peaks"), py::arg("linelets"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("xtol"), py::arg("nfa")=0, py::arg("num_threads")=1);
    m.def("detect_streaks", &detect_streaks<float>, py::arg("labels"), py::arg("peaks"), py::arg("linelets"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("xtol"), py::arg("nfa")=0, py::arg("num_threads")=1);

    m.def("p_values", &p_values<double>, py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("p0"), py::arg("vmin"), py::arg("xtol"), py::arg("num_threads")=1);
    m.def("p_values", &p_values<float>, py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("p0"), py::arg("vmin"), py::arg("xtol"), py::arg("num_threads")=1);

    m.def("n_signal", &n_signal<double>, py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("num_threads")=1);
    m.def("n_signal", &n_signal<float>, py::arg("streaks"), py::arg("labels"), py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("vmin"), py::arg("num_threads")=1);

    m.def("streak_labels", &streak_labels, py::arg("out"), py::arg("streaks"), py::arg("indices"), py::arg("labels"), py::arg("peaks"), py::arg("structure"), py::arg("num_threads")=1);
}
