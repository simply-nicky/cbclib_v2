#include "numpy.hpp"

namespace cbclib {

template <typename RandomIt, typename Compare, typename T = typename std::iterator_traits<RandomIt>::value_type>
std::common_type_t<T, decltype(0.5 * std::declval<T &>())> median_1d(RandomIt first, RandomIt last, Compare comp)
{
    auto n = std::distance(first, last);
    if (n & 1)
    {
        auto nth = std::next(first, n / 2);
        std::nth_element(first, nth, last, comp);
        return *nth;
    }
    else
    {
        auto low = std::next(first, n / 2 - 1), high = std::next(first, n / 2);
        std::nth_element(first, low, last, comp);
        std::nth_element(high, high, last, comp);
        return 0.5 * (*low + *high);
    }
}

template <typename T, typename U>
py::array_t<double> median(py::array_t<T> inp, U axis, unsigned threads)
{
    Sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_back(inp);

    auto ax = inp.ndim() - seq.size();
    auto out_shape = std::vector<py::ssize_t>(inp.shape(), inp.shape() + ax);
    auto out = py::array_t<double>(out_shape);

    auto oarr = array<double>(out.request());
    auto iarr = array<T>(inp.request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > oarr.size()) ? oarr.size() : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<T> buffer;

        #pragma omp for
        for (size_t i = 0; i < oarr.size(); i++)
        {
            e.run([&]
            {
                auto islice = iarr.slice_back(i, seq.size());

                buffer.clear();
                for (size_t index = 0; index < islice.size(); index++) buffer.push_back(islice[index]);

                if (buffer.size()) oarr[i] = median_1d(buffer.begin(), buffer.end(), std::less<T>());
                else oarr[i] = NAN;
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename U, typename D = std::common_type_t<T, float>>
py::array_t<D> robust_mean(py::array_t<T> inp, U axis, double r0, double r1, int n_iter, double lm,
                           bool return_std, unsigned threads)
{
    Sequence<long> seq (axis);
    seq = seq.unwrap(inp.ndim());
    inp = seq.swap_back(inp);

    auto ibuf = inp.request();
    auto ax = ibuf.ndim - seq.size();
    auto out_shape = std::vector<py::ssize_t>(ibuf.shape.begin(), std::next(ibuf.shape.begin(), ax));
    size_t n_rows = std::reduce(out_shape.begin(), out_shape.end(), 1, std::multiplies());
    size_t n_reduce = ibuf.size / n_rows;

    if (std::reduce(inp.shape() + ax, inp.shape() + inp.ndim(), 1, std::multiplies()) != n_reduce)
        throw std::invalid_argument("shape of input array is incompatible with the specified axis");

    if (return_std) out_shape.insert(out_shape.begin(), 2);
    auto out = py::array_t<D>(out_shape);

    if (!n_rows) return out;

    auto oarr = array<D>(out.request());
    auto iarr = array<T>(inp.request());

    thread_exception e;

    py::gil_scoped_release release;

    threads = (threads > n_rows) ? n_rows : threads;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<D, size_t>> buffer (n_reduce);

        size_t j0 = r0 * n_reduce, j1 = r1 * n_reduce;
        D mean;

        #pragma omp for
        for (size_t i = 0; i < n_rows; i++)
        {
            e.run([&]
            {
                auto islice = iarr.slice_back(i, seq.size());

                for (size_t j = 0; j < n_reduce; j++) buffer[j] = {islice[j], 0};

                if (buffer.size())
                {
                    if (buffer.size() & 1)
                    {
                        auto nth = std::next(buffer.begin(), buffer.size() / 2);
                        std::nth_element(buffer.begin(), nth, buffer.end());
                        mean = nth->first;
                    }
                    else
                    {
                        auto low = std::next(buffer.begin(), buffer.size() / 2 - 1);
                        auto high = std::next(low);
                        std::nth_element(buffer.begin(), low, buffer.end());
                        std::nth_element(high, high, buffer.end());
                        mean = 0.5 * (low->first + high->first);
                    }
                }
                else mean = D();

                for (int n = 0; n < n_iter; n++)
                {
                    for (size_t j = 0; j < n_reduce; j++)
                    {
                        buffer[j] = {(islice[j] - mean) * (islice[j] - mean), j};
                    }
                    std::sort(buffer.begin(), buffer.end());

                    if (j0 != j1)
                    {
                        D sum = D();
                        for (size_t j = j0; j < j1; j++) sum += islice[buffer[j].second];
                        mean = sum / (j1 - j0);
                    }
                    else mean = islice[buffer[j0].second];
                }

                for (size_t j = 0; j < n_reduce; j++)
                {
                    buffer[j] = {(islice[j] - mean) * (islice[j] - mean), j};
                }
                std::sort(buffer.begin(), buffer.end());

                D cumsum = D(); D var = D(); D sum = D(); size_t n_inliers = 0, j = 0;
                for (auto [error, index] : buffer)
                {
                    cumsum += error;
                    if (lm * cumsum < j++ * error) break;

                    sum += islice[index];
                    var += error;
                    n_inliers++;
                }

                if (n_inliers)
                {
                    oarr[i] = sum / n_inliers;
                    if (return_std) oarr[i + n_rows] = std::sqrt(var / n_inliers);
                }
                else
                {
                    oarr[i] = D();
                    if (return_std) oarr[i + n_rows] = D();
                }
            });
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename InputIt, typename Compare, typename = typename std::enable_if_t<
    std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<InputIt>::iterator_category>
>>
void merge_sort_parallel(InputIt left, InputIt right, Compare comp)
{
    if (left < right)
    {
        if (std::distance(left, right) >= 32)
        {
            InputIt mid = left + std::distance(left, right) / 2;
            #pragma omp taskgroup
            {
                #pragma omp task untied if (std::distance(left, right) >= (1<<14))
                merge_sort_parallel(left, mid, comp);
                #pragma omp task untied if (std::distance(left, right) >= (1<<14))
                merge_sort_parallel(mid + 1, right, comp);
                #pragma omp taskyield
            }
            std::inplace_merge(left, mid + 1, right + 1, comp);
        }
        else std::sort(left, right + 1, comp);
    }
}

template <typename InputIt, typename T = typename std::iterator_traits<InputIt>::value_type, typename = typename std::enable_if_t<
    std::is_base_of_v<std::random_access_iterator_tag, typename std::iterator_traits<InputIt>::iterator_category>
>>
void merge_sort_parallel(InputIt left, InputIt right)
{
    merge_sort_parallel(left, right, std::less<T>());
}

template <typename T, typename U, typename D = std::common_type_t<T, float>>
py::array_t<D> robust_lsq(py::array_t<T> W, py::array_t<T> y, U axis, double r0, double r1,
                          int n_iter, double lm, unsigned threads)
{
    Sequence<long> seq (axis);
    seq = seq.unwrap(y.ndim());
    y = seq.swap_back(y);

    py::buffer_info Wbuf = W.request();
    py::buffer_info ybuf = y.request();
    auto ax = ybuf.ndim - seq.size();
    check_equal("W and y arrays have incompatible shapes",
                std::make_reverse_iterator(ybuf.shape.end()), std::make_reverse_iterator(ybuf.shape.begin() + ax),
                std::make_reverse_iterator(Wbuf.shape.end()), std::make_reverse_iterator(Wbuf.shape.begin()));

    if (!ybuf.size || !Wbuf.size)
        throw std::invalid_argument("W and y must have a positive size");

    size_t n_rows = std::reduce(ybuf.shape.begin(), std::next(ybuf.shape.begin(), ax), 1, std::multiplies());
    size_t n_chunks = threads / n_rows + (threads % n_rows > 0);

    size_t n_reduce = ybuf.size / n_rows;
    // chunk size must satisfy chunk_size * n_chunks >= n_reduce to fully cover the reduction dimension
    size_t chunk_size = n_reduce / n_chunks + (n_reduce % n_chunks > 0);
    size_t n_features = Wbuf.size / n_reduce;

    W = W.reshape({n_features, n_reduce});

    auto out_shape = std::vector<py::ssize_t>(ybuf.shape.begin(), std::next(ybuf.shape.begin(), ax));
    out_shape.push_back(n_features);
    auto out = py::array_t<D>(out_shape);

    auto oarr = array<D>(out.request());
    auto Warr = array<T>(W.request());
    auto yarr = array<T>(y.request());

    // Shared result fits for the parallel reduction, indexed by [k * n_rows + row]
    std::vector<std::vector<std::pair<D, size_t>>> error_buffers(n_rows, std::vector<std::pair<D, size_t>>(n_reduce));
    std::vector<std::pair<D, D>> fit_pairs (n_features * n_rows, {D(), D()});
    std::vector<D> fits (n_features * n_rows);
    std::vector<size_t> cutoffs(n_rows, 0);

    size_t j0 = r0 * n_reduce, j1 = r1 * n_reduce;
    size_t truncated_chunk_size = (j1 - j0) / n_chunks + ((j1 - j0) % n_chunks > 0);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads) shared(error_buffers, fits, fit_pairs)
    {
        std::vector<std::pair<D, D>> local_pairs (n_features * n_rows, {D(), D()});
        auto sum_pairs = [](const std::pair<D, D> & a, const std::pair<D, D> & b)
        {
            return std::pair<D, D>{a.first + b.first, a.second + b.second};
        };
        auto get_fit = [](const std::pair<D, D> & pair)
        {
            return (pair.second > D()) ? pair.first / pair.second : D();
        };

        // ========== Initial fit computation ==========
        #pragma omp for collapse(2) nowait
        for (size_t row = 0; row < n_rows; row++)
        {
            for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++)
            {
                size_t start = chunk_id * chunk_size, end = std::min(start + chunk_size, n_reduce);

                for (size_t k = 0; k < n_features; k++)
                {
                    for (size_t j = start; j < end; j++)
                    {
                        auto Wval = Warr.at(k, j);
                        local_pairs[row * n_features + k].first += yarr[row * n_reduce + j] * Wval;
                        local_pairs[row * n_features + k].second += Wval * Wval;
                    }
                }
            }
        }

        #pragma omp critical
        std::transform(local_pairs.begin(), local_pairs.end(), fit_pairs.begin(), fit_pairs.begin(), sum_pairs);

        // Aggregate initial fits
        #pragma omp barrier
        #pragma omp for collapse(2)
        for (size_t i = 0; i < n_rows; i++)
        {
            for (size_t k = 0; k < n_features; k++)
            {
                fits[i * n_features + k] = get_fit(fit_pairs[i * n_features + k]);
            }
        }

        // ========== Iterative refinement loop ==========
        for (int n = 0; n < n_iter; n++)
        {
            // Reset local_pairs and fit_pairs for this iteration
            for (size_t i = 0; i < local_pairs.size(); i++) local_pairs[i] = {D(), D()};

            #pragma omp single
            for (size_t i = 0; i < fit_pairs.size(); i++) fit_pairs[i] = {D(), D()};

            // Phase 1: Compute errors in parallel
            #pragma omp for collapse(2)
            for (size_t row = 0; row < n_rows; row++)
            {
                for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++)
                {
                    size_t start = chunk_id * chunk_size, end = std::min(start + chunk_size, n_reduce);

                    for (size_t j = start; j < end; j++)
                    {
                        auto error = yarr[row * n_reduce + j];
                        for (size_t k = 0; k < n_features; k++)
                            error -= Warr.at(k, j) * fits[row * n_features + k];

                        error_buffers[row][j] = {error * error, j};
                    }
                }
            }

            // Phase 2: Sort errors per row (single thread dispatches per-row GNU parallel sort)
            for (size_t row = 0; row < n_rows; row++)
            {
                #pragma omp single
                merge_sort_parallel(error_buffers[row].begin(), error_buffers[row].end() - 1);
            }

            // Phase 3: Accumulate inlier fit contributions by chunking inlier range [j0, j1)
            #pragma omp for collapse(2) nowait
            for (size_t row = 0; row < n_rows; row++)
            {
                for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++)
                {
                    size_t j_start = j0 + chunk_id * truncated_chunk_size;
                    size_t j_end = std::min(j_start + truncated_chunk_size, j1);

                    for (size_t k = 0; k < n_features; k++)
                    {
                        D sum = D(), weight = D();
                        for (size_t j = j_start; j < j_end; j++)
                        {
                            size_t actual_j = error_buffers[row][j].second;
                            auto Wval = Warr.at(k, actual_j);
                            sum += yarr[row * n_reduce + actual_j] * Wval;
                            weight += Wval * Wval;
                        }
                        local_pairs[row * n_features + k].first += sum;
                        local_pairs[row * n_features + k].second += weight;
                    }
                }
            }

            #pragma omp critical
            std::transform(local_pairs.begin(), local_pairs.end(), fit_pairs.begin(), fit_pairs.begin(), sum_pairs);

            // Phase 4: Aggregate fits for next iteration
            #pragma omp barrier
            #pragma omp for collapse(2)
            for (size_t i = 0; i < n_rows; i++)
            {
                for (size_t k = 0; k < n_features; k++)
                {
                    fits[i * n_features + k] = get_fit(fit_pairs[i * n_features + k]);
                }
            }
        }

        // ========== Final output computation ==========
        // Phase 1: Compute final errors in parallel
        #pragma omp for collapse(2)
        for (size_t row = 0; row < n_rows; row++)
        {
            for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++)
            {
                size_t start = chunk_id * chunk_size, end = std::min(start + chunk_size, n_reduce);

                for (size_t j = start; j < end; j++)
                {
                    auto error = yarr[row * n_reduce + j];
                    for (size_t k = 0; k < n_features; k++) error -= Warr.at(k, j) * fits[row * n_features + k];

                    error_buffers[row][j] = {error * error, j};
                }
            }
        }

        // Phase 2: Sort final errors per row (single thread dispatches per-row GNU parallel sort)
        for (size_t row = 0; row < n_rows; row++)
        {
            #pragma omp single
            merge_sort_parallel(error_buffers[row].begin(), error_buffers[row].end() - 1);
        }

        // Phase 3: Compute final robust output with adaptive threshold
        // Step 3a: Determine inlier cutoff per row (sequential cumsum threshold)
        #pragma omp for
        for (size_t row = 0; row < n_rows; row++)
        {
            D cumsum = D();

            size_t j = 0;
            for (auto [error, index] : error_buffers[row])
            {
                cumsum += error;
                if (lm * cumsum < j++ * error)
                {
                    cutoffs[row] = j - 1; // first outlier index is j-1 because of post increment
                    break;
                }
            }
        }

        // Reset local_pairs and fit_pairs for the final computation
        for (size_t i = 0; i < local_pairs.size(); i++) local_pairs[i] = {D(), D()};

        #pragma omp single
        for (size_t i = 0; i < fit_pairs.size(); i++) fit_pairs[i] = {D(), D()};

        // Step 3b: Accumulate inlier contributions in parallel by chunking inlier range
        #pragma omp for collapse(2) nowait
        for (size_t row = 0; row < n_rows; row++)
        {
            for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++)
            {
                size_t cutoff = cutoffs[row];

                size_t start = std::min(chunk_id * chunk_size, cutoff);
                size_t end = std::min(start + chunk_size, cutoff);

                if (start == end) continue;

                for (size_t k = 0; k < n_features; k++)
                {
                    for (size_t idx = start; idx < end; idx++)
                    {
                        size_t actual_j = error_buffers[row][idx].second;
                        auto Wval = Warr.at(k, actual_j);
                        local_pairs[row * n_features + k].first += yarr[row * n_reduce + actual_j] * Wval;
                        local_pairs[row * n_features + k].second += Wval * Wval;
                    }
                }
            }
        }

        #pragma omp critical
        std::transform(local_pairs.begin(), local_pairs.end(), fit_pairs.begin(), fit_pairs.begin(), sum_pairs);

        // Phase 4: Aggregate final fits and store in output
        #pragma omp barrier
        #pragma omp for collapse(2)
        for (size_t i = 0; i < n_rows; i++)
        {
            for (size_t k = 0; k < n_features; k++)
            {
                oarr[i * n_features + k] = get_fit(fit_pairs[i * n_features + k]);
            }
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

}

PYBIND11_MODULE(median, m)
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

    m.def("median", &median<double, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<double, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<float, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<float, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<int, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<int, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);
    m.def("median", &median<long, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("num_threads") = 1);
    m.def("median", &median<long, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("num_threads") = 1);

    m.def("robust_mean", &robust_mean<double, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<double, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<float, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<int, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, int>, py::arg("inp"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);
    m.def("robust_mean", &robust_mean<long, std::vector<int>>, py::arg("inp"), py::arg("axis") = std::vector<int>{-1}, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("return_std") = false, py::arg("num_threads") = 1);

    m.def("robust_lsq", &robust_lsq<double, int>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<double, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<float, int>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<float, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<int, int>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<int, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<long, int>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);
    m.def("robust_lsq", &robust_lsq<long, std::vector<int>>, py::arg("W"), py::arg("y"), py::arg("axis") = -1, py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0, py::arg("num_threads") = 1);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
