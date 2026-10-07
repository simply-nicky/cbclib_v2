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

    if (std::reduce(inp.shape() + ax, inp.shape() + inp.ndim(), size_t(1), std::multiplies()) != n_reduce)
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
        std::vector<D> values (n_reduce);
        std::vector<std::pair<D, size_t>> buffer (n_reduce);

        size_t j0 = r0 * n_reduce, j1 = r1 * n_reduce;
        D mean;

        #pragma omp for
        for (size_t i = 0; i < n_rows; i++)
        {
            e.run([&]
            {
                size_t start = i * n_reduce, end = start + n_reduce;

                for (size_t index = start, j = 0; index < end; index++, j++)
                {
                    values[j] = iarr[index];
                    buffer[j] = {values[j], 0};
                }

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
                    for (size_t index = start, j = 0; index < end; index++, j++)
                    {
                        buffer[j] = {(values[j] - mean) * (values[j] - mean), j};
                    }

                    if (j0 != j1)
                    {
                        std::nth_element(buffer.begin(), buffer.begin() + j1 - 1, buffer.end());
                        if (j0) std::nth_element(buffer.begin(), buffer.begin() + j0, buffer.begin() + j1);

                        D sum = D();
                        for (size_t j = j0; j < j1; j++) sum += values[buffer[j].second];
                        mean = sum / (j1 - j0);
                    }
                    else
                    {
                        std::nth_element(buffer.begin(), buffer.begin() + j0, buffer.end());
                        mean = values[buffer[j0].second];
                    }
                }

                for (size_t index = start, j = 0; index < end; index++, j++)
                {
                    buffer[j] = {(values[j] - mean) * (values[j] - mean), j};
                }
                std::sort(buffer.begin(), buffer.end());

                D cumsum = D(); D var = D(); D sum = D(); size_t n_inliers = 0, j = 0;
                for (auto [error, index] : buffer)
                {
                    cumsum += error;
                    if (lm * cumsum < j++ * error) break;

                    sum += values[index];
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

}

PYBIND11_MODULE(median, m)
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


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
