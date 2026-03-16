#include "bresenham.hpp"

namespace cbclib {

template <typename I, int ExtraFlags>
void fill_indices(std::string name, size_t xsize, size_t isize, std::optional<py::array_t<I, ExtraFlags>> & idxs)
{
    idxs = py::array_t<I, ExtraFlags>(isize);

    if (isize)
    {
        if (xsize == 1) fill_array(idxs.value(), I());
        else if (xsize == isize)
        {
            for (size_t i = 0; i < isize; i++) idxs.value().mutable_data()[i] = i;
        }
        else throw std::invalid_argument(name + " has an icnompatible size (" + std::to_string(isize) + " != " +
                                         std::to_string(xsize) + ")");
    }
}

template <typename I, int ExtraFlags>
void check_indices(std::string name, size_t imax, size_t isize, const py::array_t<I, ExtraFlags> & idxs)
{
    if (idxs.size())
    {
        if (static_cast<size_t>(idxs.size()) != isize)
            throw std::invalid_argument(name + " has an invalid size (" + std::to_string(idxs.size()) +
                                        " != " + std::to_string(isize) + ")");

        auto [min, max] = std::minmax_element(idxs.data(), idxs.data() + idxs.size());
        if (*max >= static_cast<I>(imax) || *min < I())
            throw std::out_of_range(name + " range (" + std::to_string(*min) + ", " + std::to_string(*max) +
                                    ") is outside of (0, " + std::to_string(imax) + ")");
    }
}

template <typename T, typename I, size_t N, int Update, kernels::type K>
py::array_t<T> draw_lines_nd_impl(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                                  unsigned threads)
{
    constexpr size_t L = 2 * N + 1;
    constexpr auto kernel = kernels_t<T>::template select<K>();

    array<T> oarr {out.request()};
    array<T> larr (lines.request());

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimension("lines", larr.ndim() - 1, larr.shape().begin(), L);
    auto n_lines = larr.size() / larr.shape(larr.ndim() - 1);

    if (!idxs) fill_indices("idxs", n_frames, n_lines, idxs);
    else check_indices("idxs", n_frames, n_lines, idxs.value());
    auto iarr = array<I>(idxs.value().request());

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        detail::ImageBuffer<size_t, T> buffer (shape);

        auto write = [&oarr, size = buffer.size()](const std::tuple<size_t, size_t, T> & values)
        {
            auto [idx, frame, value] = values;
            size_t index = idx + size * frame;

            if constexpr (Update) oarr[index] = std::max(oarr[index], value);
            else oarr[index] = oarr[index] + value;
        };

        #pragma omp for nowait
        for (size_t i = 0; i < n_lines; i++)
        {
            e.run([&]()
            {
                auto draw_pixel = [&buffer, &kernel, max_val, frame = iarr[i]](const PointND<long, N> & pt, T error)
                {
                    if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                    {
                        buffer.emplace_back(pt, frame, max_val * kernel(std::sqrt(error)));
                    }
                };

                draw_line_nd(LineND<T, N>{to_point<N>(larr, L * i), to_point<N>(larr, L * i + N)}, larr[L * i + 2 * N], draw_pixel);
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), write);
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I, size_t N, int Update>
py::array_t<T> draw_lines_nd(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                             std::string kernel_name, unsigned threads)
{
    auto ktype = kernels::get_type(kernel_name);
    switch (ktype)
    {
        case kernels::biweight:
            return draw_lines_nd_impl<T, I, N, Update, kernels::biweight>(out, lines, idxs, max_val, threads);
        case kernels::gaussian:
            return draw_lines_nd_impl<T, I, N, Update, kernels::gaussian>(out, lines, idxs, max_val, threads);
        case kernels::parabolic:
            return draw_lines_nd_impl<T, I, N, Update, kernels::parabolic>(out, lines, idxs, max_val, threads);
        case kernels::rectangular:
            return draw_lines_nd_impl<T, I, N, Update, kernels::rectangular>(out, lines, idxs, max_val, threads);
        case kernels::triangular:
            return draw_lines_nd_impl<T, I, N, Update, kernels::triangular>(out, lines, idxs, max_val, threads);
        default:
            throw std::invalid_argument("Invalid kernel type");
    }
}

template <typename T, typename I, int Update>
py::array_t<T> draw_lines_2d_3d(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                                std::string kernel_name, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return draw_lines_nd<T, I, 2, Update>(out, lines, idxs, max_val, kernel_name, threads);
        case 7:
            return draw_lines_nd<T, I, 3, Update>(out, lines, idxs, max_val, kernel_name, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I>
py::array_t<T> draw_lines(py::array_t<T> out, py::array_t<T> lines, std::optional<py::array_t<I>> idxs, T max_val,
                          std::string kernel_name, std::string overlap, unsigned threads)
{
    if (overlap == "sum") return draw_lines_2d_3d<T, I, 0>(out, lines, idxs, max_val, kernel_name, threads);
    if (overlap == "max") return draw_lines_2d_3d<T, I, 1>(out, lines, idxs, max_val, kernel_name, threads);
    throw std::invalid_argument("Invalid overlap keyword: " + overlap);
}

template <typename T, typename I, size_t N, int Update, kernels::type K>
py::array_t<T> accumulate_lines_nd_impl(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> terms, py::array_t<I> frames,
                                        T max_val, unsigned threads)
{
    constexpr size_t L = 2 * N + 1;
    constexpr int TermUpdate = Update & 1;
    constexpr int GlobalUpdate = (Update >> 1) & 1;
    constexpr auto kernel = kernels_t<T>::template select<K>();

    array<T> oarr {out.request()};
    array<T> larr {lines.request()};
    array<I> tarr {terms.request()};

    auto n_frames = std::reduce(oarr.shape().begin(), std::prev(oarr.shape().end(), N), size_t(1), std::multiplies());
    std::vector<size_t> shape {std::prev(oarr.shape().end(), N), oarr.shape().end()};

    check_dimension("lines", larr.ndim() - 1, larr.shape().begin(), L);
    auto n_lines = larr.size() / larr.shape(larr.ndim() - 1);

    check_indices("frames", n_frames, frames.size(), frames);
    if (terms.size() != n_lines)
        throw std::invalid_argument("Term indices (" + std::to_string(terms.size()) +
                                    ") is incompatible with number of lines (" + std::to_string(n_lines) + ")");

    array<I> farr {frames.request()};

    // Sorting lines by terms
    std::vector<size_t> sort_idxs (tarr.size());
    std::iota(sort_idxs.begin(), sort_idxs.end(), 0);
    std::sort(sort_idxs.begin(), sort_idxs.end(),[&tarr](size_t a, size_t b){ return tarr[a] < tarr[b]; });

    // Counting number of lines per term
    std::vector<size_t> counts (farr.size(), 0);
    for (size_t i = 0; i < tarr.size(); i++)
    {
        if (tarr[i] >= static_cast<I>(farr.size()))
            throw std::out_of_range("Term index " + std::to_string(tarr[i]) + " is out of range (0, " +
                                    std::to_string(farr.size()) + ")");
        counts[tarr[i]]++;
    }

    // Creating CSR offsets
    std::vector<size_t> offsets (farr.size() + 1, 0);
    std::partial_sum(counts.begin(), counts.end(), std::next(offsets.begin()));

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        vector_array<T> buffer {shape};
        std::vector<size_t> indices;

        #pragma omp for nowait
        for (size_t term_idx = 0; term_idx < farr.size(); term_idx++)
        {
            T value;
            size_t shift = farr[term_idx] * buffer.size();

            for (size_t i = offsets[term_idx]; i < offsets[term_idx + 1]; i++)
            {
                e.run([&]()
                {
                    auto draw_pixel = [&buffer, &indices, &kernel, max_val](const PointND<long, N> & pt, T error)
                    {
                        if (error <= 1.0 && buffer.is_inbound(pt.rbegin(), pt.rend()))
                        {
                            size_t index = buffer.index_at(pt.rbegin(), pt.rend());
                            if (buffer[index] == T()) indices.push_back(index);
                            if constexpr (TermUpdate) buffer[index] = std::max(buffer[index], max_val * kernel(std::sqrt(error)));
                            else buffer[index] = buffer[index] + max_val * kernel(std::sqrt(error));
                        }
                    };

                    draw_line_nd(LineND<T, N>{to_point<N>(larr, L * sort_idxs[i]), to_point<N>(larr, L * sort_idxs[i] + N)},
                                 larr[L * sort_idxs[i] + 2 * N], draw_pixel);
                });
            }

            for (auto index : indices)
            {
                if constexpr (GlobalUpdate)
                {
                    #pragma omp atomic read
                    value = oarr[index + shift];

                    #pragma omp atomic write
                    oarr[index + shift] = std::max(value, buffer[index]);
                }
                else
                {
                    #pragma omp atomic update
                    oarr[index + shift] = buffer[index] + oarr[index + shift];
                }

                buffer[index] = T();
            }

            indices.clear();
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    return out;
}

template <typename T, typename I, size_t N, int Update>
py::array_t<T> accumulate_lines_nd(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> frames, py::array_t<I> counts,
                                   T max_val, std::string kernel_name, unsigned threads)
{
    auto ktype = kernels::get_type(kernel_name);
    switch (ktype)
    {
        case kernels::biweight:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::biweight>(out, lines, frames, counts, max_val, threads);
        case kernels::gaussian:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::gaussian>(out, lines, frames, counts, max_val, threads);
        case kernels::parabolic:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::parabolic>(out, lines, frames, counts, max_val, threads);
        case kernels::rectangular:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::rectangular>(out, lines, frames, counts, max_val, threads);
        case kernels::triangular:
            return accumulate_lines_nd_impl<T, I, N, Update, kernels::triangular>(out, lines, frames, counts, max_val, threads);
        default:
            throw std::invalid_argument("Invalid kernel type");
    }
}

template <typename T, typename I, int Update>
py::array_t<T> accumulate_lines_2d_3d(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> frames, py::array_t<I> counts,
                                      T max_val, std::string kernel_name, unsigned threads)
{
    size_t L = lines.shape(lines.ndim() - 1);
    switch (L)
    {
        case 5:
            return accumulate_lines_nd<T, I, 2, Update>(out, lines, frames, counts, max_val, kernel_name, threads);
        case 7:
            return accumulate_lines_nd<T, I, 3, Update>(out, lines, frames, counts, max_val, kernel_name, threads);
        default:
            throw std::runtime_error("Invalid lines size (" + std::to_string(L) + ") at axis " +
                                     std::to_string(lines.ndim() - 1));
    }
}

template <typename T, typename I>
py::array_t<T> accumulate_lines(py::array_t<T> out, py::array_t<T> lines, py::array_t<I> frames, py::array_t<I> counts,
                                T max_val, std::string kernel_name, std::string in_overlap, std::string out_overlap, unsigned threads)
{
    if (in_overlap == "sum" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 0>(out, lines, frames, counts, max_val, kernel_name, threads);
    if (in_overlap == "max" && out_overlap == "sum") return accumulate_lines_2d_3d<T, I, 1>(out, lines, frames, counts, max_val, kernel_name, threads);
    if (in_overlap == "sum" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 2>(out, lines, frames, counts, max_val, kernel_name, threads);
    if (in_overlap == "max" && out_overlap == "max") return accumulate_lines_2d_3d<T, I, 3>(out, lines, frames, counts, max_val, kernel_name, threads);
    throw std::invalid_argument("Invalid in_overlap and out_overlap keywords: " + in_overlap + " and " + out_overlap);
}

}

PYBIND11_MODULE(bresenham, m)
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

    m.def("accumulate_lines", &accumulate_lines<double, long>, py::arg("out"), py::arg("lines"), py::arg("terms"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);
    m.def("accumulate_lines", &accumulate_lines<float, int>, py::arg("out"), py::arg("lines"), py::arg("terms"), py::arg("frames"), py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("in_overlap") = "sum", py::arg("out_overlap") = "sum", py::arg("num_threads") = 1);

    m.def("draw_lines", &draw_lines<double, long>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
    m.def("draw_lines", &draw_lines<float, int>, py::arg("out"), py::arg("lines"), py::arg("idxs") = nullptr, py::arg("max_val") = 1, py::arg("kernel") = "rectangular", py::arg("overlap") = "sum", py::arg("num_threads") = 1);
}
