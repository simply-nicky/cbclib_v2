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
            int thread_id = omp_get_thread_num();
            long chunk = (current.size() + threads - 1) / threads;

            long thread_start = thread_id * chunk;
            long thread_end = std::min<long>((thread_id + 1) * chunk, current.size());

            std::vector<size_t> coord(shape.size());
            if (thread_start < thread_end) inp.coord_at(coord.begin(), thread_start);

            for (long index = thread_start; index < thread_end; ++index)
            {
                if (current[index])
                {
                    next[index] = 1;
                }
                else if (has_mask && !(*marr)[index])
                {
                    next[index] = 0;
                }
                else
                {
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

                detail::next_coord(coord, shape);
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

template <size_t N>
struct LabelShift
{
    std::array<int, N> delta;
    std::array<int, N> lower;
    std::array<int, N> upper;
    int offset = 0;
};

template <size_t N>
struct Bounds
{
    std::array<int, N> lower;
    std::array<int, N> upper;

    bool contains(const std::array<int, N> & coord) const
    {
        for (size_t dim = 0; dim < N; ++dim)
        {
            if (coord[dim] < lower[dim] || coord[dim] >= upper[dim]) return false;
        }
        return true;
    }
};

template <size_t N>
std::array<int, N> int_shape(const array_indexer & inp)
{
    std::array<int, N> shape;
    for (size_t dim = 0; dim < N; ++dim) shape[dim] = static_cast<int>(inp.shape(dim));
    return shape;
}

template <size_t N>
std::array<int, N> int_strides(const std::array<int, N> & shape)
{
    std::array<int, N> strides;
    int stride = 1;
    for (size_t dim = N; dim-- > 0;)
    {
        strides[dim] = stride;
        stride *= shape[dim];
    }
    return strides;
}

template <size_t N>
std::array<int, N> coord_at(int index, const std::array<int, N> & shape)
{
    std::array<int, N> coord;
    for (size_t dim = N; dim-- > 0;)
    {
        coord[dim] = index % shape[dim];
        index /= shape[dim];
    }
    return coord;
}

template <size_t N>
void next_coord(std::array<int, N> & coord, const std::array<int, N> & shape)
{
    for (size_t dim = N; dim-- > 0;)
    {
        if (++coord[dim] < shape[dim]) return;
        coord[dim] = 0;
    }
}

template <size_t N>
bool is_inbound_shift(const std::array<int, N> & coord, const LabelShift<N> & shift)
{
    for (size_t dim = 0; dim < N; ++dim)
    {
        if (coord[dim] < shift.lower[dim] || coord[dim] >= shift.upper[dim]) return false;
    }
    return true;
}

template <size_t N>
std::vector<LabelShift<N>> label_shifts(const Structure & structure, const std::array<int, N> & shape,
                                        const std::array<int, N> & strides)
{
    std::vector<LabelShift<N>> shifts;
    shifts.reserve(structure.size() - 1);

    for (const auto & shift : structure.shifts())
    {
        LabelShift<N> result;
        for (size_t dim = 0; dim < N; ++dim)
        {
            result.delta[dim] = static_cast<int>(shift[dim]);
            result.offset += result.delta[dim] * strides[dim];
            result.lower[dim] = std::max(0, -result.delta[dim]);
            result.upper[dim] = shape[dim] - std::max(0, result.delta[dim]);
        }

        if (result.offset < 0) shifts.emplace_back(result);
    }
    return shifts;
}

template <size_t N>
Bounds<N> interior_bounds(const std::vector<LabelShift<N>> & shifts, const std::array<int, N> & shape)
{
    Bounds<N> bounds;
    for (size_t dim = 0; dim < N; ++dim)
    {
        bounds.lower[dim] = 0;
        bounds.upper[dim] = shape[dim];
    }

    for (const auto & shift : shifts)
    {
        for (size_t dim = 0; dim < N; ++dim)
        {
            bounds.lower[dim] = std::max(bounds.lower[dim], shift.lower[dim]);
            bounds.upper[dim] = std::min(bounds.upper[dim], shift.upper[dim]);
        }
    }
    return bounds;
}

int find_root(array<int> & out, int index)
{
    int root = index;
    while (out[root] != root) root = out[root];

    while (out[index] != index)
    {
        int next = out[index];
        out[index] = root;
        index = next;
    }
    return root;
}

int root_of(const array<int> & out, int index)
{
    int root = index;
    while (out[root] != root) root = out[root];
    return root;
}

int merge_roots(array<int> & out, int lhs, int rhs)
{
    lhs = find_root(out, lhs);
    rhs = find_root(out, rhs);
    if (lhs == rhs) return lhs;

    if (lhs < rhs)
    {
        out[rhs] = lhs;
        return lhs;
    }

    out[lhs] = rhs;
    return rhs;
}

template <typename I, size_t N>
LabelResult label_impl(py::array_t<I> input, Structure structure, size_t npts, unsigned threads)
{
    array<I> inp {input.request()};
    py::array_t<int> labels {std::vector<py::ssize_t>(input.shape(), input.shape() + input.ndim())};
    array<int> out {labels.request()};

    if (inp.size() == 0)
    {
        return std::make_tuple(std::move(labels), py::array_t<int>{std::vector<py::ssize_t>{0}});
    }

    threads = std::max<unsigned>(1, std::min<unsigned>(threads, inp.size()));
    std::vector<std::vector<std::pair<int, int>>> boundary_edges (threads);
    auto shape = int_shape<N>(inp);
    auto strides = int_strides<N>(shape);
    auto shifts = label_shifts<N>(structure, shape, strides);
    auto interior = interior_bounds<N>(shifts, shape);

    thread_exception e;

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        int thread_id = omp_get_thread_num();
        int chunk = static_cast<int>((inp.size() + threads - 1) / threads);

        int thread_start = thread_id * chunk;
        int thread_end = std::min<int>((thread_id + 1) * chunk, static_cast<int>(inp.size()));

        for (int index = thread_start; index < thread_end; ++index)
        {
            out[index] = inp[index] ? index : -1;
        }

        #pragma omp barrier

        auto coord = coord_at<N>(thread_start, shape);
        for (int index = thread_start; index < thread_end; ++index)
        {
            if (out[index] >= 0)
            {
                int root = index;
                bool is_interior = interior.contains(coord);
                for (const auto & shift : shifts)
                {
                    if (!is_interior && !is_inbound_shift(coord, shift)) continue;

                    int neighbour = index + shift.offset;
                    if constexpr (std::is_same_v<I, bool>)
                    {
                        if (out[neighbour] < 0) continue;
                    }
                    else
                    {
                        if (out[neighbour] < 0 || inp[neighbour] != inp[index]) continue;
                    }

                    if (neighbour >= thread_start && neighbour < thread_end)
                    {
                        root = merge_roots(out, root, neighbour);
                    }
                    else
                    {
                        boundary_edges[thread_id].emplace_back(index, neighbour);
                    }
                }
            }
            next_coord(coord, shape);
        }
    }

    for (auto & edges : boundary_edges)
    {
        for (auto [lhs, rhs] : edges) merge_roots(out, lhs, rhs);
    }

    #pragma omp parallel for num_threads(threads)
    for (int index = 0; index < static_cast<int>(out.size()); ++index)
    {
        if (out[index] >= 0) out[index] = root_of(out, index);
    }

    int n_labels = 0;
    if (npts <= 1)
    {
        for (int index = 0; index < static_cast<int>(out.size()); ++index)
        {
            if (out[index] == index) out[index] = -(++n_labels + 1);
        }
    }
    else
    {
        std::vector<int> roots;
        for (int index = 0; index < static_cast<int>(out.size()); ++index)
        {
            if (out[index] == index)
            {
                out[index] = -static_cast<int>(roots.size()) - 2;
                roots.push_back(index);
            }
        }

        std::vector<size_t> label_sizes (roots.size(), 0);
        for (int index = 0; index < static_cast<int>(out.size()); ++index)
        {
            int value = out[index];
            if (value == -1) continue;

            int slot = (value < -1) ? -value - 2 : -out[value] - 2;
            label_sizes[slot]++;
        }

        for (size_t slot = 0; slot < roots.size(); ++slot)
        {
            int root = roots[slot];
            out[root] = (label_sizes[slot] >= npts) ? -(++n_labels + 1) : -1;
        }
    }

    if (threads == 1)
    {
        for (int index = static_cast<int>(out.size()); index-- > 0;)
        {
            int value = out[index];
            if (value == -1)
            {
                out[index] = 0;
            }
            else if (value < -1)
            {
                out[index] = -value - 1;
            }
            else
            {
                value = out[value];
                out[index] = (value < -1) ? -value - 1 : 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(threads)
        for (int index = 0; index < static_cast<int>(out.size()); ++index)
        {
            int value = out[index];
            if (value >= 0)
            {
                value = out[value];
                out[index] = (value < -1) ? -value - 1 : 0;
            }
        }

        #pragma omp parallel for num_threads(threads)
        for (int index = 0; index < static_cast<int>(out.size()); ++index)
        {
            int value = out[index];
            if (value == -1) out[index] = 0;
            else if (value < -1) out[index] = -value - 1;
        }
    }

    py::gil_scoped_acquire acquire;

    e.rethrow();

    py::array_t<int> index {std::vector<py::ssize_t>{n_labels}};
    array<int> iarr {index.request()};
    for (int i = 0; i < n_labels; ++i) iarr[i] = i + 1;

    return std::make_tuple(std::move(labels), std::move(index));
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

    if (inp.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("input array is too large for int32 label indices");
    }

    switch (input.ndim())
    {
        case 2: return label_impl<I, 2>(std::move(input), std::move(structure), npts, threads);
        case 3: return label_impl<I, 3>(std::move(input), std::move(structure), npts, threads);
        case 4: return label_impl<I, 4>(std::move(input), std::move(structure), npts, threads);
        case 5: return label_impl<I, 5>(std::move(input), std::move(structure), npts, threads);
        case 6: return label_impl<I, 6>(std::move(input), std::move(structure), npts, threads);
        case 7: return label_impl<I, 7>(std::move(input), std::move(structure), npts, threads);
        default:
            throw std::invalid_argument("Unsupported number of dimensions: " + std::to_string(input.ndim()));
    }
}

struct LabelLookup
{
    py::ssize_t max_label = 0;
    std::vector<int> slot;
};

LabelLookup make_label_to_slot(const array<int> & index)
{
    LabelLookup lookup;
    for (auto label_id : index) if (label_id > lookup.max_label) lookup.max_label = label_id;

    lookup.slot.assign(lookup.max_label + 1, -1);
    for (size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] > 0) lookup.slot[index[i]] = static_cast<int>(i);
    }
    return lookup;
}

template <typename T, size_t N>
std::vector<MomentsND<T, N>> moments_from_labels(const LabelResult & labels, py::array_t<T> data, unsigned threads)
{
    array<int> larr {std::get<0>(labels).request()};
    array<int> iarr {std::get<1>(labels).request()};
    array<T> darr {data.request()};

    check_equal("labels and data must have the same shape",
                larr.shape().begin(), larr.shape().end(),
                darr.shape().begin(), darr.shape().end());

    auto label_lookup = make_label_to_slot(iarr);

    std::vector<int> first_index (iarr.size(), -1);
    for (size_t i = 0; i < larr.size(); ++i)
    {
        auto label_id = larr[i];
        if (label_id <= 0 || label_id > label_lookup.max_label) continue;

        auto slot = label_lookup.slot[label_id];
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
            if (label_id <= 0 || label_id > label_lookup.max_label) continue;

            auto slot = label_lookup.slot[label_id];
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
py::array_t<int> maximum_position_nd(const LabelResult & labels, py::array_t<T> data, unsigned threads)
{
    array<int> labels_array {std::get<0>(labels).request()};
    array<int> index_array {std::get<1>(labels).request()};
    array<T> darr {data.request()};

    check_equal("labels and data must have the same shape",
                labels_array.shape().begin(), labels_array.shape().end(),
                darr.shape().begin(), darr.shape().end());

    auto label_lookup = make_label_to_slot(index_array);

    std::vector<int> best_index (index_array.size(), 0);
    std::vector<unsigned char> found (index_array.size(), 0);
    std::vector<T> best_value (index_array.size(), T());

    threads = std::max(1u, threads);

    py::gil_scoped_release release;

    #pragma omp parallel num_threads(threads)
    {
        std::vector<int> local_best_index (index_array.size(), 0);
        std::vector<unsigned char> local_found (index_array.size(), 0);
        std::vector<T> local_best_value (index_array.size(), T());

        #pragma omp for
        for (long i = 0; i < static_cast<long>(labels_array.size()); ++i)
        {
            auto label_id = labels_array[i];
            if (label_id <= 0 || label_id > label_lookup.max_label) continue;

            auto slot = label_lookup.slot[label_id];
            if (slot < 0) continue;

            auto value = darr[i];
            if (!local_found[slot] || value > local_best_value[slot])
            {
                local_found[slot] = 1;
                local_best_value[slot] = value;
                local_best_index[slot] = static_cast<int>(i);
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < index_array.size(); ++i)
            {
                if (!local_found[i]) continue;

                if (!found[i] || local_best_value[i] > best_value[i] ||
                    (local_best_value[i] == best_value[i] && local_best_index[i] < best_index[i]))
                {
                    found[i] = 1;
                    best_value[i] = local_best_value[i];
                    best_index[i] = local_best_index[i];
                }
            }
        }
    }

    py::gil_scoped_acquire acquire;

    py::array_t<int> result (std::vector<py::ssize_t>{py::ssize_t(index_array.size()),
                                                      py::ssize_t(data.ndim())});
    array<int> out {result.request()};
    for (size_t i = 0; i < index_array.size(); ++i)
    {
        auto point = make_point<N>(best_index[i], darr.shape());
        for (size_t dim = 0; dim < N; ++dim) out[i * N + dim] = static_cast<int>(point[N - dim - 1]);
    }
    return result;
}

template <typename T>
py::array_t<int> maximum_position(const LabelResult & labels, py::array_t<T> data, unsigned threads)
{
    switch (data.ndim())
    {
        case 2: return maximum_position_nd<T, 2>(labels, data, threads);
        case 3: return maximum_position_nd<T, 3>(labels, data, threads);
        case 4: return maximum_position_nd<T, 4>(labels, data, threads);
        case 5: return maximum_position_nd<T, 5>(labels, data, threads);
        case 6: return maximum_position_nd<T, 6>(labels, data, threads);
        case 7: return maximum_position_nd<T, 7>(labels, data, threads);
        default:
            throw std::invalid_argument("Unsupported number of dimensions: " + std::to_string(data.ndim()));
    }
}

template <typename T, size_t N>
py::array_t<T> p_values_nd(const LabelResult & labels, py::array_t<T> larray, py::array_t<T> data, T p0, T vmin, T xtol,
                           unsigned threads)
{
    array<int> labels_array {std::get<0>(labels).request()};
    array<int> index_array {std::get<1>(labels).request()};

    py::array_t<T> result (std::vector<py::ssize_t>{py::ssize_t(index_array.size())});
    array<T> out {result.request()};
    array<T> lines {larray.request()};
    array<T> darr {data.request()};

    check_equal("labels and data must have the same shape",
                labels_array.shape().begin(), labels_array.shape().end(),
                darr.shape().begin(), darr.shape().end());

    auto label_lookup = make_label_to_slot(index_array);

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
            if (label_id <= 0 || label_id > label_lookup.max_label) continue;

            auto slot = label_lookup.slot[label_id];
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

    m.def("maximum_position", &maximum_position<double>, py::arg("labels"), py::arg("data"), py::arg("num_threads") = 1);
    m.def("maximum_position", &maximum_position<float>, py::arg("labels"), py::arg("data"), py::arg("num_threads") = 1);
    m.def("maximum_position", &maximum_position<int>, py::arg("labels"), py::arg("data"), py::arg("num_threads") = 1);
    m.def("maximum_position", &maximum_position<py::ssize_t>, py::arg("labels"), py::arg("data"), py::arg("num_threads") = 1);

    m.def("p_values", &p_values<double>, py::arg("labels"), py::arg("lines"), py::arg("data"), py::arg("p0"),
          py::arg("vmin"), py::arg("xtol"), py::arg("num_threads") = 1);
    m.def("p_values", &p_values<float>, py::arg("labels"), py::arg("lines"), py::arg("data"), py::arg("p0"),
          py::arg("vmin"), py::arg("xtol"), py::arg("num_threads") = 1);
}
