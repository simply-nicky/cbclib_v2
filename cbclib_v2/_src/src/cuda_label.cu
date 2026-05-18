#include <cub/cub.cuh>
#include "label.hpp"
#include "cupy_array.hpp"

namespace cbclib::cuda {

namespace detail {

// Return log(binomial_tail(n, k, p))
// binomial_tail(n, k, p) = sum_{i = k}^n bincoef(n, i) * p^i * (1 - p)^{n - i}
// bincoef(n, k) = gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

template <typename T>
HOST_DEVICE T logaddexp(T a, T b)
{
    const T neg_inf = -numeric_limits<T>::infinity();

    if (a == neg_inf) return b;
    if (b == neg_inf) return a;

    T m = (a > b) ? a : b;
    return m + math_traits<T>::log(math_traits<T>::exp(a - m) + math_traits<T>::exp(b - m));
};

template <typename I, typename T>
HOST_DEVICE T logbinom(I n, I k, T p)
{
    const T neg_inf = -numeric_limits<T>::infinity();

    if (k <= 0) return T(0.0);
    if (k > n) return neg_inf;

    if (p <= T(0.0)) return (k <= 0) ? T(0.0) : neg_inf;
    if (p >= T(1.0)) return (k <= n) ? T(0.0) : neg_inf;

    T log_p = math_traits<T>::log(p);
    T log_q = math_traits<T>::log1p(-p);

    T log_term = math_traits<T>::lgamma(n + 1) - math_traits<T>::lgamma(k + 1) - math_traits<T>::lgamma(n - k + 1) +
                 k * log_p + (n - k) * log_q;
    T log_tail = log_term;

    for (I i = k + 1; i < n + 1; ++i)
    {
        log_term += math_traits<T>::log(n - i + 1) - math_traits<T>::log(i) + log_p - log_q;
        log_tail = logaddexp(log_tail, log_term);
    }

    return log_tail;
}

} // namespace detail

DeviceVector<py::ssize_t> negative_shifts(const Structure & structure)
{
    std::vector<py::ssize_t> shifts;
    for (auto shift : structure)
    {
        py::ssize_t offset = 0;
        for (csize_t n = 0; n < structure.rank(); n++)
        {
            offset = offset * structure.shape(n) + shift[n];
        }
        if (offset < 0) shifts.insert(shifts.end(), shift.begin(), shift.end());
    }
    return DeviceVector<py::ssize_t>::from_host(shifts.data(), shifts.size());
}

DeviceVector<py::ssize_t> all_shifts(const Structure & structure)
{
    std::vector<py::ssize_t> shifts;
    for (auto shift : structure)
    {
        py::ssize_t offset = 0;
        for (csize_t n = 0; n < structure.rank(); n++)
        {
            offset = offset * structure.shape(n) + shift[n];
        }
        if (offset != 0) shifts.insert(shifts.end(), shift.begin(), shift.end());
    }
    return DeviceVector<py::ssize_t>::from_host(shifts.data(), shifts.size());
}

// CUDA kernel : initialize output array
template <typename I, csize_t N>
__global__ void init_kernel(ArrayViewND<int, N> out, ArrayViewND<I, N> inp)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (inp[idx]) out[idx] = idx;
    else out[idx] = -1;
}

// CUDA kernel : connect neighbouring pixels
template <typename I, csize_t N>
__global__ void connect_kernel(ArrayViewND<int, N> out, ArrayViewND<I, N> inp, DeviceRange<py::ssize_t> shifts, csize_t n_shifts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx] < 0) return;

    // k is a shifted index and j is an original index
    int j = idx;

    for (csize_t i = 0; i < n_shifts; i++)
    {
        int k = detail::shift_index(idx, out.shape(), N, shifts.data(i * N), N);

        if (k < 0 || out[k] < 0 || inp[k] != inp[idx]) continue;

        // Union-Find: connect the two components k and j
        while (true)
        {
            // Traversing up the tree to find the root of j and k
            // The root points to itself, i.e., out[p] == p
            while (j != out[j]) j = out[j];
            while (k != out[k]) k = out[k];

            // If they are already in the same component, break
            if (j == k) break;

            // Union by rank: attach the smaller tree to the root of the larger tree
            if (j < k)
            {
                int old = atomicCAS(&out[k], k, j); // Attempt to set out[k] to j if it is currently k
                if (old == k) break; // Successfully connected k to j
                k = old; // Another thread has connected k to a different root, update k to the new root and try again
            }
            else
            {
                int old = atomicCAS(&out[j], j, k); // Attempt to set out[j] to k if it is currently j
                if (old == j) break; // Successfully connected j to k
                j = old; // Another thread has connected j to a different root, update j to the new root and try again
            }
        }
    }
}

// CUDA kernel : flatten the union-find trees and count the number of labels
template <csize_t N>
__global__ void flatten_kernel(ArrayViewND<int, N> out, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx] < 0) return;

    // Flatten the tree by making every node point directly to the root
    int root = idx;
    while (root != out[root]) root = out[root];

    if (root != idx) out[idx] = root;
    else atomicAdd(&counts[0], 1);
}

// CUDA kernel : yield the label roots and store them in the labels array
template <csize_t N>
__global__ void label_kernel(ArrayViewND<int, N> out, DeviceRange<csize_t> labels, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx] != idx) return; // Only process root nodes
    csize_t label_idx = atomicAdd(&counts[1], 1); // Get a unique index for this label
    labels[label_idx] = idx; // Store the root index of the label
}

// CUDA kernel : relabel the output array to have consecutive labels starting from 1 and background pixels remain 0
template <csize_t N>
__global__ void finalise_kernel(ArrayViewND<int, N> out, DeviceRange<csize_t> labels)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx] < 0) { out[idx] = 0; return; } // Background pixels remain 0;

    // Find the index of the root in the labels array
    int root = out[idx];
    csize_t left = 0;
    csize_t right = labels.size();
    csize_t mid = (left + right) / 2;
    while (left < right)
    {
        if (labels[mid] == root) break;
        if (labels[mid] > root) right = mid - 1;
        else left = mid + 1;
        mid = (left + right) / 2;
    }

    out[idx] = mid + 1; // Labels start from 1
}

// CUDA kernel : count the size of each label (labels are 1..n_labels)
template <csize_t N>
__global__ void count_kernel(ArrayViewND<int, N> out, DeviceRange<csize_t> label_sizes)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    int label = out[idx];
    if (label <= 0) return;

    atomicAdd(&label_sizes[label - 1], 1);
}

// CUDA kernel : apply label remapping (0 means discard)
template <csize_t N>
__global__ void remap_kernel(ArrayViewND<int, N> out, DeviceRange<csize_t> label_map)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    int label = out[idx];
    if (label <= 0) return;

    out[idx] = static_cast<int>(label_map[label - 1]);
}

DeviceVector<csize_t> sort_labels(const DeviceVector<csize_t> & labels, csize_t n_labels)
{
    DeviceVector<csize_t> sorted_labels(n_labels);

    // Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, labels.data(), sorted_labels.data(), n_labels);

    // Allocate temporary storage
    DeviceVector<char> temp_storage(temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(temp_storage.data(), temp_storage_bytes, labels.data(), sorted_labels.data(), n_labels);
    handle_cuda_error(cudaGetLastError());

    return sorted_labels;
}

// CUDA kernel : mark labels to keep (1) or discard (0)
__global__ void keep_kernel(DeviceRange<const csize_t> label_sizes, DeviceRange<csize_t> keep_flags, csize_t npts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= label_sizes.size()) return;

    keep_flags[idx] = (label_sizes[idx] >= npts) ? 1 : 0;
}

// CUDA kernel : build label remap (0 means discard, else compact id starting at 1) and count the number of labels to keep
__global__ void build_map_kernel(DeviceRange<csize_t> keep_flags, DeviceRange<csize_t> prefix,
                                 DeviceRange<csize_t> label_map, DeviceRange<csize_t> counts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= label_map.size()) return;

    label_map[idx] = keep_flags[idx] ? (prefix[idx] + 1) : 0;

    if (keep_flags[idx]) atomicAdd(&counts[0], 1);
}

DeviceVector<csize_t> filter_labels(const DeviceVector<csize_t> & label_sizes, DeviceVector<csize_t> & counts, csize_t & n_labels, py::ssize_t npts)
{
    // Allocate device memory for keep flags and label map
    DeviceVector<csize_t> keep_flags (n_labels);
    DeviceVector<csize_t> label_map (n_labels);

    // Launch kernel to mark labels to keep (1) or discard (0)
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_labels + block_size - 1) / block_size;
    keep_kernel<<<n_blocks, block_size>>>(label_sizes.view(), keep_flags.view(), npts);
    handle_cuda_error(cudaGetLastError());

    // Compute prefix sum (exclusive scan) of keep flags to build the label map
    DeviceVector<csize_t> prefix(n_labels);

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, keep_flags.data(), prefix.data(), n_labels);

    DeviceVector<char> temp_storage (temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(temp_storage.data(), temp_storage_bytes, keep_flags.data(), prefix.data(), n_labels);
    handle_cuda_error(cudaGetLastError());

    // Launch kernel to build label remap (0 means discard, else compact id starting at 1)
    counts.fill(0); // counts[0] will hold the number of labels to keep
    build_map_kernel<<<n_blocks, block_size>>>(keep_flags.view(), prefix.view(), label_map.view(), counts.view());
    handle_cuda_error(cudaGetLastError());

    // Copy the number of labels to keep back to host
    handle_cuda_error(cudaMemcpy(&n_labels, counts.data(), sizeof(csize_t), cudaMemcpyDeviceToHost));

    return label_map;
}

template <typename I, csize_t N>
std::tuple<array_t<int>, py::ssize_t> label_nd(array_t<int> output, array_t<I> input, Structure structure, py::ssize_t npts)
{
    DeviceVector<py::ssize_t> shifts = negative_shifts(structure);

    auto output_view = cast_to_nd<int, N>(output.view());
    auto input_view = cast_to_nd<I, N>(input.view());

    // Launch kernel to initialize output array
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (output.size() + block_size - 1) / block_size;
    init_kernel<<<n_blocks, block_size>>>(output_view, input_view);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to connect neighbouring pixels
    connect_kernel<<<n_blocks, block_size>>>(output_view, input_view, shifts.view(), shifts.size() / N);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to flatten the union-find trees and count the number of labels
    DeviceVector<csize_t> counts (2); // counts[0] is the number of labels, counts[1] is the temporary counter
    counts.fill(0);

    flatten_kernel<<<n_blocks, block_size>>>(output_view, counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to yield the label roots and store them in the labels array
    csize_t n_labels = 0;
    handle_cuda_error(cudaMemcpy(&n_labels, counts.data(), sizeof(csize_t), cudaMemcpyDeviceToHost));

    DeviceVector<csize_t> labels (n_labels);

    label_kernel<<<n_blocks, block_size>>>(output_view, labels.view(), counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Sort the labels array to enable binary search during relabeling
    DeviceVector<csize_t> sorted_labels = sort_labels(labels, n_labels);

    // Launch kernel to relabel the output array to have consecutive labels starting from 0
    finalise_kernel<<<n_blocks, block_size>>>(output_view, sorted_labels.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    if (npts <= 1 || n_labels == 0) return std::make_tuple(output, py::ssize_t(n_labels)); // No need to filter labels if npts <= 1 or no labels found

    // Count label sizes and discard small labels
    DeviceVector<csize_t> label_sizes (n_labels);
    label_sizes.fill(0);
    count_kernel<<<n_blocks, block_size>>>(output_view, label_sizes.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    auto label_map = filter_labels(label_sizes, counts, n_labels, npts);

    // Launch kernel to apply label remapping
    remap_kernel<<<n_blocks, block_size>>>(output_view, label_map.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return std::make_tuple(output, py::ssize_t(n_labels));
}

template <typename I>
std::tuple<array_t<int>, py::ssize_t> label(array_t<int> output, array_t<I> input, Structure structure, py::ssize_t npts)
{
    if (input.ndim() != structure.rank())
    {
        throw std::invalid_argument("input array dimension (" + std::to_string(input.ndim()) +
                                    ") does not match structure rank (" + std::to_string(structure.rank()) + ")");
    }

    switch (structure.rank())
    {
        case 2: return label_nd<I, 2>(output, input, structure, npts);
        case 3: return label_nd<I, 3>(output, input, structure, npts);
        case 4: return label_nd<I, 4>(output, input, structure, npts);
        case 5: return label_nd<I, 5>(output, input, structure, npts);
        case 6: return label_nd<I, 6>(output, input, structure, npts);
        case 7: return label_nd<I, 7>(output, input, structure, npts);
        default: throw std::runtime_error("Unsupported number of dimensions");
    }
}

// Pass 1: build label -> slot map; one thread per slot in index
__global__ void index_map_kernel(DeviceRange<int> label_to_slot, ArrayViewND<py::ssize_t, 1> index)
{
    csize_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= index.size()) return;

    csize_t label = index[j];
    if (label > 0 && label < label_to_slot.size()) label_to_slot[label] = static_cast<int>(j);
}

template <typename T, csize_t N>
struct FirstMoment
{
    int origin = -1;
    T mu = T();
    PointND<T, N> mu_x {};

    // Needed for DeviceVector initialisation
    HOST_DEVICE friend bool operator==(const FirstMoment & lhs, const FirstMoment & rhs)
    {
        if (lhs.origin != rhs.origin || lhs.mu != rhs.mu) return false;
        for (csize_t d = 0; d < N; ++d)
        {
            if (lhs.mu_x[d] != rhs.mu_x[d]) return false;
        }
        return true;
    }

    HOST_DEVICE friend bool operator!=(const FirstMoment & lhs, const FirstMoment & rhs)
    {
        return !(lhs == rhs);
    }
};

// Pass 2: scatter-add using prebuilt label -> slot map; one thread per pixel
template <typename T, csize_t N>
__global__ void first_moment_kernel(DeviceRange<FirstMoment<T, N>> out, ArrayViewND<int, N> labels,
                                    DeviceRange<int> label_to_slot, ArrayViewND<T, N> data)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= labels.size()) return;

    int label = labels[idx];
    if (label <= 0) return;

    int j = label_to_slot[static_cast<csize_t>(label)];
    if (j < 0) return;

    PointND<csize_t, N> point = make_point<N, N>(idx, labels.shape());
    T value = data[idx];

    atomicCAS(&out[j].origin, -1, idx); // Set origin to the first point of the label (arbitrary but deterministic)

    if (value <= T()) return; // Skip non-positive values

    atomicAdd(&out[j].mu, value);
    for (csize_t d = 0; d < N; ++d)
    {
        atomicAdd(&out[j].mu_x[d], value * point[d]);
    }
}

template <typename T, csize_t N>
__global__ void center_of_mass_kernel(ArrayViewND<T, 2> out, DeviceRange<FirstMoment<T, N>> moments, ArrayViewND<int, N> labels)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= moments.size()) return;

    if (moments[idx].origin < 0) return; // Skip empty labels

    T mu = moments[idx].mu;

    if (mu == T())
    {
        PointND<csize_t, N> origin = make_point<N, N>(moments[idx].origin, labels.shape());

        // Output an origin point
        for (csize_t d = 0; d < N; ++d)
        {
            out.at(idx, N - d - 1) = origin[d];
        }
    }
    else
    {
        for (csize_t d = 0; d < N; ++d)
        {
            out.at(idx, N - d - 1) = moments[idx].mu_x[d] / mu;
        }
    }
}

template <typename T, csize_t N>
array_t<T> center_of_mass_nd(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> data)
{
    auto out_view = cast_to_nd<T, 2>(out.view());
    auto labels_view = cast_to_nd<int, N>(labels.view());
    auto index_view = cast_to_nd<py::ssize_t, 1>(index.view());
    auto data_view = cast_to_nd<T, N>(data.view());

    csize_t block_size = BLOCK_SIZE;

    // Find max label value to size the lookup table
    DeviceVector<int> max_label_d (1, 0);
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    DeviceVector<char> temp_storage(temp_bytes);
    cub::DeviceReduce::Max(temp_storage.data(), temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    handle_cuda_error(cudaGetLastError());

    int max_label = 0;
    handle_cuda_error(cudaMemcpy(&max_label, max_label_d.data(), sizeof(int), cudaMemcpyDeviceToHost));

    if (max_label <= 0) return out;

    // Pass 1: build label -> slot map (sentinel -1 means label not in index)
    DeviceVector<int> label_to_slot (static_cast<csize_t>(max_label) + 1, -1);

    csize_t n_blocks = (index_view.size() + block_size - 1) / block_size;
    index_map_kernel<<<n_blocks, block_size>>>(label_to_slot.view(), index_view);
    handle_cuda_error(cudaGetLastError());

    // Pass 2: scatter-add using label -> slot map
    DeviceVector<FirstMoment<T, N>> moments (index_view.size(), FirstMoment<T, N>{});
    n_blocks = (labels_view.size() + block_size - 1) / block_size;
    first_moment_kernel<T, N><<<n_blocks, block_size>>>(moments.view(), labels_view, label_to_slot.view(), data_view);
    handle_cuda_error(cudaGetLastError());

    // Pass 3: compute center of mass from moments
    n_blocks = (index_view.size() + block_size - 1) / block_size;
    center_of_mass_kernel<T, N><<<n_blocks, block_size>>>(out_view, moments.view(), labels_view);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T>
array_t<T> center_of_mass(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> data)
{
    if (labels.ndim() != data.ndim())
    {
        throw std::invalid_argument("labels array dimension (" + std::to_string(labels.ndim()) +
                                    ") does not match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (out.ndim() != 2 || out.shape(0) != index.size() || out.shape(1) != labels.ndim())
    {
        throw std::invalid_argument("output array shape (" + std::to_string(out.shape(0)) + ", " + std::to_string(out.shape(1)) +
                                    ") does not match expected shape (" + std::to_string(index.size()) + ", " + std::to_string(labels.ndim()) + ")");
    }

    switch (labels.ndim())
    {
        case 2: return center_of_mass_nd<T, 2>(out, labels, index, data);
        case 3: return center_of_mass_nd<T, 3>(out, labels, index, data);
        case 4: return center_of_mass_nd<T, 4>(out, labels, index, data);
        case 5: return center_of_mass_nd<T, 5>(out, labels, index, data);
        case 6: return center_of_mass_nd<T, 6>(out, labels, index, data);
        case 7: return center_of_mass_nd<T, 7>(out, labels, index, data);
        default: throw std::runtime_error("Unsupported number of dimensions: labels.ndim = " + std::to_string(labels.ndim()));
    }
}

template <typename T, csize_t N>
struct SecondMoment
{
    T mu = T();
    PointND<T, N> mu_x {};
    PointND<T, N * N> mu_xx {};

    // Needed for DeviceVector initialisation
    HOST_DEVICE friend bool operator==(const SecondMoment& lhs, const SecondMoment& rhs)
    {
        if (lhs.mu != rhs.mu) return false;
        for (csize_t d = 0; d < N; ++d)
        {
            if (lhs.mu_x[d] != rhs.mu_x[d]) return false;
        }
        for (csize_t i = 0; i < N * N; ++i)
        {
            if (lhs.mu_xx[i] != rhs.mu_xx[i]) return false;
        }
        return true;
    }

    HOST_DEVICE friend bool operator!=(const SecondMoment& lhs, const SecondMoment& rhs)
    {
        return !(lhs == rhs);
    }
};

template <typename T, csize_t N>
__global__ void second_moment_kernel(DeviceRange<SecondMoment<T, N>> out, ArrayViewND<int, N> labels,
                                     DeviceRange<int> label_to_slot, ArrayViewND<T, N> data)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= labels.size()) return;

    int label = labels[idx];
    if (label <= 0) return;

    int j = label_to_slot[static_cast<csize_t>(label)];
    if (j < 0) return;

    PointND<csize_t, N> point = make_point<N, N>(idx, labels.shape());
    T value = data[idx];

    if (value <= T()) return; // Skip non-positive values

    atomicAdd(&out[j].mu, value);
    for (csize_t d1 = 0; d1 < N; ++d1)
    {
        atomicAdd(&out[j].mu_x[d1], value * point[d1]);
        for (csize_t d2 = 0; d2 < N; ++d2)
        {
            atomicAdd(&out[j].mu_xx[d1 * N + d2], value * point[d1] * point[d2]);
        }
    }
};

template <typename T, csize_t N>
__global__ void covariance_kernel(ArrayViewND<T, 3> out, DeviceRange<SecondMoment<T, N>> moments, ArrayViewND<int, N> labels)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= moments.size()) return;

    T mu = moments[idx].mu;

    if (mu == T())
    {
        // Output a zero covariance matrix for empty labels
        for (csize_t d1 = 0; d1 < N; ++d1)
        {
            for (csize_t d2 = 0; d2 < N; ++d2)
            {
                out.at(idx, N - d1 - 1, N - d2 - 1) = T();
            }
        }
        return;
    }
    else
    {
        for (csize_t d1 = 0; d1 < N; ++d1)
        {
            for (csize_t d2 = 0; d2 < N; ++d2)
            {
                out.at(idx, N - d1 - 1, N - d2 - 1) = moments[idx].mu_xx[d1 * N + d2] / mu - (moments[idx].mu_x[d1] / mu) * (moments[idx].mu_x[d2] / mu);
            }
        }
    }
};

template <typename T, csize_t N>
array_t<T> covariance_matrix_nd(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> data)
{
    auto out_view = cast_to_nd<T, 3>(out.view());
    auto labels_view = cast_to_nd<int, N>(labels.view());
    auto index_view = cast_to_nd<py::ssize_t, 1>(index.view());
    auto data_view = cast_to_nd<T, N>(data.view());

    csize_t block_size = BLOCK_SIZE;

    // Find max label value to size the lookup table
    DeviceVector<int> max_label_d (1, 0);
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    DeviceVector<char> temp_storage(temp_bytes);
    cub::DeviceReduce::Max(temp_storage.data(), temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    handle_cuda_error(cudaGetLastError());

    int max_label = 0;
    handle_cuda_error(cudaMemcpy(&max_label, max_label_d.data(), sizeof(int), cudaMemcpyDeviceToHost));

    if (max_label <= 0) return out;

    // Pass 1: build label -> slot map (sentinel -1 means label not in index)
    DeviceVector<int> label_to_slot (static_cast<csize_t>(max_label) + 1, -1);

    csize_t n_blocks = (index_view.size() + block_size - 1) / block_size;
    index_map_kernel<<<n_blocks, block_size>>>(label_to_slot.view(), index_view);
    handle_cuda_error(cudaGetLastError());

    // Pass 2: scatter-add using label -> slot map
    DeviceVector<SecondMoment<T, N>> moments (index_view.size(), SecondMoment<T, N>{});
    n_blocks = (labels_view.size() + block_size - 1) / block_size;
    second_moment_kernel<T, N><<<n_blocks, block_size>>>(moments.view(), labels_view, label_to_slot.view(), data_view);
    handle_cuda_error(cudaGetLastError());

    // Pass 3: compute covariance from moments
    n_blocks = (index_view.size() + block_size - 1) / block_size;
    covariance_kernel<T, N><<<n_blocks, block_size>>>(out_view, moments.view(), labels_view);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T>
array_t<T> covariance_matrix(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> data)
{
    if (labels.ndim() != data.ndim())
    {
        throw std::invalid_argument("labels array dimension (" + std::to_string(labels.ndim()) +
                                    ") does not match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (out.ndim() != 3 || out.shape(0) != index.size() || out.shape(1) != labels.ndim() || out.shape(2) != labels.ndim())
    {
        throw std::invalid_argument("output array shape (" + std::to_string(out.shape(0)) + ", " + std::to_string(out.shape(1)) + ", " + std::to_string(out.shape(2)) +
                                    ") does not match expected shape (" + std::to_string(index.size()) + ", " + std::to_string(labels.ndim()) + ", " + std::to_string(labels.ndim()) + ")");
    }

    switch (labels.ndim())
    {
        case 2: return covariance_matrix_nd<T, 2>(out, labels, index, data);
        case 3: return covariance_matrix_nd<T, 3>(out, labels, index, data);
        case 4: return covariance_matrix_nd<T, 4>(out, labels, index, data);
        case 5: return covariance_matrix_nd<T, 5>(out, labels, index, data);
        case 6: return covariance_matrix_nd<T, 6>(out, labels, index, data);
        case 7: return covariance_matrix_nd<T, 7>(out, labels, index, data);
        default: throw std::runtime_error("Unsupported number of dimensions: labels.ndim = " + std::to_string(labels.ndim()));
    }
}

struct StreakCounts
{
    csize_t n_signal = 0;
    csize_t n_total = 0;

    // Needed for DeviceVector initialisation
    HOST_DEVICE friend bool operator==(const StreakCounts & lhs, const StreakCounts & rhs)
    {
        return lhs.n_signal == rhs.n_signal && lhs.n_total == rhs.n_total;
    }

    HOST_DEVICE friend bool operator!=(const StreakCounts & lhs, const StreakCounts & rhs)
    {
        return !(lhs == rhs);
    }
};

template <typename T, csize_t N>
__global__ void count_kernel(DeviceRange<StreakCounts> counts, ArrayViewND<int, N> labels,
                             DeviceRange<int> label_to_slot, ArrayViewND<T, 2> lines,
                             ArrayViewND<T, N> data, T p0, T vmin, T xtol)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= labels.size()) return;

    int label = labels[idx];
    if (label <= 0) return;

    int id = label_to_slot[static_cast<csize_t>(label)];
    if (id < 0) return;

    // Lines are indexed by label slot in `index`, not by flattened pixel index.
    LineND<T, N> line {to_point<N>(lines.data(2 * id * N)), to_point<N>(lines.data(2 * id * N + N))};
    PointND<csize_t, N> point = make_point<N, N>(idx, labels.shape());

    if (line.distance(point) < xtol)
    {
        T val = data[idx];
        atomicAdd(&counts[id].n_total, 1);
        if (val >= vmin) atomicAdd(&counts[id].n_signal, 1);
    }
}

template <typename T>
__global__ void logbinom_kernel(ArrayViewND<T, 1> out, DeviceRange<StreakCounts> counts, T p0)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    out[idx] = detail::logbinom(counts[idx].n_total, counts[idx].n_signal, p0);
}

template <typename T, csize_t N>
array_t<T> p_values_nd(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> lines, array_t<T> data, T p0, T vmin, T xtol)
{
    auto out_view = cast_to_nd<T, 1>(out.view());
    auto labels_view = cast_to_nd<int, N>(labels.view());
    auto index_view = cast_to_nd<py::ssize_t, 1>(index.view());
    auto lines_view = cast_to_nd<T, 2>(lines.view());
    auto data_view = cast_to_nd<T, N>(data.view());

    csize_t n_labels = index_view.size();
    if (n_labels == 0) return out;

    DeviceVector<StreakCounts> counts (n_labels, StreakCounts{});

    csize_t block_size = BLOCK_SIZE;

    // Find max label value to size the lookup table
    DeviceVector<int> max_label_d (1, 0);
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    DeviceVector<char> temp_storage(temp_bytes);
    cub::DeviceReduce::Max(temp_storage.data(), temp_bytes, labels_view.data(0), max_label_d.data(), labels_view.size());
    handle_cuda_error(cudaGetLastError());

    int max_label = 0;
    handle_cuda_error(cudaMemcpy(&max_label, max_label_d.data(), sizeof(int), cudaMemcpyDeviceToHost));

    // Pass 1: build label -> slot map
    DeviceVector<int> label_to_slot (static_cast<csize_t>(max_label) + 1, -1);
    csize_t n_blocks = (index_view.size() + block_size - 1) / block_size;
    index_map_kernel<<<n_blocks, block_size>>>(label_to_slot.view(), index_view);
    handle_cuda_error(cudaGetLastError());

    // Pass 2: count signal/total pixels per label using O(1) lookup
    csize_t size = labels_view.size();
    n_blocks = (size + block_size - 1) / block_size;
    count_kernel<<<n_blocks, block_size>>>(counts.view(), labels_view, label_to_slot.view(), lines_view, data_view, p0, vmin, xtol);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    n_blocks = (n_labels + block_size - 1) / block_size;
    logbinom_kernel<<<n_blocks, block_size>>>(out_view, counts.view(), p0);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return out;
}

template <typename T>
array_t<T> p_values(array_t<T> out, array_t<int> labels, array_t<py::ssize_t> index, array_t<T> lines, array_t<T> data, T p0, T vmin, T xtol)
{
    if (labels.ndim() != data.ndim())
    {
        throw std::invalid_argument("labels array dimension (" + std::to_string(labels.ndim()) +
                                    ") does not match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (lines.ndim() != 2 || lines.shape(1) != 2 * data.ndim())
    {
        throw std::invalid_argument("lines array must have shape (n_lines, 2 * data.ndim())");
    }
    if (out.ndim() != 1 || out.shape(0) != index.size())
    {
        throw std::invalid_argument("output array shape (" + std::to_string(out.shape(0)) + ") does not match expected shape (" + std::to_string(index.size()) + ")");
    }

    switch (labels.ndim())
    {
        case 2: return p_values_nd<T, 2>(out, labels, index, lines, data, p0, vmin, xtol);
        case 3: return p_values_nd<T, 3>(out, labels, index, lines, data, p0, vmin, xtol);
        case 4: return p_values_nd<T, 4>(out, labels, index, lines, data, p0, vmin, xtol);
        case 5: return p_values_nd<T, 5>(out, labels, index, lines, data, p0, vmin, xtol);
        case 6: return p_values_nd<T, 6>(out, labels, index, lines, data, p0, vmin, xtol);
        case 7: return p_values_nd<T, 7>(out, labels, index, lines, data, p0, vmin, xtol);
        default: throw std::runtime_error("Unsupported number of dimensions: labels.ndim = " + std::to_string(labels.ndim()));
    }
}

}; // namespace cbclib::cuda

PYBIND11_MODULE(cuda_label, m)
{
    using namespace cbclib;
    namespace cu = cbclib::cuda;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    m.def("label", &cu::label<bool>, py::arg("out"), py::arg("inp"), py::arg("structure"), py::arg("npts") = 1);
    m.def("label", &cu::label<int>, py::arg("out"), py::arg("inp"), py::arg("structure"), py::arg("npts") = 1);
    m.def("label", &cu::label<py::ssize_t>, py::arg("out"), py::arg("inp"), py::arg("structure"), py::arg("npts") = 1);

    m.def("center_of_mass", &cu::center_of_mass<float>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("data"));
    m.def("center_of_mass", &cu::center_of_mass<double>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("data"));

    m.def("covariance_matrix", &cu::covariance_matrix<float>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("data"));
    m.def("covariance_matrix", &cu::covariance_matrix<double>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("data"));

    m.def("p_values", &cu::p_values<float>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("lines"), py::arg("data"), py::arg("p0"), py::arg("vmin"), py::arg("xtol"));
    m.def("p_values", &cu::p_values<double>, py::arg("out"), py::arg("labels"), py::arg("index"), py::arg("lines"), py::arg("data"), py::arg("p0"), py::arg("vmin"), py::arg("xtol"));
}
