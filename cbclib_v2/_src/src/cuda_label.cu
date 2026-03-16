#include <cub/cub.cuh>
#include "label.hpp"
#include "cupy_array.hpp"

namespace cbclib::cuda {

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
template <csize_t N>
__global__ void init_kernel(ArrayViewND<int, N> out)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx]) out[idx] = idx;
    else out[idx] = -1;
}

// CUDA kernel : connect neighbouring pixels
template <csize_t N>
__global__ void connect_kernel(ArrayViewND<int, N> out, DeviceRange<py::ssize_t> shifts, csize_t n_shifts)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out.size()) return;

    if (out[idx] < 0) return;

    // k is a shifted index and j is an original index
    int j = idx;

    for (csize_t i = 0; i < n_shifts; i++)
    {
        csize_t rest = idx;
        int k = 0;
        csize_t stride = 1;

        // Converting linear index to multi-dimensional index
        // Applying shifts and converting back to linear index
        for (csize_t n = out.ndim(); n > 0; --n)
        {
            auto coord = rest % out.shape(n - 1) + shifts[i * out.ndim() + (n - 1)];
            if (coord < 0 || coord >= out.shape(n - 1))
            {
                k = -1; // Out of bounds
                break;
            }
            k += coord * stride;
            rest /= out.shape(n - 1);
            stride *= out.shape(n - 1);
        }

        if (k < 0 || out[k] < 0) continue;

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
    DeviceVector<csize_t> keep_flags(n_labels);
    DeviceVector<csize_t> label_map(n_labels);

    // Launch kernel to mark labels to keep (1) or discard (0)
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_labels + block_size - 1) / block_size;
    keep_kernel<<<n_blocks, block_size>>>(label_sizes.view(), keep_flags.view(), npts);
    handle_cuda_error(cudaGetLastError());

    // Compute prefix sum (exclusive scan) of keep flags to build the label map
    DeviceVector<csize_t> prefix(n_labels);

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, keep_flags.data(), prefix.data(), n_labels);

    DeviceVector<char> temp_storage(temp_storage_bytes);

    cub::DeviceScan::ExclusiveSum(temp_storage.data(), temp_storage_bytes, keep_flags.data(), prefix.data(), n_labels);
    handle_cuda_error(cudaGetLastError());

    // Launch kernel to build label remap (0 means discard, else compact id starting at 1)
    counts.set(0); // counts[0] will hold the number of labels to keep
    build_map_kernel<<<n_blocks, block_size>>>(keep_flags.view(), prefix.view(), label_map.view(), counts.view());
    handle_cuda_error(cudaGetLastError());

    // Copy the number of labels to keep back to host
    handle_cuda_error(cudaMemcpy(&n_labels, counts.data(), sizeof(csize_t), cudaMemcpyDeviceToHost));

    return label_map;
}

template <csize_t N>
std::tuple<array_t<int>, py::ssize_t> label_nd(array_t<bool> input, Structure structure, py::ssize_t npts)
{
    array_t<int> output = input.astype<int>();
    DeviceVector<py::ssize_t> shifts = negative_shifts(structure);

    auto output_view = cast_to_nd<int, N>(output.view());

    // Launch kernel to initialize output array
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (output.size() + block_size - 1) / block_size;
    init_kernel<<<n_blocks, block_size>>>(output_view);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to connect neighbouring pixels
    connect_kernel<<<n_blocks, block_size>>>(output_view, shifts.view(), shifts.size() / N);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to flatten the union-find trees and count the number of labels
    DeviceVector<csize_t> counts (2); // counts[0] is the number of labels, counts[1] is the temporary counter
    counts.set(0);

    flatten_kernel<<<n_blocks, block_size>>>(output_view, counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Launch kernel to yield the label roots and store them in the labels array
    csize_t n_labels = 0;
    handle_cuda_error(cudaMemcpy(&n_labels, counts.data(), sizeof(csize_t), cudaMemcpyDeviceToHost));

    DeviceVector<csize_t> labels(n_labels);

    label_kernel<<<n_blocks, block_size>>>(output_view, labels.view(), counts.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    // Sort the labels array to enable binary search during relabeling
    DeviceVector<csize_t> sorted_labels = sort_labels(labels, n_labels);

    // Launch kernel to relabel the output array to have consecutive labels starting from 0
    finalise_kernel<<<n_blocks, block_size>>>(output_view, sorted_labels.view());
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    if (npts <= 1) return std::make_tuple(output, py::ssize_t(n_labels)); // No need to filter labels if npts <= 1

    // Count label sizes and discard small labels
    DeviceVector<csize_t> label_sizes(n_labels);
    label_sizes.set(0);
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

std::tuple<array_t<int>, py::ssize_t> label(array_t<bool> input, Structure structure, py::ssize_t npts)
{
    if (input.ndim() != structure.rank())
    {
        throw std::invalid_argument("input array dimension (" + std::to_string(input.ndim()) +
                                    ") does not match structure rank (" + std::to_string(structure.rank()) + ")");
    }

    switch (structure.rank())
    {
        case 2: return label_nd<2>(input, structure, npts);
        case 3: return label_nd<3>(input, structure, npts);
        case 4: return label_nd<4>(input, structure, npts);
        case 5: return label_nd<5>(input, structure, npts);
        case 6: return label_nd<6>(input, structure, npts);
        case 7: return label_nd<7>(input, structure, npts);
        default: throw std::runtime_error("Unsupported number of dimensions");
    }
}

template <typename T, csize_t N>
__global__ void detect_peaks_kernel(ArrayViewND<py::ssize_t, N> peaks, ArrayViewND<T, N> data, DeviceRange<py::ssize_t> shifts, csize_t n_shifts, size_t radius, T vmin)
{
    csize_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= peaks.size()) return;

    // Find the coordinates of the 2D square neighbourhood
    auto start = peaks.coord_at(idx);
    start[N - 2] *= radius;
    start[N - 1] *= radius;

    auto end = start;
    end[N - 2] += radius;
    end[N - 1] += radius;

    end[N - 2] = min(end[N - 2], data.shape(N - 2));
    end[N - 1] = min(end[N - 1], data.shape(N - 1));

    // Find local maximum in the 2D neighbourhood
    // A local maximum is a pixel whose value is higher than all its neighbors (defined by shifts)
    py::ssize_t best_idx = -1, running_idx = -1;
    T best_val = vmin;

    // Find the index of the (coord[0], ..., coord[N - 3], 0, 0) pixel
    auto running = start;
    running[N - 2] = 0;
    running[N - 1] = 0;
    auto zero_idx = data.index_at(running);

    for (csize_t i = start[N - 2]; i < end[N - 2]; ++i)
    {
        for (csize_t j = start[N - 1]; j < end[N - 1]; ++j)
        {
            running[N - 2] = i;
            running[N - 1] = j;
            running_idx = data.index_at(running);

            T val = data.at(running);
            if (val <= vmin) continue; // Skip if below threshold

            // Check if this is a local maximum by comparing to all neighbors
            for (csize_t k = 0; k < n_shifts; ++k)
            {
                py::ssize_t neighbour_idx = 0;
                csize_t stride = 1;

                // Converting last two dimensions of running coordinate
                // to linear index and applying shifts
                for (csize_t n = 2; n > 0; --n)
                {
                    auto coord = running[N + n - 3] + shifts[2 * k + (n - 1)];
                    if (coord < 0 || coord >= data.shape(data.ndim() + n - 3))
                    {
                        neighbour_idx = -1; // Out of bounds
                        break;
                    }
                    neighbour_idx += coord * stride;
                    stride *= data.shape(data.ndim() + n - 3);
                }

                if (neighbour_idx < 0 || data[neighbour_idx + zero_idx] >= val)
                {
                    running_idx = -1; // Not a local maximum
                    break;
                }
            }

            // Keep track of the best local maximum found
            if (running_idx >= 0 && val > best_val)
            {
                best_val = val;
                best_idx = running_idx;
            }
        }
    }

    peaks[idx] = best_idx; // Will be -1 if no local maximum above threshold found
}

template <typename T, csize_t N>
array_t<py::ssize_t> detect_peaks_nd(array_t<py::ssize_t> peaks, array_t<T> data, Structure structure, size_t radius, T vmin)
{
    auto peaks_view = cast_to_nd<py::ssize_t, N>(peaks.view());
    auto data_view = cast_to_nd<T, N>(data.view());

    auto shifts = all_shifts(structure); // Use the provided structure
    csize_t n_shifts = shifts.size() / structure.rank();

    // Launch kernel to detect peaks
    csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (peaks.size() + block_size - 1) / block_size;
    detect_peaks_kernel<<<n_blocks, block_size>>>(peaks_view, data_view, shifts.view(), n_shifts, radius, vmin);
    handle_cuda_error(cudaGetLastError());
    handle_cuda_error(cudaDeviceSynchronize());

    return peaks;
}

template <typename T>
array_t<py::ssize_t> detect_peaks(array_t<py::ssize_t> peaks, array_t<T> data, Structure structure, size_t radius, T vmin)
{
    if (peaks.ndim() != data.ndim())
    {
        throw std::invalid_argument("peaks array dimension (" + std::to_string(peaks.ndim()) +
                                    ") does not match data array dimension (" + std::to_string(data.ndim()) + ")");
    }
    if (structure.rank() != 2)
    {
        throw std::invalid_argument("structure rank (" + std::to_string(structure.rank()) + ") must be 2");
    }

    switch (peaks.ndim())
    {
        case 2: return detect_peaks_nd<T, 2>(peaks, data, structure, radius, vmin);
        case 3: return detect_peaks_nd<T, 3>(peaks, data, structure, radius, vmin);
        case 4: return detect_peaks_nd<T, 4>(peaks, data, structure, radius, vmin);
        case 5: return detect_peaks_nd<T, 5>(peaks, data, structure, radius, vmin);
        case 6: return detect_peaks_nd<T, 6>(peaks, data, structure, radius, vmin);
        case 7: return detect_peaks_nd<T, 7>(peaks, data, structure, radius, vmin);
        default: throw std::runtime_error("Unsupported number of dimensions: peaks.ndim = " + std::to_string(peaks.ndim()));
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

    m.def("label", &cu::label, py::arg("inp"), py::arg("structure"), py::arg("npts") = 1);

    m.def("detect_peaks", &cu::detect_peaks<float>, py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"));
    m.def("detect_peaks", &cu::detect_peaks<double>, py::arg("peaks"), py::arg("data"), py::arg("structure"), py::arg("radius"), py::arg("vmin"));
}
