#include <cub/cub.cuh>
#include "cupy_array.hpp"

namespace cbclib::cuda {

static constexpr csize_t SMALL_REDUCE_SIZE = 64;

template <typename T>
struct less
{
    __device__ bool operator()(const T & lhs, const T & rhs) const { return lhs < rhs; }
};

template <typename D>
struct order_less
{
    const D * values;

    __device__ bool operator()(csize_t lhs, csize_t rhs) const
    {
        return values[lhs] < values[rhs];
    }
};

template <typename Iter>
__device__ void iter_swap(Iter lhs, Iter rhs)
{
    auto tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}

template <typename Iter, typename Compare>
__device__ Iter partition_pivot(Iter first, Iter last, Iter pivot, Compare comp)
{
    auto pivot_value = *pivot;
    iter_swap(pivot, last - 1);

    Iter store = first;
    for (Iter it = first; it + 1 < last; ++it)
    {
        if (comp(*it, pivot_value))
        {
            iter_swap(store, it);
            ++store;
        }
    }

    iter_swap(store, last - 1);
    return store;
}

template <typename Iter, typename Compare>
__device__ void nth_element(Iter first, Iter nth, Iter last, Compare comp)
{
    Iter left = first;
    Iter right = last;

    while (right - left > 1)
    {
        Iter pivot = left + (right - left) / 2;
        pivot = partition_pivot(left, right, pivot, comp);

        if (nth == pivot) return;
        if (nth < pivot) right = pivot;
        else left = pivot + 1;
    }
}

template <typename Iter>
__device__ void nth_element(Iter first, Iter nth, Iter last)
{
    nth_element(first, nth, last, less<remove_cvref_t<decltype(*first)>>());
}

template <typename Iter, typename Compare>
__device__ void sort(Iter first, Iter last, Compare comp)
{
    for (Iter it = first; it + 1 < last; ++it)
    {
        nth_element(it, it, last, comp);
    }
}

template <typename Iter>
__device__ void sort(Iter first, Iter last)
{
    sort(first, last, less<remove_cvref_t<decltype(*first)>>());
}

template <typename D>
__device__ D median(D * first, D * last)
{
    csize_t n_values = last - first;
    D * mid = first + n_values / 2;
    nth_element(first, mid, last);

    if (n_values & 1) return *mid;

    D low = *first;
    for (D * it = first + 1; it < mid; ++it)
    {
        if (*it > low) low = *it;
    }
    return (low + *mid) / D(2);
}

template <typename T, typename D, csize_t N, bool RETURN_STD>
__global__ void robust_mean_kernel(ArrayViewND<D, N> mean, ArrayViewND<D, N> std,
                                         ArrayViewND<T, N> inp, csize_t n_rows,
                                         csize_t n_reduce, csize_t j0, csize_t j1,
                                         csize_t n_iter, D lm)
{
    csize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    csize_t start = row * n_reduce;
    T * inp_ptr = inp.data(start);
    csize_t inp_stride = inp.strides(N - 1) / sizeof(T);

    D values[SMALL_REDUCE_SIZE];    // mutable array for median computation
    D errors[SMALL_REDUCE_SIZE];
    csize_t order[SMALL_REDUCE_SIZE];

    order_less<D> comp {errors};

    for (csize_t j = 0; j < n_reduce; ++j)
    {
        values[j] = static_cast<D>(inp_ptr[j * inp_stride]);
    }

    D center = median(values, values + n_reduce);

    for (csize_t i = 0; i < n_iter; ++i)
    {
        for (csize_t j = 0; j < n_reduce; ++j)
        {
            D diff = inp_ptr[j * inp_stride] - center;
            errors[j] = diff * diff;
            order[j] = j;
        }

        nth_element(order, order + j1 - 1, order + n_reduce, comp);
        if (j0) nth_element(order, order + j0, order + j1, comp);

        D sum = D();
        for (csize_t j = j0; j < j1; ++j) sum += inp_ptr[order[j] * inp_stride];
        center = sum / static_cast<D>(j1 - j0);
    }

    for (csize_t j = 0; j < n_reduce; ++j)
    {
        D diff = inp_ptr[j * inp_stride] - center;
        errors[j] = diff * diff;
        order[j] = j;
    }

    sort(order, order + n_reduce, comp);

    csize_t cutoff = n_reduce;
    D cumsum = D();
    for (csize_t j = 0; j < n_reduce; ++j)
    {
        D error = errors[order[j]];
        cumsum += error;

        if (lm * cumsum < static_cast<D>(j) * error)
        {
            cutoff = j;
            break;
        }
    }

    D sum = D();
    D var = D();
    for (csize_t j = 0; j < cutoff; ++j)
    {
        csize_t index = order[j];
        sum += inp_ptr[index * inp_stride];
        if constexpr (RETURN_STD) var += errors[index];
    }

    if (cutoff > 0)
    {
        mean[row] = sum / static_cast<D>(cutoff);
        if constexpr (RETURN_STD) std[row] = math_traits<D>::sqrt(var / static_cast<D>(cutoff));
    }
    else
    {
        mean[row] = D();
        if constexpr (RETURN_STD) std[row] = D();
    }
}

template <typename T, typename D, csize_t N, bool RETURN_STD>
std::tuple<array_t<D>, array_t<D>> robust_mean_nd(array_t<D> mean, array_t<D> std,
                                                        array_t<T> inp, D r0, D r1,
                                                        csize_t n_iter, D lm)
{
    csize_t n_reduce = inp.shape(N - 1);
    csize_t n_rows = inp.size() / n_reduce;
    if (n_reduce > SMALL_REDUCE_SIZE)
        throw std::runtime_error("Small robust mean only supports reduction sizes up to " + std::to_string(SMALL_REDUCE_SIZE));

    csize_t j0 = static_cast<csize_t>(r0 * static_cast<D>(n_reduce));
    csize_t j1 = static_cast<csize_t>(r1 * static_cast<D>(n_reduce));
    if (j1 <= j0 || j1 > n_reduce)
        throw std::runtime_error("Invalid robust mean quantile range for reduction size " + std::to_string(n_reduce));

    constexpr csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = (n_rows + block_size - 1) / block_size;

    ArrayViewND<D, N> std_view;
    if constexpr (RETURN_STD) std_view = cast_to_nd<D, N>(std.view());

    robust_mean_kernel<T, D, N, RETURN_STD><<<n_blocks, block_size>>>(
        cast_to_nd<D, N>(mean.view()), std_view,
        cast_to_nd<T, N>(inp.view()), n_rows, n_reduce, j0, j1, n_iter, lm
    );
    handle_cuda_error(cudaGetLastError());

    return std::make_tuple(mean, std);
}

template <typename T, typename D>
array_t<D> robust_mean(array_t<D> mean, array_t<T> inp, D r0, D r1, csize_t n_iter, D lm)
{
    check_equal("inp and mean shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim() - 1,
                mean.shape(), mean.shape() + mean.ndim());
    if (mean.shape(mean.ndim() - 1) != 1)
        throw std::runtime_error("Last dimension of mean must equal to 1");

    switch (inp.ndim())
    {
        case 1: return std::get<0>(robust_mean_nd<T, D, 1, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 2: return std::get<0>(robust_mean_nd<T, D, 2, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 3: return std::get<0>(robust_mean_nd<T, D, 3, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 4: return std::get<0>(robust_mean_nd<T, D, 4, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 5: return std::get<0>(robust_mean_nd<T, D, 5, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 6: return std::get<0>(robust_mean_nd<T, D, 6, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        case 7: return std::get<0>(robust_mean_nd<T, D, 7, false>(mean, array_t<D>(), inp, r0, r1, n_iter, lm));
        default: throw std::runtime_error("Unsupported number of dimensions of mean and input: " + std::to_string(mean.ndim()) +
                                          " and " + std::to_string(inp.ndim()));
    }
}

template <typename T, typename D>
std::tuple<array_t<D>, array_t<D>> robust_mean_std(array_t<D> mean, array_t<D> std,
                                                        array_t<T> inp, D r0, D r1,
                                                        csize_t n_iter, D lm)
{
    check_equal("mean and std shapes are incompatible",
                mean.shape(), mean.shape() + mean.ndim(),
                std.shape(), std.shape() + std.ndim());
    check_equal("inp and mean shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim() - 1,
                mean.shape(), mean.shape() + mean.ndim());
    if (mean.shape(mean.ndim() - 1) != 1 || std.shape(std.ndim() - 1) != 1)
        throw std::runtime_error("Last dimensions of mean and std must equal to 1");

    switch (inp.ndim())
    {
        case 1: return robust_mean_nd<T, D, 1, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 2: return robust_mean_nd<T, D, 2, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 3: return robust_mean_nd<T, D, 3, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 4: return robust_mean_nd<T, D, 4, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 5: return robust_mean_nd<T, D, 5, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 6: return robust_mean_nd<T, D, 6, true>(mean, std, inp, r0, r1, n_iter, lm);
        case 7: return robust_mean_nd<T, D, 7, true>(mean, std, inp, r0, r1, n_iter, lm);
        default: throw std::runtime_error("Unsupported number of dimensions of mean, std and input: " + std::to_string(mean.ndim()) +
                                          ", " + std::to_string(std.ndim()) + " and " + std::to_string(inp.ndim()));
    }
}

// Pass 1: Compute per-chunk error sums (parallelized over reduction axis)
// Input: errors array of size [n_rows, n_reduce] (indirectly indexed via indices)
// Output: chunk_sums array of size [n_rows, n_chunks]
// Grid: (n_rows, n_chunks) blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Each block handles one (row, chunk) pair. Within the block, threads
//           cooperatively sum BLOCK consecutive error values using CUB BlockReduce.
//           This divides the large reduction axis (n_reduce ~ 1e6) into manageable
//           BLOCK_SIZE-element chunks that can be reduced in parallel.
template <typename T, typename D, csize_t N, int BLOCK>
__global__ void sum_chunks_kernel(ArrayViewND<D, N> errors, ArrayViewND<py::ssize_t, N> indices,
                                  D * chunk_sums, csize_t n_reduce, csize_t n_chunks)
{
    csize_t row = blockIdx.x;
    csize_t chunk = blockIdx.y;
    csize_t tid = threadIdx.x;

    if (chunk >= n_chunks) return;

    csize_t start = row * n_reduce;
    csize_t chunk_start = chunk * BLOCK;
    csize_t j = chunk_start + tid;  // Each thread handles one element within the chunk

    D * err_ptr = errors.data(start);
    csize_t err_stride = errors.strides(N - 1) / sizeof(D);

    // Load one error value per thread (zero-padded if beyond n_reduce)
    D val = D();
    if (j < n_reduce)
    {
        py::ssize_t idx_j = indices[start + j];
        val = err_ptr[idx_j * err_stride];
    }

    // Use CUB BlockReduce to sum all BLOCK values in parallel using shared memory
    using BlockReduce = cub::BlockReduce<D, BLOCK>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;

    D total = BlockReduce(reduce_storage).Sum(val);
    if (tid == 0)
    {
        chunk_sums[row * n_chunks + chunk] = total;  // Thread 0 writes the result
    }
}

// Pass 2: Compute exclusive prefix scan of chunk sums (one block per row)
// Input: chunk_sums array of size [n_rows, n_chunks] from Pass 1
// Output: chunk_prefix array of size [n_rows, n_chunks]
// Grid: n_rows blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Each block handles one row with n_chunks values. Since n_chunks can exceed
//           BLOCK size, we use a tiled approach: process n_chunks in tiles of BLOCK chunks.
//           Within each tile, use CUB BlockScan for exclusive prefix scan. The 'running'
//           accumulator carries the sum from all previous tiles, enabling correct
//           prefix computation across tile boundaries.
// Result: chunk_prefix[row, chunk] = sum of chunk_sums[row, 0..chunk-1]
template <typename D, int BLOCK>
__global__ void exclusive_scan_kernel(const D * chunk_sums, D * chunk_prefix, csize_t n_chunks)
{
    csize_t row = blockIdx.x;
    csize_t tid = threadIdx.x;

    using BlockScan = cub::BlockScan<D, BLOCK>;
    using BlockReduce = cub::BlockReduce<D, BLOCK>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ D running;  // Accumulates sum from all previous tiles

    if (tid == 0) running = D();
    __syncthreads();

    // Process n_chunks in tiles of size BLOCK (one chunk per thread per iteration)
    for (csize_t base = 0; base < n_chunks; base += BLOCK)
    {
        csize_t idx = base + tid;  // Each thread processes one chunk index
        D val = (idx < n_chunks) ? chunk_sums[row * n_chunks + idx] : D();

        // ExclusiveSum: each thread gets sum of all values from lower thread IDs
        // Example: inputs [5,3,7,2] -> outputs [0,5,8,15]
        D prefix = D();
        BlockScan(scan_storage).ExclusiveSum(val, prefix);

        if (idx < n_chunks)
        {
            // Add 'running' to get true prefix across all tiles
            chunk_prefix[row * n_chunks + idx] = running + prefix;
        }

        // Sum all values in current tile and update running accumulator
        D tile_sum = BlockReduce(reduce_storage).Sum(val);
        if (tid == 0)
        {
            running += tile_sum;  // Carry forward for next tile
        }
        __syncthreads();
    }
}

// Pass 3 for inliers_mean_parallel: Detect cutoff and compute mean (optionally std)
// Grid: n_rows blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Detect the cutoff and reduce the accepted values:
//   Phase A: Find cutoff using chunk-based scanning
//   Phase B: Compute sum of values and optionally variance for j < cutoff using strided access
//            Use BlockReduce to combine partial sums from all threads
//            Compute mean = sum/cutoff, optionally std = sqrt(var/cutoff)
template <typename T, typename D, csize_t N, int BLOCK>
__global__ void inliers_mean_kernel(ArrayViewND<D, N> mean, ArrayViewND<D, N> std,
                                    ArrayViewND<T, N> inp, ArrayViewND<D, N> errors,
                                    ArrayViewND<py::ssize_t, N> indices, const D * chunk_prefix,
                                    csize_t n_reduce, csize_t n_chunks, D lm)
{
    csize_t row = blockIdx.x;
    csize_t tid = threadIdx.x;

    csize_t start = row * n_reduce;

    T * inp_ptr = inp.data(start);
    csize_t inp_stride = inp.strides(N - 1) / sizeof(T);

    D * err_ptr = errors.data(start);
    csize_t err_stride = errors.strides(N - 1) / sizeof(D);

    bool compute_std = (std.data() != nullptr);

    using BlockScan = cub::BlockScan<D, BLOCK>;
    using BlockReduceInt = cub::BlockReduce<int, BLOCK>;
    using BlockReduceD = cub::BlockReduce<D, BLOCK>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ typename BlockReduceInt::TempStorage reduce_int_storage;
    __shared__ typename BlockReduceD::TempStorage reduce_storage;
    __shared__ csize_t cutoff; // Shared cutoff index (first j where condition fails)

    if (tid == 0) cutoff = n_reduce;  // Initialize to "no cutoff found"
    __syncthreads();

    // Phase A: Find the first residual that violates the inlier condition
    for (csize_t chunk = 0; chunk < n_chunks; ++chunk)
    {
        if (cutoff != n_reduce) break;  // Early exit if cutoff already found

        csize_t chunk_start = chunk * BLOCK;
        csize_t j = chunk_start + tid;  // Each thread handles one element in chunk

        py::ssize_t idx_j = 0;
        D error = D();
        if (j < n_reduce)
        {
            idx_j = indices[start + j];
            error = err_ptr[idx_j * err_stride];
        }

        // InclusiveSum: each thread gets sum of its value + all lower thread IDs
        // Example: inputs [5,3,7,2] -> outputs [5,8,15,17] (includes current element)
        D cumsum = D();
        BlockScan(scan_storage).InclusiveSum(error, cumsum);
        __syncthreads();

        // Compute global cumulative sum by adding base offset from previous chunks
        D base = chunk_prefix[row * n_chunks + chunk];
        int candidate = static_cast<int>(n_reduce);
        if (j < n_reduce)
        {
            // Check stopping condition: lm * cumsum < j * error means we've gone too far
            // Note: j is row-relative (0 to n_reduce-1), matching the row-relative cumsum
            if (lm * (base + cumsum) < static_cast<D>(j) * error)
            {
                candidate = static_cast<int>(j);  // This thread found a cutoff candidate
            }
        }

        // Find minimum cutoff candidate across all threads in the block
        int chunk_cutoff = BlockReduceInt(reduce_int_storage).Reduce(candidate, cub::Min());
        if (tid == 0 && chunk_cutoff < static_cast<int>(n_reduce))
        {
            cutoff = static_cast<csize_t>(chunk_cutoff);
        }
        __syncthreads();
    }

    // Phase B: Compute sum of values and optionally variance using strided access
    D local_sum = D();
    D local_var = D();
    for (csize_t j = tid; j < cutoff; j += BLOCK)
    {
        py::ssize_t idx_j = indices[start + j];
        D val = static_cast<D>(inp_ptr[idx_j * inp_stride]);
        local_sum += val;
        if (compute_std)
        {
            D error = err_ptr[idx_j * err_stride];
            local_var += error;
        }
    }

    // Reduce sum across all threads
    D total_sum = BlockReduceD(reduce_storage).Sum(local_sum);

    D total_var = D();
    if (compute_std)
    {
        __syncthreads();  // Sync before reusing shared memory for second reduction
        total_var = BlockReduceD(reduce_storage).Sum(local_var);
    }

    if (tid == 0)
    {
        if (cutoff > 0)
        {
            mean[row] = total_sum / static_cast<D>(cutoff);
            if (compute_std) std[row] = math_traits<D>::sqrt(total_var / static_cast<D>(cutoff));
        }
        else
        {
            mean[row] = D();
            if (compute_std) std[row] = D();
        }
    }
}

template <typename T, typename D, csize_t N>
__global__ void small_inliers_mean_kernel(ArrayViewND<D, N> mean, ArrayViewND<D, N> std,
                                          ArrayViewND<T, N> inp, ArrayViewND<D, N> errors,
                                          ArrayViewND<py::ssize_t, N> indices,
                                          csize_t n_rows, csize_t n_reduce, D lm)
{
    csize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    csize_t start = row * n_reduce;

    T * inp_ptr = inp.data(start);
    csize_t inp_stride = inp.strides(N - 1) / sizeof(T);

    D * err_ptr = errors.data(start);
    csize_t err_stride = errors.strides(N - 1) / sizeof(D);

    csize_t cutoff = n_reduce;
    D cumsum = D();

    for (csize_t j = 0; j < n_reduce; ++j)
    {
        py::ssize_t idx_j = indices[start + j];
        D error = err_ptr[idx_j * err_stride];
        cumsum += error;

        if (lm * cumsum < static_cast<D>(j) * error)
        {
            cutoff = j;
            break;
        }
    }

    D sum = D();
    D var = D();
    bool compute_std = (std.data() != nullptr);

    for (csize_t j = 0; j < cutoff; ++j)
    {
        py::ssize_t idx_j = indices[start + j];
        sum += static_cast<D>(inp_ptr[idx_j * inp_stride]);
        if (compute_std) var += err_ptr[idx_j * err_stride];
    }

    if (cutoff > 0)
    {
        mean[row] = sum / static_cast<D>(cutoff);
        if (compute_std) std[row] = math_traits<D>::sqrt(var / static_cast<D>(cutoff));
    }
    else
    {
        mean[row] = D();
        if (compute_std) std[row] = D();
    }
}

template <typename T, typename D, csize_t N>
array_t<D> inliers_mean_nd(array_t<D> mean, array_t<T> inp, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    csize_t n_reduce = inp.shape(N - 1);
    csize_t n_rows = inp.size() / n_reduce;

    constexpr csize_t block_size = BLOCK_SIZE;
    if (n_reduce <= block_size)
    {
        constexpr csize_t rows_per_block = 256;
        csize_t n_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

        small_inliers_mean_kernel<T, D, N><<<n_blocks, rows_per_block>>>(
            cast_to_nd<D, N>(mean.view()), ArrayViewND<D, N>(),
            cast_to_nd<T, N>(inp.view()), cast_to_nd<D, N>(errors.view()),
            cast_to_nd<py::ssize_t, N>(indices.view()), n_rows, n_reduce, lm
        );
        handle_cuda_error(cudaGetLastError());

        return mean;
    }

    csize_t n_chunks = (n_reduce + block_size - 1) / block_size;

    DeviceVector<D> chunk_sums(n_rows * n_chunks);
    DeviceVector<D> chunk_prefix(n_rows * n_chunks);

    // Reuse the shared error scan passes
    dim3 grid_chunks(n_rows, n_chunks);
    sum_chunks_kernel<T, D, N, block_size><<<grid_chunks, block_size>>>(
        cast_to_nd<D, N>(errors.view()), cast_to_nd<py::ssize_t, N>(indices.view()),
        chunk_sums.data(), n_reduce, n_chunks
    );
    handle_cuda_error(cudaGetLastError());

    exclusive_scan_kernel<D, block_size><<<n_rows, block_size>>>(
        chunk_sums.data(), chunk_prefix.data(), n_chunks
    );
    handle_cuda_error(cudaGetLastError());

    // Pass 3: Compute mean (no std)
    inliers_mean_kernel<T, D, N, block_size><<<n_rows, block_size>>>(
        cast_to_nd<D, N>(mean.view()), ArrayViewND<D, N>(),
        cast_to_nd<T, N>(inp.view()), cast_to_nd<D, N>(errors.view()),
        cast_to_nd<py::ssize_t, N>(indices.view()), chunk_prefix.data(),
        n_reduce, n_chunks, lm
    );
    handle_cuda_error(cudaGetLastError());

    return mean;
}

template <typename T, typename D>
array_t<D> inliers_mean(array_t<D> mean, array_t<T> inp, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    check_equal("inp and indices shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim(),
                indices.shape(), indices.shape() + indices.ndim());
    check_equal("inp and errors shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim(),
                errors.shape(), errors.shape() + errors.ndim());
    check_equal("inp and mean shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim() - 1,
                mean.shape(), mean.shape() + mean.ndim());
    if (mean.shape(mean.ndim() - 1) != 1)
        throw std::runtime_error("Last dimensions of mean must equal to 1");

    switch (inp.ndim())
    {
        case 1: return inliers_mean_nd<T, D, 1>(mean, inp, errors, indices, lm);
        case 2: return inliers_mean_nd<T, D, 2>(mean, inp, errors, indices, lm);
        case 3: return inliers_mean_nd<T, D, 3>(mean, inp, errors, indices, lm);
        case 4: return inliers_mean_nd<T, D, 4>(mean, inp, errors, indices, lm);
        case 5: return inliers_mean_nd<T, D, 5>(mean, inp, errors, indices, lm);
        case 6: return inliers_mean_nd<T, D, 6>(mean, inp, errors, indices, lm);
        case 7: return inliers_mean_nd<T, D, 7>(mean, inp, errors, indices, lm);
        default: throw std::runtime_error("Unsupported number of dimensions of mean and input: " + std::to_string(mean.ndim()) +
                                          " and " + std::to_string(inp.ndim()));
    }
}

template <typename T, typename D, csize_t N>
std::tuple<array_t<D>, array_t<D>> inliers_mean_std_nd(array_t<D> mean, array_t<D> std, array_t<T> inp, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    csize_t n_reduce = inp.shape(N - 1);
    csize_t n_rows = inp.size() / n_reduce;

    constexpr csize_t block_size = BLOCK_SIZE;
    if (n_reduce <= block_size)
    {
        constexpr csize_t rows_per_block = 256;
        csize_t n_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

        small_inliers_mean_kernel<T, D, N><<<n_blocks, rows_per_block>>>(
            cast_to_nd<D, N>(mean.view()), cast_to_nd<D, N>(std.view()),
            cast_to_nd<T, N>(inp.view()), cast_to_nd<D, N>(errors.view()),
            cast_to_nd<py::ssize_t, N>(indices.view()), n_rows, n_reduce, lm
        );
        handle_cuda_error(cudaGetLastError());

        return std::make_tuple(mean, std);
    }

    csize_t n_chunks = (n_reduce + block_size - 1) / block_size;

    DeviceVector<D> chunk_sums(n_rows * n_chunks);
    DeviceVector<D> chunk_prefix(n_rows * n_chunks);

    // Reuse the shared error scan passes
    dim3 grid_chunks(n_rows, n_chunks);
    sum_chunks_kernel<T, D, N, block_size><<<grid_chunks, block_size>>>(
        cast_to_nd<D, N>(errors.view()), cast_to_nd<py::ssize_t, N>(indices.view()),
        chunk_sums.data(), n_reduce, n_chunks
    );
    handle_cuda_error(cudaGetLastError());

    exclusive_scan_kernel<D, block_size><<<n_rows, block_size>>>(
        chunk_sums.data(), chunk_prefix.data(), n_chunks
    );
    handle_cuda_error(cudaGetLastError());

    // Pass 3: Compute mean + std
    inliers_mean_kernel<T, D, N, block_size><<<n_rows, block_size>>>(
        cast_to_nd<D, N>(mean.view()), cast_to_nd<D, N>(std.view()),
        cast_to_nd<T, N>(inp.view()), cast_to_nd<D, N>(errors.view()),
        cast_to_nd<py::ssize_t, N>(indices.view()), chunk_prefix.data(),
        n_reduce, n_chunks, lm
    );
    handle_cuda_error(cudaGetLastError());

    return std::make_tuple(mean, std);
}

template <typename T, typename D>
std::tuple<array_t<D>, array_t<D>> inliers_mean_std(array_t<D> mean, array_t<D> std, array_t<T> inp, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    check_equal("inp and indices shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim(),
                indices.shape(), indices.shape() + indices.ndim());
    check_equal("inp and errors shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim(),
                errors.shape(), errors.shape() + errors.ndim());
    check_equal("mean and std shapes are incompatible",
                mean.shape(), mean.shape() + mean.ndim(),
                std.shape(), std.shape() + std.ndim());
    check_equal("inp and mean shapes are incompatible",
                inp.shape(), inp.shape() + inp.ndim() - 1,
                mean.shape(), mean.shape() + mean.ndim());
    if (mean.shape(mean.ndim() - 1) != 1 || std.shape(std.ndim() - 1) != 1)
        throw std::runtime_error("Last dimensions of mean and std must equal to 1");

    switch (inp.ndim())
    {
        case 1: return inliers_mean_std_nd<T, D, 1>(mean, std, inp, errors, indices, lm);
        case 2: return inliers_mean_std_nd<T, D, 2>(mean, std, inp, errors, indices, lm);
        case 3: return inliers_mean_std_nd<T, D, 3>(mean, std, inp, errors, indices, lm);
        case 4: return inliers_mean_std_nd<T, D, 4>(mean, std, inp, errors, indices, lm);
        case 5: return inliers_mean_std_nd<T, D, 5>(mean, std, inp, errors, indices, lm);
        case 6: return inliers_mean_std_nd<T, D, 6>(mean, std, inp, errors, indices, lm);
        case 7: return inliers_mean_std_nd<T, D, 7>(mean, std, inp, errors, indices, lm);
        default: throw std::runtime_error("Unsupported number of dimensions of mean, std and input: " + std::to_string(mean.ndim()) +
                                          ", " + std::to_string(std.ndim()) + " and " + std::to_string(inp.ndim()));
    }
}

} // namespace cbclib::cuda

PYBIND11_MODULE(cuda_median, m)
{
    using namespace cbclib;
    namespace cu = cbclib::cuda;
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

    // Parallel versions (3-pass with chunked reduction and prefix scan)
    m.def("robust_mean", &cu::robust_mean<float, float>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean", &cu::robust_mean<double, double>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);
    m.def("robust_mean", &cu::robust_mean<int, float>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean", &cu::robust_mean<int, double>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);
    m.def("robust_mean", &cu::robust_mean<long, float>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean", &cu::robust_mean<long, double>, py::arg("mean"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);

    m.def("robust_mean_std", &cu::robust_mean_std<float, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean_std", &cu::robust_mean_std<double, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);
    m.def("robust_mean_std", &cu::robust_mean_std<int, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean_std", &cu::robust_mean_std<int, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);
    m.def("robust_mean_std", &cu::robust_mean_std<long, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0f, py::arg("r1") = 0.5f, py::arg("n_iter") = 12, py::arg("lm") = 9.0f);
    m.def("robust_mean_std", &cu::robust_mean_std<long, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("r0") = 0.0, py::arg("r1") = 0.5, py::arg("n_iter") = 12, py::arg("lm") = 9.0);

    m.def("inliers_mean", &cu::inliers_mean<float, float>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean", &cu::inliers_mean<double, double>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);
    m.def("inliers_mean", &cu::inliers_mean<int, float>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean", &cu::inliers_mean<int, double>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);
    m.def("inliers_mean", &cu::inliers_mean<long, float>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean", &cu::inliers_mean<long, double>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);

    m.def("inliers_mean_std", &cu::inliers_mean_std<float, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean_std", &cu::inliers_mean_std<double, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);
    m.def("inliers_mean_std", &cu::inliers_mean_std<int, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean_std", &cu::inliers_mean_std<int, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);
    m.def("inliers_mean_std", &cu::inliers_mean_std<long, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean_std", &cu::inliers_mean_std<long, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);


}
