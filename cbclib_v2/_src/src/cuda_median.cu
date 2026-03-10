#include <cub/cub.cuh>
#include "cupy_array.hpp"

namespace cbclib::cuda {

// Linear least squares fit with weights, computed in parallel over the reduction axis (last dimension of input arrays)
//
// fits[i, k] = sum_j(y[i, j] * W[k, indices[i, j]]) / sum_j(W[k, indices[i, j]]^2)
//
// The input arrays are expected to have the reduction dimension as the last one:
// 1. y and indices have shape [..., n_reduce]
// 2. W has shape [n_features, n_reduce]
// 3. fits has shape [..., n_features], i.e. the same shape as y and indices except the last dimension replaced by n_features.

template <typename D>
struct SumPair
{
    D sum;
    D weight;
};

template <typename D>
struct SumPairReduce
{
    __device__ SumPair<D> operator()(const SumPair<D> & lhs, const SumPair<D> & rhs) const
    {
        return {lhs.sum + rhs.sum, lhs.weight + rhs.weight};
    }
};

// Parallel LSQ kernel: one block per (row, feature) pair, threads cooperatively reduce over n_reduce
// Grid: (n_rows * n_features) blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Each block handles one (row, feature) pair. Threads use strided access to accumulate
//           partial SumPair{sum, weight} values. BlockReduce combines all partial sums.
//           Much simpler than inliers_lsq since there's no cutoff detection needed.
template <typename T, typename D, csize_t N, int BLOCK>
__global__ void lsq_kernel(ArrayViewND<D, N> fits, ArrayViewND<T, 2> W, ArrayViewND<T, N> y,
                           ArrayViewND<py::ssize_t, N> indices, csize_t n_features, csize_t n_reduce, csize_t n_truncated)
{
    csize_t block_idx = blockIdx.x;
    csize_t n_rows = fits.size() / n_features;

    csize_t row = block_idx / n_features;
    csize_t feature = block_idx % n_features;

    if (row >= n_rows) return;

    csize_t tid = threadIdx.x;
    csize_t start = row * n_truncated;

    T * y_ptr = y.data(row * n_reduce);
    csize_t y_stride = y.strides(N - 1) / sizeof(T);

    // Each thread accumulates partial sums using strided access
    SumPair<D> local = {D(), D()};
    for (csize_t j = tid; j < n_truncated; j += BLOCK)
    {
        py::ssize_t idx_j = indices[start + j];
        T w = W.at(feature, idx_j);
        local.sum += static_cast<D>(y_ptr[idx_j * y_stride]) * w;
        local.weight += static_cast<D>(w * w);
    }

    // Combine partial sums from all threads using BlockReduce
    using BlockReducePair = cub::BlockReduce<SumPair<D>, BLOCK>;
    __shared__ typename BlockReducePair::TempStorage reduce_storage;

    SumPair<D> total = BlockReducePair(reduce_storage).Reduce(local, SumPairReduce<D>());

    if (tid == 0)
    {
        fits[row * n_features + feature] = total.weight > D() ? total.sum / total.weight : D();
    }
}

template <typename T, typename D, csize_t N>
array_t<D> lsq_impl(array_t<D> fits, array_t<T> W, array_t<T> y, array_t<py::ssize_t> indices)
{
    csize_t n_reduce = W.shape(1);
    csize_t n_features = W.shape(0);
    csize_t n_truncated = indices.shape(indices.ndim() - 1);  // Truncated reduction length from indices
    csize_t n_rows = fits.size() / n_features;

    if (n_rows) check_equal(
        "fits shape is incompatible with y", fits.shape(), fits.shape() + fits.ndim() - 1, y.shape(), y.shape() + y.ndim() - 1);
    if (y.shape(y.ndim() - 1) != n_reduce)
        throw std::runtime_error("Last dimension of y must match to last dimension of W");
    if (fits.shape(fits.ndim() - 1) != n_features)
        throw std::runtime_error("Last dimension of fits must match to first dimension of W");

    constexpr csize_t block_size = BLOCK_SIZE;
    csize_t n_blocks = n_rows * n_features;

    lsq_kernel<T, D, N, block_size><<<n_blocks, block_size>>>(
        cast_to_nd<D, N>(fits.view()), cast_to_nd<T, 2>(W.view()), cast_to_nd<T, N>(y.view()),
        cast_to_nd<py::ssize_t, N>(indices.view()), n_features, n_reduce, n_truncated
    );
    handle_cuda_error(cudaGetLastError());

    return fits;
}

template <typename T, typename D>
array_t<D> lsq(array_t<D> fits, array_t<T> W, array_t<T> y, array_t<py::ssize_t> indices)
{
    switch (y.ndim())
    {
        case 1: return lsq_impl<T, D, 1>(fits, W, y, indices);
        case 2: return lsq_impl<T, D, 2>(fits, W, y, indices);
        case 3: return lsq_impl<T, D, 3>(fits, W, y, indices);
        case 4: return lsq_impl<T, D, 4>(fits, W, y, indices);
        case 5: return lsq_impl<T, D, 5>(fits, W, y, indices);
        case 6: return lsq_impl<T, D, 6>(fits, W, y, indices);
        case 7: return lsq_impl<T, D, 7>(fits, W, y, indices);
        default: throw std::runtime_error("Unsupported number of dimensions of y: " + std::to_string(y.ndim()));
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

// Pass 3: Detect cutoff and compute weighted least squares (one block per row)
// Input: chunk_prefix from Pass 2, plus original data arrays (W, y, errors, indices)
// Output: fits array of size [n_rows, n_features]
// Grid: n_rows blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Two-phase algorithm within each block:
//   Phase A (cutoff detection): Walk chunks sequentially until stopping condition is met.
//            Use InclusiveSum within each chunk + chunk_prefix as base offset to get
//            global cumulative error sum. Check condition lm * cumsum < j * error for
//            each element. Use BlockReduce(Min) to find earliest violation in chunk.
//   Phase B (weighted LSQ): Once cutoff is known, threads cooperatively compute
//            sum(y*w) and sum(w*w) for j < cutoff using strided access pattern.
//            BlockReduce combines partial sums from all threads. Repeat for each feature.
template <typename T, typename D, csize_t N, int BLOCK>
__global__ void inliers_lsq_kernel(ArrayViewND<D, N> fits, ArrayViewND<T, 2> W, ArrayViewND<T, N> y,
                                   ArrayViewND<D, N> errors, ArrayViewND<py::ssize_t, N> indices,
                                   const D * chunk_prefix, csize_t n_features, csize_t n_reduce,
                                   csize_t n_chunks, D lm)
{
    csize_t row = blockIdx.x;
    csize_t tid = threadIdx.x;

    csize_t start = row * n_reduce;

    T * y_ptr = y.data(start);
    csize_t y_stride = y.strides(N - 1) / sizeof(T);

    D * err_ptr = errors.data(start);
    csize_t err_stride = errors.strides(N - 1) / sizeof(D);

    using BlockScan = cub::BlockScan<D, BLOCK>;
    using BlockReduceInt = cub::BlockReduce<int, BLOCK>;
    using BlockReducePair = cub::BlockReduce<SumPair<D>, BLOCK>;

    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ typename BlockReduceInt::TempStorage reduce_int_storage;
    __shared__ typename BlockReducePair::TempStorage reduce_pair_storage;
    __shared__ csize_t cutoff;  // Shared cutoff index (first j where condition fails)

    if (tid == 0) cutoff = n_reduce;  // Initialize to "no cutoff found"
    __syncthreads();

    // Phase A: Find cutoff point by scanning chunks sequentially
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
        int candidate = static_cast<int>(n_reduce);  // Default: no cutoff in this thread
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
            cutoff = static_cast<csize_t>(chunk_cutoff);  // Store cutoff in shared memory
        }
        __syncthreads();
    }

    // Phase B: Compute weighted least squares for each feature using elements j < cutoff
    for (size_t k = 0; k < n_features; ++k)
    {
        // Each thread accumulates partial sums using strided access (tid, tid+BLOCK, tid+2*BLOCK, ...)
        SumPair<D> local = {D(), D()};
        for (csize_t j = tid; j < cutoff; j += BLOCK)
        {
            py::ssize_t idx_j = indices[start + j];
            T w = W.at(k, idx_j);  // Get weight for this feature
            local.sum += static_cast<D>(y_ptr[idx_j * y_stride]) * w;  // Accumulate sum(y*w)
            local.weight += static_cast<D>(w * w);  // Accumulate sum(w*w)
        }

        // Combine partial sums from all threads using custom reduction operator
        SumPair<D> total = BlockReducePair(reduce_pair_storage).Reduce(local, SumPairReduce<D>());
        if (tid == 0)
        {
            // Compute final weighted least squares fit: sum(y*w) / sum(w*w)
            fits[row * n_features + k] = total.weight > D() ? total.sum / total.weight : D();
        }
        __syncthreads();
    }
}

template <typename T, typename D, csize_t N>
array_t<D> inliers_lsq_impl(array_t<D> fits, array_t<T> W, array_t<T> y, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    csize_t n_reduce = W.shape(1);
    csize_t n_features = W.shape(0);
    csize_t n_rows = y.size() / n_reduce;

    if (n_rows) check_equal(
        "fits shape is incompatible with y", fits.shape(), fits.shape() + fits.ndim() - 1, y.shape(), y.shape() + y.ndim() - 1);
    if (y.shape(y.ndim() - 1) != n_reduce)
        throw std::runtime_error("Last dimension of y must match to last dimension of W");
    if (fits.shape(fits.ndim() - 1) != n_features)
        throw std::runtime_error("Last dimension of fits must match to first dimension of W");

    constexpr csize_t block_size = BLOCK_SIZE;
    csize_t n_chunks = (n_reduce + block_size - 1) / block_size;

    DeviceVector<D> chunk_sums(n_rows * n_chunks);
    DeviceVector<D> chunk_prefix(n_rows * n_chunks);

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

    inliers_lsq_kernel<T, D, N, block_size><<<n_rows, block_size>>>(
        cast_to_nd<D, N>(fits.view()), cast_to_nd<T, 2>(W.view()), cast_to_nd<T, N>(y.view()),
        cast_to_nd<D, N>(errors.view()), cast_to_nd<py::ssize_t, N>(indices.view()),
        chunk_prefix.data(), n_features, n_reduce, n_chunks, lm
    );
    handle_cuda_error(cudaGetLastError());

    return fits;
}

template <typename T, typename D>
array_t<D> inliers_lsq(array_t<D> fits, array_t<T> W, array_t<T> y, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    switch (y.ndim())
    {
        case 1: return inliers_lsq_impl<T, D, 1>(fits, W, y, errors, indices, lm);
        case 2: return inliers_lsq_impl<T, D, 2>(fits, W, y, errors, indices, lm);
        case 3: return inliers_lsq_impl<T, D, 3>(fits, W, y, errors, indices, lm);
        case 4: return inliers_lsq_impl<T, D, 4>(fits, W, y, errors, indices, lm);
        case 5: return inliers_lsq_impl<T, D, 5>(fits, W, y, errors, indices, lm);
        case 6: return inliers_lsq_impl<T, D, 6>(fits, W, y, errors, indices, lm);
        case 7: return inliers_lsq_impl<T, D, 7>(fits, W, y, errors, indices, lm);
        default: throw std::runtime_error("Unsupported number of dimensions of y: " + std::to_string(y.ndim()));
    }
}

// Pass 3 for inliers_mean_parallel: Detect cutoff and compute mean (optionally std)
// Grid: n_rows blocks, each with BLOCK=BLOCK_SIZE threads
// Strategy: Similar to inliers_lsq_kernel but simpler:
//   Phase A: Find cutoff using chunk-based scanning (same algorithm as inliers_lsq)
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

    // Phase A: Find cutoff point (identical to inliers_lsq Phase A)
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
array_t<D> inliers_mean_nd(array_t<D> mean, array_t<T> inp, array_t<D> errors, array_t<py::ssize_t> indices, D lm)
{
    csize_t n_reduce = inp.shape(N - 1);
    csize_t n_rows = inp.size() / n_reduce;

    constexpr csize_t block_size = BLOCK_SIZE;
    csize_t n_chunks = (n_reduce + block_size - 1) / block_size;

    DeviceVector<D> chunk_sums(n_rows * n_chunks);
    DeviceVector<D> chunk_prefix(n_rows * n_chunks);

    // Reuse Pass 1 & 2 from inliers_lsq
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
    csize_t n_chunks = (n_reduce + block_size - 1) / block_size;

    DeviceVector<D> chunk_sums(n_rows * n_chunks);
    DeviceVector<D> chunk_prefix(n_rows * n_chunks);

    // Reuse Pass 1 & 2 from inliers_lsq
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

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    // Parallel versions (3-pass with chunked reduction and prefix scan)
    m.def("inliers_mean", &cu::inliers_mean<float, float>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean", &cu::inliers_mean<double, double>, py::arg("mean"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);

    m.def("inliers_mean_std", &cu::inliers_mean_std<float, float>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_mean_std", &cu::inliers_mean_std<double, double>, py::arg("mean"), py::arg("std"), py::arg("inp"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);

    m.def("lsq", &cu::lsq<float, float>, py::arg("fits"), py::arg("W"), py::arg("y"), py::arg("indices"));
    m.def("lsq", &cu::lsq<double, double>, py::arg("fits"), py::arg("W"), py::arg("y"), py::arg("indices"));

    m.def("inliers_lsq", &cu::inliers_lsq<float, float>, py::arg("fits"), py::arg("W"), py::arg("y"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0f);
    m.def("inliers_lsq", &cu::inliers_lsq<double, double>, py::arg("fits"), py::arg("W"), py::arg("y"), py::arg("errors"), py::arg("indices"), py::arg("lm") = 9.0);
}
