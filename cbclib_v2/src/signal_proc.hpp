#ifndef SIGNAL_PROC_
#define SIGNAL_PROC_
#include "array.hpp"

namespace cbclib {

/*----------------------------------------------------------------------------*/
/*------------------------- Bilinear interpolation ---------------------------*/
/*----------------------------------------------------------------------------*/

/* points (integer) follow the convention:      [..., k, j, i], where {i <-> x, j <-> y, k <-> z}
   coordinates (float) follow the convention:   [x, y, z, ...]
 */

template <typename T, typename U>
T bilinear(const array<T> & arr, const std::vector<array<U>> & grid, const std::vector<U> & coord)
{
    std::vector<size_t> lbound, ubound;
    std::vector<T> dx;

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        // liter is GREATER OR EQUAL
        auto liter = std::lower_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // uiter is GREATER
        auto uiter = std::upper_bound(grid[index].begin(), grid[index].end(), coord[index]);
        // lbound is LESS OR EQUAL
        lbound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), uiter) - 1, 0, grid[index].size - 1));
        // rbound is GREATER OR EQUAL
        ubound.push_back(std::clamp<size_t>(std::distance(grid[index].begin(), liter), 0, grid[index].size - 1));
    }

    for (size_t n = 0; n < coord.size(); n++)
    {
        auto index = coord.size() - 1 - n;
        if (lbound[index] != ubound[index])
        {
            dx.push_back((coord[n] - grid[n][lbound[index]]) / (grid[n][ubound[index]] - grid[n][lbound[index]]));
        }
        else dx.push_back(T());
    }

    T out = T();
    std::vector<size_t> point (coord.size());

    // Iterating over a square around coord
    for (size_t i = 0; i < (1ul << coord.size()); i++)
    {
        T factor = 1.0;
        for (size_t n = 0; n < coord.size(); n++)
        {
            // If the index is odd
            if ((i >> n) & 1)
            {
                point[point.size() - 1 - n] = ubound[ubound.size() - 1 - n];
                factor *= dx[n];
            }
            else
            {
                point[point.size() - 1 - n] = lbound[lbound.size() - 1 - n];
                factor *= 1.0 - dx[n];
            }

        }

        if (arr.is_inbound(point)) out += factor * arr.at(point);
        else
        {
            std::ostringstream oss;
            oss << "Invalid index: {";
            std::copy(point.begin(), point.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
            oss << "}";
            throw std::runtime_error(oss.str());
        }
    }

    return out;
}

template <typename T>
py::array_t<T> binterpolate(py::array_t<T, py::array::c_style | py::array::forcecast> inp,
                            std::vector<py::array_t<T, py::array::c_style | py::array::forcecast>> grid,
                            py::array_t<T, py::array::c_style | py::array::forcecast> coords, unsigned threads);

/*----------------------------------------------------------------------------*/
/*---------------------------- Kernel regression -----------------------------*/
/*----------------------------------------------------------------------------*/

template <typename T>
py::array_t<T> kr_predict(py::array_t<T, py::array::c_style | py::array::forcecast> y,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x,
                          py::array_t<T, py::array::c_style | py::array::forcecast> x_hat, T sigma, std::string kernel,
                          std::optional<py::array_t<T, py::array::c_style | py::array::forcecast>> w, unsigned threads);

template <typename InputIt, typename T, typename Axes, class UnaryFunction>
UnaryFunction maxima_nd(InputIt first, InputIt last, UnaryFunction unary_op, const array<T> & arr, const Axes & axes, size_t order)
{
    // First element can't be a maximum
    auto iter = (first != last) ? std::next(first) : first;
    last = (iter != last) ? std::prev(last) : last;

    while (iter != last)
    {
        if (*std::prev(iter) < *iter)
        {
            // ahead can be last
            auto ahead = std::next(iter);

            while (ahead != last && *ahead == *iter) ++ahead;

            if (*ahead < *iter)
            {
                auto index = arr.index(iter);

                size_t n = 1;
                for (; n < axes.size(); n++)
                {
                    auto coord = arr.index_along_dim(index, axes[n]);
                    if (coord > 1 && coord < arr.shape[axes[n]] - 1)
                    {
                        if (arr[index - arr.stride(axes[n])] < *iter && arr[index + arr.stride(axes[n])] < *iter)
                        {
                            continue;
                        }
                    }

                    break;
                }

                if (n >= order) unary_op(index);

                // Skip samples that can't be maximum, check if it's not last
                if (ahead != last) iter = ahead;
            }
        }

        iter = std::next(iter);
    }

    return unary_op;
}

template <typename T, typename U>
py::array_t<size_t> local_maxima(py::array_t<T, py::array::c_style | py::array::forcecast> inp, U axis, unsigned threads);

}

#endif
