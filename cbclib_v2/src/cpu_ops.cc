#include "image_proc.hpp"

namespace cbclib_jax {

using namespace cbclib;

template <typename T>
pybind11::capsule EncapsulateFunction(T * fn)
{
    return pybind11::capsule(std::bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
T line_value(BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter, T width)
{
    T length = amplitude(liter.tau);
    auto r1 = liter.error / length, r2 = eiter.error / length;

    if (r2 < T())
    {
        return std::sqrt(r1 * r1 + r2 * r2) / width;
    }
    else if (r2 > length)
    {
        return std::sqrt(r1 * r1 + (r2 - length) * (r2 - length)) / width;
    }
    else
    {
        return std::abs(r1) / width;
    }
}

template <typename T, typename I, class Func>
void draw_bresenham(const point_t & ubound, I frame, I index, const Line<T> & line, T width, Func && draw_pixel)
{
    width = std::clamp(width, T(), std::numeric_limits<T>::max());

    auto get_val = [width](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        return line_value(liter, eiter, width);
    };

    auto draw = [&get_val, &draw_pixel, frame, index](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        std::forward<Func>(draw_pixel)(liter.point, frame, index, get_val(liter, eiter));
    };

    draw_bresenham_func(ubound, line, width, draw);
}

template <typename T, typename I>
void line_distances_fwd(void * out_tuple, const void ** input)
{
    // Parse the inputs
    const std::int64_t nframes = *reinterpret_cast<const std::int64_t *>(input[0]);
    const std::int64_t Y = *reinterpret_cast<const std::int64_t *>(input[1]);
    const std::int64_t X = *reinterpret_cast<const std::int64_t *>(input[2]);
    const std::int64_t nlines = *reinterpret_cast<const std::int64_t *>(input[3]);
    const std::int64_t num_devices = *reinterpret_cast<const std::int64_t *>(input[4]);
    const T * lines = reinterpret_cast<const T *>(input[5]);
    const I * idxs = reinterpret_cast<const I *>(input[6]);

    // The output is stored as a list of pointers since we have multiple outputs
    void ** out = reinterpret_cast<void **>(out_tuple);
    T * image_out = reinterpret_cast<T *>(out[0]);
    I * line_idxs = reinterpret_cast<I *>(out[1]);

    std::vector<std::int64_t> shape {nframes, Y, X};
    array<T> image_arr {shape, image_out};
    array<I> lidxs_arr {shape, line_idxs};

    auto size = image_arr.size;

    std::fill_n(image_out, size, T(1.0));
    std::fill_n(line_idxs, size, I());

    auto ubound = get_ubound(shape);

    thread_exception e;

    #pragma omp parallel num_threads(std::min(num_devices, nlines / 100 + 1))
    {
        detail::ImageBuffer<std::tuple<I, I, T>> buffer (shape);
        auto draw_pixel = [&buffer](const point_t & pt, I frame, I index, T val)
        {
            detail::draw_pixel(buffer, pt, frame, index, val);
        };
        auto write = [&buffer, &image_arr, &lidxs_arr](const std::tuple<I, I, T> & value)
        {
            if (std::get<2>(value) < image_arr[std::get<0>(value)])
            {
                image_arr[std::get<0>(value)] = std::get<2>(value);
                lidxs_arr[std::get<0>(value)] = std::get<1>(value);
            }
        };

        #pragma omp for nowait schedule(dynamic,100)
        for (I i = 0; i < nlines; i++)
        {
            e.run([&]()
            {
                if (idxs[i] < nframes)
                {
                    auto line = std::make_from_tuple<Line<T>>(to_tuple<4>(lines, 5 * i));
                    draw_bresenham(ubound, idxs[i], i + 1, line, lines[5 * i + 4], draw_pixel);
                }
            });
        }

        #pragma omp critical
        std::for_each(buffer.begin(), buffer.end(), write);
    }

    e.rethrow();
}

template <typename T>
void line_value_vjp(std::array<T, 5> & ct_line, T ct_pt, const Line<T> & line, T width, BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
{
    T length = amplitude(line.tau);
    auto mag = length * length;
    auto r1 = liter.error / length, r2 = eiter.error / length;
    auto v0 = liter.point - line.pt0, v1 = liter.point - line.pt1;

    T ct_r1, ct_r2, val;
    if (r2 < T())
    {
        val = std::sqrt(r1 * r1 + r2 * r2);
        ct_r1 = ct_pt * r1 / (val * width);
        ct_r2 = ct_pt * r2 / (val * width);
    }
    else if (r2 > length)
    {
        val = std::sqrt(r1 * r1 + (r2 - length) * (r2 - length));
        ct_r1 = ct_pt * r1 / (val * width);
        ct_r2 = ct_pt * (r2 - length) / (val * width);

        ct_line[0] += ct_r2 * line.tau.x() / length;
        ct_line[1] += ct_r2 * line.tau.y() / length;
        ct_line[2] -= ct_r2 * line.tau.x() / length;
        ct_line[3] -= ct_r2 * line.tau.y() / length;
    }
    else
    {
        val = std::abs(r1);
        ct_r1 = ct_pt * detail::signum(r1) / width;
        ct_r2 = T();
    }

    ct_line[0] += ct_r1 * (line.tau.x() * r1 / mag + v1.y() / length) +
                  ct_r2 * (line.tau.x() * r2 / mag - (line.tau.x() + v0.x()) / length);
    ct_line[1] += ct_r1 * (line.tau.y() * r1 / mag - v1.x() / length) +
                  ct_r2 * (line.tau.y() * r2 / mag - (line.tau.y() + v0.y()) / length);
    ct_line[2] += ct_r1 * (-line.tau.x() * r1 / mag - v0.y() / length) +
                  ct_r2 * (-line.tau.x() * r2 / mag + v0.x() / length);
    ct_line[3] += ct_r1 * (-line.tau.y() * r1 / mag + v0.x() / length) +
                  ct_r2 * (-line.tau.y() * r2 / mag + v0.y() / length);
    ct_line[4] -= ct_pt * val / (width * width);
}

template <typename T, typename I, class Func>
auto draw_bresenham_vjp(const point_t & ubound, I frame, I index, const Line<T> & line, T width, Func && get_pixel)
{
    width = std::clamp(width, T(), std::numeric_limits<T>::max());
    std::array<T, 5> ct_line {};

    auto propagate = [&ct_line, &line, width](T ct_pt, BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        line_value_vjp(ct_line, ct_pt, line, width, liter, eiter);
    };

    auto get_cotangent = [&propagate, &get_pixel, frame, index](BresenhamIterator<T, true> liter, BresenhamIterator<T, true> eiter)
    {
        auto [ct_pt, flag] = std::forward<Func>(get_pixel)(liter.point, frame, index);
        if (flag) propagate(ct_pt, liter, eiter);
    };

    draw_bresenham_func(ubound, line, width, get_cotangent);

    return to_tuple(ct_line);
}

template <typename T, typename I>
void line_distances_bwd(void * out, const void ** input)
{
    // Parse the inputs
    const std::int64_t nframes = *reinterpret_cast<const std::int64_t *>(input[0]);
    const std::int64_t Y = *reinterpret_cast<const std::int64_t *>(input[1]);
    const std::int64_t X = *reinterpret_cast<const std::int64_t *>(input[2]);
    const std::int64_t nlines = *reinterpret_cast<const std::int64_t *>(input[3]);
    const std::int64_t num_devices = *reinterpret_cast<const std::int64_t *>(input[4]);
    const T * cotangents = reinterpret_cast<const T *>(input[5]);
    const I * line_idxs = reinterpret_cast<const I *>(input[6]);
    const T * lines = reinterpret_cast<const T *>(input[7]);
    const I * idxs = reinterpret_cast<const I *>(input[8]);

    // Parse the output
    T * ct_lines = reinterpret_cast<T *>(out);

    std::vector<std::int64_t> shape {nframes, Y, X};
    array<const T> ct_arr {shape, cotangents};
    array<const I> lidxs_arr {shape, line_idxs};

    auto ubound = get_ubound(shape);

    auto get_pixel = [&ct_arr, &lidxs_arr](const point_t & pt, I frame, I index)
    {
        using integer_type = typename point_t::value_type;
        std::array<integer_type, 3> coord {static_cast<integer_type>(frame), pt.y(), pt.x()};
        if (ct_arr.is_inbound(coord)) return std::make_pair(ct_arr.at(coord), lidxs_arr.at(coord) == index);
        return std::make_pair(T(), false);
    };

    thread_exception e;

    #pragma omp parallel for num_threads(std::min(num_devices, nlines / 100)) schedule(dynamic,100)
    for (I i = 0; i < nlines; i++)
    {
        e.run([&]()
        {
            if (idxs[i] < nframes)
            {
                auto line = std::make_from_tuple<Line<T>>(to_tuple<4>(lines, 5 * i));
                to_tie<5>(ct_lines, 5 * i) = draw_bresenham_vjp(ubound, idxs[i], i + 1, line, lines[5 * i + 4], get_pixel);
            }
        });
    }

    e.rethrow();
}

pybind11::dict Registrations()
{
    pybind11::dict dict;
    dict["line_distances_fwd_f32_i32"] = EncapsulateFunction(line_distances_fwd<float, int>);
    dict["line_distances_fwd_f64_i32"] = EncapsulateFunction(line_distances_fwd<double, int>);
    dict["line_distances_fwd_f32_i64"] = EncapsulateFunction(line_distances_fwd<float, long>);
    dict["line_distances_fwd_f64_i64"] = EncapsulateFunction(line_distances_fwd<double, long>);
    dict["line_distances_bwd_f32_i32"] = EncapsulateFunction(line_distances_bwd<float, int>);
    dict["line_distances_bwd_f64_i32"] = EncapsulateFunction(line_distances_bwd<double, int>);
    dict["line_distances_bwd_f32_i64"] = EncapsulateFunction(line_distances_bwd<float, long>);
    dict["line_distances_bwd_f64_i64"] = EncapsulateFunction(line_distances_bwd<double, long>);
    return dict;
}

PYBIND11_MODULE(cpu_ops, m)
{
    m.def("registrations", &Registrations);
}

}