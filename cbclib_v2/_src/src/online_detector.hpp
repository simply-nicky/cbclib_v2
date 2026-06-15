#ifndef ONLINE_DETECTOR_
#define ONLINE_DETECTOR_
#include "numpy.hpp"

namespace cbclib {

template <typename R, typename I>
struct PanelGeometry
{
    I offset = 0;
    std::array<I, 4> bounds = {0, 0, 0, 0}; // fs_min, fs_max, ss_min, ss_max
    std::array<I, 2> shape = {0, 0};        // ss_size, fs_size
    std::array<I, 2> stride = {0, 0};       // ss_stride, fs_stride
    std::array<R, 2> corner = {R(), R()};   // x, y
    std::array<R, 3> ss = {R(), R(), R()};  // ss_x, ss_y, ss_z
    std::array<R, 3> fs = {R(), R(), R()};  // fs_x, fs_y, fs_z

    I size() const
    {
        return shape[0] * shape[1];
    }

    R lab_x(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return corner[0] + shift + ss_index * ss[0] + fs_index * fs[0];
    }

    R lab_y(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return corner[1] + shift + ss_index * ss[1] + fs_index * fs[1];
    }

    R lab_z(I ss_index, I fs_index, bool half_pixel_shift) const
    {
        R shift = half_pixel_shift ? R(0.5) : R();
        return shift + ss_index * ss[2] + fs_index * fs[2];
    }
};

template <typename R, typename I>
struct DetectorGeometry
{
    std::vector<PanelGeometry<R, I>> panels;
    std::vector<I> panel_offsets;
    std::array<I, 2> shape = {0, 0};
    std::array<R, 4> bounds = {R(), R(), R(), R()}; // x_min, x_max, y_min, y_max
    bool half_pixel_shift = true;

    R radius(const PanelGeometry<R, I> & panel, const std::tuple<R, R> & center,
             I ss, I fs) const
    {
        R dx = panel.lab_x(ss, fs, half_pixel_shift) - bounds[0] - std::get<0>(center);
        R dy = panel.lab_y(ss, fs, half_pixel_shift) - bounds[2] - std::get<1>(center);
        return std::sqrt(dx * dx + dy * dy);
    }

    R max_radius(const std::tuple<R, R> & center) const
    {
        R width = bounds[1] - bounds[0];
        R height = bounds[3] - bounds[2];
        std::array<std::tuple<R, R>, 4> corners {{
            std::make_tuple(R(), R()),
            std::make_tuple(width, R()),
            std::make_tuple(R(), height),
            std::make_tuple(width, height),
        }};

        R result = R();
        for (auto corner : corners)
        {
            R dx = std::get<0>(corner) - std::get<0>(center);
            R dy = std::get<1>(corner) - std::get<1>(center);
            result = std::max(result, std::sqrt(dx * dx + dy * dy));
        }
        return result;
    }

    I ndim() const
    {
        return 2;
    }

    I size() const
    {
        return shape[0] * shape[1];
    }

    I panel_size() const
    {
        return panel_offsets.back();
    }

    void validate() const
    {
        if (panels.empty()) throw std::invalid_argument("detector geometry must not be empty");
        if (shape[0] <= 0 || shape[1] <= 0)
            throw std::invalid_argument("detector frame shape must not be empty");

        auto size = this->size();
        if (panel_offsets.size() != panels.size() + 1 || panel_offsets.front() != 0)
        {
            throw std::invalid_argument("detector panel offsets mismatch");
        }

        for (size_t index = 0; index < panels.size(); ++index)
        {
            const auto & panel = panels[index];
            if (panel_offsets[index + 1] - panel_offsets[index] != panel.size())
            {
                throw std::invalid_argument("detector panel offset size mismatch");
            }
            I panel_end = panel.offset + (panel.shape[0] - 1) * panel.stride[0] + (panel.shape[1] - 1) * panel.stride[1] + 1;
            if (panel_end > size)
            {
                throw std::invalid_argument("panel geometry exceeds detector frame size");
            }
        }
    }
};

using PyDetectorGeometry = DetectorGeometry<double, py::ssize_t>;

template <typename R, typename I>
DetectorGeometry<R, I> cast_detector_geometry(const PyDetectorGeometry & geometry)
{
    DetectorGeometry<R, I> result;
    result.panels.reserve(geometry.panels.size());
    result.panel_offsets.reserve(geometry.panel_offsets.size());
    result.shape = {static_cast<I>(geometry.shape[0]), static_cast<I>(geometry.shape[1])};
    result.bounds = {static_cast<R>(geometry.bounds[0]), static_cast<R>(geometry.bounds[1]),
                     static_cast<R>(geometry.bounds[2]), static_cast<R>(geometry.bounds[3])};
    result.half_pixel_shift = geometry.half_pixel_shift;

    for (const auto & panel : geometry.panels)
    {
        PanelGeometry<R, I> cast_panel;
        cast_panel.offset = static_cast<I>(panel.offset);
        cast_panel.bounds = {static_cast<I>(panel.bounds[0]), static_cast<I>(panel.bounds[1]),
                             static_cast<I>(panel.bounds[2]), static_cast<I>(panel.bounds[3])};
        cast_panel.shape = {static_cast<I>(panel.shape[0]), static_cast<I>(panel.shape[1])};
        cast_panel.stride = {static_cast<I>(panel.stride[0]), static_cast<I>(panel.stride[1])};
        cast_panel.corner = {static_cast<R>(panel.corner[0]), static_cast<R>(panel.corner[1])};
        cast_panel.ss = {static_cast<R>(panel.ss[0]), static_cast<R>(panel.ss[1]),
                         static_cast<R>(panel.ss[2])};
        cast_panel.fs = {static_cast<R>(panel.fs[0]), static_cast<R>(panel.fs[1]),
                         static_cast<R>(panel.fs[2])};
        result.panels.push_back(cast_panel);
    }

    for (auto offset : geometry.panel_offsets)
    {
        result.panel_offsets.push_back(static_cast<I>(offset));
    }
    return result;
}

} // namespace cbclib

namespace pybind11::detail {

template <typename R, typename I>
struct type_caster<cbclib::DetectorGeometry<R, I>>
{
public:
    using Geometry = cbclib::DetectorGeometry<R, I>;
    using Panel = cbclib::PanelGeometry<R, I>;
    PYBIND11_TYPE_CASTER(Geometry, const_name("DetectorGeometry"));

    bool load(handle src, bool)
    {
        try
        {
            pybind11::object detector = pybind11::reinterpret_borrow<pybind11::object>(src);
            pybind11::dict protocol = detector.attr("__geometry_protocol__")()
                                           .cast<pybind11::dict>();
            value = Geometry();
            auto shape = protocol["shape"].cast<std::vector<I>>();
            if (shape.size() != 2)
            {
                return false;
            }
            value.shape = {shape[0], shape[1]};

            auto bounds = protocol["bounds"].cast<std::tuple<double, double, double, double>>();
            value.bounds[0] = static_cast<R>(std::get<0>(bounds));
            value.bounds[1] = static_cast<R>(std::get<1>(bounds));
            value.bounds[2] = static_cast<R>(std::get<2>(bounds));
            value.bounds[3] = static_cast<R>(std::get<3>(bounds));

            auto panels = protocol["panels"].cast<pybind11::list>();
            for (auto item : panels)
            {
                value.panels.push_back(panel_from_protocol(item.cast<pybind11::dict>(),
                                                           value.shape));
            }
            value.panel_offsets.reserve(value.panels.size() + 1);
            value.panel_offsets.push_back(0);
            for (const auto & panel : value.panels)
            {
                value.panel_offsets.push_back(value.panel_offsets.back() + panel.size());
            }
        }
        catch (const pybind11::error_already_set &)
        {
            return false;
        }
        catch (const std::exception &)
        {
            return false;
        }
        return true;
    }

private:
    static Panel panel_from_protocol(const pybind11::dict & protocol,
                                     const std::array<I, 2> & shape)
    {
        Panel panel;
        auto region = protocol["region"].cast<std::tuple<I, I, I, I>>();
        panel.bounds[0] = std::get<0>(region);
        panel.bounds[1] = std::get<1>(region);
        panel.bounds[2] = std::get<2>(region);
        panel.bounds[3] = std::get<3>(region);
        panel.offset = panel.bounds[2] * shape[1] + panel.bounds[0];
        panel.shape = {panel.bounds[3] - panel.bounds[2] + 1,
                       panel.bounds[1] - panel.bounds[0] + 1};
        panel.stride = {shape[1], 1};

        auto corner = protocol["corner"].cast<std::tuple<double, double>>();
        panel.corner = {static_cast<R>(std::get<0>(corner)),
                        static_cast<R>(std::get<1>(corner))};

        auto ss = protocol["ss"].cast<std::tuple<double, double, double>>();
        panel.ss = {static_cast<R>(std::get<0>(ss)),
                    static_cast<R>(std::get<1>(ss)),
                    static_cast<R>(std::get<2>(ss))};

        auto fs = protocol["fs"].cast<std::tuple<double, double, double>>();
        panel.fs = {static_cast<R>(std::get<0>(fs)),
                    static_cast<R>(std::get<1>(fs)),
                    static_cast<R>(std::get<2>(fs))};
        return panel;
    }
};

} // namespace pybind11::detail

#endif // ONLINE_DETECTOR_
