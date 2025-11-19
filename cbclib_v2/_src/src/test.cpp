#include "array.hpp"

PYBIND11_MODULE(test, m)
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

    py::class_<array<double>>(m, "ArrayView")
        .def(py::init([](py::array_t<double> arr){return array<double>(arr.request());}), py::arg("array"))
        .def_property_readonly("itemsize", [](const array<double> & arr){return arr.itemsize();})
        .def_property_readonly("ndim", [](const array<double> & arr){return arr.ndim();})
        .def_property_readonly("size", [](const array<double> & arr){return arr.size();})
        .def_property_readonly("shape", [](const array<double> & arr){return arr.shape();})
        .def_property_readonly("strides", [](const array<double> & arr){return arr.strides();})
        .def("__getitem__", [](const array<double> & arr, size_t index) { return arr[index]; })
        .def("__setitem__", [](array<double> & arr, size_t index, double value) { arr[index] = value; })
        .def("coord_at", [](const array<double> & arr, size_t index)
        {
            std::vector<size_t> coord(arr.ndim());
            arr.coord_at(coord.begin(), index);
            return coord;
        }, py::arg("index"))
        .def("at", [](const array<double> & arr, py::iterable coord)
        {
            std::vector<size_t> coord_vec;
            for (auto item : coord) coord_vec.push_back(py::cast<size_t>(item));
            return arr.at(coord_vec);
        }, py::arg("coord"))
        .def("at", [](const array<double> & arr, size_t x)
        {
            if (arr.ndim() != 1)
                throw std::runtime_error("at with 1 argument requires a 1D array");
            return arr.at(x);
        }, py::arg("x"))
        .def("at", [](const array<double> & arr, size_t y, size_t x)
        {
            if (arr.ndim() != 2)
                throw std::runtime_error("at with 2 arguments requires a 2D array");
            return arr.at(y, x);
        }, py::arg("y"), py::arg("x"))
        .def("at", [](const array<double> & arr, size_t z, size_t y, size_t x)
        {
            if (arr.ndim() != 3)
                throw std::runtime_error("at with 3 arguments requires a 3D array");
            return arr.at(z, y, x);
        }, py::arg("z"), py::arg("y"), py::arg("x"))
        .def("index_at", [](const array<double> & arr, py::iterable coord)
        {
            std::vector<size_t> coord_vec;
            for (auto item : coord) coord_vec.push_back(py::cast<size_t>(item));
            return arr.index_at(coord_vec);
        }, py::arg("coord"))
        .def("index_at", [](const array<double> & arr, size_t x)
        {
            if (arr.ndim() != 1)
                throw std::runtime_error("index_at with 1 argument requires a 1D array");
            return arr.index_at(x);
        }, py::arg("x"))
        .def("index_at", [](const array<double> & arr, size_t y, size_t x)
        {
            if (arr.ndim() != 2)
                throw std::runtime_error("index_at with 2 arguments requires a 2D array");
            return arr.index_at(y, x);
        }, py::arg("y"), py::arg("x"))
        .def("index_at", [](const array<double> & arr, size_t z, size_t y, size_t x)
        {
            if (arr.ndim() != 3)
                throw std::runtime_error("index_at with 3 arguments requires a 3D array");
            return arr.index_at(z, y, x);
        }, py::arg("z"), py::arg("y"), py::arg("x"));

    py::class_<rectangle_range<std::vector<size_t>>>(m, "RectangleRange")
        .def(py::init<const std::vector<size_t> &>(), py::arg("shape"))
        .def("size", &rectangle_range<std::vector<size_t>>::size)
        .def("__iter__", [](const rectangle_range<std::vector<size_t>> & rr)
        {
            return py::make_iterator(rr.begin(), rr.end());
        }, py::keep_alive<0, 1>());

}
