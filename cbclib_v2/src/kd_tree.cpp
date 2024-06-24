#include "kd_tree.hpp"

namespace cbclib {

template <typename T, typename I>
KDTree<array<T>, I> build_tree(py::array_t<T> database)
{
    auto dbuf = database.request();
    size_t ndim = dbuf.shape[dbuf.ndim - 1];
    size_t dsize = dbuf.size / ndim;

    array<T> darr {dbuf};

    std::vector<std::pair<array<T>, I>> items;
    for (size_t i = 0; i < dsize; i++)
    {
        items.emplace_back(array<T>(ndim, darr.data() + i * ndim), i);
    }
    return {items.begin(), items.end(), ndim};
}

template <typename T, typename I>
void declare_kd_tree(py::module & m, const std::string & typestr)
{
    py::class_<KDTree<array<T>, I>>(m, (std::string("KDTree") + typestr).c_str())
        .def(py::init([](py::array_t<T> database)
            {
                return build_tree<T, I>(database);
            }), py::arg("database"))
        .def_property("high", [](const KDTree<array<T>, I> & tree)
            {
                return tree.rectangle().high;
            }, nullptr)
        .def_property("low", [](const KDTree<array<T>, I> & tree)
            {
                return tree.rectangle().low;
            }, nullptr)
        .def_readonly("ndim", &KDTree<array<T>, I>::ndim)
        .def_property("size", [](const KDTree<array<T>, I> & tree)
            {
                return std::distance(tree.begin(), tree.end());
            }, nullptr)
        .def("find_nearest", [](const KDTree<array<T>, I> & tree, py::array_t<T> query, size_t k, unsigned threads)
            {
                array<T> qarr {query.request()};
                size_t ndim = qarr.shape[qarr.ndim - 1];
                size_t qsize = qarr.size / ndim;
                if (ndim != tree.ndim)
                    throw std::invalid_argument("query has invalid number of dimensions (" + std::to_string(ndim) + ")");

                std::vector<size_t> shape {qarr.shape.begin(), std::prev(qarr.shape.end())};
                shape.push_back(k);
                py::array_t<I> result {shape};
                array<I> rarr {result.request()};
                py::array_t<double> dist {shape};
                array<double> darr {dist.request()};

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads) schedule(dynamic,20)
                for (size_t i = 0; i < qsize; i++)
                {
                    auto stack = tree.find_k_nearest(array<T>(ndim, qarr.data() + i * ndim), k);
                    size_t j = 0;
                    for (auto [iter, dist]: stack)
                    {
                        rarr[k * i + j] = iter->data(); darr[k * i + j] = std::sqrt(dist);
                        j++;
                    }
                }

                py::gil_scoped_acquire acquire;

                return std::make_tuple(dist, result);
            }, py::arg("query"), py::arg("k")=1, py::arg("num_threads")=1)
        .def("find_range", [](const KDTree<array<T>, I> & tree, py::array_t<T> query, T range, unsigned threads)
            {
                array<T> qarr {query.request()};
                size_t ndim = qarr.shape[qarr.ndim - 1];
                size_t qsize = qarr.size / ndim;
                if (ndim != tree.ndim)
                    throw std::invalid_argument("query has invalid number of dimensions (" + std::to_string(ndim) + ")");

                std::vector<std::vector<I>> result;

                auto get_index = [](const std::pair<typename KDTree<array<T>, I>::const_iterator, T> & item)
                {
                    return item.first->data();
                };

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads)
                for (size_t i = 0; i < qsize; i++)
                {
                    auto stack = tree.find_range(array<T>(ndim, qarr.data() + i * ndim), range * range);
                    auto & neighbours = result.emplace_back();
                    std::transform(stack.begin(), stack.end(), std::back_inserter(neighbours), get_index);
                }

                py::gil_scoped_acquire acquire;

                return result;
            }, py::arg("query"), py::arg("range"), py::arg("num_threads")=1);
}

}

PYBIND11_MODULE(kd_tree, m)
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

    declare_kd_tree<float, long>(m, "Float");
    declare_kd_tree<double, long>(m, "Double");
    declare_kd_tree<long, long>(m, "Int");

    m.def("build_tree", &build_tree<float, long>, py::arg("database"));
    m.def("build_tree", &build_tree<double, long>, py::arg("database"));
    m.def("build_tree", &build_tree<long, long>, py::arg("database"));


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
