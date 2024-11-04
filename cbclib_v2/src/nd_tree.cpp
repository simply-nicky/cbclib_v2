#include "nd_tree.hpp"

namespace cbclib {

template <typename T, typename I, size_t N>
NDTree<array<T>, I, N> build_nd_tree(py::array_t<T> database, std::array<T, 2 * N> box)
{
    auto dbuf = database.request();
    check_dimensions("database", dbuf.ndim - 1, dbuf.shape, 2 * N);
    size_t dsize = dbuf.size / dbuf.shape[dbuf.ndim - 1];

    auto get_box = [](const array<T> & elem)
    {
        PointND<T, N> min, max;
        for (size_t i = 0; i < N; i++) {min[i] = elem[i]; max[i] = elem[N + i];}
        return BoxND<T, N>{std::move(min), std::move(max)};
    };

    PointND<T, N> min, max;
    for (size_t i = 0; i < N; i++) {min[i] = box[2 * (N - 1 - i)]; max[i] = box[2 * (N - 1 - i) + 1];}

    NDTree<array<T>, I, N> tree (get_box, BoxND<T, N>{std::move(min), std::move(max)});

    array<T> darr {dbuf};

    for (size_t i = 0; i < dsize; i++)
    {
        tree.insert(std::make_pair(array<T>(2 * N, darr.data() + 2 * N * i), i));
    }

    return tree;
}

template <typename T, typename I, size_t N>
void declare_nd_tree(py::module & m, const std::string & class_str)
{
    py::class_<NDTree<array<T>, I, N>>(m, class_str.c_str())
        .def(py::init([](py::array_t<T> database, std::array<T, 2 * N> box)
        {
            return build_nd_tree<T, I, N>(database, box);
        }), py::arg("database"), py::arg("box"))
        .def_property("box", [](const NDTree<array<T>, I, N> & tree)
        {
            return tree.box().to_array();
        }, nullptr)
        .def_property("size", [](const NDTree<array<T>, I, N> & tree)
        {
            return std::distance(tree.begin(), tree.end());
        }, nullptr)
        .def("find_k_nearest", [](const NDTree<array<T>, I, N> & tree, py::array_t<T> query, size_t k, unsigned threads)
            {
                array<T> qarr {query.request()};
                check_dimensions("query", qarr.ndim - 1, qarr.shape, 2 * N);
                size_t qsize = qarr.size / (2 * N);

                std::vector<size_t> shape {qarr.shape.begin(), std::prev(qarr.shape.end())};
                shape.push_back(k);

                py::array_t<I> result {shape};
                array<I> rarr {result.request()};
                fill_array(result, I(-1));

                py::array_t<double> dist {shape};
                array<double> darr {dist.request()};
                fill_array(dist, 0.0);

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads) schedule(dynamic,20)
                for (size_t i = 0; i < qsize; i++)
                {
                    auto stack = tree.find_k_nearest(tree.to_box(array<T>(2 * N, qarr.data() + 2 * N * i)), k);
                    size_t j = 0;

                    for (auto [iter, dist] : stack)
                    {
                        rarr[k * i + j] = tree.element(iter).second;
                        darr[k * i + j] = std::sqrt(dist);
                        j++;
                    }
                }

                py::gil_scoped_acquire acquire;

                return std::make_tuple(dist, result);
            }, py::arg("query"), py::arg("k")=1, py::arg("num_threads")=1)
        .def("find_range", [](const NDTree<array<T>, I, N> & tree, py::array_t<T> query, T range, unsigned threads)
            {
                array<T> qarr {query.request()};
                check_dimensions("query", qarr.ndim - 1, qarr.shape, 2 * N);
                size_t qsize = qarr.size / (2 * N);

                std::vector<std::vector<double>> dist;
                std::vector<std::vector<I>> result;

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads)
                for (size_t i = 0; i < qsize; i++)
                {
                    auto stack = tree.find_range(tree.to_box(array<T>(2 * N, qarr.data() + 2 * N * i)), range * range);

                    auto & rvec = result.emplace_back();
                    auto & dvec = dist.emplace_back();

                    for (auto [iter, dist] : stack)
                    {
                        rvec.push_back(tree.element(iter).second);
                        dvec.push_back(std::sqrt(dist));
                    }
                }

                py::gil_scoped_acquire acquire;

                return std::make_tuple(dist, result);
            }, py::arg("query"), py::arg("range"), py::arg("num_threads")=1);
}

template <typename T, typename I, size_t N>
NDStack<array<T>, I, N> build_nd_stack(py::array_t<T> database, py::array_t<I> indices, std::array<T, 2 * N> box)
{
    auto dbuf = database.request(), ibuf = indices.request();
    check_equal("database shape is incompatible with indices",
                dbuf.shape.begin(), std::prev(dbuf.shape.end()),
                ibuf.shape.begin(), ibuf.shape.end());
    check_dimensions("database", dbuf.ndim - 1, dbuf.shape, 2 * N);
    size_t dsize = dbuf.size / dbuf.shape[dbuf.ndim - 1];

    auto get_box = [](const array<T> & elem)
    {
        PointND<T, N> min, max;
        for (size_t i = 0; i < N; i++) {min[i] = elem[i]; max[i] = elem[N + i];}
        return BoxND<T, N>{std::move(min), std::move(max)};
    };
    NDStack<array<T>, I, N> stack {};

    array<T> darr {dbuf};
    array<I> iarr {ibuf};

    PointND<T, N> min, max;
    for (size_t i = 0; i < N; i++) {min[i] = box[2 * (N - 1 - i)]; max[i] = box[2 * (N - 1 - i) + 1];}
    BoxND<T, N> roi {std::move(min), std::move(max)};

    for (size_t i = 0; i < dsize; i++)
    {
        auto [iter, is_inserted] = stack.trees.try_emplace(iarr[i], NDTree<array<T>, I, N>(get_box, roi));
        iter->second.insert(std::make_pair(array<T>(2 * N, darr.data() + 2 * N * i), i));
    }

    return stack;
}

template <typename T, typename I, size_t N>
void declare_nd_stack(py::module & m, const std::string & class_str)
{
    py::class_<NDStack<array<T>, I, N>>(m, class_str.c_str())
        .def(py::init([](py::array_t<T> database, py::array_t<I> indices, std::array<T, 2 * N> box)
            {
                return build_nd_stack<T, I, N>(database, indices, box);
            }), py::arg("database"), py::arg("indices"), py::arg("box"))
        .def_property("trees", [](const NDStack<array<T>, I, N> & stack)
            {
                return stack.trees;
            }, nullptr)
        .def("find_k_nearest", [](const NDStack<array<T>, I, N> & stack, py::array_t<T> query, py::array_t<I> indices, size_t k, unsigned threads)
            {
                array<T> qarr {query.request()};
                array<I> iarr {indices.request()};
                check_equal("query shape is incompatible with indices",
                            qarr.shape.begin(), std::prev(qarr.shape.end()),
                            iarr.shape.begin(), iarr.shape.end());
                check_dimensions("query", qarr.ndim - 1, qarr.shape, 2 * N);
                size_t qsize = qarr.size / (2 * N);

                std::vector<size_t> shape {qarr.shape.begin(), std::prev(qarr.shape.end())};
                shape.push_back(k);

                py::array_t<I> results {shape};
                array<I> rarr {results.request()};
                fill_array(results, I(-1));

                py::array_t<double> dist {shape};
                array<double> darr {dist.request()};
                fill_array(dist, 0.0);

                thread_exception e;

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads) schedule(dynamic,20)
                for (size_t i = 0; i < qsize; i++)
                {
                    e.run([&]()
                    {
                        const auto & tree = stack.trees.at(iarr[i]);
                        auto result = tree.find_k_nearest(tree.to_box(array<T>(2 * N, qarr.data() + 2 * N * i)), k);
                        size_t j = 0;

                        for (auto [iter, dist] : result)
                        {
                            rarr[k * i + j] = tree.element(iter).second;
                            darr[k * i + j] = std::sqrt(dist);
                            j++;
                        }
                    });
                }

                py::gil_scoped_acquire acquire;

                e.rethrow();

                return std::make_tuple(dist, results);
            }, py::arg("query"), py::arg("indices"), py::arg("k")=1, py::arg("num_threads")=1)
        .def("find_range", [](const NDStack<array<T>, I, N> & stack, py::array_t<T> query, py::array_t<I> indices, T range, unsigned threads)
            {
                array<T> qarr {query.request()};
                array<I> iarr {indices.request()};
                check_equal("query shape is incompatible with indices",
                            qarr.shape.begin(), std::prev(qarr.shape.end()),
                            iarr.shape.begin(), iarr.shape.end());
                check_dimensions("query", qarr.ndim - 1, qarr.shape, 2 * N);
                size_t qsize = qarr.size / (2 * N);

                std::vector<std::vector<double>> dist;
                std::vector<std::vector<I>> results;

                thread_exception e;

                py::gil_scoped_release release;

                #pragma omp parallel for num_threads(threads)
                for (size_t i = 0; i < qsize; i++)
                {
                    e.run([&]()
                    {
                        const auto & tree = stack.trees.at(iarr[i]);
                        auto result = tree.find_range(tree.to_box(array<T>(2 * N, qarr.data() + 2 * N * i)), range * range);

                        auto & rvec = results.emplace_back();
                        auto & dvec = dist.emplace_back();

                        for (auto [iter, dist] : result)
                        {
                            rvec.push_back(tree.element(iter).second);
                            dvec.push_back(std::sqrt(dist));
                        }
                    });
                }

                py::gil_scoped_acquire acquire;

                e.rethrow();

                return std::make_tuple(dist, results);
            }, py::arg("query"), py::arg("indices"), py::arg("range"), py::arg("num_threads")=1);
}

}

PYBIND11_MODULE(nd_tree, m)
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

    declare_nd_tree<float, long, 2>(m, "QuadTreeFloat");
    declare_nd_tree<double, long, 2>(m, "QuadTreeDouble");
    declare_nd_tree<long, long, 2>(m, "QuadTreeInt");

    m.def("build_quad_tree", &build_nd_tree<float, long, 2>, py::arg("database"), py::arg("box"));
    m.def("build_quad_tree", &build_nd_tree<double, long, 2>, py::arg("database"), py::arg("box"));
    m.def("build_quad_tree", &build_nd_tree<long, long, 2>, py::arg("database"), py::arg("box"));

    declare_nd_stack<float, long, 2>(m, "QuadStackFloat");
    declare_nd_stack<double, long, 2>(m, "QuadStackDouble");
    declare_nd_stack<long, long, 2>(m, "QuadStackInt");

    m.def("build_quad_stack", &build_nd_stack<float, long, 2>, py::arg("database"), py::arg("indices"), py::arg("box"));
    m.def("build_quad_stack", &build_nd_stack<double, long, 2>, py::arg("database"), py::arg("indices"), py::arg("box"));
    m.def("build_quad_stack", &build_nd_stack<long, long, 2>, py::arg("database"), py::arg("indices"), py::arg("box"));

    declare_nd_tree<float, long, 3>(m, "OctreeFloat");
    declare_nd_tree<double, long, 3>(m, "OctreeDouble");
    declare_nd_tree<long, long, 3>(m, "OctreeInt");

    m.def("build_octree", &build_nd_tree<float, long, 3>, py::arg("database"), py::arg("box"));
    m.def("build_octree", &build_nd_tree<double, long, 3>, py::arg("database"), py::arg("box"));
    m.def("build_octree", &build_nd_tree<long, long, 3>, py::arg("database"), py::arg("box"));

    declare_nd_stack<float, long, 3>(m, "OctStackFloat");
    declare_nd_stack<double, long, 3>(m, "OctStackDouble");
    declare_nd_stack<long, long, 3>(m, "OctStackInt");

    m.def("build_oct_stack", &build_nd_stack<float, long, 3>, py::arg("database"), py::arg("indices"), py::arg("box"));
    m.def("build_oct_stack", &build_nd_stack<double, long, 3>, py::arg("database"), py::arg("indices"), py::arg("box"));
    m.def("build_oct_stack", &build_nd_stack<long, long, 3>, py::arg("database"), py::arg("indices"), py::arg("box"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}