#include "array.hpp"

namespace cbclib {

class Indexer
{
public:
    py::array m_index, m_unique;
    bool is_increasing, is_decreasing, is_unique;

    template <typename I>
    Indexer(py::array_t<I> arr) : m_index(arr)
    {
        is_monotonic(arr);
        if (!is_increasing && !is_decreasing)
            throw std::invalid_argument("array must be monotonic");
        unique(arr);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    py::slice get_index(I value) const
    {
        auto index = py::array_t<I>::ensure(m_index);
        if (!index) throw std::invalid_argument("array and indices have incompatible dtypes");

        const I * first = index.data();
        const I * last = index.data() + index.size();

        const I * start;
        const I * end;
        if (is_increasing)
        {
            start = std::lower_bound(first, last, value, std::less());
            end = std::upper_bound(first, last, value, std::less());
        }
        else
        {
            start = std::lower_bound(first, last, value, std::greater());
            end = std::upper_bound(first, last, value, std::greater());
        }

        if (start == last)
            throw std::out_of_range(std::to_string(value) + " is out of range");
        if (start == end)
            throw std::out_of_range(std::to_string(value) + " is not present");
        return py::slice(py::ssize_t(std::distance(first, start)), py::ssize_t(std::distance(first, end)), std::nullopt);
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    std::tuple<py::array_t<py::ssize_t>, py::array_t<py::ssize_t>> get_index(py::array_t<I> idxs) const
    {
        auto index = py::array_t<I>::ensure(m_index);
        if (!index) throw std::invalid_argument("array and indices have incompatible dtypes");

        std::vector<py::ssize_t> indexer, new_index;
        const I * first = index.data();
        const I * last = index.data() + index.size();

        py::ssize_t idx = py::ssize_t();
        for (const I * ptr = idxs.data(); ptr != idxs.data() + idxs.size(); ptr++, idx++)
        {
            const I * start;
            const I * end;
            if (is_increasing)
            {
                start = std::lower_bound(first, last, *ptr, std::less());
                end = std::upper_bound(first, last, *ptr, std::less());
            }
            else
            {
                start = std::lower_bound(first, last, *ptr, std::greater());
                end = std::upper_bound(first, last, *ptr, std::greater());
            }

            if (start == last)
                throw std::out_of_range(std::to_string(*ptr) + " is out of range");
            if (start == end)
                throw std::out_of_range(std::to_string(*ptr) + " is not present");

            for (auto elem = start; elem != end; elem++)
            {
                indexer.push_back(std::distance(first, elem));
                new_index.push_back(idx);
            }
        }

        return std::make_tuple(as_pyarray(std::move(indexer)), as_pyarray(std::move(new_index)));
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    std::tuple<py::array_t<py::ssize_t>, py::array_t<py::ssize_t>> insert_index(py::array_t<I> idxs) const
    {
        auto index = py::array_t<I>::ensure(m_index);
        if (!index) throw std::invalid_argument("array and indices have incompatible dtypes");

        std::vector<py::ssize_t> to, from;
        const I * first = index.data();
        const I * last = index.data() + index.size();

        py::ssize_t idx = py::ssize_t();
        for (const I * ptr = idxs.data(); ptr != idxs.data() + idxs.size(); ptr++, idx++)
        {
            const I * start;
            const I * end;
            if (is_increasing)
            {
                start = std::lower_bound(first, last, *ptr, std::less());
                end = std::upper_bound(first, last, *ptr, std::less());
            }
            else
            {
                start = std::lower_bound(first, last, *ptr, std::greater());
                end = std::upper_bound(first, last, *ptr, std::greater());
            }

            if (start == last)
            {
                if (!first)
                {
                    to.push_back(py::ssize_t());
                    from.push_back(idx);
                }
                else if (*ptr < *first)
                {
                    to.push_back(py::ssize_t());
                    from.push_back(idx);
                }
                else
                {
                    to.push_back(py::ssize_t(index.size()));
                    from.push_back(idx);
                }
            }

            else if (start == end)
            {
                to.push_back(std::distance(first, start));
                from.push_back(idx);
            }
        }

        return std::make_tuple(as_pyarray(std::move(to)), as_pyarray(std::move(from)));
    }

protected:
    template <typename I>
    void is_monotonic(py::array_t<I> arr)
    {
        size_t size = arr.size();

        is_increasing = true, is_decreasing = true, is_unique = true;

        if (size > 2)
        {
            I prev = *arr.data(), cur = I();
            for (const I * ptr = arr.data(); ptr != arr.data() + size; ptr++)
            {
                cur = *ptr;
                if (cur < prev) is_increasing = false;
                else if (cur > prev) is_decreasing = false;
                else if (cur == prev) is_unique = false;
                else /* cur or prev is NaN */
                {
                    is_increasing = false;
                    is_decreasing = false;
                    break;
                }
                if (!is_increasing && !is_decreasing) break;
                prev = cur;
            }
        }

        is_unique = is_unique && (is_increasing || is_decreasing);
    }

    template <typename I>
    void unique(py::array_t<I> arr)
    {
        std::vector<I> values;

        const I * ptr = arr.data();
        const I * first = arr.data();
        const I * last = arr.data() + arr.size();
        while (ptr != last)
        {
            values.push_back(*ptr);
            if (is_increasing) ptr = std::upper_bound(first, last, values.back(), std::less());
            else ptr = std::upper_bound(first, last, values.back(), std::greater());
        }

        m_unique = as_pyarray(std::move(values));
    }
};

}

PYBIND11_MODULE(index, m)
{
    using namespace cbclib;

    try
    {
        import_numpy();
    }
    catch (const py::error_already_set & e)
    {
        return;
    }

    py::class_<Indexer>(m, "Indexer")
        .def(py::init<py::array_t<int>>(), py::arg("array"))
        .def(py::init<py::array_t<unsigned>>(), py::arg("array"))
        .def(py::init<py::array_t<long>>(), py::arg("array"))
        .def(py::init<py::array_t<size_t>>(), py::arg("array"))
        .def_readonly("is_increasing", &Indexer::is_increasing)
        .def_readonly("is_decreasing", &Indexer::is_decreasing)
        .def_readonly("is_unique", &Indexer::is_unique)
        .def_readonly("array", &Indexer::m_index)
        .def("__getitem__", [](Indexer & indexer, int value){return indexer.get_index(value);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, unsigned value){return indexer.get_index(value);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, long value){return indexer.get_index(value);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, size_t value){return indexer.get_index(value);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, py::array_t<int> idxs){return indexer.get_index(idxs);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, py::array_t<unsigned> idxs){return indexer.get_index(idxs);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, py::array_t<long> idxs){return indexer.get_index(idxs);}, py::arg("indices"))
        .def("__getitem__", [](Indexer & indexer, py::array_t<size_t> idxs){return indexer.get_index(idxs);}, py::arg("indices"))
        .def("insert_index", [](Indexer & indexer, py::array_t<int> idxs){return indexer.insert_index(idxs);}, py::arg("indices"))
        .def("insert_index", [](Indexer & indexer, py::array_t<unsigned> idxs){return indexer.insert_index(idxs);}, py::arg("indices"))
        .def("insert_index", [](Indexer & indexer, py::array_t<long> idxs){return indexer.insert_index(idxs);}, py::arg("indices"))
        .def("insert_index", [](Indexer & indexer, py::array_t<size_t> idxs){return indexer.insert_index(idxs);}, py::arg("indices"))
        .def("unique", [](Indexer & indexer){return indexer.m_unique;});
}
