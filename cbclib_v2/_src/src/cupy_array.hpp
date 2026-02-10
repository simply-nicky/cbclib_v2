#ifndef CUPY_ARRAY_
#define CUPY_ARRAY_
#include <cuda_runtime.h>
#include "include.hpp"
#include "array_view.hpp"
#include "cuda_geometry.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace cbclib::cuda {

void handle_cuda_error(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

namespace detail {

// Convert linear index to N-D coordinate (row-major, last dimension fastest)
template <typename I, csize_t N>
HOST_DEVICE inline void index_to_coord(I idx, I * coord, const I * dims)
{
    for (csize_t n = N; n > 0; --n)
    {
        coord[n - 1] = idx % dims[n - 1];
        idx /= dims[n - 1];
    }
}

template <typename I, csize_t N>
HOST_DEVICE inline PointND<I, N> index_to_coord(I idx, const PointND<I, N> & dims)
{
    PointND<I, N> coord;
    index_to_coord<I, N>(idx, coord.data(), dims.data());
    return coord;
}

// Flatten a coordinate to linear index given dims (row-major, last dimension fastest)
template <typename I, csize_t N>
HOST_DEVICE inline csize_t coord_to_index(const PointND<I, N> & coord, const PointND<I, N> & dims)
{
    I idx = 0;
    for (csize_t i = 0; i < N; i++) idx = idx * dims[i] + coord[i];
    return idx;
}

} // namespace detail

// POD structures for easier use in device code.
// Both ShapeND and ArrayIndexerND own shape and stride as C-arrays (PointND).

// ShapeND: fixed-size N-dimensional shape holder
// Can be used for c-contiguous arrays only.
template <csize_t N>
struct ShapeND
{
    PointND<csize_t, N> _M_shape;   // follows zyx ordering
    csize_t _M_size;

    HOST_DEVICE ShapeND() : _M_shape(), _M_size(0) {}
    HOST_DEVICE ShapeND(const PointND<csize_t, N> & other) : _M_shape(other), _M_size(1)
    {
        for (csize_t i = 0; i < N; i++) _M_size *= _M_shape[i];
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    HOST_DEVICE ShapeND(const I * dims) : _M_shape(), _M_size(1)
    {
        for (csize_t i = 0; i < N; i++)
        {
            _M_shape[i] = dims[i];
            _M_size *= _M_shape[i];
        }
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    HOST_DEVICE ShapeND(const I (&dims)[N]) : ShapeND(dims) {}

    HOST_DEVICE bool is_inbound(const PointND<csize_t, N> & coord) const
    {
        bool flag = true;
        for (csize_t i = 0; i < N; i++)
        {
            flag &= coord[i] >= 0 && coord[i] < _M_shape[i];
        }
        return flag;
    }

    HOST_DEVICE csize_t index_at(const PointND<csize_t, N> & coord) const
    {
        return detail::coord_to_index<csize_t, N>(coord, _M_shape);
    }

    HOST_DEVICE PointND<csize_t, N> coord_at(csize_t index) const
    {
        return detail::index_to_coord<csize_t, N>(index, _M_shape);
    }

    HOST_DEVICE PointND<csize_t, N> & coord_at(PointND<csize_t, N> & coord, csize_t index) const
    {
        detail::index_to_coord<csize_t, N>(index, coord.data(), _M_shape.data());
        return coord;
    }

    HOST_DEVICE csize_t coord_along_dim(csize_t index, csize_t dim) const
    {
        csize_t coord = 0;
        for (csize_t n = N; n > dim; --n)
        {
            coord = index % _M_shape[n - 1];
            index /= _M_shape[n - 1];
        }
        return coord;
    }

    constexpr HOST_DEVICE csize_t ndim() const { return N; }
    HOST_DEVICE csize_t size() const { return _M_size; }

    HOST_DEVICE const csize_t * shape() const { return _M_shape.data(); }
    HOST_DEVICE csize_t shape(csize_t dim) const { return _M_shape[dim]; }
};

// ArrayIndexerND: fixed-size N-dimensional array indexer
// Can be used for any array with given number of dimensions.
template <csize_t N>
struct ArrayIndexerND : public ShapeND<N>
{
    using ShapeND<N>::_M_shape;
    PointND<csize_t, N> _M_strides;    // follows zyx ordering

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    HOST_DEVICE ArrayIndexerND(const I * dims, const I * strides) : ShapeND<N>(dims), _M_strides(strides) {}

    HOST_DEVICE csize_t offset_at(const PointND<csize_t, N> & coord) const
    {
        return cbclib::detail::coord_to_offset(coord.data(), coord.data() + N, _M_strides.data());
    }

    HOST_DEVICE csize_t offset_at(csize_t index) const
    {
        return cbclib::detail::index_to_offset(index, _M_shape.data(), _M_strides.data(), 0, N);
    }

    HOST_DEVICE csize_t * strides() const { return _M_strides.data(); }
    HOST_DEVICE csize_t strides(csize_t dim) const { return _M_strides[dim]; }
};

// ArrayViewND: fixed-size N-dimensional array view
// Lightweight owning wrapper around raw pointer and shape/stride info.
// StridedIterator: iterator adapter for iterating over a given axis of a multi-dimensional array
template <typename T>
struct StridedIterator
{
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using difference_type = csize_t;

    pointer _M_ptr;
    difference_type _M_stride;

    HOST_DEVICE StridedIterator() : _M_ptr(nullptr), _M_stride(0) {}
    HOST_DEVICE StridedIterator(pointer ptr, difference_type stride) : _M_ptr(ptr), _M_stride(stride) {}

    HOST_DEVICE operator bool() const {return bool(_M_ptr);}

    bool operator==(const StridedIterator & rhs) const {return _M_ptr == rhs._M_ptr;}
    bool operator!=(const StridedIterator & rhs) const {return _M_ptr != rhs._M_ptr;}
    bool operator<=(const StridedIterator & rhs) const {return _M_ptr <= rhs._M_ptr;}
    bool operator>=(const StridedIterator & rhs) const {return _M_ptr >= rhs._M_ptr;}
    bool operator<(const StridedIterator & rhs) const {return _M_ptr < rhs._M_ptr;}
    bool operator>(const StridedIterator & rhs) const {return _M_ptr > rhs._M_ptr;}

    HOST_DEVICE StridedIterator & operator+=(const difference_type & step) {_M_ptr += step * _M_stride; return *this;}
    HOST_DEVICE StridedIterator & operator-=(const difference_type & step) {_M_ptr -= step * _M_stride; return *this;}
    HOST_DEVICE StridedIterator & operator++() {_M_ptr += _M_stride; return *this;}
    HOST_DEVICE StridedIterator & operator--() {_M_ptr -= _M_stride; return *this;}
    HOST_DEVICE StridedIterator operator++(int) {StridedIterator temp = *this; ++(*this); return temp;}
    HOST_DEVICE StridedIterator operator--(int) {StridedIterator temp = *this; --(*this); return temp;}
    HOST_DEVICE StridedIterator operator+(const difference_type & step) const
    {
        return {_M_ptr + step * _M_stride, _M_stride};
    }
    HOST_DEVICE StridedIterator operator-(const difference_type & step) const
    {
        return {_M_ptr - step * _M_stride, _M_stride};
    }

    HOST_DEVICE difference_type operator-(const StridedIterator & rhs) const {return (_M_ptr - rhs._M_ptr) / _M_stride;}

    HOST_DEVICE reference operator[] (size_t index) const {return _M_ptr[index * _M_stride];}
    HOST_DEVICE reference operator*() const {return *_M_ptr;}
    HOST_DEVICE pointer operator->() const {return _M_ptr;}
};

template <typename T, csize_t N>
struct ArrayViewND : public ArrayIndexerND<N>
{
    using ArrayIndexerND<N>::_M_shape;
    using ArrayIndexerND<N>::_M_strides;
    T * _M_ptr;

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    HOST_DEVICE ArrayViewND(T * ptr, const I * shape, const I * strides)
        : ArrayIndexerND<N>(shape, strides), _M_ptr(ptr) {}

    HOST_DEVICE T * data() { return _M_ptr; }
    HOST_DEVICE const T * data() const { return _M_ptr; }

    HOST_DEVICE T & operator[] (csize_t index)
    {
        return *(_M_ptr + cbclib::detail::index_to_offset(index, _M_shape.data(), _M_strides.data(), csize_t(), N) / sizeof(T));
    }
    HOST_DEVICE const T & operator[] (csize_t index) const
    {
        return *(_M_ptr + cbclib::detail::index_to_offset(index, _M_shape.data(), _M_strides.data(), csize_t(), N) / sizeof(T));
    }

    HOST_DEVICE T * data(const PointND<csize_t, N> & coord)
    {
        return _M_ptr + offset_at(coord) / sizeof(T);
    }
    HOST_DEVICE const T * data(const PointND<csize_t, N> & coord) const
    {
        return _M_ptr + offset_at(coord) / sizeof(T);
    }

    HOST_DEVICE T & at(const PointND<csize_t, N> & coord)
    {
        return *data(coord);
    }
    HOST_DEVICE const T & at(const PointND<csize_t, N> & coord) const
    {
        return *data(coord);
    }

    HOST_DEVICE StridedIterator<T> begin_at(csize_t offset, csize_t dim)
    {
        return StridedIterator<T>(_M_ptr + offset / sizeof(T), _M_strides[dim] / sizeof(T));
    }
    HOST_DEVICE StridedIterator<const T> begin_at(csize_t offset, csize_t dim) const
    {
        return StridedIterator<const T>(_M_ptr + offset / sizeof(T), _M_strides[dim] / sizeof(T));
    }
    HOST_DEVICE StridedIterator<T> end_at(csize_t offset, csize_t dim)
    {
        return StridedIterator<T>(_M_ptr + (offset + _M_shape[dim] * _M_strides[dim]) / sizeof(T));
    }
    HOST_DEVICE StridedIterator<const T> end_at(csize_t offset, csize_t dim) const
    {
        return StridedIterator<const T>(_M_ptr + (offset + _M_shape[dim] * _M_strides[dim]) / sizeof(T));
    }
};

// Convert a general array_view to fixed-size N-D ArrayViewND used on the device
// Do it at your own risk: no checks are performed, make sure the input view has the correct number of dimensions
template <typename T, csize_t N>
ArrayViewND<T, N> cast_to_nd(array_view<T, py::ssize_t> view)
{
    if (N == view.ndim()) return ArrayViewND<T, N>(view.data(), view.shape(), view.strides());
    if (N > view.ndim())
    {
        std::vector<py::ssize_t> shape (view.shape(), view.shape() + view.ndim());
        std::vector<py::ssize_t> strides (view.strides(), view.strides() + view.ndim());
        // pad with ones
        for (py::ssize_t i = view.ndim(); i < N; ++i)
        {
            shape.insert(shape.begin(), 1);
            strides.insert(strides.begin(), sizeof(T));
        }
        return ArrayViewND<T, N>(view.data(), shape.data(), strides.data());
    }
    return ArrayViewND<T, N>(view.data(), view.shape() + (view.ndim() - N), view.strides() + (view.ndim() - N));
}

template <typename T, csize_t N>
ArrayViewND<const T, N> cast_to_nd(array_view<const T, py::ssize_t> view)
{
    if (N == view.ndim()) return ArrayViewND<const T, N>(view.data(), view.shape(), view.strides());
    if (N > view.ndim())
    {
        std::vector<py::ssize_t> shape (view.shape(), view.shape() + view.ndim());
        std::vector<py::ssize_t> strides (view.strides(), view.strides() + view.ndim());
        // pad with ones
        for (py::ssize_t i = view.ndim(); i < N; ++i)
        {
            shape.insert(shape.begin(), 1);
            strides.insert(strides.begin(), sizeof(T));
        }
        return ArrayViewND<const T, N>(view.data(), shape.data(), strides.data());
    }
    return ArrayViewND<const T, N>(view.data(), view.shape() + (view.ndim() - N), view.strides() + (view.ndim() - N));
}

// Lightweight non-owning wrapper around a CuPy array
// CuPy arrays expose the __cuda_array_interface__ attribute
// https://docs.cupy.dev/en/stable/user_guide/interoperability.html#cuda-array-interface
// CuPy array stores data on the GPU and all other metadata (shape, strides, dtype) on the host.
template <typename T>
class array_t : public array_view<T, py::ssize_t>
{
public:
    array_t() : array_view<T, py::ssize_t>(), m_base() {}

    array_t(const std::vector<py::ssize_t> & shape,
            const std::vector<py::ssize_t> & strides,
            T * ptr, py::handle base = py::handle())
        : m_shape(shape), m_strides(strides), m_base(py::reinterpret_borrow<py::object>(base))
    {
        this->m_shape_ptr = m_shape.data();
        this->m_strides_ptr = m_strides.data();
        this->m_itemsize = sizeof(T);
        this->m_ndim = m_shape.size();
        this->m_ptr = ptr;
    }

    py::dtype dtype() const {return py::dtype::of<T>();}

    py::object base() const {return m_base;}

    array_view<T, py::ssize_t> view()
    {
        return array_view<T, py::ssize_t>(this->m_ptr, this->m_shape_ptr, this->m_strides_ptr, this->m_ndim);
    }

    array_view<const T, py::ssize_t> view() const
    {
        return array_view<const T, py::ssize_t>(this->m_ptr, this->m_shape_ptr, this->m_strides_ptr, this->m_ndim);
    }

    template <typename U>
    array_t<U> astype()
    {
        // call CuPy’s astype on the original Python object
        py::object astyped = m_base.attr("astype")(py::dtype::of<U>());
        // wrap back into array_t; uses __cuda_array_interface__ from CuPy
        return array_t<U>::ensure(astyped);
    }

    array_t reshape(const std::vector<py::ssize_t> & new_shape) const
    {
        // call CuPy’s reshape on the original Python object
        py::tuple shape(new_shape.size());
        for (size_t i = 0; i < new_shape.size(); ++i) shape[i] = new_shape[i];

        py::object reshaped = m_base.attr("reshape")(shape);
        // wrap back into array_t; uses __cuda_array_interface__ from CuPy
        return array_t::ensure(reshaped);
    }

    array_t reshape(std::initializer_list<py::ssize_t> new_shape) const
    {
        return reshape(std::vector<py::ssize_t>(new_shape));
    }

    static array_t ensure(py::handle h)
    {
        auto interface = h.attr("__cuda_array_interface__");

        auto py_typestr = interface["typestr"];

        auto py_data = interface["data"];
        auto data_tuple = py::reinterpret_borrow<py::tuple>(py_data);

        auto ptr = reinterpret_cast<T *>(py::cast<uintptr_t>(data_tuple[0]));

        auto py_shape = interface["shape"];
        auto shape_tuple = py::reinterpret_borrow<py::tuple>(py_shape);

        std::vector<py::ssize_t> shape;
        for (auto item : shape_tuple) shape.push_back(py::cast<py::ssize_t>(item));

        py::object py_strides;
        if (interface.contains("strides")) py_strides = interface["strides"];
        else py_strides = py::none();

        std::vector<py::ssize_t> strides;
        if (!py_strides.is_none() && py::isinstance<py::tuple>(py_strides))
        {
            auto strides_tuple = py::reinterpret_borrow<py::tuple>(py_strides);
            for (auto item : strides_tuple) strides.push_back(py::cast<py::ssize_t>(item));
        }
        else
        {
            // If strides are not provided or None, assume a contiguous array
            strides.resize(shape.size());
            strides.back() = sizeof(T);
            for (ssize_t n = shape.size() - 1; n > 0; --n) strides[n - 1] = strides[n] * shape[n];
        }
        // Remember original Python object as owner so we can cast back safely
        return array_t(shape, strides, ptr, h);
    }

    static bool check_(py::handle h)
    {
        if (!py::hasattr(h, "__cuda_array_interface__")) return false;

        auto interface = h.attr("__cuda_array_interface__");

        // Fallback: parse typestr manually
        auto py_typestr = interface["typestr"];
        if (!py::isinstance<py::str>(py_typestr)) return false;

        auto typestr = py::reinterpret_borrow<py::str>(py_typestr).cast<std::string>();
        return py::detail::npy_api::get().PyArray_EquivTypes_(py::dtype::of<T>().ptr(), py::dtype(typestr).ptr());
    }

protected:
    std::vector<py::ssize_t> m_shape, m_strides;
    py::object m_base;  // Owns reference to original Python object
};

// Device array and range wrappers

template <typename T>
struct DeviceRange
{
    T * _M_begin;
    T * _M_end;

    HOST_DEVICE DeviceRange() : _M_begin(nullptr), _M_end(nullptr) {}
    HOST_DEVICE DeviceRange(T * begin, T * end) : _M_begin(begin), _M_end(end) {}
    HOST_DEVICE T * begin() { return _M_begin; }
    HOST_DEVICE const T * begin() const { return _M_begin; }
    HOST_DEVICE T * end() { return _M_end; }
    HOST_DEVICE const T * end() const { return _M_end; }
    HOST_DEVICE T * data() { return _M_begin; }
    HOST_DEVICE const T * data() const { return _M_begin; }

    HOST_DEVICE const T & operator[](csize_t i) const { return _M_begin[i]; }
    HOST_DEVICE T & operator[](csize_t i) { return _M_begin[i]; }

    HOST_DEVICE csize_t size() const { return static_cast<csize_t>(_M_end - _M_begin);}
};

template <typename T, typename T_mutable = typename std::remove_const_t<T>>
class DeviceVector
{
public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using iterator = T *;
    using const_iterator = const T *;

    DeviceVector() : m_data(nullptr), m_size(0), m_owns(true) {}
    DeviceVector(csize_t size) : m_size(size), m_owns(true)
    {
        handle_cuda_error(cudaMalloc(&m_data, m_size * sizeof(T)));
    }
    DeviceVector(csize_t size, int ch) : DeviceVector(size)
    {
        handle_cuda_error(cudaMemset(m_data, ch, m_size * sizeof(T)));
    }

    static DeviceVector<T> from_host(const T * host_data, csize_t size)
    {
        DeviceVector<T> dev_array(size);
        handle_cuda_error(cudaMemcpy(dev_array.m_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
        return dev_array;
    }

    static DeviceVector<T> from_device(T_mutable * device_data, csize_t size)
    {
        DeviceVector<T> dev_array;
        dev_array.m_data = device_data;
        dev_array.m_size = size;
        dev_array.m_owns = false;
        return dev_array;
    }

    ~DeviceVector()
    {
        free();
    }

    DeviceVector(const DeviceVector &) = delete;
    DeviceVector & operator=(const DeviceVector &) = delete;

    DeviceVector(DeviceVector && other) noexcept : m_data(other.m_data), m_size(other.m_size), m_owns(other.m_owns)
    {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_owns = true;
    }

    DeviceVector & operator=(DeviceVector && other) noexcept
    {
        if (this != &other)
        {
            free();

            m_data = other.m_data;
            m_size = other.m_size;
            m_owns = other.m_owns;
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_owns = true;
        }
        return *this;
    }

    T * data() { return m_data; }
    const T * data() const { return m_data; }

    iterator begin() { return m_data; }
    const_iterator begin() const { return m_data; }
    iterator end() { return m_data + m_size; }
    const_iterator end() const { return m_data + m_size; }

    csize_t size() const { return m_size; }
    reference operator[](csize_t index) { return m_data[index]; }
    const_reference operator[](csize_t index) const { return m_data[index]; }

    DeviceRange<const T> view() const
    {
        return DeviceRange<const T>(m_data, m_data + m_size);
    }
    DeviceRange<T> view()
    {
        return DeviceRange<T>(m_data, m_data + m_size);
    }

    void to_device(T_mutable * dest)
    {
        handle_cuda_error(cudaMemcpy(dest, m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void to_device(DeviceVector<T> & dest)
    {
        if (dest.size() != m_size)
        {
            throw std::runtime_error("DeviceVector::to_device: destination array has incorrect size");
        }
        handle_cuda_error(cudaMemcpy(dest.m_data, m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    void to_host(T_mutable * dest) const
    {
        handle_cuda_error(cudaMemcpy(dest, m_data, m_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void clear() noexcept
    {
        free();
    }

    void resize(csize_t new_size)
    {
        if (new_size != m_size)
        {
            free();
            m_size = new_size;
            handle_cuda_error(cudaMalloc(&m_data, m_size * sizeof(T)));
            m_owns = true;
        }
    }

    void set(int ch = 0)
    {
        handle_cuda_error(cudaMemset(m_data, ch, m_size * sizeof(T)));
    }

private:
    T_mutable * m_data;
    csize_t m_size = 0;
    bool m_owns = true;

    void free() noexcept
    {
        if (m_data && m_owns)
        {
            // Destructors must not throw. Ignore cudaFree errors here.
            cudaError_t _err = cudaFree(m_data);
            (void)_err;
            m_data = nullptr;
            m_size = 0;
        }
    }
};

// Debug helper: print device vector to stdout
template <typename T>
void print_device_vector(const char * label, const DeviceVector<T> & vec, csize_t max_print = 20)
{
    csize_t n = std::min(vec.size(), max_print);
    std::vector<T> host_data(n);
    handle_cuda_error(cudaMemcpy(host_data.data(), vec.data(), n * sizeof(T), cudaMemcpyDeviceToHost));

    printf("%s: [", label);
    for (csize_t i = 0; i < n; ++i)
    {
        if (i > 0) printf(" ");
        if constexpr (std::is_floating_point_v<T>)
            printf("%.3f", static_cast<double>(host_data[i]));
        else
            printf("%zu", static_cast<size_t>(host_data[i]));
    }

    if (vec.size() > n)
        printf(" ... (total %zu)", vec.size());
    printf("]\n");
}

} // namespace cbclib::cuda

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<cbclib::cuda::array_t<T>>
{
    using type = cbclib::cuda::array_t<T>;

    // Python -> C++: convert a `PyObject` into a `cuda::array<T>` and return false upon failure. The
    // second argument indicates whether implicit conversions should be allowed.
    // The accepted types should reflect the type hint specified by the first argument of
    // `io_name`.
    bool load(handle src, bool convert)
    {
        if (!type::check_(src)) return false;

        value = type::ensure(src);
        return static_cast<bool>(value.base());
    }

    // C++ -> Python: convert `cuda::array<T>` to `cupy.ndarray`. The second and third arguments
    // are used to indicate the return value policy and parent object (for
    // return_value_policy::reference_internal) and are often ignored by custom casters.
    // The return value should reflect the type hint specified by the second argument of `io_name`.
    static handle cast(const type & src,
                       return_value_policy /* policy */,
                       handle /* parent */)
    {
        return src.base().inc_ref();
    }

    PYBIND11_TYPE_CASTER(type, io_name("cupy.ndarray", "cupy.ndarray"));
};

} // namespace detail
} // namespace pybind11

#endif
