#ifndef MEDIAN_
#define MEDIAN_
#include "array.hpp"

namespace cbclib {

namespace detail{

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt mirror(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = mirror(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt reflect(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = reflect(*first, *min, *max);
    }
    return d_first;
}

template <typename InputIt1, typename InputIt2, typename InputIt3, typename OutputIt>
OutputIt wrap(InputIt1 first, InputIt1 last, OutputIt d_first, InputIt2 min, InputIt3 max)
{
    for (; first != last; ++first, ++d_first, ++min, ++max)
    {
        *d_first = wrap(*first, *min, *max);
    }
    return d_first;
}

}

template <typename RandomIt, typename Compare, typename T = typename std::iterator_traits<RandomIt>::value_type>
std::common_type_t<T, decltype(0.5 * std::declval<T &>())> median_1d(RandomIt first, RandomIt last, Compare comp)
{
    auto n = std::distance(first, last);
    if (n & 1)
    {
        auto nth = std::next(first, n / 2);
        std::nth_element(first, nth, last, comp);
        return *nth;
    }
    else
    {
        auto low = std::next(first, n / 2 - 1), high = std::next(first, n / 2);
        std::nth_element(first, low, last, comp);
        std::nth_element(high, high, last, comp);
        return 0.5 * (*low + *high);
    }
}

template <typename Container, typename T, typename = std::enable_if_t<std::is_integral_v<typename Container::value_type>>>
T extend_point(const Container & coord, const array<T> & arr, extend mode, const T & cval)
{
    using I = typename Container::value_type;

    /* kkkkkkkk|abcd|kkkkkkkk */
    if (mode == extend::constant) return cval;

    std::vector<I> close;
    std::vector<I> min (arr.ndim(), I());

    switch (mode)
    {
        /* aaaaaaaa|abcd|dddddddd */
        case extend::nearest:

            for (size_t n = 0; n < arr.ndim(); n++)
            {
                if (coord[n] >= static_cast<I>(arr.shape(n))) close.push_back(arr.shape(n) - 1);
                else if (coord[n] < I()) close.push_back(I());
                else close.push_back(coord[n]);
            }

            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case extend::mirror:

            detail::mirror(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape().begin());

            break;

        /* abcddcba|abcd|dcbaabcd */
        case extend::reflect:

            detail::reflect(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape().begin());

            break;

        /* abcdabcd|abcd|abcdabcd */
        case extend::wrap:

            detail::wrap(coord.begin(), coord.end(), std::back_inserter(close), min.begin(), arr.shape().begin());

            break;

        default:
            throw std::invalid_argument("Invalid extend argument: " + std::to_string(static_cast<int>(mode)));
    }

    return arr.at(close.begin(), close.end());
}

template <typename T>
class ImageFilter
{
public:
    ImageFilter(std::vector<std::vector<long>> offsets) : offsets(std::move(offsets)) {}

    ImageFilter(const array<bool> & footprint)
    {
        size_t index = 0;
        for (const auto & pt : rectangle_range(footprint.shape()))
        {
            if (footprint[index++])
            {
                auto & offset = offsets.emplace_back();
                std::transform(pt.begin(), pt.end(), footprint.shape().begin(), std::back_inserter(offset),
                               [](long crd, size_t dim){return crd - dim / 2;});
            }
        }

        if (!offsets.size())
            throw std::runtime_error("zero number of points in a ImageFilter");
    }

    template <typename Container, typename = std::enable_if_t<std::is_convertible_v<typename Container::value_type, long>>>
    void update(const Container & coord, const array<T> & image, extend mode, const T & cval)
    {
        data.clear();
        std::vector<long> current;

        for (const auto & offset : offsets)
        {
            current.clear();
            bool extend = false;

            for (size_t n = 0; n < offset.size(); n++)
            {
                current.push_back(coord[n] + offset[n]);
                extend |= (current.back() >= static_cast<long>(image.shape(n))) || (current.back() < 0);
            }

            if (extend)
            {
                auto val = extend_point(current, image, mode, cval);
                data.push_back(val);
            }
            else data.push_back(image.at(current));
        }
    }

    T nth_element(size_t rank)
    {
        if (rank >= data.size()) return T();
        auto nth = std::next(data.begin(), rank);
        std::nth_element(data.begin(), nth, data.end());
        return *nth;
    }

    auto size() const {return offsets.size();}

private:
    std::vector<std::vector<long>> offsets;
    std::vector<T> data;
};

}

#endif
