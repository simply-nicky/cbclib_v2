#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "array.hpp"

namespace cbclib {

// 2D Point class

template <typename T>
class Point
{
public:
    using value_type = T;
    using size_type = size_t;

    using const_iterator = std::array<T, 2>::const_iterator;
    using iterator = std::array<T, 2>::iterator;

    using const_reference = std::array<T, 2>::const_reference;
    using reference = std::array<T, 2>::reference;

    const_iterator begin() const {return pt.begin();}
    const_iterator end() const {return pt.end();}
    iterator begin() {return pt.begin();}
    iterator end() {return pt.end();}

    const_reference operator[](size_type index) const {return pt[index];}
    reference operator[](size_type index) {return pt[index];}

    const_reference x() const {return pt[0];}
    const_reference y() const {return pt[1];}
    reference x() {return pt[0];}
    reference y() {return pt[1];}

    size_type size() const {return pt.size();}

    Point() : pt{} {}
    Point(T x, T y) : pt{x, y} {}

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    Point(const std::array<V, 2> & point) : pt{static_cast<T>(point[0]), static_cast<T>(point[1])} {}

    template <typename V, typename = std::enable_if_t<std::is_constructible_v<T, V>>>
    Point(const Point<V> & point) : Point(point.to_array()) {}

    Point(std::array<T, 2> && point) : pt(std::move(point)) {}

    template <typename V>
    Point<std::common_type_t<T, V>> operator+(const Point<V> & rhs) const {return {x() + rhs.x(), y() + rhs.y()};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator+(V rhs) const {return {x() + rhs, y() + rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator+(V lhs, const Point<T> & rhs) {return {lhs + rhs.x(), lhs + rhs.y()};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator-(const Point<V> & rhs) const {return {x() - rhs.x(), y() - rhs.y()};}
    template <typename V>
    Point<std::common_type_t<T, V>> operator-(V rhs) const {return {x() - rhs, y() - rhs};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator-(V lhs, const Point<T> & rhs) {return {lhs - rhs.x(), lhs - rhs.y()};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator*(V rhs) const {return {rhs * x(), rhs * y()};}
    template <typename V>
    friend Point<std::common_type_t<T, V>> operator*(V lhs, const Point<T> & rhs) {return {lhs * rhs.x(), lhs * rhs.y()};}

    template <typename V>
    Point<std::common_type_t<T, V>> operator/(V rhs) const {return {x() / rhs, y() / rhs};}

    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(const Point<V> & rhs) {x() += rhs.x(); y() += rhs.y(); return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator+=(V rhs) {x() += rhs; y() += rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(const Point<V> & rhs) {x() -= rhs.x(); y() -= rhs.y(); return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> & operator-=(V rhs) {x() -= rhs; y() -= rhs; return *this;}
    template <typename V, typename = std::enable_if_t<std::is_convertible_v<T, V>>>
    Point<T> operator/=(V rhs) {x() /= rhs; y() /= rhs; return *this;}

    bool operator<(const Point<T> & rhs) const {return pt < rhs.pt;}
    bool operator==(const Point<T> & rhs) const {return pt == rhs.pt;}
    bool operator!=(const Point<T> & rhs) const {return !operator==(rhs);}

    friend std::ostream & operator<<(std::ostream & os, const Point<T> & pt)
    {
        os << "{" << pt.x() << ", " << pt.y() << "}";
        return os;
    }

    Point<T> clamp(const Point<T> & lo, const Point<T> & hi) const
    {
        return {std::clamp(x(), lo.x(), hi.x()), std::clamp(y(), lo.y(), hi.y())};
    }

    std::array<T, 2> coordinate() const
    {
        return {y(), x()};
    }

    std::array<T, 2> & to_array() & {return pt;}
    const std::array<T, 2> & to_array() const & {return pt;}
    std::array<T, 2> && to_array() && {return std::move(pt);}

    Point<T> round() const {return {static_cast<T>(std::round(x())), static_cast<T>(std::round(y()))};}

private:
    std::array<T, 2> pt;
};

template <class Container, typename T = typename Container::value_type>
constexpr auto to_point(Container & a, size_t start)
{
    return Point<T>(a[start], a[start + 1]);
}

template <typename T, typename V, typename U = std::common_type_t<T, V>, size_t N>
constexpr U dot(const std::array<T, N> & a, const std::array<V, N> & b)
{
    return apply_to_sequence<N>([&a, &b](auto... idxs){return ((a[idxs] * b[idxs]) + ...);});
}

template <typename T, typename V, typename U = std::common_type_t<T, V>>
constexpr U dot(const Point<T> & a, const Point<V> & b)
{
    return dot(a.to_array(), b.to_array());
}

template <typename T, size_t N>
constexpr T magnitude(const std::array<T, N> & a)
{
    return apply_to_sequence<N>([&a](auto... idxs){return ((a[idxs] * a[idxs]) + ...);});
}

template <typename T>
constexpr T magnitude(const Point<T> & p)
{
    return magnitude(p.to_array());
}

template <typename T, size_t N>
constexpr auto amplitude(const std::array<T, N> & a) -> decltype(std::sqrt(std::declval<T &>()))
{
    return std::sqrt(magnitude(a));
}

template <typename T>
constexpr auto amplitude(const Point<T> & p) -> decltype(std::sqrt(std::declval<T &>()))
{
    return amplitude(p.to_array());
}

template <typename Pt, typename = void>
struct is_point : std::false_type {};

template <typename Pt>
struct is_point <Pt,
    typename std::enable_if_t<std::is_base_of_v<Point<typename Pt::value_type>, std::remove_cvref_t<Pt>>>
> : std::true_type {};

template <typename Pt>
constexpr bool is_point_v = is_point<Pt>::value;

// 2D Line class

template <typename T>
struct Line
{
    Point<T> pt0, pt1;
    Point<T> tau;

    Line() = default;

    template <typename Pt0, typename Pt1, typename = std::enable_if_t<
        std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt0>> && std::is_base_of_v<Point<T>, std::remove_cvref_t<Pt1>>
    >>
    Line(Pt0 && pt0, Pt1 && pt1) : pt0(std::forward<Pt0>(pt0)), pt1(std::forward<Pt1>(pt1)), tau(pt1 - pt0) {}

    Line(T x0, T y0, T x1, T y1) : Line(Point<T>{x0, y0}, Point<T>{x1, y1}) {}

    Point<T> norm() const {return {tau.y(), -tau.x()};}

    auto theta() const {return std::atan(tau.y(), tau.x());}

    template <typename U, typename V = std::common_type_t<T, U>, typename W = decltype(amplitude(std::declval<Point<V> &>()))>
    W distance(const Point<U> & point) const
    {
        if (magnitude(tau))
        {
            auto compare_point = [](const Point<V> & a, const Point<V> & b){return magnitude(a) < magnitude(b);};
            auto r = std::min(point - pt0, pt1 - point, compare_point);

            // need to divide by magnitude(tau) : dist = amplitude(norm() * dot(norm(), r) / magnitude(norm()))
            auto r_tau = static_cast<W>(dot(tau, r)) / magnitude(tau);
            auto r_norm = static_cast<W>(dot(norm(), r)) / magnitude(norm());
            if (r_tau > 1) return amplitude(norm() * r_norm + tau * (r_tau - 1));
            if (r_tau < 0) return amplitude(norm() * r_norm + tau * r_tau);
            return amplitude(norm() * r_norm);
        }
        return amplitude(pt0 - point);
    }

    template <typename U, typename V = std::common_type_t<T, U>, typename W = decltype(amplitude(std::declval<Point<V> &>()))>
    W normal_distance(const Point<U> & point) const
    {
        if (magnitude(tau))
        {
            auto compare_point = [](const Point<V> & a, const Point<V> & b){return magnitude(a) < magnitude(b);};
            auto r = std::min(point - pt0, pt1 - point, compare_point);
            return abs(dot(norm(), r) / amplitude(norm()));
        }
        return amplitude(pt0 - point);
    }

    friend std::ostream & operator<<(std::ostream & os, const Line<T> & line)
    {
        os << "{" << line.pt0 << ", " << line.pt1 << "}";
        return os;
    }

    std::array<T, 4> to_array() const {return {pt0.x(), pt0.y(), pt1.x(), pt1.y()};}
};

using point_t = Point<long>;

template <typename T, typename I = typename point_t::value_type>
using table_t = std::map<std::pair<I, I>, T>;

namespace detail {

template <typename T>
class ImageBuffer : public shape_handler
{
public:
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    ImageBuffer() : shape_handler(), data() {}
    ImageBuffer(ShapeContainer shape) : shape_handler(std::move(shape)), data() {}

    iterator begin() {return data.begin();}
    const_iterator begin() const {return data.begin();}
    iterator end() {return data.end();}
    const_iterator end() const {return data.begin();}

    template <typename Container, class... Args, typename =
        std::enable_if_t<std::is_integral_v<typename Container::value_type>>
    >
    void emplace_back(const Container & coord, Args &&... args)
    {
        data.emplace_back(ravel_index(coord), std::forward<Args>(args)...);
    }

private:
    std::vector<T> data;
};

template <typename T, typename I>
void draw_pixel(ImageBuffer<std::pair<I, T>> & buffer, const point_t & pt, I frame, T val)
{
    using integer_type = typename point_t::value_type;
    if (val)
    {
        std::array<integer_type, 3> coord {static_cast<integer_type>(frame), pt.y(), pt.x()};
        if (buffer.is_inbound(coord))
        {
            buffer.emplace_back(coord, val);
        }
        else throw std::runtime_error("Invalid pixel index: {" + std::to_string(frame) + ", " +
                                      std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
    }
}

template <typename T, typename I>
void draw_pixel(ImageBuffer<std::tuple<I, I, T>> & buffer, const point_t & pt, I frame, I index, T val)
{
    using integer_type = typename point_t::value_type;
    if (val)
    {
        std::array<integer_type, 3> coord {static_cast<integer_type>(frame), pt.y(), pt.x()};
        if (buffer.is_inbound(coord))
        {
            buffer.emplace_back(coord, index, val);
        }
        else throw std::runtime_error("Invalid pixel index: {" + std::to_string(frame) + ", " +
                                      std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
    }
}

template <typename T>
T get_pixel(const array<T> & image, const point_t & pt)
{
    if (image.is_inbound(pt.coordinate()))
    {
        return image.at(pt.coordinate());
    }
    else throw std::runtime_error("Invalid pixel index: {" + std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
}

template <typename T, typename Key = typename table_t<T>::key_type>
void draw_pixel(table_t<T> & table, const Key & key , T val)
{
    if (val) table[key] = val;
}

template <typename T, typename Key = typename table_t<T>::key_type>
T get_pixel(const table_t<T> & table, const Key & key)
{
    if (table.contains(key)) return table.at(key);
    else return T();
}

}

/*----------------------------------------------------------------------------*/
/*-------------------------- Bresenham's Algorithm ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------
    Function :  plot_line_width()
    In       :  A 2d line defined by two (float) points (x0, y0, x1, y1)
                and a (float) width wd
    Out      :  A rasterized image of the line

    Reference:
        Author: Alois Zingl
        Title: A Rasterizing Algorithm for Drawing Curves
        pdf: http://members.chello.at/%7Eeasyfilter/Bresenham.pdf
        url: http://members.chello.at/~easyfilter/bresenham.html
------------------------------------------------------------------------------*/
template <typename T, bool IsForward>
struct BresenhamTraits;

template <typename T>
struct BresenhamTraits<T, true>
{
    static point_t start(const Line<T> & line) {return line.pt0.round();}
    static point_t end(const Line<T> & line) {return line.pt1.round();}
    static point_t step(const Point<T> & tau)
    {
        point_t point {};
        point.x() = (tau.x() >= 0) ? 1 : -1;
        point.y() = (tau.y() >= 0) ? 1 : -1;
        return point;
    }
};

template <typename T>
struct BresenhamTraits<T, false>
{
    static point_t start(const Line<T> & line) {return line.pt1.round();}
    static point_t end(const Line<T> & line) {return line.pt0.round();}
    static point_t step(const Point<T> & tau)
    {
        point_t point {};
        point.x() = (tau.x() >= 0) ? -1 : 1;
        point.y() = (tau.y() >= 0) ? -1 : 1;
        return point;
    }
};

// Iterator for a rasterizing algorithm for drawing lines
// See -> http://members.chello.at/~easyfilter/bresenham.html
template <typename T, bool IsForward>
struct BresenhamIterator
{
    point_t step;       /* Unit step                                                        */
    Point<T> tau;       /* Line norm, derivative of a line error function:                  */
                        /* error(x + dx, y + dy) = error(x, y) + tau.x * dx + tau.y * dy    */

    /* Temporal loop variables */

    T error;            /* Current error value                                              */
    point_t next_step;  /* Next step                                                        */
    point_t point;      /* Current point position                                           */

    template <typename Point1, typename Point2>
    BresenhamIterator(Point1 && tau, Point2 && start, const Point<T> & origin, const Line<T> & line) :
        step(BresenhamTraits<T, IsForward>::step(line.tau)), tau(std::forward<Point1>(tau)),
        error(dot(start - origin, tau)), next_step(), point(start) {}

    template <typename Point1, typename Point2>
    BresenhamIterator(Point1 && tau, Point2 && start, const Line<T> & line) :
        BresenhamIterator(std::forward<Point1>(tau), std::forward<Point2>(start), line.pt0, line) {}

    template <typename Point1>
    BresenhamIterator(Point1 && tau, const Line<T> & line) :
        BresenhamIterator(std::forward<Point1>(tau), BresenhamTraits<T, IsForward>::start(line), line.pt0, line) {}

    BresenhamIterator move_x() const
    {
        auto pix = *this;
        pix.add_x(1);
        return pix;
    }

    BresenhamIterator move_y() const
    {
        auto pix = *this;
        pix.add_y(1);
        return pix;
    }

    BresenhamIterator & step_xy()
    {
        if (next_step.x())
        {
            add_x(next_step.x());
            next_step.x() = 0;
        }
        if (next_step.y())
        {
            add_y(next_step.y());
            next_step.y() = 0;
        }
        return *this;
    }

    BresenhamIterator & step_x()
    {
        add_x(1);
        return *this;
    }

    BresenhamIterator & step_y()
    {
        add_y(1);
        return *this;
    }

    // Increment x if:
    //      e(x + sx, y + sy) + e(x, y + sy) < 0    if sx * tau.x() > 0
    //      e(x + sx, y + sy) + e(x, y + sy) > 0    if sx * tau.x() < 0

    bool is_xnext() const
    {
        if (step.x() * tau.x() == 0) return true;
        if (step.x() * tau.x() > 0) return 2 * e_xy() <= step.x() * tau.x();
        return 2 * e_xy() >= step.x() * tau.x();
    }

    // Increment y if:
    //      e(x + sx, y + sy) + e(x + sx, y) < 0    if sy * tau.y() > 0
    //      e(x + sx, y + sy) + e(x + sx, y) > 0    if sy * tau.y() < 0

    bool is_ynext() const
    {
        if (step.y() * tau.y() == 0) return true;
        if (step.y() * tau.y() > 0) return 2 * e_xy() <= step.y() * tau.y();
        return 2 * e_xy() >= step.y() * tau.y();
    }

    bool is_end(const point_t & end) const
    {
        bool isend = (next_step == point_t{});
        if (next_step.x()) isend |= (end.x() - point.x()) * step.x() <= 0;
        if (next_step.y()) isend |= (end.y() - point.y()) * step.y() <= 0;
        return isend;
    }

    void x_is_next() {next_step.x() = 1;}
    void y_is_next() {next_step.y() = 1;}

private:

    void add_x(int x)
    {
        point.x() += x * step.x(); error += x * step.x() * tau.x();
    }

    void add_y(int y)
    {
        point.y() += y * step.y(); error += y * step.y() * tau.y();
    }

    // Return e(x + sx, y + sy)

    T e_xy() const
    {
        return error + step.x() * tau.x() + step.y() * tau.y();
    }
};

template <typename T, class Func, typename = std::enable_if_t<
    std::is_invocable_v<std::remove_cvref_t<Func>, BresenhamIterator<T, true>, BresenhamIterator<T, true>>
>>
void draw_bresenham_func(const point_t & ubound, const Line<T> & line, T width, Func && func)
{
    /* Discrete line */
    point_t pt0 (line.pt0.round());
    point_t pt1 (line.pt1.round());

    T length = amplitude(line.tau);
    int wi = std::ceil(width) + 1;

    if (!length) return;

    /* Define bounds of the line plot */
    auto step = BresenhamTraits<T, true>::step(line.tau);
    auto bnd0 = (pt0 - wi * step).clamp(point_t{}, ubound);
    auto bnd1 = (pt1 + wi * step).clamp(point_t{}, ubound);

    BresenhamIterator<T, true> lpix {line.norm(), bnd0, line};
    BresenhamIterator<T, true> epix {line.tau, bnd0, line};

    do
    {
        // Perform a step
        lpix.step_xy();
        epix.step_xy();

        // Draw a pixel
        std::forward<Func>(func)(lpix, epix);

        if (lpix.is_xnext())
        {
            // x step
            for (auto liter = lpix.move_y(), eiter = epix.move_y();
                 std::abs(liter.error) < length * width && liter.point.y() != bnd1.y() + step.y();
                 liter.step_y(), eiter.step_y())
            {
                std::forward<Func>(func)(liter, eiter);
            }
            lpix.x_is_next();
            epix.x_is_next();
        }
        if (lpix.is_ynext())
        {
            // y step
            for (auto liter = lpix.move_x(), eiter = epix.move_x();
                 std::abs(liter.error) < length * width && liter.point.x() != bnd1.x() + step.x();
                 liter.step_x(), eiter.step_x())
            {
                std::forward<Func>(func)(liter, eiter);
            }
            lpix.y_is_next();
            epix.y_is_next();
        }
    }
    while (!lpix.is_end(bnd1));
}

template <typename I>
point_t get_ubound(const std::vector<I> & shape)
{
    using integer_type = typename point_t::value_type;
    return point_t{static_cast<integer_type>((shape[shape.size() - 1]) ? shape[shape.size() - 1] - 1 : 0),
                   static_cast<integer_type>((shape[shape.size() - 2]) ? shape[shape.size() - 2] - 1 : 0)};
}

}

#endif
