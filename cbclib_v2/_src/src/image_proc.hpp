#ifndef IMAGE_PROC_
#define IMAGE_PROC_
#include "array.hpp"
#include "geometry.hpp"

namespace cbclib {

template <typename T>
using table_t = std::map<std::pair<long, long>, T>;

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
void draw_pixel(ImageBuffer<std::pair<I, T>> & buffer, const Point<long> & pt, I frame, T val)
{
    if (val)
    {
        std::array<long, 3> coord {static_cast<long>(frame), pt.y(), pt.x()};
        if (buffer.is_inbound(coord))
        {
            buffer.emplace_back(coord, val);
        }
        else throw std::runtime_error("Invalid pixel index: {" + std::to_string(frame) + ", " +
                                      std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
    }
}

template <typename T, typename I>
void draw_pixel(ImageBuffer<std::tuple<I, I, T>> & buffer, const Point<long> & pt, I frame, I index, T val)
{
    if (val)
    {
        std::array<long, 3> coord {static_cast<long>(frame), pt.y(), pt.x()};
        if (buffer.is_inbound(coord))
        {
            buffer.emplace_back(coord, index, val);
        }
        else throw std::runtime_error("Invalid pixel index: {" + std::to_string(frame) + ", " +
                                      std::to_string(pt.y()) + ", " + std::to_string(pt.x()) + "}");
    }
}

template <typename T>
T get_pixel(const array<T> & image, const Point<long> & pt)
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
    static Point<long> start(const Line<T> & line) {return line.pt0.round();}
    static Point<long> end(const Line<T> & line) {return line.pt1.round();}
    static Point<long> step(const Point<T> & tau)
    {
        Point<long> point {};
        point.x() = (tau.x() >= 0) ? 1 : -1;
        point.y() = (tau.y() >= 0) ? 1 : -1;
        return point;
    }
};

template <typename T>
struct BresenhamTraits<T, false>
{
    static Point<long> start(const Line<T> & line) {return line.pt1.round();}
    static Point<long> end(const Line<T> & line) {return line.pt0.round();}
    static Point<long> step(const Point<T> & tau)
    {
        Point<long> point {};
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
    Point<long> step;       /* Unit step                                                        */
    Point<T> tau;       /* Line norm, derivative of a line error function:                  */
                        /* error(x + dx, y + dy) = error(x, y) + tau.x * dx + tau.y * dy    */

    /* Temporal loop variables */

    T error;            /* Current error value                                              */
    Point<long> next_step;  /* Next step                                                        */
    Point<long> point;      /* Current point position                                           */

    template <typename Point1, typename Point2>
    BresenhamIterator(Point1 && tau, Point2 && start, const Point<T> & origin, const Line<T> & line) :
        step(BresenhamTraits<T, IsForward>::step(line.tangent())), tau(std::forward<Point1>(tau)),
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

    bool is_end(const Point<long> & end) const
    {
        bool isend = (next_step == Point<long>{});
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
void draw_bresenham_func(const Point<long> & ubound, const Line<T> & line, T width, Func && func)
{
    /* Discrete line */
    Point<long> pt0 (line.pt0.round());
    Point<long> pt1 (line.pt1.round());

    T length = amplitude(line.tangent());
    int wi = std::ceil(width) + 1;

    if (!length) return;

    /* Define bounds of the line plot */
    auto step = BresenhamTraits<T, true>::step(line.tangent());
    auto bnd0 = (pt0 - wi * step).clamp(Point<long>{}, ubound);
    auto bnd1 = (pt1 + wi * step).clamp(Point<long>{}, ubound);

    BresenhamIterator<T, true> lpix {line.normal(), bnd0, line};
    BresenhamIterator<T, true> epix {line.tangent(), bnd0, line};

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
Point<long> get_ubound(const std::vector<I> & shape)
{
    return Point<long>{static_cast<long>((shape[shape.size() - 1]) ? shape[shape.size() - 1] - 1 : 0),
                       static_cast<long>((shape[shape.size() - 2]) ? shape[shape.size() - 2] - 1 : 0)};
}

}

#endif
