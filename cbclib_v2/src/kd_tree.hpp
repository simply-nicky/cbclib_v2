#ifndef KD_TREE_
#define KD_TREE_
#include "array.hpp"

namespace cbclib {

template<typename Point, typename Data>
class KDTree
{
public:
    using point_type = typename Point::value_type;
    using item_type = std::pair<Point, Data>;

    class KDNode
    {
    public:
        using node_ptr = KDNode *;

        item_type item;
        int cut_dim;

        KDNode() = default;

        template <typename Item, typename = std::enable_if_t<std::is_same_v<item_type, std::remove_cvref_t<Item>>>>
        KDNode(Item && item, int dir, node_ptr lt = node_ptr(), node_ptr rt = node_ptr(), node_ptr par = node_ptr()) :
            item(std::forward<Item>(item)), cut_dim(dir), left(lt), right(rt), parent(par) {}

        Point & point() {return item.first;}
        const Point & point() const {return item.first;}

        Data & data() {return item.second;}
        const Data & data() const {return item.second;}

        template <typename Pt>
        bool is_left(const Pt & pt) const
        {
            return pt[cut_dim] < point()[cut_dim];
        }

    private:
        node_ptr left;
        node_ptr right;
        node_ptr parent;

        friend class KDTree<Point, Data>;
    };

    using node_t = KDNode;
    using node_ptr = KDNode *;

    class KDIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = node_t;
        using difference_type = std::ptrdiff_t;
        using pointer = node_ptr;
        using reference = const node_t &;

        KDIterator() : ptr(nullptr), root(nullptr) {}

        bool operator==(const KDIterator & rhs) const
        {
            return root == rhs.root && ptr == rhs.ptr;
        }

        bool operator!=(const KDIterator & rhs) const {return !operator==(rhs);}

        KDIterator & operator++()
        {
            if (!ptr)
            {
                // ++ from end(). Get the root of the tree
                ptr = root;

                // error! ++ requested for an empty tree
                while (ptr && ptr->left) ptr = ptr->left;

            }
            else if (ptr->right)
            {
                // successor is the farthest left node of right subtree
                ptr = ptr->right;

                while (ptr->left) ptr = ptr->left;
            }
            else
            {
                // have already processed the left subtree, and
                // there is no right subtree. move up the tree,
                // looking for a parent for which nodePtr is a left child,
                // stopping if the parent becomes NULL. a non-NULL parent
                // is the successor. if parent is NULL, the original node
                // was the last node inorder, and its successor
                // is the end of the list
                node_ptr p = ptr->parent;
                while (p && ptr == p->right)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the right-most node in
                // the tree, nodePtr = nullptr, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        KDIterator & operator--()
        {
            if (!ptr)
            {
                // -- from end(). Get the root of the tree
                ptr = root;

                // move to the largest value in the tree,
                // which is the last node inorder
                while (ptr && ptr->right) ptr = ptr->right;
            }
            else if (ptr->left)
            {
                // must have gotten here by processing all the nodes
                // on the left branch. predecessor is the farthest
                // right node of the left subtree
                ptr = ptr->left;

                while (ptr->right) ptr = ptr->right;
            }
            else
            {
                // must have gotten here by going right and then
                // far left. move up the tree, looking for a parent
                // for which ptr is a right child, stopping if the
                // parent becomes nullptr. a non-nullptr parent is the
                // predecessor. if parent is nullptr, the original node
                // was the first node inorder, and its predecessor
                // is the end of the list
                node_ptr p = ptr->parent;
                while (p && ptr == p->left)
                {
                    ptr = p; p = p->parent;
                }

                // if we were previously at the left-most node in
                // the tree, ptr = NULL, and the iterator specifies
                // the end of the list
                ptr = p;
            }

            return *this;
        }

        KDIterator operator--(int)
        {
            auto saved = *this;
            operator--();
            return saved;
        }

        reference operator*() const {return *ptr;}
        pointer operator->() const {return ptr;}

    private:
        friend class KDTree<Point, Data>;

        node_ptr ptr;
        node_ptr root;

        KDIterator(node_ptr ptr, node_ptr root) : ptr(ptr), root(root) {}
    };

    using const_iterator = KDIterator;
    using iterator = const_iterator;

    class Rectangle
    {
    public:
        std::vector<point_type> low, high;

        Rectangle() = default;

        Rectangle(const KDTree<Point, Data> & tree)
        {
            for (size_t i = 0; i < tree.ndim; i++)
            {
                low.push_back(tree.find_min(i)->point()[i]);
                high.push_back(tree.find_max(i)->point()[i]);
            }
        }

        void update(const Point & pt)
        {
            for (size_t i = 0; i < pt.size(); ++i)
            {
                low[i] = std::min(low[i], pt[i]);
                high[i] = std::max(high[i], pt[i]);
            }
        }

        template <typename Pt, typename T = std::common_type_t<point_type, typename Pt::value_type>> 
        T distance(const Pt & point) const
        {
            T dist = T();
            for (size_t i = 0; i < low.size(); i++)
            {
                if (point[i] < low[i]) dist += std::pow(low[i] - point[i], 2);
                if (point[i] > high[i]) dist += std::pow(point[i] - high[i], 2);
            }
            return dist;
        }

    private:
        friend class KDTree<Point, Data>;

        template <bool IsLeft>
        void trim_back(node_ptr node, const point_type & value)
        {
            if constexpr (IsLeft) high[node->cut_dim] = value;
            else low[node->cut_dim] = value;
        }

        point_type trim_left(node_ptr node)
        {
            auto value = high[node->cut_dim];
            high[node->cut_dim] = node->point()[node->cut_dim];
            return value;
        }

        point_type trim_right(node_ptr node)
        {
            auto value = low[node->cut_dim];
            low[node->cut_dim] = node->point()[node->cut_dim];
            return value;
        }
    };

    using rect_t = Rectangle;
    using rect_ptr = rect_t *;

    size_t ndim;

    template <typename T>
    using query_t = std::pair<const_iterator, T>;
    template <typename T>
    using stack_t = std::vector<std::pair<const_iterator, T>>;

    KDTree() : ndim(), root(nullptr), rect(nullptr) {}
    KDTree(size_t ndim) : ndim(ndim), root(nullptr), rect(nullptr) {}

    template <typename InputIt, typename = std::enable_if_t<
        std::is_same_v<item_type, typename std::iterator_traits<InputIt>::value_type>
    >>
    KDTree(InputIt first, InputIt last, size_t ndim) : ndim(ndim), root(nullptr), rect(nullptr)
    {
        root = build_tree(first, last, node_ptr(), 0);
        if (root) rect = new rect_t{*this};
    }

    KDTree(const KDTree<Point, Data> & rhs) : ndim(rhs.ndim), root(clone_node(rhs.root)), rect(clone_rect(rhs.rect)) {}
    KDTree(KDTree<Point, Data> && rhs) : ndim(rhs.ndim), root(rhs.root), rect(rhs.rect)
    {
        rhs.root = node_ptr(); rhs.rect = rect_ptr();
    }

    ~KDTree() {clear();}

    KDTree<Point, Data> & operator=(const KDTree<Point, Data> & rhs)
    {
        if (&rhs != this)
        {
            KDTree<Point, Data> copy {rhs};
            swap(copy);
        }
        return *this;
    }

    KDTree<Point, Data> & operator=(KDTree<Point, Data> && rhs)
    {
        swap(rhs);
        return *this;
    }

    bool is_empty() const {return !root;}

    void clear()
    {
        root = clear_node(root);
        rect = clear_rect(rect);
    }

    const_iterator begin() const
    {
        return {begin_node(root), root};
    }

    const_iterator end() const
    {
        return {node_ptr(), root};
    }

    const_iterator insert(item_type && item)
    {
        const_iterator inserted;
        if (root) std::tie(root, inserted) = insert_node(root, std::move(item), root, root->cut_dim);
        else std::tie(root, inserted) = insert_node(root, std::move(item), root, 0);

        if (inserted != end())
        {
            if (!rect) rect = new rect_t{*this};
            else rect->update(item.first);
        }

        return inserted;
    }

    size_t erase(const Point & point)
    {
        size_t removed;
        std::tie(root, removed) = remove_node(root, point);

        if (rect && removed)
        {
            if (root) *rect = rect_t{*this};
            else rect = rect_ptr();
        }

        return removed;
    }

    const_iterator erase(const_iterator pos)
    {
        if (pos != end())
        {
            erase((pos++)->point());
        }
        return pos;
    }

    const_iterator find(const Point & point) const
    {
        // I clone the rect to avoid the race condition so that rect stays unchanged
        auto copy = clone_rect(rect);
        const_iterator result {find_node(root, point, copy, node_ptr()), root};
        clear_rect(copy);
        return result;
    }

    const_iterator find_min(int axis) const
    {
        return {find_min_node(root, axis), root};
    }

    const_iterator find_max(int axis) const
    {
        return {find_max_node(root, axis), root};
    }

    template <typename Pt, typename T = std::common_type_t<point_type, typename Pt::value_type>>
    query_t<T> find_nearest(const Pt & point) const
    {
        auto copy = clone_rect(rect);
        query_t<T> result {const_iterator(root, root), std::numeric_limits<T>::max()};
        nearest_node<Pt, T>(result, root, copy, point);
        clear_rect(copy);
        return result;
    }

    template <typename Pt, typename T = std::common_type_t<point_type, typename Pt::value_type>>
    stack_t<T> find_k_nearest(const Pt & point, size_t k) const
    {
        if (k)
        {
            auto copy = clone_rect(rect);
            stack_t<T> result;
            nearest_k_nodes<Pt, T>(result, root, copy, point, k);
            clear_rect(copy);
            return result;
        }
        return {};
    }

    template <typename Pt, typename T = std::common_type_t<point_type, typename Pt::value_type>>
    stack_t<T> find_range(const Pt & point, T range_sq) const
    {
        auto copy = clone_rect(rect);
        stack_t<T> result;
        find_range_node(result, root, copy, point, range_sq);
        clear_rect(copy);
        return result;
    }

    Rectangle rectangle() const
    {
        if (rect) return *rect;
        else return Rectangle();
    }

    Point & point(iterator pos) {return pos->point();}
    const Point & point(const_iterator pos) const {return pos->point();}

    Data & data(iterator pos) {return pos->data();}
    const Data & data(const_iterator pos) const {return pos->data();}

    void print(std::ostream & os) const
    {
        print_node(os, root);
        print_rect(os);
    }

private:
    node_ptr root;
    rect_ptr rect;

    void swap(KDTree<Point, Data> & rhs)
    {
        std::swap(rhs.ndim, ndim);
        std::swap(rhs.root, root);
        std::swap(rhs.rect, rect);
    }

    template <class InputIt>
    node_ptr build_tree(InputIt first, InputIt last, node_ptr par, int dir) const
    {
        using value_t = typename std::iterator_traits<InputIt>::value_type;

        if (last <= first) return node_ptr();
        else if (last == std::next(first))
        {
            return new node_t{*first, dir, node_ptr(), node_ptr(), par};
        }
        else
        {
            auto compare = [dir](const value_t & a, const value_t & b){return a.first[dir] < b.first[dir];};
            auto iter = median_element(first, last, compare);

            node_ptr node = new node_t{*iter, dir, node_ptr(), node_ptr(), par};
            node->left = build_tree(first, iter, node, (dir + 1) % ndim);
            node->right = build_tree(std::next(iter), last, node, (dir + 1) % ndim);
            return node;
        }
    }

    rect_ptr clear_rect(rect_ptr rect) const
    {
        if (rect) delete rect;
        return rect_ptr();
    }

    node_ptr clear_node(node_ptr node) const
    {
        if (node)
        {
            node->left = clear_node(node->left);
            node->right = clear_node(node->right);
            delete node;
        }

        return node_ptr();
    }

    node_ptr clone_node(node_ptr node) const
    {
        if (!node)
        {
            return node;
        }
        else
        {
            return new node_t{node->item, node->cut_dim, clone_node(node->left), clone_node(node->right), node->parent};
        }
    }

    rect_ptr clone_rect(rect_ptr rect) const
    {
        if (!rect) return rect;
        else return new rect_t(*rect);
    }

    std::tuple<node_ptr, const_iterator> insert_node(node_ptr node, item_type && item, node_ptr par, int dir) const
    {
        // Create new node if empty
        if (!node)
        {
            node = new node_t{std::move(item), dir, node_ptr(), node_ptr(), par};
            return {node, const_iterator(node, root)};
        }

        // Duplicate data point, no insertion
        if (item.first == node->point())
        {
            return {node, end()};
        }

        const_iterator inserted;

        if (node->is_left(item.first))
        {
            // left of splitting line
            std::tie(node->left, inserted) = insert_node(node->left, std::move(item), node, (node->cut_dim + 1) % ndim);
        }
        else
        {
            // on or right of splitting line
            std::tie(node->right, inserted) = insert_node(node->right, std::move(item), node, (node->cut_dim + 1) % ndim);
        }

        return {node, inserted};
    }

    std::tuple<node_ptr, size_t> remove_node(node_ptr node, const Point & point) const
    {
        // Fell out of tree
        if (!node) return {node, 0};

        size_t removed;

        // Found the node
        if (node->point() == point)
        {
            // Take replacement from right
            if (node->right)
            {
                // Swapping the node
                node->item = find_min_node(node->right, node->cut_dim)->item;

                std::tie(node->right, removed) = remove_node(node->right, node->point());
            }
            // Take replacement from left
            else if (node->left)
            {
                // Swapping the nodes 
                node->item = find_min_node(node->left, node->cut_dim)->item;

                // move left subtree to right!
                std::tie(node->right, removed) = remove_node(node->left, node->point());
                // left subtree is now empty
                node->left = node_ptr();
            }
            // Remove this leaf
            else
            {
                node = node_ptr(); removed = 1;
            }
        }
        // Search left subtree
        else if (node->is_left(point))
        {
            std::tie(node->left, removed) = remove_node(node->left, point);
        }
        // Search right subtree
        else std::tie(node->right, removed) = remove_node(node->right, point);

        return {node, removed};
    }

    /* Node a is always not null */
    node_ptr min_node(node_ptr a, node_ptr b, node_ptr c, int axis) const
    {
        if (b && b->point()[axis] < a->point()[axis])
        {
            if (c && c->point()[axis] < b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] < a->point()[axis]) return c;
        return a;
    }

    /* Node a is always not null */
    node_ptr max_node(node_ptr a, node_ptr b, node_ptr c, int axis) const
    {
        if (b && b->point()[axis] > a->point()[axis])
        {
            if (c && c->point()[axis] > b->point()[axis]) return c;
            return b;
        }
        if (c && c->point()[axis] > a->point()[axis]) return c;
        return a;
    }

    node_ptr find_min_node(node_ptr node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->left) return node;
            else return find_min_node(node->left, axis);
        }
        else return min_node(node, find_min_node(node->left, axis), find_min_node(node->right, axis), axis);
    }

    node_ptr find_max_node(node_ptr node, int axis) const
    {
        // Fell out of tree
        if (!node) return node;

        if (node->cut_dim == axis)
        {
            if (!node->right) return node;
            else return find_max_node(node->right, axis);
        }
        else return max_node(node, find_max_node(node->left, axis), find_max_node(node->right, axis), axis);
    }

    node_ptr begin_node(node_ptr node) const
    {
        if (!node) return node;
        if (!node->left) return node;
        return begin_node(node->left);
    }

    template <typename Point1, typename Point2,
        typename T = std::common_type_t<typename Point1::value_type, typename Point2::value_type>
    >
    T distance(const Point1 & a, const Point2 & b) const
    {
        T dist = T();
        for (size_t i = 0; i < ndim; i++) dist += std::pow(a[i] - b[i], 2);
        return dist;
    }

    node_ptr find_node(node_ptr node, const Point & point, rect_ptr rect, node_ptr query) const
    {
        // Fell out of tree
        if (!node || !rect) return query;
        // This cell is too far away
        if (rect->distance(point) > point_type()) return query;

        // We found the node
        if (point == node->point()) query = node;

        // pt is close to left child
        if (node->is_left(point))
        {
            // First left then right
            auto lvalue = rect->trim_left(node);
            query = find_node(node->left, point, rect, query);
            rect->template trim_back<true>(node, lvalue);

            auto rvalue = rect->trim_right(node);
            query = find_node(node->right, point, rect, query);
            rect->template trim_back<false>(node, rvalue);
        }
        // pt is closer to right child
        else
        {
            // First right then left
            auto rvalue = rect->trim_right(node);
            query = find_node(node->right, point, rect, query);
            rect->template trim_back<false>(node, rvalue);

            auto lvalue = rect->trim_left(node);
            query = find_node(node->left, point, rect, query);
            rect->template trim_back<true>(node, lvalue);
        }

        return query;
    }

    template <typename Pt, typename T>
    void nearest_node(query_t<T> & query, node_ptr node, rect_ptr rect, const Pt & point) const
    {
        // Fell out of tree
        if (!node || !rect) return;
        // This cell is too far away
        if (rect->distance(point) >= query.second) return;

        // Update if the root is closer
        auto dist_sq = distance(node->point(), point);
        if (dist_sq < query.second) query = std::make_pair(const_iterator(node, root), dist_sq);

        // pt is close to left child
        if (node->is_left(point))
        {
            // First left then right
            auto lvalue = rect->trim_left(node);
            nearest_node(query, node->left, rect, point);
            rect->template trim_back<true>(node, lvalue);

            auto rvalue = rect->trim_right(node);
            nearest_node(query, node->right, rect, point);
            rect->template trim_back<false>(node, rvalue);
        }
        // pt is closer to right child
        else
        {
            // First right then left
            auto rvalue = rect->trim_right(node);
            nearest_node(query, node->right, rect, point);
            rect->template trim_back<false>(node, rvalue);

            auto lvalue = rect->trim_left(node);
            nearest_node(query, node->left, rect, point);
            rect->template trim_back<true>(node, lvalue);
        }
    }

    template <typename T>
    void insert_to_stack(stack_t<T> & stack, node_ptr node, T dist_sq) const
    {
        auto compare = [](const std::pair<const_iterator, T> & elem, T dist_sq)
        {
            return elem.second < dist_sq;
        };

        auto iter = std::lower_bound(stack.begin(), stack.end(), dist_sq, compare);
        stack.insert(iter, std::make_pair(const_iterator(node, root), dist_sq));
    }

    template <typename Pt, typename T>
    void nearest_k_nodes(stack_t<T> & stack, node_ptr node, rect_ptr rect, const Pt & point, size_t k) const
    {
        // Fell out of tree
        if (!node || !rect) return;

        // The stack is not full yet
        if (stack.size() < k)
        {
            // Insert in the stack according to its distance
            auto dist_sq = distance(node->point(), point);
            insert_to_stack(stack, node, dist_sq);
        }
        // The stack is full
        else
        {
            // This cell is too far away
            if (rect->distance(point) >= stack.back().second) return;

            // Update if the root is close
            auto dist_sq = distance(node->point(), point);
            if (dist_sq < stack.back().second)
            {
                insert_to_stack(stack, node, dist_sq);
                stack.pop_back();
            }
        }

        // pt is close to left child
        if (node->is_left(point))
        {
            // First left then right
            auto lvalue = rect->trim_left(node);
            nearest_k_nodes(stack, node->left, rect, point, k);
            rect->template trim_back<true>(node, lvalue);

            auto rvalue = rect->trim_right(node);
            nearest_k_nodes(stack, node->right, rect, point, k);
            rect->template trim_back<false>(node, rvalue);
        }
        // pt is closer to right child
        else
        {
            // First right then left
            auto rvalue = rect->trim_right(node);
            nearest_k_nodes(stack, node->right, rect, point, k);
            rect->template trim_back<false>(node, rvalue);

            auto lvalue = rect->trim_left(node);
            nearest_k_nodes(stack, node->left, rect, point, k);
            rect->template trim_back<true>(node, lvalue);
        }
    }

    template <typename Pt, typename T>
    void stack_push_node(stack_t<T> & stack, node_ptr node, const Pt & point) const
    {
        if (node->left) stack_push_node(stack, node->left, point);
        stack.emplace_back(const_iterator(node, root), distance(node->point(), point));
        if (node->right) stack_push_node(stack, node->right, point);
    }

    template <typename Pt, typename T>
    void find_range_node(stack_t<T> & stack, node_ptr node, rect_ptr rect, const Pt & point, T range_sq) const
    {
        // Fell out of tree
        if (!node || !rect) return;
        // The cell doesn't overlap the query
        if (rect->distance(point) > range_sq) return;

        // The query contains the cell
        if (distance(point, rect->low) < range_sq && distance(point, rect->high) < range_sq)
        {
            stack_push_node(stack, node, point);
            return;
        }

        auto dist_sq = distance(point, node->point());
        // Put this item to stack
        if (dist_sq < range_sq) stack.emplace_back(const_iterator(node, root), dist_sq);

        // Search left subtree
        auto lvalue = rect->trim_left(node);
        find_range_node(stack, node->left, rect, point, range_sq);
        rect->template trim_back<true>(node, lvalue);

        // Search right subtree
        auto rvalue = rect->trim_right(node);
        find_range_node(stack, node->right, rect, point, range_sq);
        rect->template trim_back<false>(node, rvalue);
    }

    std::ostream & print_rect(std::ostream & os) const
    {
        if (!rect) return os;

        os << "low  : [";
        std::copy(rect->low.begin(), rect->low.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;

        os << "high : [";
        std::copy(rect->high.begin(), rect->high.end(), std::experimental::make_ostream_joiner(os, ", "));
        os << "]" << std::endl;
        return os;
    }

    std::ostream & print_node(std::ostream & os, node_ptr node, size_t level = 0) const
    {
        if (!node) return os;

        print_node(os, node->left, level + 1);

        os << std::string(level, '\t') << "(";
        std::copy(node->point().begin(), node->point().end(), std::experimental::make_ostream_joiner(os, ", "));
        os << ")" << " axis = " << node->cut_dim << std::endl;

        print_node(os, node->right, level + 1);
        return os;
    }
};

}

#endif
