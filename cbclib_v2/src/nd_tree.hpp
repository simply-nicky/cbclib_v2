#ifndef ND_TREE_
#define ND_TREE_
#include "geometry.hpp"

namespace cbclib {

template <typename T, size_t N>
class BoxND
{
public:
    using value_type = T;
    using point_type = PointND<T, N>;

    BoxND() : min(), max()
    {
        for (size_t i = 0; i < N; i++)
        {
            min[i] = std::numeric_limits<T>::max();
            max[i] = std::numeric_limits<T>::lowest();
        }
    }

    template <typename PtA, typename PtB, typename = std::enable_if_t<
        std::is_same_v<point_type, std::remove_cvref_t<PtA>> &&
        std::is_same_v<point_type, std::remove_cvref_t<PtB>>
    >>
    BoxND(PtA && a, PtB && b) : min(std::forward<PtA>(a)), max(std::forward<PtB>(b))
    {
        for (size_t i = 0; i < N; i++) if (min[i] > max[i]) std::swap(min[i], max[i]);
    }

    point_type size() const {return max - min;}

    point_type center() const {return min + 0.5 * size();}

    bool contains(const BoxND & rhs) const &
    {
        bool flag = true;
        for (size_t i = 0; i < N; i++) flag &= min[i] <= rhs.min[i] && max[i] >= rhs.max[i];
        return flag;
    }

    bool intersects(const BoxND & rhs) const &
    {
        bool flag = true;
        for (size_t i = 0; i < N; i++) flag &= min[i] < rhs.max[i] && max[i] > rhs.min[i];
        return flag;
    }

    std::array<BoxND, 1 << N> quadrants() const
    {
        auto ctr = center();
        std::array<BoxND, 1 << N> result;
        for (size_t i = 0; i < (1 << N); i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                if ((i >> j) & 1)
                {
                    result[i].min[j] = ctr[j]; result[i].max[j] = max[j];
                }
                else
                {
                    result[i].min[j] = min[j]; result[i].max[j] = ctr[j];
                }
            }
        }
        return result;
    }

    BoxND & operator|=(const point_type & rhs) &
    {
        for (size_t i = 0; i < N; i++)
        {
            min[i] = std::min(min[i], rhs[i]);
            max[i] = std::max(max[i], rhs[i]);
        }
        return *this;
    }

    BoxND & operator|=(const BoxND & rhs) &
    {
        for (size_t i = 0; i < N; i++)
        {
            min[i] = std::min(min[i], rhs.min[i]);
            max[i] = std::max(max[i], rhs.max[i]);
        }
        return *this;
    }

    template <typename V, typename U = std::common_type_t<T, V>>
    friend U distance(const BoxND & a, const BoxND<V, N> & b)
    {
        U dist = U();
        for (size_t i = 0; i < N; i++)
        {
            if (a.min[i] > b.max[i]) dist += (a.min[i] - b.max[i]) * (a.min[i] - b.max[i]);
            else if (a.max[i] < b.min[i]) dist += (a.max[i] - b.min[i]) * (a.max[i] - b.min[i]);
        }
        return dist;
    }

    friend std::ostream & operator<<(std::ostream & os, const BoxND & box)
    {
        os << "{" << box.min << ", " << box.max << "}";
        return os;
    }

    std::array<T, 2 * N> to_array() const {return concatenate(min.to_array(), max.to_array());}

private:
    point_type min, max;
};

template <typename Element, typename Data, size_t N, size_t MaxDepth = 8, size_t Threshold = 8>
class NDTree
{
public:
    using value_type = std::pair<Element, Data>;

private:
    using box_t = BoxND<typename Element::value_type, N>;
    using box_ptr = BoxND<typename Element::value_type, N> *;

    // Represents an element in the quadtree
    struct Item
    {
        // The pair of an element and it's corresponding data
        value_type value;
        // The bounding box
        box_t box;
    };

    using item_t = Item;
    using item_ptr = std::list<item_t>::const_iterator;
    using item_stack_t = std::vector<item_ptr>;

    class Node
    {
    public:
        using node_ptr = Node *;

        Node() : box(), items(), depth(), children(), parent(nullptr)
        {
            set_children();
        }

        template <typename Bx, typename = std::enable_if_t<
            std::is_same_v<std::remove_cvref_t<Bx>, box_t>
        >>
        Node(Bx && box) : box(std::forward<Bx>(box)), items(), depth(), children(), parent(nullptr)
        {
            set_children();
        }

        bool is_leaf() const
        {
            bool flag = true;
            for (auto child : children) flag &= !child;
            return flag;
        }

        bool is_empty() const {return !items.size();}

    private:
        box_t box;
        std::list<item_ptr> items;
        size_t depth;
        std::array<node_ptr, 1 << N> children;
        node_ptr parent;

        Node(box_t && box, size_t depth, node_ptr parent) : box(std::move(box)), items(), depth(depth), children(), parent(parent)
        {
            set_children();
        }

        Node(const std::list<item_ptr> & items, const box_t & box, size_t depth, node_ptr parent)
            : box(box), items(items), depth(depth), children(), parent(parent)
        {
            set_children();
        }

        void set_children() {for (auto & child : children) child = nullptr;}

        friend class NDTree;
    };

    using node_t = Node;
    using node_ptr = Node *;

public:
    class NDIterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = node_t;
        using difference_type = std::ptrdiff_t;
        using pointer = node_ptr;
        using reference = const node_t &;

        NDIterator() : ptr(nullptr), index(), root(nullptr) {}

        bool operator==(const NDIterator & rhs) const
        {
            return root == rhs.root && ptr == rhs.ptr;
        }

        bool operator!=(const NDIterator & rhs) const {return !operator==(rhs);}

        NDIterator & operator++()
        {
            if (!ptr)
            {
                // ++ from end(). Get the root of the tree
                ptr = root;

                // error! ++ requested for an empty tree
                while (ptr && !ptr->is_leaf()) {ptr = ptr->children[0]; index = 0;}

            }
            else if (ptr->parent)
            {
                if (index != (1 << N) - 1)
                {
                    // successor is the farthest left node of the next subtree
                    ptr = ptr->parent->children[++index];

                    while (!ptr->is_leaf()) {ptr = ptr->children[0]; index = 0;}
                }
                else
                {
                    ptr = ptr->parent;
                    set_index();
                }
            }
            else ptr = node_ptr();

            return *this;
        }

        NDIterator operator++(int)
        {
            auto saved = *this;
            operator++();
            return saved;
        }

        NDIterator & operator--()
        {
            if (!ptr)
            {
                // -- from end(). Get the root of the tree
                ptr = root;

                // error! -- requested for an empty tree
                while (ptr && !ptr->is_leaf()) {ptr = ptr->children[(1 << N) - 1]; index = (1 << N) - 1;}
            }
            else if (ptr->parent)
            {
                if (index != 0)
                {
                    // successor is the farthest right node of the next subtree)
                    ptr = ptr->parent->children[--index];

                    while (!ptr->is_leaf()) {ptr = ptr->children[(1 << N) - 1]; index = (1 << N) - 1;}
                }
                else
                {
                    ptr = ptr->parent;
                    set_index();
                }
            }
            else ptr = node_ptr();

            return *this;
        }

        NDIterator operator--(int)
        {
            auto saved = *this;
            operator--();
            return saved;
        }

        reference operator*() const {return *ptr;}
        pointer operator->() const {return ptr;}

    private:
        node_ptr ptr;
        size_t index;
        node_ptr root;

        NDIterator(node_ptr ptr, node_ptr root) : ptr(ptr), index(), root(root)
        {
            set_index();
        }

        void set_index()
        {
            if (ptr && ptr->parent)
            {
                for (size_t i = 0; i < (1 << N); i++) if (ptr->parent->children[i] == ptr) {index = i; break;}
            }
            else index = 0; 
        }
        
        friend class NDTree;
    };

    using const_iterator = NDIterator;
    using iterator = const_iterator;

    template <typename Bx, typename Func, typename = std::enable_if_t<
        std::is_same_v<box_t, std::remove_cvref_t<Bx>> &&
        std::is_invocable_r_v<box_t, std::remove_cvref_t<Func>, const Element &>
    >>
    NDTree(Func && func, Bx && box) : NDTree(std::forward<Func>(func))
    {
        root = new node_t{std::forward<Bx>(box)};
    }

    template <typename Bx, typename = std::enable_if_t<
        std::is_same_v<box_t, std::remove_cvref_t<Bx>>
    >>
    NDTree(box_t (*func)(const Element &), Bx && box) : NDTree(func)
    {
        root = new node_t{std::forward<Bx>(box)};
    }

    template <typename InputIt, typename Func, typename = std::enable_if_t<
        std::is_same_v<value_type, typename std::iterator_traits<InputIt>::value_type> &&
        std::is_invocable_r_v<box_t, std::remove_cvref_t<Func>, const Element &>
    >>
    NDTree(InputIt first, InputIt last, Func && func) : NDTree(std::forward<Func>(func))
    {
        auto make_item = [this](const value_type & value){item_t{value, get_box(value.first)};};
        std::transform(first, last, std::back_inserter(tree_items), make_item);

        root = new node_t{};
        for (const auto & item : tree_items) root->box |= item.box;

        for (auto iter = tree_items.begin(); iter != tree_items.end(); ++iter) insert_node(root, iter, size_t());
    }

    NDTree(const NDTree & rhs) : get_box(rhs.get_box), root(clone_node(rhs.root)), tree_items(rhs.tree_items) {}
    NDTree(NDTree && rhs) : get_box(std::move(rhs.get_box)), root(rhs.root), tree_items(std::move(rhs.tree_items))
    {
        rhs.root = node_ptr();
    }

    ~NDTree() {root = clear_node(root); tree_items.clear();}

    NDTree & operator=(const NDTree & rhs)
    {
        if (&rhs != this)
        {
            NDTree copy {rhs};
            swap(copy);
        }
        return *this;
    }

    NDTree & operator=(NDTree && rhs)
    {
        swap(rhs);
        return *this;
    }

    bool is_empty() const {return empty_leaves(root);}

    void clear()
    {
        box_t box = root->box;
        root = clear_node(root);
        root = new node_t{std::move(box)};

        tree_items.clear();
    }

    size_t insert(value_type && value)
    {
        auto elem_box = get_box(value.first);
        if (!root->box.intersects(elem_box)) return size_t();

        auto iter = tree_items.insert(tree_items.end(), {std::move(value), std::move(elem_box)});

        return insert_node(root, iter, size_t());
    }

    size_t erase(const Element & elem)
    {
        auto elem_box = get_box(elem);
        if (!root->box.intersects(elem_box)) return size_t();

        auto [item, count] = erase_node(root, elem, elem_box, size_t());

        if (item != item_ptr()) tree_items.erase(item);
        return count;
    }

    template <typename T>
    using knn_stack_t = std::vector<std::pair<item_ptr, T>>;

    template <typename T, typename V = std::common_type_t<T, typename Element::value_type>>
    knn_stack_t<V> find_k_nearest(const BoxND<T, N> & query, size_t k) const
    {
        knn_stack_t<V> result;
        nearest_nodes(result, root, query, k);
        return result;
    }

    template <typename T, typename V = std::common_type_t<T, typename Element::value_type>>
    knn_stack_t<V> find_range(const BoxND<T, N> & query, T range_sq) const
    {
        knn_stack_t<V> result;
        nodes_in_range(result, root, query, range_sq);
        return result;
    }

    template <typename T>
    item_stack_t find_intersections(const BoxND<T, N> & query) const
    {
        item_stack_t result;
        intersected_nodes(result, root, query);
        return result;
    }

    const_iterator begin() const
    {
        return {begin_node(root), root};
    }

    const_iterator end() const
    {
        return {node_ptr(), root};
    }
    
    const box_t & box() const {return root->box;}

    const box_t & box(const_iterator pos) const {return pos->box;}

    const value_type & element(item_ptr pos) const {return pos->value;}

    const box_t & element_box(item_ptr pos) const {return pos->box;}

    const std::list<item_ptr> & items(const_iterator pos) const {return pos->items;}

    box_t to_box(const Element & elem) const {return get_box(elem);}

private:
    std::function<box_t(const Element &)> get_box;
    node_ptr root;
    std::list<item_t> tree_items;

    template <typename Func, typename = std::enable_if_t<
        std::is_invocable_r_v<box_t, std::remove_cvref_t<Func>, const Element &>
    >>
    NDTree(Func && func) : get_box(std::forward<Func>(func)), root(nullptr), tree_items() {}

    NDTree(box_t (*func)(const Element &)) : get_box(func), root(nullptr), tree_items() {}

    void swap(NDTree & rhs)
    {
        std::swap(rhs.get_box, get_box);
        std::swap(rhs.root, root);
        std::swap(rhs.tree_items, tree_items);
    }

    node_ptr clear_node(node_ptr node) const
    {
        if (node)
        {
            if (!node->is_leaf()) for (auto child : node->children) child = clear_node(child);
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
            auto clone = new node_t{node->items, node->box, node->depth, node->parent};
            for (auto i = 0; i < (1 << N); i++) clone->children[i] = clone_node(node->children[i]);
            return clone;
        }
    }

    node_ptr begin_node(node_ptr node) const
    {
        if (!node || node->is_leaf()) return node;
        return begin_node(node->children[0]);
    }

    bool empty_leaves(node_ptr node) const
    {
        bool flag = true;

        if (node->is_leaf())
        {
            flag &= node->is_empty();
        }
        else
        {
            for (auto child : node->children)
            {
                if (child) flag &= empty_leaves(child);
            }
        }

        return flag;
    }

    node_ptr split_node(node_ptr leaf)
    {
        auto quadrants = leaf->box.quadrants();

        for (auto & child : leaf->children)
        {
            auto index = std::addressof(child) - leaf->children.data();
            child = new node_t{std::move(quadrants[index]), leaf->depth + 1, leaf};
        }

        for (auto item : leaf->items)
        {
            for (auto child : leaf->children)
            {
                if (child->box.intersects(item->box)) child->items.push_back(item);
            }
        }

        leaf->items.clear();

        return leaf;
    }

    node_ptr collapse_node(node_ptr node)
    {
        for (auto child : node->children) if (child) child = clear_node(child);

        return node;
    }

    size_t insert_node(node_ptr node, item_ptr item, size_t count)
    {
        if (node->is_leaf())
        {
            auto compare = [&item](item_ptr ptr){return ptr->box != item->box;};
            auto is_absent = std::all_of(node->items.begin(), node->items.end(), compare);
            if (is_absent) {node->items.push_back(item); count++;}

            if (node->items.size() >= Threshold && node->depth < MaxDepth)
            {
                node = split_node(node);
            }
        }
        else
        {
            for (auto child : node->children)
            {
                if (child->box.intersects(item->box))
                {
                    count = insert_node(child, item, count);
                }
            }
        }

        return count;
    }

    std::pair<item_ptr, size_t> erase_node(node_ptr node, const Element & elem, const box_t & elem_box, size_t count)
    {
        item_ptr item = item_ptr();

        if (node->is_leaf())
        {
            for (auto iter = node->items.begin(); iter != node->items.end(); ++iter)
            {
                if ((*iter)->value.first == elem)
                {
                    item = *iter;
                    iter = node->items.erase(iter);
                    count++;
                }
            }
        }
        else
        {
            for (auto child : node->children)
            {
                if (child->box.intersects(elem_box))
                {
                    std::tie(item, count) = erase_node(child, elem, elem_box, count);
                }
            }

            if (empty_leaves(node)) node = collapse_node(node);
        }

        return {item, count};
    }

    template <typename U>
    void insert_to_stack(knn_stack_t<U> & stack, item_ptr item, U dist_sq) const
    {
        using elem_type = std::pair<item_ptr, U>;

        auto comp_lb = [](const elem_type & elem, U dist_sq)
        {
            return elem.second < dist_sq;
        };
        auto comp_ub = [](U dist_sq, const elem_type & elem)
        {
            return dist_sq < elem.second;
        };

        // lower_bound returns an iterator to the first element NOT LESS than dist_sq
        auto liter = std::lower_bound(stack.begin(), stack.end(), dist_sq, comp_lb);
        // upper_bound returns an iterator to the first element GREATER than dist_sq
        auto uiter = std::upper_bound(stack.begin(), stack.end(), dist_sq, comp_ub);

        auto compare = [item](const elem_type & elem){return elem.first != item;};

        auto is_absent = std::transform_reduce(liter, uiter, true, std::logical_and(), compare);

        if (is_absent) stack.insert(liter, std::make_pair(std::move(item), dist_sq));
    }

    template <typename V, typename U>
    void nearest_nodes(knn_stack_t<U> & stack, node_ptr node, const BoxND<V, N> & query, size_t k) const
    {
        if (node->is_leaf())
        {
            for (auto item : node->items)
            {
                auto dist_sq = distance(item->box, query);
                if (stack.size() < k) insert_to_stack(stack, item, dist_sq);
                else if (dist_sq < stack.back().second)
                {
                    insert_to_stack(stack, item, dist_sq);
                    stack.pop_back();
                }
            }
        }
        else
        {
            std::array<std::pair<U, size_t>, 1 << N> indices;
            for (size_t i = 0; i < (1 << N); i++) indices[i] = {distance(node->children[i]->box, query), i};

            auto compare = [](const std::pair<U, size_t> & a, const std::pair<U, size_t> & b)
            {
                return a.first < b.first;
            };
            std::sort(indices.begin(), indices.end(), compare);

            for (auto [dist_sq, index] : indices)
            {
                if (stack.size() != k || dist_sq < stack.back().second) nearest_nodes(stack, node->children[index], query, k);
            }
        }
    }

    template <typename V, typename U>
    void nodes_in_range(knn_stack_t<U> & stack, node_ptr node, const BoxND<V, N> & query, V range_sq) const
    {
        if (node->is_leaf())
        {
            for (auto item : node->items)
            {
                auto dist_sq = distance(item->box, query);
                if (dist_sq < range_sq) insert_to_stack(stack, item, dist_sq);
            }
        }
        else
        {
            for (auto child : node->children)
            {
                if (distance(child->box, query) < range_sq) nodes_in_range(stack, child, query, range_sq);
            }
        }
    }

    template <typename V>
    void intersected_nodes(item_stack_t & stack, node_ptr node, const BoxND<V, N> & query) const
    {
        if (node->is_leaf())
        {
            for (auto && iter : node->items)
            {
                if (iter->box.intersects(query)) stack.emplace_back(std::forward<decltype(iter)>(iter));
            }
        }
        else
        {
            for (auto child : node->children)
            {
                if (child->box.intersects(query))
                {
                    intersected_nodes(stack, child, query);
                }
            }
        }
    }
};

template <typename Element, typename Index, size_t N>
struct NDStack
{
    std::map<Index, NDTree<Element, Index, N>> trees;
};

}

#endif