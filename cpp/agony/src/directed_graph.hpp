#pragma once // Ensures this header is only included once during compilation

// Standard C++ library includes
#include <unordered_map> // Similar to Python dictionaries, provides O(1) lookups
#include <unordered_set> // Similar to Python sets, provides O(1) lookups
#include <vector>        // Similar to Python lists, but with static typing
#include <utility>       // Provides std::pair, similar to Python tuples
#include <stdexcept>     // Provides C++ exceptions like std::runtime_error

namespace agony
{ // Namespace to prevent naming collisions, similar to Python modules

    /**
     * @brief Custom hash function for std::pair<int, int>
     *
     * In C++, unlike Python, we need to define our own hash functions for custom types
     * or combinations of types when using them as keys in hash-based containers.
     * This is similar to implementing __hash__ in Python, but more explicit.
     */
    struct pair_hash
    {
        /**
         * @brief Hashing operator for pairs
         * @tparam T1 Type of first element (typically int in our usage)
         * @tparam T2 Type of second element (typically int in our usage)
         * @param p The pair to hash
         * @return A hash value for the pair
         *
         * Template parameters are like Python's generics, allowing this to work with
         * different types of pairs. The ^ operator is bitwise XOR, and << is bit shifting.
         */
        template <typename T1, typename T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const
        {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    /**
     * @brief Directed graph implementation with weighted edges
     *
     * This class implements a directed graph with integer node IDs and weighted edges.
     * It's similar to a NetworkX DiGraph in Python, but using C++ data structures.
     */
    class DirectedGraph
    {
    private:
        // Set of all node IDs (similar to Python set)
        std::unordered_set<int> _nodes;

        // Adjacency list: maps each node to a list of (neighbor, weight) pairs
        // Similar to {node: [(neighbor1, weight1), (neighbor2, weight2), ...]} in Python
        std::unordered_map<int, std::vector<std::pair<int, double>>> _adj_list;

        // Edge weight lookup: maps (source, target) pairs to edge weights
        // The pair_hash struct is needed to make the pairs usable as keys
        // Similar to {(source, target): weight} in Python
        std::unordered_map<std::pair<int, int>, double, pair_hash> _edge_weights;
    
    public:
        /**
         * @brief Add a node to the graph
         * @param id The integer identifier for the node
         *
         * In Python, you might add a node to a NetworkX graph with G.add_node(id)
         */
        void add_node(int id);

        /**
         * @brief Add a directed edge between two nodes
         * @param source The source node ID
         * @param target The target node ID
         * @param weight The weight of the edge (defaults to 1.0)
         *
         * Similar to G.add_edge(source, target, weight=1.0) in NetworkX
         */
        void add_edge(int source, int target, double weight = 1.0);

        /**
         * @brief Check if an edge exists between two nodes
         * @param source The source node ID
         * @param target The target node ID
         * @return true if the edge exists, false otherwise
         *
         * Equivalent to checking if (source, target) in G.edges in NetworkX
         */
        bool has_edge(int source, int target) const;

        /**
         * @brief Get the weight of an edge
         * @param source The source node ID
         * @param target The target node ID
         * @return The weight of the edge
         * @throws std::runtime_error if the edge doesn't exist
         *
         * Similar to G[source][target]['weight'] in NetworkX
         */
        double get_edge_weight(int source, int target) const;

        /**
         * @brief Get all nodes in the graph
         * @return A constant reference to the set of node IDs
         *
         * The return type is similar to a Python set, but is returned as a reference
         * to avoid copying. This is a common C++ pattern to improve performance.
         * Equivalent to G.nodes() in NetworkX.
         */
        const std::unordered_set<int> &nodes() const;

        /**
         * @brief Get all neighbors of a node along with edge weights
         * @param node The node ID to get neighbors for
         * @return A constant reference to a vector of (neighbor, weight) pairs
         * @throws std::out_of_range if the node doesn't exist
         *
         * The return type is like a list of tuples in Python, where each tuple
         * contains (neighbor_id, edge_weight). Similar to G[node] in NetworkX
         * but with weights included.
         */
        const std::vector<std::pair<int, double>> &neighbors(int node) const;
    };

} // namespace agony