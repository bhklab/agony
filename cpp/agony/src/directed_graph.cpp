// Implementation of DirectedGraph class

#include "directed_graph.hpp"

namespace agony
{

    void DirectedGraph::add_node(int id)
    {
        _nodes.insert(id);
    }

    bool DirectedGraph::has_edge(int source, int target) const
    {
        return _edge_weights.find({source, target}) != _edge_weights.end();
    }

    void DirectedGraph::add_edge(int source, int target, double weight)
    {
        // Add nodes if they don't exist
        add_node(source);
        add_node(target);

        // dont support self-loops
        if (source == target)
        {
            throw std::runtime_error("Self-loops are not supported");
        }

        // If the edge already exists, update its weight
        if (_edge_weights.find({source, target}) != _edge_weights.end())
        {
            // Update the weight in the adjacency list
            for (auto &neighbor : _adj_list[source])
            {
                if (neighbor.first == target)
                {
                    neighbor.second = weight;
                    break;
                }
            }
        }
        else
        {
            // Add the new edge to the adjacency list
            _adj_list[source].emplace_back(target, weight);
        }

        // Update the edge weight lookup map
        _edge_weights[{source, target}] = weight;
    }

    double DirectedGraph::get_edge_weight(int source, int target) const
    {
        auto it = _edge_weights.find({source, target});

        if (it == _edge_weights.end())
        {
            throw std::runtime_error("Edge does not exist");
        }

        return it->second;
    }

    const std::unordered_set<int> &DirectedGraph::nodes() const
    {
        return _nodes;
    }

    const std::vector<std::pair<int, double>> &DirectedGraph::neighbors(int node) const
    {
        static const std::vector<std::pair<int, double>> empty;
        auto it = _adj_list.find(node);
        return it != _adj_list.end() ? it->second : empty;
    }

} // namespace agony