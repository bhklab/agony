#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <vector>

#include "agony.hh"
#include "cycle_dfs.hh"
#include "heapu.hh"

// Add Nanobind includes
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

using namespace std;
namespace nb = nanobind;

// here we have graph in the adjacency list representation and euler_subgraph in
// a adjacency matrix representation
// this is important for fast computation
void relief(const vector<vector<int>> &graph, const vector<vector<int>> &rgraph,
            vector<vector<bool>> &euler_subgraph, vector<int> &r, int p,
            int s) {
  int n = graph.size(); // i am lazy
  vector<int> r2(r.begin(),
                 r.end()), // make a copy of the current ranking
      t(n, 0),             // the increase in the current ranks
      parent(n, -2 * n),   // the new parents of each node in the graph
      dir(n, 0);           // to store the directionality of the edges
  t[p] = edge_slack(p, s, r);
  heapu S(n);
  S.update(p, t[p]);
  while (!S.empty()) {
    pair<int, int> pp = S.top();
    S.pop();
    int u = pp.first, tu = pp.second;
    r2[u] += tu;
    for (int v : graph[u])
      if (!euler_subgraph[u][v] && r2[v] <= r2[u]) {
        int tt = r2[u] - r2[v] + 1;
        if (tt > t[v]) {
          t[v] = tt;
          S.update(v, tt);
          parent[v] = u;
          dir[v] = 1;
        }
      }
    for (int w : rgraph[u])
      if (euler_subgraph[w][u] && edge_slack(w, u, r2) > edge_slack(w, u, r)) {
        int tt = edge_slack(w, u, r2) - edge_slack(w, u, r);
        if (tt > t[w]) {
          t[w] = tt;
          S.update(w, tt);
          parent[w] = u;
          dir[w] = 2;
        }
      }
  }
  if (edge_slack(p, s, r2) > 0) {
    int x = s, y;
    while (x != p) {
      y = parent[x];
      if (dir[x] == 1) {
        euler_subgraph[y][x] = euler_subgraph[y][x] != true;
      } else {
        euler_subgraph[x][y] = euler_subgraph[x][y] != true;
      }
      x = y;
    }
    euler_subgraph[p][s] = false;
  }
  for (size_t i = 0; i < r.size(); i++)
    r[i] = r2[i];
}

vector<int> get_ranks(const vector<vector<int>> &graph,
                      const vector<vector<bool>> &euler_subgraph) {
  int n = graph.size();
  vector<int> r(n, 0);
  vector<int> topsort, indegree(n, 0);
  queue<int> que;
  for (int i = 0; i < n; i++)
    for (int j : graph[i])
      if (!euler_subgraph[i][j])
        indegree[j]++;
  for (int i = 0; i < n; i++)
    if (indegree[i] == 0)
      que.push(i);
  while (!que.empty()) {
    int i = que.front();
    que.pop();
    topsort.push_back(i);
    for (int j : graph[i])
      if (!euler_subgraph[i][j]) {
        indegree[j]--;
        if (indegree[j] == 0)
          que.push(j);
      }
  }
  for (int i : topsort)
    for (int j : graph[i])
      if (!euler_subgraph[i][j])
        r[j] = max(r[j], r[i] + 1);
  return r;
}

pair<vector<vector<bool>>, vector<int>>
minimum_agony(const vector<vector<int>> &graph) {
  int n = graph.size();
  vector<vector<bool>> euler_subgraph = cycle_dfs(graph);
  vector<int> r = get_ranks(graph, euler_subgraph);
  vector<vector<int>> rgraph(n);
  for (int i = 0; i < n; i++)
    for (int j : graph[i])
      rgraph[j].push_back(i);
  while (agony(graph, r) > number_of_edges(graph, euler_subgraph)) {
    pair<int, int> edge = largest_slack(graph, euler_subgraph, r);
    relief(graph, rgraph, euler_subgraph, r, edge.first, edge.second);
    int mmin = *min_element(r.begin(), r.end());
    for (size_t i = 0; i < r.size(); i++)
      r[i] -= mmin;
  }
  return make_pair(euler_subgraph, r);
}

// Helper function to visualize a graph with its eulerian subgraph as ASCII art
std::string visualize_euler_subgraph(const std::vector<std::vector<int>>& graph, 
                                    const std::vector<std::vector<bool>>& euler_subgraph) {
    std::stringstream ss;
    int n = graph.size();
    
    // Header
    ss << "Graph visualization with " << n << " nodes:\n";
    ss << "  - Regular edges: ---->\n";
    ss << "  - Eulerian edges: ====>\n\n";
    
    // For each node
    for (int i = 0; i < n; i++) {
        ss << "Node " << i << ":\n";
        
        // Show outgoing edges
        for (int j : graph[i]) {
            ss << "  ";
            if (euler_subgraph[i][j]) {
                ss << "====>";
            } else {
                ss << "---->";
            }
            ss << " " << j << "\n";
        }
        
        if (graph[i].empty()) {
            ss << "  (no outgoing edges)\n";
        }
        ss << "\n";
    }
    
    return ss.str();
}

// Nanobind module definition
NB_MODULE(agony, m) {
    m.doc() = "Agony algorithm implementation for Python";
    
    // Add binding for cycle_dfs function
    m.def("cycle_dfs", [](const std::vector<std::vector<int>>& graph) {
        return cycle_dfs(graph);
    }, "Compute a maximal eulerian subgraph of the given graph using DFS.\n\n"
       "Parameters:\n"
       "    graph: List of lists representing an adjacency list of the graph\n\n"
       "Returns:\n"
       "    A 2D boolean matrix representing the eulerian subgraph\n\n"
       "Example:\n"
       "    >>> import agony\n"
       "    >>> # Create a graph with 3 nodes and edges 0->1, 1->2, 2->0\n"
       "    >>> graph = [[1], [2], [0]]\n"
       "    >>> result = agony.cycle_dfs(graph)\n"
       "    >>> # Result will be a boolean adjacency matrix of the eulerian subgraph\n"
       "    >>> # In this case, it should contain all edges as they form a cycle");
    
    // Add a binding that visualizes the eulerian subgraph as ASCII art
    m.def("visualize_cycle_dfs", [](const std::vector<std::vector<int>>& graph) {
        std::vector<std::vector<bool>> euler_subgraph = cycle_dfs(graph);
        return visualize_euler_subgraph(graph, euler_subgraph);
    }, "Compute a maximal eulerian subgraph of the given graph and return an ASCII visualization.\n\n"
       "Parameters:\n"
       "    graph: List of lists representing an adjacency list of the graph\n\n"
       "Returns:\n"
       "    A string containing an ASCII visualization of the graph and its eulerian subgraph\n\n"
       "Example:\n"
       "    >>> import agony\n"
       "    >>> graph = [[1], [2], [0]]\n"
       "    >>> print(agony.visualize_cycle_dfs(graph))\n"
       "    # This will display a text visualization showing all edges as eulerian");
}
