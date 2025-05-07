#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

#include "agony.hh"
#include "cycle_dfs.hh"
#include "heapu.hh"

// Add Nanobind includes
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

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

// Helper functions for Nanobind

std::vector<std::vector<int>> convert_to_graph(const nb::list& py_graph) {
    std::vector<std::vector<int>> graph;
    for (size_t i = 0; i < len(py_graph); i++) {
        nb::list adjacency = py_graph[i];
        std::vector<int> neighbors;
        for (size_t j = 0; j < len(adjacency); j++) {
            neighbors.push_back(adjacency[j].cast<int>());
        }
        graph.push_back(neighbors);
    }
    return graph;
}

nb::list convert_from_euler_subgraph(const std::vector<std::vector<bool>>& euler_subgraph) {
    nb::list result;
    for (const auto& row : euler_subgraph) {
        nb::list py_row;
        for (const auto& val : row) {
            py_row.append(val);
        }
        result.append(py_row);
    }
    return result;
}

nb::list convert_from_ranks(const std::vector<int>& ranks) {
    nb::list result;
    for (const auto& rank : ranks) {
        result.append(rank);
    }
    return result;
}

// Nanobind module definition
NB_MODULE(agony_project, m) {
    m.doc() = "Agony algorithm implementation for Python";
    
    m.def("edge_agony", [](int a, int b, const nb::list& py_r) {
        std::vector<int> r;
        for (size_t i = 0; i < len(py_r); i++) {
            r.push_back(py_r[i].cast<int>());
        }
        return edge_agony(a, b, r);
    }, "Calculate the agony of an edge a -> b with respect to rank function r");
    
    m.def("edge_slack", [](int a, int b, const nb::list& py_r) {
        std::vector<int> r;
        for (size_t i = 0; i < len(py_r); i++) {
            r.push_back(py_r[i].cast<int>());
        }
        return edge_slack(a, b, r);
    }, "Calculate the slack of an edge a -> b with respect to rank function r");
    
    m.def("minimum_agony", [](const nb::list& py_graph) {
        auto graph = convert_to_graph(py_graph);
        auto result = minimum_agony(graph);
        
        nb::list euler_subgraph = convert_from_euler_subgraph(result.first);
        nb::list ranks = convert_from_ranks(result.second);
        
        return nb::make_tuple(euler_subgraph, ranks);
    }, "Calculate the minimum agony of a graph");
    
    m.def("agony", [](const nb::list& py_graph, const nb::list& py_r) {
        std::vector<std::vector<int>> graph = convert_to_graph(py_graph);
        std::vector<int> r;
        for (size_t i = 0; i < len(py_r); i++) {
            r.push_back(py_r[i].cast<int>());
        }
        return agony(graph, r);
    }, "Calculate the total agony of a graph with respect to rank function r");
}
