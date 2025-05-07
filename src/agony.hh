#ifndef AGONY_
#define AGONY_

#include <vector>

#define max(a, b) (((a) > (b)) ? (a) : (b))

// gives back the agony of an edge a -> b w.r.t. r
// a, b: nodes
// r: a rank function
inline int edge_agony(int a, int b, const std::vector<int> &r) {
  return max(r[a] - r[b] + 1, 0);
}

// gives back the slack of an edge a -> b w.r.t. r
// a, b: nodes
// r: a rank function
inline int edge_slack(int a, int b, const std::vector<int> &r) {
  return max(r[b] - r[a] - 1, 0);
}

// return the edge with the largest slack in the euler_subgraph w.r.t r
inline std::pair<int, int>
largest_slack(const std::vector<std::vector<int>> &graph,
              const std::vector<std::vector<bool>> &euler_subgraph,
              const std::vector<int> &r) {
  int a, b, cslack = 0;
  for (std::size_t i = 0; i < graph.size(); i++)
    for (int j : graph[i])
      if (euler_subgraph[i][j] == true && cslack <= edge_slack(i, j, r)) {
        cslack = edge_slack(i, j, r);
        a = i, b = j;
      }
  return std::make_pair(a, b);
}

// returns the current agony of the graph w.r.t r
inline int agony(const std::vector<std::vector<int>> &graph,
                 const std::vector<int> &r) {
  int total_agony = 0;
  for (std::size_t i = 0; i < graph.size(); i++)
    for (int j : graph[i])
      total_agony += edge_agony(i, j, r);
  return total_agony;
}

inline int
number_of_edges(const std::vector<std::vector<int>> &graph,
                const std::vector<std::vector<bool>> &euler_subgraph) {
  int num_edges = 0;
  for (std::size_t i = 0; i < graph.size(); i++)
    for (int j : graph[i])
      if (euler_subgraph[i][j] == true)
        num_edges++;
  return num_edges;
}

// updates the euler_subgraph by increasing its size using the relief method
void relief(const std::vector<std::vector<int>> &graph,
            const std::vector<std::vector<int>> &rgraph,
            std::vector<std::vector<bool>> &euler_subgraph, std::vector<int> &r,
            int p, int s);

// calculates the minimum agony of a graph
std::pair<std::vector<std::vector<bool>>, std::vector<int>>
minimum_agony(const std::vector<std::vector<int>> &graph);

#endif // AGONY_
