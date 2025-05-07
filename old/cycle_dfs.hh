#ifndef CYCLE_DFS_
#define CYCLE_DFS_

#include <list>
#include <vector>

using namespace std;

// this part does not need to be fast as the second part of the algorithm is
// much slower and will encompass this complextiy
// BUT I DID IT FAST ANYWAY!!! (though it still takes O(n^2) as we want an
// adjacency matrix representation of the final eulerian graph)
// this algorithm gives back A MAXIMAL eulerian subgraph of the given graph
vector<vector<bool>> cycle_dfs(const vector<vector<int>> &ngraph) {
  int n = ngraph.size();
  // we make another copy of the graph that will be used to delete edges
  vector<list<int>> graph(n);
  for (int i = 0; i < n; i++)
    for (int j : ngraph[i])
      graph[i].push_back(j);
  // the euler subgraph will be represented as a adjacency matrix
  vector<vector<bool>> euler_subgraph(n, vector<bool>(n, false));
  // what is the current parent of the node in the path
  vector<int> parent(n, -1);
  // is the vertex in the current path
  vector<bool> in_path(n, false);
  list<int> nodes;
  for (int i = 0; i < n; i++)
    nodes.push_back(i);
  for (list<int>::iterator root = nodes.begin(); root != nodes.end(); root++) {
    if (graph[*root].empty()) // if we have already parsed
                              // all the neighbors
      continue;
    list<list<int>::iterator> path;
    path.push_back(root);
    in_path[*root] = true;
    while (!path.empty()) {
      int curr_node = *path.back();
      if (graph[curr_node].empty()) {
        path.pop_back();
        continue;
      }
      list<int>::iterator it = graph[curr_node].begin();
      while (it != graph[curr_node].end() && graph[*it].size() == 0)
        it = graph[curr_node].erase(it);
      if (it == graph[curr_node].end()) {
        path.pop_back();
        continue;
      }
      if (in_path[*it]) {
        int start_node = *it;
        while (*path.back() != start_node) {
          list<int>::iterator last = path.back();
          euler_subgraph[*last][*it] = true;
          graph[*last].erase(it);
          in_path[*last] = false;
          it = last;
          path.pop_back();
        }
        euler_subgraph[start_node][*it] = true;
        in_path[*it] = false;
        graph[start_node].erase(it);
      } else {
        path.push_back(it);
        in_path[*it] = true;
      }
    }
  }
  return euler_subgraph;
}

#endif // CYCLE_DFS_
