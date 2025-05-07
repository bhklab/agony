#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>

#include "agony.hh"
#include "cycle_dfs.hh"
#include "heapu.hh"

using namespace std;

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
    //cout << "looping...\n";
    pair<int, int> pp = S.top();
    S.pop();
    //cout << "could pop\n";
    int u = pp.first, tu = pp.second;
    //cout << u << " " << tu << endl;
    r2[u] += tu;
    for (int v : graph[u])
      if (!euler_subgraph[u][v] && r2[v] <= r2[u]) {
        //cout << "successor: " << v << endl;
        int tt = r2[u] - r2[v] + 1;
        if (tt > t[v]) {
          t[v] = tt;
          S.update(v, tt);
          parent[v] = u;
          dir[v] = 1;
        }
      }
    // moving u forward may have made some of its pred's have a larger than
    // needed slack so we should move them forward in rank as well
    for (int w : rgraph[u])
      if (euler_subgraph[w][u] && edge_slack(w, u, r2) > edge_slack(w, u, r)) {
        //cout << "predecessor: " << w << endl;
        int tt = edge_slack(w, u, r2) - edge_slack(w, u, r);
        if (tt > t[w]) {
          t[w] = tt;
          S.update(w, tt);
          parent[w] = u;
          dir[w] = 2;
        }
      }
    //cout << "parent array is:\n";
    //for (int i : parent)
    //  cout << i << " ";
    //cout << endl;
  }
  if (edge_slack(p, s, r2) > 0) {
    //cout << "slack was (not 0): " << edge_slack(p, s, r2) << endl;
    int x = s, y;
    while (x != p) {
      y = parent[x];
      if (dir[x] == 1) {
        //cout << y << " " << x << " " << euler_subgraph[y][x] << endl;
        euler_subgraph[y][x] = euler_subgraph[y][x] != true;
        //cout << euler_subgraph[y][x] << endl;
      } else {
        //cout << x << " " << y << " " << euler_subgraph[x][y] << endl;
        euler_subgraph[x][y] = euler_subgraph[x][y] != true;
        //cout << euler_subgraph[x][y] << endl;
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
  // we will now first construct the maximal subgraph and the corresponding rank
  // for the graph
  vector<vector<bool>> euler_subgraph = cycle_dfs(graph);
  vector<int> r = get_ranks(graph, euler_subgraph);
  vector<vector<int>> rgraph(n);
  for (int i = 0; i < n; i++)
    for (int j : graph[i])
      rgraph[j].push_back(i);
  // cout << "started with\n";
  // vector<int> ind(n, 0), outd(n, 0);
  // for (int i = 0; i < n; i++)
  //  for (int j = 0; j < n; j++)
  //    if (euler_subgraph[i][j])
  //      ind[j]++, outd[i]++;
  // for (int i : ind)
  //  cout << i << " ";
  // cout << endl;
  // for (int i : outd)
  //  cout << i << " ";
  // cout << endl;
  // int iter = 0;
  while (agony(graph, r) > number_of_edges(graph, euler_subgraph)) {
    // if (iter > 1)
    //  break;
    // cout << "in iteration: " << iter++ << endl;
    // cout << agony(graph, r) << " " << number_of_edges(graph, euler_subgraph)
    //     << endl;
    pair<int, int> edge = largest_slack(graph, euler_subgraph, r);
    // cout << "largest edge: " << edge.first << " " << edge.second << " "
    //     << edge_slack(edge.first, edge.second, r) << endl;
    relief(graph, rgraph, euler_subgraph, r, edge.first, edge.second);
    // we will also change the ranks of everyone to start from 0
    int mmin = *min_element(r.begin(), r.end());
    for (size_t i = 0; i < r.size(); i++)
      r[i] -= mmin;
    /*
    // small check to see if the graph is actually eulerian or not
    vector<int> ind(n, 0), outd(n, 0);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (euler_subgraph[i][j])
          ind[j]++, outd[i]++;
    cout << "degrees\n";
    for (int i : ind)
      cout << i << " ";
    cout << endl;
    for (int i : outd)
      cout << i << " ";
    cout << endl;*/
  } /*
   cout << "ranks\n";
   for (int i : r)
     cout << i << " ";
   cout << endl;
   cout << "new euler subgraph is\n";
   for (int i = 0; i < n; i++) {
     for (int j : euler_subgraph[i])
       cout << j << " ";
     cout << endl;
   }*/
  return make_pair(euler_subgraph, r);
}
