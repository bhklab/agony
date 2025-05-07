#include <iostream>

#include "cycle_dfs.hh"

using namespace std;

int main() {
  int n, m, a, b;
  cin >> n >> m;
  vector<vector<int>> graph(n, vector<int>());
  for (int i = 0; i < m; i++) {
    cin >> a >> b;
    graph[a].push_back(b);
  }
  auto eul = cycle_dfs(graph);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      cout << eul[i][j] << " ";
    cout << endl;
  }
}
