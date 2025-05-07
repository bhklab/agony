#include "agony.hh"
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main() {
  int n, m, a, b;
  cin >> n >> m;
  vector<vector<int>> graph(n);
  for(int i = 0; i < m; i++){
    cin >> a >> b;
    graph[a].push_back(b);
  }
  auto ans = minimum_agony(graph);
  ofstream fout("agony_temp.txt");
  fout << agony(graph, ans.second) << endl;
}
