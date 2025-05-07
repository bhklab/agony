#include <iostream>

#include "heapu.hh"

using namespace std;

int main() {
  int n, m, t, b, c, cnt = 0;
  cin >> n >> m;
  heapu h(n);
  for (int i = 0; i < m; i++) {
    cin >> t;
    if (t == 0) {
      cin >> b >> c;
      h.update(b, c);
    } else {
      cnt++;
      auto pp = h.top();
      h.pop();
      cout << pp.second << endl;
      //      cout << "popped " << cnt << endl;
    }
  }
}
