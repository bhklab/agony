#ifndef HEAPU_
#define HEAPU_

#include <algorithm>
#include <list>
#include <stdexcept>
#include <vector>

using namespace std;

// this is a minimal max-heap class for the agony code (only for integers) which
// allows for random updates as well
class heapu {
private:
  vector<list<int>> heap;
  list<int> null_list;
  vector<list<int>::iterator> pointers;
  vector<int> priority_;
  int current_priority;
  int number_of_elements;

public:
  // you should only insert elements 0...n-1 with priorities in 0...n-1
  // else this will give a lot of errors
  heapu(int n)
      : heap(n), null_list(), pointers(n, null_list.end()), priority_(n, -1),
        current_priority(n - 1), number_of_elements(0) {}
  // I tried to give a stl like interface for this structure
  bool empty() { return number_of_elements == 0; }
  int size() { return number_of_elements; }
  pair<int, int> top() {
    while (heap[current_priority].empty())
      current_priority--;
    return make_pair(heap[current_priority].front(), current_priority);
  }
  void push(int key, int priority) {
    current_priority = max(priority, current_priority);
    heap[priority].push_front(key);
    pointers[key] = heap[priority].begin();
    priority_[key] = priority;
    number_of_elements++;
  }
  void pop() {
    pair<int, int> top_key = top();
    pointers[top_key.first] = null_list.end();
    heap[current_priority].pop_front();
    priority_[top_key.first] = -1;
    number_of_elements--;
  }
  void update(int key, int priority) {
    if (pointers[key] != null_list.end()) {
      number_of_elements--;
      current_priority = max(priority, current_priority);
      heap[priority_[key]].erase(pointers[key]);
    }
    push(key, priority);
  }
};

#endif // HEAPU_
