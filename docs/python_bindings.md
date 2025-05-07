# Python Bindings in Agony

This document explains how the C++ code in Agony is made accessible to Python users through bindings.

## What are Python Bindings?

Python bindings are code that allows Python to interact with libraries written in other languages, such as C++. They serve as a bridge between Python and C++, enabling Python code to call C++ functions and work with C++ objects.

In the Agony project, we use bindings to make our efficient C++ graph algorithms available to Python users, combining C++'s performance with Python's ease of use.

## How Bindings Work

When you use a binding-enabled library in Python, you're actually:

1. Importing a Python module that contains special code to interface with C++
2. Using Python objects that represent the underlying C++ objects
3. Calling methods that forward your requests to the C++ implementations
4. Getting results back that have been converted from C++ types to Python types

### Type Conversions

The bindings automatically handle conversions between C++ and Python types:

| C++ Type | Python Type |
|----------|-------------|
| `int`, `double` | `int`, `float` |
| `std::string` | `str` |
| `std::vector<T>` | `list` |
| `std::unordered_map<K,V>` | `dict` |
| `std::unordered_set<T>` | `set` |
| Custom C++ classes | Custom Python classes |

## Using the Python API

From Python, you would use the bound library like this:

```python
# Import the module exposed by the bindings
import agony

# Create a directed graph
graph = agony.DirectedGraph()

# Add nodes
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)

# Add edges
graph.add_edge(1, 2, 0.5)  # Edge from node 1 to 2 with weight 0.5
graph.add_edge(2, 3)       # Edge from node 2 to 3 with default weight 1.0

# Check if an edge exists
if graph.has_edge(1, 2):
    print(f"Edge weight: {graph.get_edge_weight(1, 2)}")  # Should print 0.5

# Get all nodes
all_nodes = graph.nodes()  # Returns a Python set: {1, 2, 3}

# Get neighbors of a node
neighbors_of_2 = graph.neighbors(2)  # Returns a list of (neighbor, weight) tuples
```
