# Agony Project Documentation

This document provides an overview of the Agony project, aimed at helping Python developers understand the C++ codebase and its components.

## Project Structure

The Agony project contains both C++ and Python code, with the core algorithms implemented in C++ and Python bindings that make these algorithms accessible from Python.

```console
cpp/agony/          - Core C++ implementation
  ├── include/      - Header files that define interfaces
  ├── src/          - Implementation files
  ├── bindings/     - Python binding code
  └── old/          - Legacy implementations (for reference)
src/                - Python package code
docs/               - Documentation
```

## Core Components

### DirectedGraph Class

The `DirectedGraph` class is the primary data structure used in the project. It implements a directed graph with weighted edges, similar to NetworkX's DiGraph in Python.

#### What is a Directed Graph?

A directed graph (or digraph) is a set of vertices (nodes) connected by edges, where the edges have a direction from one vertex to another. In our implementation:

- Nodes are represented by integer IDs
- Edges connect a source node to a target node
- Edges can have weights (default is 1.0)

#### C++ Types and Their Python Equivalents

| C++ Type | Python Equivalent | Description |
|----------|-------------------|-------------|
| `std::unordered_set<int>` | `set()` | A collection of unique node IDs |
| `std::unordered_map<K,V>` | `dict()` | A key-value mapping, like Python dictionaries |
| `std::vector<T>` | `list()` | A sequence container, similar to Python lists |
| `std::pair<T1,T2>` | `tuple()` | A pair of values, like a 2-element Python tuple |
| `const T&` | N/A | A reference to data (avoids copying, improves performance) |

#### "Odd" Type Descriptions

Some C++ types might look confusing to Python developers:

- **Template parameters** (things in `<>` brackets): These are like Python's type hints, but required by the compiler. For example, `std::vector<int>` is a list that can only contain integers.

- **`const` keyword**: Indicates that something won't be modified. When you see `const` in function parameters or return types, it's saying "this won't change the data."

- **`&` symbol**: Indicates a reference. Instead of copying data, C++ can pass a reference to the original data for efficiency.

- **Function names with `::` operator**: The `::` operator in C++ shows namespace or class membership, similar to Python's dot notation.

## Basic Usage

While Python developers won't directly use the C++ code, understanding the underlying implementation helps when using the Python bindings.

The graph implementation allows:

1. Adding nodes to the graph
2. Adding weighted directed edges between nodes
3. Checking if edges exist
4. Getting weights of edges
5. Retrieving all nodes in the graph
6. Finding all neighbors of a node

## Advanced C++ Concepts for Python Developers

### Namespaces

C++ uses namespaces (like `namespace agony`) to prevent naming collisions. This is similar to Python modules and helps organize code.

### Custom Hash Functions

In C++, unlike Python, you need to define hash functions for custom types when using them as keys in hash tables. The `pair_hash` struct serves this purpose for pairs of integers.

### Function Modifiers

- `const` after a function declaration means the function won't modify the object.
- Function declarations without definitions are the C++ equivalent of abstract methods.

### Memory Management

C++ requires more explicit memory management than Python. Our implementation uses:

- References (with `&`) to avoid unnecessary copying
- Container classes that handle memory allocation/deallocation

## Building and Extending

If you need to modify the C++ code, you'll need to:

1. Update the appropriate files in `cpp/agony/include` and `cpp/agony/src`
2. Rebuild the project (follow build instructions in README)
3. Ensure Python bindings are updated if interfaces change

Python developers should focus on the Python interface rather than modifying the C++ code directly unless absolutely necessary.
