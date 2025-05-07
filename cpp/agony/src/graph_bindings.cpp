#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include "directed_graph.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace agony;

NB_MODULE(agony, m) {
	nb::class_<DirectedGraph>(m, "DirectedGraph")
		.def(nb::init<>())
		.def("add_node", &DirectedGraph::add_node)
		.def("add_edge", &DirectedGraph::add_edge, "source"_a, "target"_a, "weight"_a = 1.0)
		.def("has_edge", &DirectedGraph::has_edge)
		.def("get_edge_weight", &DirectedGraph::get_edge_weight)
		.def("nodes", &DirectedGraph::nodes)
		.def("neighbors", &DirectedGraph::neighbors);
}