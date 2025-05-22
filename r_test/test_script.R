#!/usr/bin/env Rscript

# Load the agony implementation
source("r_test/agony.R")

# Load example graph from file
cat("Loading example graph from file...\n")
graph <- load_edgelist("r_test/example_graph.txt")

# Print graph information
cat(
  "Graph loaded with",
  length(graph$nodes()),
  "nodes and",
  length(
    unlist(lapply(graph$nodes(), function(n) length(graph$neighbors(n))))
  ),
  "edges\n"
)

# Compute agony using LP method
cat("\nComputing agony using LP method:\n")
lp_result <- compute_agony(graph, method = "lp")
cat("Agony value:", lp_result$agony, "\n")
cat("Node rankings:\n")
print(lp_result$ranks)

# Compute agony using Relief method
cat("\nComputing agony using Relief method:\n")
relief_result <- compute_agony(graph, method = "relief")
cat("Agony value:", relief_result$agony, "\n")
cat("Node rankings:\n")
print(relief_result$ranks)

# Create a small custom graph example for comparison
cat("\nCreating a custom graph example:\n")
custom_graph <- DirectedGraph$new()
custom_graph$add_edge(1, 2)
custom_graph$add_edge(2, 3)
custom_graph$add_edge(3, 4)
custom_graph$add_edge(4, 1) # creates a cycle with agony

# Compute agony for custom graph
cat("\nComputing agony for custom graph:\n")
custom_result <- compute_agony(custom_graph, method = "relief")
cat("Agony value:", custom_result$agony, "\n")
cat("Node rankings:\n")
print(custom_result$ranks)

# Performance benchmark
cat("\nRunning performance benchmark...\n")

# Generate a larger random graph
random_graph <- DirectedGraph$new()
n_nodes <- 100
n_edges <- 300

# set.seed(42) # For reproducibility
for (i in 1:n_edges) {
  source_node <- sample(1:n_nodes, 1)
  target_node <- sample(1:n_nodes, 1)

  # Avoid self-loops
  while (target_node == source_node) {
    target_node <- sample(1:n_nodes, 1)
  }

  random_graph$add_edge(source_node, target_node)
}

cat(
  "Random graph created with",
  length(random_graph$nodes()),
  "nodes and",
  length(unlist(lapply(
    random_graph$nodes(),
    function(n) length(random_graph$neighbors(n))
  ))),
  "edges\n"
)

# Time relief method
start_time <- Sys.time()
random_result <- compute_agony(random_graph, method = "relief")
end_time <- Sys.time()
cat(
  "Relief method completed in",
  difftime(end_time, start_time, units = "secs"),
  "seconds\n"
)
cat("Agony value:", random_result$agony, "\n")

# Print summary
cat("\nSummary of results:\n")
cat("Example graph LP agony:", lp_result$agony, "\n")
cat("Example graph Relief agony:", relief_result$agony, "\n")
cat("Custom graph Relief agony:", custom_result$agony, "\n")
cat("Random graph Relief agony:", random_result$agony, "\n")
