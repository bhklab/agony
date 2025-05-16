#!/usr/bin/env Rscript

# Load required packages for performance
library(data.table) # For efficient data manipulation
library(igraph) # For graph operations
library(Rcpp) # For C++ integration for performance-critical code
library(assertthat) # For input validation

# DirectedGraph class implementation
DirectedGraph <- R6::R6Class(
  "DirectedGraph",

  public = list(
    # All node IDs
    nodes_set = NULL,

    # Adjacency list: maps each node to list of (neighbor, weight) pairs
    adj_list = NULL,

    # Edge weight lookup: maps (source, target) pairs to edge weights
    edge_weights = NULL,

    # Constructor
    initialize = function() {
      self$nodes_set <- new.env(hash = TRUE)
      self$adj_list <- new.env(hash = TRUE)
      self$edge_weights <- new.env(hash = TRUE)
    },

    # Add a node to the graph
    add_node = function(id) {
      self$nodes_set[[as.character(id)]] <- TRUE
      invisible(self)
    },

    # Check if edge exists
    has_edge = function(source, target) {
      edge_key <- paste(source, target, sep = "_")
      exists(edge_key, envir = self$edge_weights)
    },

    # Add an edge to the graph
    add_edge = function(source, target, weight = 1.0) {
      # Add nodes if they don't exist
      self$add_node(source)
      self$add_node(target)

      # No self-loops allowed
      if (source == target) {
        stop("Self-loops are not supported")
      }

      edge_key <- paste(source, target, sep = "_")

      # If edge exists, update weight
      if (self$has_edge(source, target)) {
        # Update weight in adjacency list
        source_key <- as.character(source)
        if (exists(source_key, envir = self$adj_list)) {
          neighbors <- self$adj_list[[source_key]]
          for (i in seq_along(neighbors)) {
            if (neighbors[[i]]$target == target) {
              neighbors[[i]]$weight <- weight
              break
            }
          }
          self$adj_list[[source_key]] <- neighbors
        }
      } else {
        # Add new edge to adjacency list
        source_key <- as.character(source)
        if (!exists(source_key, envir = self$adj_list)) {
          self$adj_list[[source_key]] <- list()
        }

        neighbors <- self$adj_list[[source_key]]
        neighbors[[length(neighbors) + 1]] <- list(
          target = target,
          weight = weight
        )
        self$adj_list[[source_key]] <- neighbors
      }

      # Update edge weight lookup map
      self$edge_weights[[edge_key]] <- weight

      invisible(self)
    },

    # Get weight of an edge
    get_edge_weight = function(source, target) {
      edge_key <- paste(source, target, sep = "_")
      if (!exists(edge_key, envir = self$edge_weights)) {
        stop("Edge does not exist")
      }

      self$edge_weights[[edge_key]]
    },

    # Get all nodes in the graph
    nodes = function() {
      as.integer(names(self$nodes_set))
    },

    # Get neighbors of a node
    neighbors = function(node) {
      node_key <- as.character(node)
      if (!exists(node_key, envir = self$adj_list)) {
        return(list())
      }

      self$adj_list[[node_key]]
    }
  )
)

# Priority Queue implementation using binary heap
PriorityQueue <- R6::R6Class(
  "PriorityQueue",

  private = list(
    heap = NULL,
    positions = NULL,
    size = 0,

    # Helper methods for heap operations
    parent = function(i) floor(i / 2),
    left_child = function(i) 2 * i,
    right_child = function(i) 2 * i + 1,

    # Swap elements at positions i and j
    swap = function(i, j) {
      tmp <- private$heap[[i]]
      private$heap[[i]] <- private$heap[[j]]
      private$heap[[j]] <- tmp

      # Update positions
      private$positions[[as.character(private$heap[[i]]$item)]] <- i
      private$positions[[as.character(private$heap[[j]]$item)]] <- j
    },

    # Maintain heap property after insertion
    heapify_up = function(i) {
      while (
        i > 1 &&
          private$heap[[private$parent(i)]]$priority >
            private$heap[[i]]$priority
      ) {
        private$swap(i, private$parent(i))
        i <- private$parent(i)
      }
    },

    # Maintain heap property after deletion
    heapify_down = function(i) {
      smallest <- i
      left <- private$left_child(i)
      right <- private$right_child(i)

      if (
        left <= private$size &&
          private$heap[[left]]$priority < private$heap[[smallest]]$priority
      ) {
        smallest <- left
      }

      if (
        right <= private$size &&
          private$heap[[right]]$priority < private$heap[[smallest]]$priority
      ) {
        smallest <- right
      }

      if (smallest != i) {
        private$swap(i, smallest)
        private$heapify_down(smallest)
      }
    }
  ),

  public = list(
    # Constructor
    initialize = function() {
      private$heap <- list()
      private$positions <- new.env(hash = TRUE)
      private$size <- 0
    },

    # Check if queue is empty
    is_empty = function() {
      private$size == 0
    },

    # Add item to queue with given priority
    enqueue = function(item, priority) {
      private$size <- private$size + 1
      private$heap[[private$size]] <- list(item = item, priority = priority)
      private$positions[[as.character(item)]] <- private$size
      private$heapify_up(private$size)
      invisible(self)
    },

    # Remove and return item with lowest priority
    dequeue = function() {
      if (self$is_empty()) {
        stop("Queue is empty")
      }

      top <- private$heap[[1]]$item
      private$positions[[as.character(top)]] <- NULL

      private$heap[[1]] <- private$heap[[private$size]]
      private$positions[[as.character(private$heap[[1]]$item)]] <- 1

      private$size <- private$size - 1
      if (private$size > 0) {
        private$heapify_down(1)
      }

      top
    },

    # Check if item exists in queue
    contains = function(item) {
      exists(as.character(item), envir = private$positions)
    },

    # Update priority of an item
    update_priority = function(item, priority) {
      if (!self$contains(item)) {
        stop("Item not in queue")
      }

      i <- private$positions[[as.character(item)]]
      old_priority <- private$heap[[i]]$priority
      private$heap[[i]]$priority <- priority

      if (priority < old_priority) {
        private$heapify_up(i)
      } else if (priority > old_priority) {
        private$heapify_down(i)
      }

      invisible(self)
    }
  )
)

# Convert DirectedGraph to adjacency list
to_adj_list <- function(graph) {
  adj_list <- new.env(hash = TRUE)

  for (node in graph$nodes()) {
    node_key <- as.character(node)
    adj_list[[node_key]] <- sapply(graph$neighbors(node), function(n) n$target)
  }

  adj_list
}

# Create inverse graph (for in-neighbors)
inverse_graph <- function(adj_list) {
  inv_adj_list <- new.env(hash = TRUE)

  # Initialize empty lists for all nodes
  for (node in names(adj_list)) {
    inv_adj_list[[node]] <- c()
  }

  # Add incoming edges
  for (node in names(adj_list)) {
    neighbors <- adj_list[[node]]
    for (neighbor in neighbors) {
      neighbor_key <- as.character(neighbor)
      if (!exists(neighbor_key, envir = inv_adj_list)) {
        inv_adj_list[[neighbor_key]] <- c()
      }
      inv_adj_list[[neighbor_key]] <- c(
        inv_adj_list[[neighbor_key]],
        as.integer(node)
      )
    }
  }

  inv_adj_list
}

# Relief implementation
edge_slack <- function(ranks, u, v) {
  max(0, ranks[[as.character(u)]] - ranks[[as.character(v)]] + 1)
}

compute_agony_from_ranks <- function(graph, ranks) {
  total <- 0

  for (u in names(graph)) {
    for (v in graph[[u]]) {
      total <- total + edge_slack(ranks, u, v)
    }
  }

  total
}

relief_rank <- function(graph, inv_graph) {
  # Initialize nodes at rank 0
  nodes <- unique(c(
    names(graph),
    unlist(lapply(graph, function(x) as.character(x)))
  ))

  ranks <- new.env(hash = TRUE)
  for (node in nodes) {
    ranks[[node]] <- 0
  }

  # Construct DAG using topological sorting
  in_degree <- new.env(hash = TRUE)
  for (node in nodes) {
    in_degree[[node]] <- 0
  }

  for (u in names(graph)) {
    for (v in graph[[u]]) {
      v_key <- as.character(v)
      in_degree[[v_key]] <- in_degree[[v_key]] + 1
    }
  }

  # Start with nodes with no incoming edges
  queue <- PriorityQueue$new()
  for (node in nodes) {
    if (in_degree[[node]] == 0) {
      queue$enqueue(as.integer(node), 0)
    }
  }

  # Perform topological sort and assign initial ranks
  while (!queue$is_empty()) {
    u <- queue$dequeue()
    u_key <- as.character(u)

    if (exists(u_key, envir = graph)) {
      for (v in graph[[u_key]]) {
        v_key <- as.character(v)
        ranks[[v_key]] <- max(ranks[[v_key]], ranks[[u_key]] + 1)
        in_degree[[v_key]] <- in_degree[[v_key]] - 1

        if (in_degree[[v_key]] == 0) {
          queue$enqueue(as.integer(v), ranks[[v_key]])
        }
      }
    }
  }

  # Iteratively improve ranks
  max_iterations <- 100
  iteration <- 0

  # Calculate initial agony
  current_agony <- compute_agony_from_ranks(graph, ranks)

  # Continue until no improvement or max iterations
  previous_agony <- Inf
  while (current_agony < previous_agony && iteration < max_iterations) {
    iteration <- iteration + 1
    previous_agony <- current_agony

    # Find the edge with maximum slack
    max_slack <- 0
    max_edge <- c(0, 0)

    for (u in names(graph)) {
      for (v in graph[[u]]) {
        slack <- edge_slack(ranks, u, v)
        if (slack > max_slack) {
          max_slack <- slack
          max_edge <- c(as.integer(u), as.integer(v))
        }
      }
    }

    if (max_slack <= 0) {
      break
    }

    s <- max_edge[2]

    # Apply relief operation
    new_ranks <- as.environment(as.list(ranks))
    relief_edges <- new.env(hash = TRUE)

    # Find all nodes reachable from s
    visited <- new.env(hash = TRUE)
    stack <- c(s)

    while (length(stack) > 0) {
      node <- stack[length(stack)]
      stack <- stack[-length(stack)]

      node_key <- as.character(node)
      if (exists(node_key, envir = visited)) {
        next
      }

      visited[[node_key]] <- TRUE

      if (exists(node_key, envir = graph)) {
        for (neighbor in graph[[node_key]]) {
          neighbor_key <- as.character(neighbor)
          edge_key <- paste(node_key, neighbor_key, sep = "_")

          if (edge_slack(ranks, node, neighbor) == 0) {
            stack <- c(stack, neighbor)
            relief_edges[[edge_key]] <- TRUE
          }
        }
      }
    }

    # Apply the minimum slack to all reachable nodes
    for (node in names(visited)) {
      new_ranks[[node]] <- new_ranks[[node]] + max_slack
    }

    # Update ranks
    ranks <- new_ranks

    # Normalize ranks to start from 0
    min_rank <- min(as.numeric(as.list(ranks)))
    for (node in names(ranks)) {
      ranks[[node]] <- ranks[[node]] - min_rank
    }

    # Recalculate agony
    current_agony <- compute_agony_from_ranks(graph, ranks)
  }

  # Convert environment to named list for easier access
  rank_list <- as.list(ranks)
  rank_values <- as.numeric(rank_list)
  names(rank_values) <- names(rank_list)

  list(agony = current_agony, ranks = rank_values)
}

# LP implementation using lpSolve
unweighted_agony_lp <- function(graph) {
  # Get unique nodes
  node_list <- graph$nodes()
  node_list <- sort(node_list)
  n <- length(node_list)

  # Create node mapping
  node_map <- new.env(hash = TRUE)
  for (i in seq_along(node_list)) {
    node_map[[as.character(node_list[i])]] <- i
  }

  # Create edge list for LP
  edge_list <- list()
  edge_count <- 0

  for (u in node_list) {
    for (neighbor in graph$neighbors(u)) {
      edge_count <- edge_count + 1
      edge_list[[edge_count]] <- c(
        node_map[[as.character(u)]],
        node_map[[as.character(neighbor$target)]]
      )
    }
  }

  m <- length(edge_list)

  # Create optimization model
  # Objective function coefficients: 1 for penalties, 0 for ranks
  obj <- c(rep(1, m), rep(0, n))

  # Constraint matrix
  const_mat <- matrix(0, nrow = m, ncol = m + n)

  # Build constraints: r[v] - r[u] + p[i] >= 1
  for (i in 1:m) {
    e <- edge_list[[i]]
    u <- e[1]
    v <- e[2]

    # Coefficient for p[i]
    const_mat[i, i] <- 1

    # Coefficient for r[v]
    const_mat[i, m + v] <- 1

    # Coefficient for r[u]
    const_mat[i, m + u] <- -1
  }

  # RHS of constraints
  rhs <- rep(1, m)

  # Direction of constraints
  direction <- rep(">=", m)

  # Solve the LP problem
  result <- lpSolve::lp("min", obj, const_mat, direction, rhs)

  if (result$status == 0) {
    # Extract solution
    ranks_array <- as.integer(round(result$solution[(m + 1):(m + n)]))

    # Shift ranks to 0-based
    min_rank <- min(ranks_array)
    ranks_array <- ranks_array - min_rank

    # Map ranks back to original nodeIDs
    node_ranks <- numeric(length = n)
    names(node_ranks) <- as.character(node_list)

    for (i in 1:n) {
      node_ranks[i] <- ranks_array[i]
    }

    list(agony = as.integer(round(result$objval)), ranks = node_ranks)
  } else {
    stop("LP solver failed")
  }
}

# Load graph from edge list file
load_edgelist <- function(filename) {
  graph <- DirectedGraph$new()

  # Use data.table for faster file reading
  edges <- data.table::fread(
    filename,
    header = FALSE,
    colClasses = "character",
    na.strings = "",
    fill = TRUE
  )

  for (i in seq_len(nrow(edges))) {
    row <- edges[i]

    # Skip comment lines or empty lines
    if (nchar(row[[1]]) == 0 || substr(row[[1]], 1, 1) == "#") {
      next
    }

    if (ncol(row) >= 2) {
      source <- as.integer(row[[1]])
      target <- as.integer(row[[2]])

      weight <- 1.0
      if (ncol(row) >= 3 && !is.na(row[[3]])) {
        weight <- as.numeric(row[[3]])
      }

      graph$add_edge(source, target, weight)
    }
  }

  graph
}

# Main function to compute agony
compute_agony <- function(graph, method = c("lp", "relief")) {
  method <- match.arg(method)

  if (method == "lp") {
    cat("Using Linear Programming method...\n")
    result <- unweighted_agony_lp(graph)
  } else if (method == "relief") {
    cat("Using Relief method...\n")
    # Convert to adjacency list format
    adj_list <- to_adj_list(graph)
    inv_adj_list <- inverse_graph(adj_list)

    result <- relief_rank(adj_list, inv_adj_list)
  } else {
    stop("Unknown method: ", method, ". Use 'lp' or 'relief'")
  }

  result
}
