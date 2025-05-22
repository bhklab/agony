#!/usr/bin/env Rscript

# Load required packages for performance
library(data.table) # For efficient data manipulation
library(igraph) # For graph operations
library(Rcpp) # For C++ integration for performance-critical code
library(assertthat) # For input validation
library(collections) # For priority queue implementation
library(Matrix) # For sparse matrix operations

# Configuration constants
MAX_ITERATIONS <- 100
DEFAULT_WEIGHT <- 1.0
TOLERANCE <- 1e-10

# Input validation helper functions
validate_node_id <- function(node_id) {
  if (
    !is.numeric(node_id) ||
      length(node_id) != 1 ||
      is.na(node_id) ||
      node_id != as.integer(node_id)
  ) {
    stop("Node ID must be a single integer value, got: ", deparse(node_id))
  }
  if (node_id < 0 || node_id > .Machine$integer.max) {
    stop("Node ID out of valid range: ", node_id)
  }
  as.integer(node_id)
}

validate_weight <- function(weight) {
  if (
    !is.numeric(weight) || length(weight) != 1 || is.na(weight) || weight < 0
  ) {
    stop(
      "Weight must be a single non-negative numeric value, got: ",
      deparse(weight)
    )
  }
  as.numeric(weight)
}

# DirectedGraph class implementation with optimized data structures
DirectedGraph <- R6::R6Class(
  "DirectedGraph",

  public = list(
    # All node IDs as sorted vector for fast lookups
    nodes_vec = NULL,

    # Node ID to index mapping for O(1) lookups
    node_to_index = NULL,

    # Adjacency list using lists for efficient access
    adj_list = NULL,

    # Edge count for pre-allocation
    edge_count = 0,

    # Constructor
    initialize = function() {
      self$nodes_vec <- integer(0)
      self$node_to_index <- new.env(hash = TRUE)
      self$adj_list <- list()
      self$edge_count <- 0
    },

    # Add a node to the graph with validation
    add_node = function(id) {
      id <- validate_node_id(id)
      id_str <- as.character(id)

      if (!exists(id_str, envir = self$node_to_index)) {
        # Add to sorted vector maintaining order
        self$nodes_vec <- sort(c(self$nodes_vec, id))

        # Update index mapping
        for (i in seq_along(self$nodes_vec)) {
          self$node_to_index[[as.character(self$nodes_vec[i])]] <- i
        }

        # Initialize adjacency list entry
        self$adj_list[[length(self$nodes_vec)]] <- list(
          targets = integer(0),
          weights = numeric(0)
        )
      }
      invisible(self)
    },

    # Check if edge exists with validation
    has_edge = function(source, target) {
      source <- validate_node_id(source)
      target <- validate_node_id(target)

      source_str <- as.character(source)
      if (!exists(source_str, envir = self$node_to_index)) {
        return(FALSE)
      }

      source_idx <- self$node_to_index[[source_str]]
      if (source_idx > length(self$adj_list)) {
        return(FALSE)
      }

      target %in% self$adj_list[[source_idx]]$targets
    },

    # Add an edge to the graph with validation and optimization
    add_edge = function(source, target, weight = DEFAULT_WEIGHT) {
      source <- validate_node_id(source)
      target <- validate_node_id(target)
      weight <- validate_weight(weight)

      # No self-loops allowed
      if (source == target) {
        stop(
          "Self-loops are not supported: source=",
          source,
          ", target=",
          target
        )
      }

      # Add nodes if they don't exist
      self$add_node(source)
      self$add_node(target)

      source_idx <- self$node_to_index[[as.character(source)]]

      # Check if edge exists and update or add
      targets <- self$adj_list[[source_idx]]$targets
      target_pos <- match(target, targets)

      if (!is.na(target_pos)) {
        # Update existing edge weight
        self$adj_list[[source_idx]]$weights[target_pos] <- weight
      } else {
        # Add new edge
        self$adj_list[[source_idx]]$targets <- c(targets, target)
        self$adj_list[[source_idx]]$weights <- c(
          self$adj_list[[source_idx]]$weights,
          weight
        )
        self$edge_count <- self$edge_count + 1
      }

      invisible(self)
    },

    # Get weight of an edge with validation
    get_edge_weight = function(source, target) {
      source <- validate_node_id(source)
      target <- validate_node_id(target)

      if (!self$has_edge(source, target)) {
        stop("Edge does not exist: (", source, ", ", target, ")")
      }

      source_idx <- self$node_to_index[[as.character(source)]]
      targets <- self$adj_list[[source_idx]]$targets
      target_pos <- match(target, targets)

      self$adj_list[[source_idx]]$weights[target_pos]
    },

    # Get all nodes in the graph (already sorted)
    nodes = function() {
      self$nodes_vec
    },

    # Get neighbors of a node with validation
    neighbors = function(node) {
      node <- validate_node_id(node)
      node_str <- as.character(node)

      if (!exists(node_str, envir = self$node_to_index)) {
        return(list(targets = integer(0), weights = numeric(0)))
      }

      node_idx <- self$node_to_index[[node_str]]
      if (node_idx > length(self$adj_list)) {
        return(list(targets = integer(0), weights = numeric(0)))
      }

      self$adj_list[[node_idx]]
    },

    # Get total number of edges
    num_edges = function() {
      self$edge_count
    },

    # Get total number of nodes
    num_nodes = function() {
      length(self$nodes_vec)
    }
  )
)


# Convert DirectedGraph to optimized adjacency list
to_adj_list <- function(graph) {
  if (graph$num_nodes() == 0) {
    return(list())
  }

  nodes <- graph$nodes()
  adj_list <- vector("list", length(nodes))
  names(adj_list) <- as.character(nodes)

  for (i in seq_along(nodes)) {
    neighbors <- graph$neighbors(nodes[i])
    adj_list[[i]] <- neighbors$targets
  }

  adj_list
}

# Create inverse graph (for in-neighbors) - optimized
inverse_graph <- function(adj_list) {
  if (length(adj_list) == 0) {
    return(list())
  }

  all_nodes <- unique(c(names(adj_list), unlist(adj_list, use.names = FALSE)))
  inv_adj_list <- vector("list", length(all_nodes))
  names(inv_adj_list) <- as.character(all_nodes)

  # Pre-allocate with empty vectors
  for (node in names(inv_adj_list)) {
    inv_adj_list[[node]] <- integer(0)
  }

  # Build inverse adjacency list efficiently
  for (node_name in names(adj_list)) {
    node <- as.integer(node_name)
    neighbors <- adj_list[[node_name]]

    for (neighbor in neighbors) {
      neighbor_key <- as.character(neighbor)
      inv_adj_list[[neighbor_key]] <- c(inv_adj_list[[neighbor_key]], node)
    }
  }

  inv_adj_list
}

# Relief implementation with caching
# Create slack cache for performance
create_slack_cache <- function() {
  list(
    cache = new.env(hash = TRUE),
    hits = 0,
    misses = 0
  )
}

# Cached edge slack calculation
edge_slack_cached <- function(ranks, u, v, cache = NULL) {
  slack_val <- max(0, ranks[u] - ranks[v] + 1)

  if (!is.null(cache)) {
    cache_key <- paste(u, v, ranks[u], ranks[v], sep = "_")
    if (exists(cache_key, envir = cache$cache)) {
      cache$hits <- cache$hits + 1
      return(cache$cache[[cache_key]])
    } else {
      cache$cache[[cache_key]] <- slack_val
      cache$misses <- cache$misses + 1
    }
  }

  slack_val
}

# Vectorized edge slack for multiple edges
edge_slack_vectorized <- function(ranks, u_vec, v_vec) {
  pmax(0, ranks[u_vec] - ranks[v_vec] + 1)
}

# Optimized agony computation with vectorization
compute_agony_from_ranks <- function(graph, ranks) {
  if (length(graph) == 0) {
    return(0)
  }

  total <- 0

  for (u_name in names(graph)) {
    neighbors <- graph[[u_name]]
    if (length(neighbors) > 0) {
      u_rank <- ranks[u_name]
      neighbor_names <- as.character(neighbors)
      v_ranks <- ranks[neighbor_names]

      # Handle missing ranks
      if (any(is.na(v_ranks)) || is.na(u_rank)) {
        next
      }

      slacks <- pmax(0, u_rank - v_ranks + 1)
      total <- total + sum(slacks)
    }
  }

  total
}

# Optimized relief ranking with progress monitoring and early termination
relief_rank <- function(
  graph,
  inv_graph,
  max_iter = MAX_ITERATIONS,
  progress = TRUE
) {
  if (length(graph) == 0) {
    return(list(agony = 0, ranks = numeric(0)))
  }

  # Get all nodes efficiently
  nodes <- unique(c(names(graph), unlist(graph, use.names = FALSE)))
  node_names <- as.character(nodes)
  n_nodes <- length(nodes)

  # Initialize ranks as named vector for fast access
  ranks <- setNames(rep(0, n_nodes), node_names)

  if (progress) {
    cat("Initializing relief algorithm for", n_nodes, "nodes...\n")
  }

  # Construct DAG using topological sorting with pre-allocated structures
  in_degree <- setNames(rep(0, n_nodes), node_names)

  # Calculate in-degrees efficiently
  for (u_name in names(graph)) {
    neighbors <- graph[[u_name]]
    for (v in neighbors) {
      v_name <- as.character(v)
      in_degree[v_name] <- in_degree[v_name] + 1
    }
  }

  # Start with nodes with no incoming edges
  queue <- collections::priority_queue()
  zero_indegree_nodes <- node_names[in_degree == 0]

  for (node_name in zero_indegree_nodes) {
    queue$push(as.integer(node_name), 0)
  }

  # Perform topological sort and assign initial ranks
  while (queue$size() > 0) {
    u <- queue$pop()
    u_name <- as.character(u)

    if (u_name %in% names(graph)) {
      neighbors <- graph[[u_name]]
      u_rank <- ranks[u_name]

      for (v in neighbors) {
        v_name <- as.character(v)
        ranks[v_name] <- max(ranks[v_name], u_rank + 1)
        in_degree[v_name] <- in_degree[v_name] - 1

        if (in_degree[v_name] == 0) {
          queue$push(as.integer(v), -ranks[v_name])
        }
      }
    }
  }

  # Iteratively improve ranks with enhanced termination conditions
  iteration <- 0

  # Calculate initial agony
  current_agony <- compute_agony_from_ranks(graph, ranks)
  previous_agony <- Inf
  no_improvement_count <- 0

  if (progress) {
    cat("Initial agony:", current_agony, "\n")
  }

  # Continue until no improvement or max iterations
  while (iteration < max_iter) {
    iteration <- iteration + 1
    previous_agony <- current_agony

    if (progress && iteration %% 10 == 0) {
      cat("Iteration", iteration, "- Current agony:", current_agony, "\n")
    }

    # Find the edge with maximum slack efficiently
    max_slack <- 0
    max_edge <- c(0, 0)

    for (u_name in names(graph)) {
      neighbors <- graph[[u_name]]
      if (length(neighbors) > 0) {
        u <- as.integer(u_name)
        u_rank <- ranks[u_name]
        v_ranks <- ranks[as.character(neighbors)]
        slacks <- pmax(0, u_rank - v_ranks + 1)

        max_idx <- which.max(slacks)
        if (length(max_idx) > 0 && slacks[max_idx] > max_slack) {
          max_slack <- slacks[max_idx]
          max_edge <- c(u, neighbors[max_idx])
        }
      }
    }

    # Early termination conditions
    if (max_slack <= TOLERANCE) {
      if (progress) {
        cat(
          "Converged: maximum slack ≤ tolerance at iteration",
          iteration,
          "\n"
        )
      }
      break
    }

    # Check for improvement (handle NaN/NA values)
    agony_diff <- current_agony - previous_agony
    if (!is.finite(agony_diff) || abs(agony_diff) < TOLERANCE) {
      no_improvement_count <- no_improvement_count + 1
      if (no_improvement_count >= 5) {
        if (progress) {
          cat(
            "Converged: no improvement for 5 iterations at iteration",
            iteration,
            "\n"
          )
        }
        break
      }
    } else {
      no_improvement_count <- 0
    }

    s <- max_edge[2]
    s_name <- as.character(s)

    # Apply relief operation with optimized data structures
    new_ranks <- ranks
    visited <- logical(n_nodes)
    names(visited) <- node_names
    stack <- s

    # DFS to find all reachable nodes with zero slack edges
    while (length(stack) > 0) {
      node <- stack[length(stack)]
      stack <- stack[-length(stack)]
      node_name <- as.character(node)

      if (visited[node_name]) {
        next
      }

      visited[node_name] <- TRUE

      if (node_name %in% names(graph)) {
        neighbors <- graph[[node_name]]
        node_rank <- ranks[node_name]

        for (neighbor in neighbors) {
          neighbor_name <- as.character(neighbor)
          if (abs(node_rank - ranks[neighbor_name] + 1) < TOLERANCE) {
            stack <- c(stack, neighbor)
          }
        }
      }
    }

    # Apply relief: add max_slack to all visited nodes
    visited_nodes <- names(visited)[visited]
    new_ranks[visited_nodes] <- new_ranks[visited_nodes] + max_slack

    # Normalize ranks to start from 0
    min_rank <- min(new_ranks)
    new_ranks <- new_ranks - min_rank
    ranks <- new_ranks

    # Recalculate agony
    current_agony <- compute_agony_from_ranks(graph, ranks)

    # Early termination if agony increased (shouldn't happen, but safety check)
    if (
      is.finite(current_agony) &&
        is.finite(previous_agony) &&
        current_agony > previous_agony + TOLERANCE
    ) {
      if (progress) {
        cat("Warning: agony increased at iteration", iteration, "- stopping\n")
      }
      ranks <- setNames(ranks - min(ranks), node_names) # Re-normalize
      current_agony <- previous_agony
      break
    }
  }

  if (progress) {
    cat("Relief algorithm completed after", iteration, "iterations\n")
    cat("Final agony:", current_agony, "\n")
  }

  # Return results with properly formatted ranks
  list(agony = as.integer(round(current_agony)), ranks = ranks)
}

# Optimized LP implementation using sparse matrices
unweighted_agony_lp <- function(graph, progress = TRUE) {
  if (graph$num_nodes() == 0) {
    return(list(agony = 0, ranks = numeric(0)))
  }

  # Get unique nodes (already sorted from graph)
  node_list <- graph$nodes()
  n <- length(node_list)

  if (progress) {
    cat("Setting up LP problem for", n, "nodes...\n")
  }

  # Create efficient node mapping
  node_map <- setNames(seq_along(node_list), as.character(node_list))

  # Pre-calculate edge count and create edge list efficiently
  m <- graph$num_edges()

  if (m == 0) {
    # No edges, return zero agony with all ranks = 0
    ranks <- setNames(rep(0, n), as.character(node_list))
    return(list(agony = 0, ranks = ranks))
  }

  if (progress) {
    cat("Processing", m, "edges...\n")
  }

  # Pre-allocate edge list
  edge_list <- matrix(0, nrow = m, ncol = 2)
  edge_idx <- 1

  for (u in node_list) {
    neighbors_data <- graph$neighbors(u)
    targets <- neighbors_data$targets

    if (length(targets) > 0) {
      u_idx <- node_map[as.character(u)]

      for (target in targets) {
        v_idx <- node_map[as.character(target)]
        edge_list[edge_idx, ] <- c(u_idx, v_idx)
        edge_idx <- edge_idx + 1
      }
    }
  }

  # Create optimization model with sparse matrices
  if (progress) {
    cat("Building constraint matrix...\n")
  }

  # Objective function coefficients: 1 for penalties, 0 for ranks
  obj <- c(rep(1, m), rep(0, n))

  # Build sparse constraint matrix more efficiently
  # Each constraint: r[v] - r[u] + p[i] >= 1

  # Pre-allocate vectors for sparse matrix construction
  row_indices <- rep(1:m, 3) # Each constraint has 3 non-zero elements
  col_indices <- integer(3 * m)
  values <- numeric(3 * m)

  for (i in 1:m) {
    base_idx <- (i - 1) * 3
    u <- edge_list[i, 1]
    v <- edge_list[i, 2]

    # Coefficient for p[i] (penalty variable)
    col_indices[base_idx + 1] <- i
    values[base_idx + 1] <- 1

    # Coefficient for r[v] (target rank)
    col_indices[base_idx + 2] <- m + v
    values[base_idx + 2] <- 1

    # Coefficient for r[u] (source rank)
    col_indices[base_idx + 3] <- m + u
    values[base_idx + 3] <- -1
  }

  # Create sparse matrix
  const_mat <- Matrix::sparseMatrix(
    i = row_indices,
    j = col_indices,
    x = values,
    dims = c(m, m + n)
  )

  # Convert to dense matrix for lpSolve (it doesn't support sparse directly)
  const_mat <- as.matrix(const_mat)

  # RHS of constraints and directions (pre-allocated)
  rhs <- rep(1, m)
  direction <- rep(">=", m)

  if (progress) {
    cat("Solving LP problem...\n")
  }

  # Solve the LP problem with error handling
  tryCatch(
    {
      result <- lpSolve::lp("min", obj, const_mat, direction, rhs)
    },
    error = function(e) {
      stop("LP solver failed with error: ", e$message)
    }
  )

  # Process results with enhanced error handling
  if (result$status != 0) {
    stop(
      "LP solver failed with status: ",
      result$status,
      ". This may indicate an infeasible or unbounded problem."
    )
  }

  if (progress) {
    cat("LP solved successfully. Extracting solution...\n")
  }

  # Extract and process solution efficiently
  solution <- result$solution
  ranks_array <- solution[(m + 1):(m + n)]

  # Round to integers and normalize
  ranks_array <- as.integer(round(ranks_array))
  min_rank <- min(ranks_array)
  ranks_array <- ranks_array - min_rank

  # Create named result vector
  node_ranks <- setNames(ranks_array, as.character(node_list))

  agony_value <- as.integer(round(result$objval))

  if (progress) {
    cat("LP solution found. Agony:", agony_value, "\n")
  }

  list(agony = agony_value, ranks = node_ranks)
}

# Optimized graph loading from edge list file
load_edgelist <- function(filename, progress = TRUE) {
  if (!file.exists(filename)) {
    stop("File does not exist: ", filename)
  }

  if (progress) {
    cat("Loading graph from", filename, "...\n")
  }

  graph <- DirectedGraph$new()

  # Use data.table for faster file reading with better error handling
  tryCatch(
    {
      edges <- data.table::fread(
        filename,
        header = FALSE,
        colClasses = "character",
        na.strings = "",
        fill = TRUE,
        showProgress = progress
      )
    },
    error = function(e) {
      stop("Failed to read file ", filename, ": ", e$message)
    }
  )

  if (nrow(edges) == 0) {
    warning("No edges found in file: ", filename)
    return(graph)
  }

  if (progress) {
    cat("Processing", nrow(edges), "lines...\n")
  }

  # Process edges more efficiently
  valid_edges <- 0

  for (i in seq_len(nrow(edges))) {
    row <- edges[i, ]

    # Skip empty lines or comments
    first_col <- row[[1]]
    if (
      is.na(first_col) ||
        nchar(trimws(first_col)) == 0 ||
        substr(trimws(first_col), 1, 1) == "#"
    ) {
      next
    }

    if (ncol(row) >= 2 && !is.na(row[[2]])) {
      # Validate and convert node IDs
      tryCatch(
        {
          source <- as.integer(row[[1]])
          target <- as.integer(row[[2]])

          weight <- DEFAULT_WEIGHT
          if (
            ncol(row) >= 3 && !is.na(row[[3]]) && nchar(trimws(row[[3]])) > 0
          ) {
            weight <- as.numeric(row[[3]])
          }

          graph$add_edge(source, target, weight)
          valid_edges <- valid_edges + 1
        },
        error = function(e) {
          warning("Skipping invalid edge at line ", i, ": ", e$message)
        }
      )
    }
  }

  if (progress) {
    cat(
      "Loaded",
      valid_edges,
      "valid edges into graph with",
      graph$num_nodes(),
      "nodes\n"
    )
  }

  graph
}

# Main function to compute agony with enhanced options
compute_agony <- function(
  graph,
  method = c("lp", "relief"),
  progress = TRUE,
  max_iter = MAX_ITERATIONS
) {
  method <- match.arg(method)

  # Validate input graph
  if (!inherits(graph, "DirectedGraph")) {
    stop("Input must be a DirectedGraph object")
  }

  if (graph$num_nodes() == 0) {
    if (progress) {
      cat("Empty graph provided. Returning zero agony.\n")
    }
    return(list(agony = 0, ranks = numeric(0)))
  }

  if (progress) {
    cat(
      "Computing agony for graph with",
      graph$num_nodes(),
      "nodes and",
      graph$num_edges(),
      "edges\n"
    )
  }

  # Record start time for performance monitoring
  start_time <- Sys.time()

  if (method == "lp") {
    if (progress) {
      cat("Using Linear Programming method...\n")
    }
    result <- unweighted_agony_lp(graph, progress = progress)
  } else if (method == "relief") {
    if (progress) {
      cat("Using Relief method...\n")
    }

    # Convert to adjacency list format efficiently
    adj_list <- to_adj_list(graph)
    inv_adj_list <- inverse_graph(adj_list)

    result <- relief_rank(
      adj_list,
      inv_adj_list,
      max_iter = max_iter,
      progress = progress
    )
  } else {
    stop("Unknown method: '", method, "'. Use 'lp' or 'relief'")
  }

  # Performance reporting
  if (progress) {
    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cat("Computation completed in", round(elapsed, 3), "seconds\n")
  }

  # Validate results
  if (!is.list(result) || !all(c("agony", "ranks") %in% names(result))) {
    stop("Invalid result format from ", method, " method")
  }

  if (!is.numeric(result$agony) || length(result$agony) != 1) {
    stop("Invalid agony value from ", method, " method")
  }

  result
}
