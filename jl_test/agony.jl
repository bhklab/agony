#!/usr/bin/env julia

using SparseArrays
using LinearAlgebra
using JuMP
using HiGHS
using DataStructures

mutable struct DirectedGraph
    # All nodeIDs
    nodes_set::Set{Int}

    # Adjacency list: maps each node to list of (neighbor, weight) pairs
    adj_list::Dict{Int,Vector{Tuple{Int,Float64}}}

    # Edge weight lookup: maps (source, target) pairs to edge weights
    edge_weights::Dict{Tuple{Int,Int},Float64}

    # Constructor
    function DirectedGraph()
        new(
            Set{Int}(),
            Dict{Int,Vector{Tuple{Int,Float64}}}(),
            Dict{Tuple{Int,Int},Float64}()
        )
    end
end

function add_node!(graph::DirectedGraph, id::Int)
    push!(graph.nodes_set, id)
end

function has_edge(graph::DirectedGraph, source::Int, target::Int)
    haskey(graph.edge_weights, (source, target))
end

function add_edge!(graph::DirectedGraph, source::Int, target::Int, weight::Float64=1.0)
    # Add nodes if they don't exist
    add_node!(graph, source)
    add_node!(graph, target)

    # No self-loops!!
    if source == target
        throw(ArgumentError("Self-loops are not supported"))
    end

    # If edge exists, update weight
    if haskey(graph.edge_weights, (source, target))
        # Update weight in adjacency list
        for i in eachindex(graph.adj_list[source])
            if graph.adj_list[source][i][1] == target
                graph.adj_list[source][i] = (target, weight)
                break
            end
        end
    else
        # Add new edge to adjacency list
        if !haskey(graph.adj_list, source)
            graph.adj_list[source] = Tuple{Int,Float64}[]
        end
        push!(graph.adj_list[source], (target, weight))
    end

    # Update edge weight lookup map
    graph.edge_weights[(source, target)] = weight
end

function get_edge_weight(graph::DirectedGraph, source::Int, target::Int)
    if !haskey(graph.edge_weights, (source, target))
        throw(KeyError("Edge does not exist"))
    end

    return graph.edge_weights[(source, target)]
end

function nodes(graph::DirectedGraph)
    return graph.nodes_set
end

function neighbors(graph::DirectedGraph, node::Int)
    if !haskey(graph.adj_list, node)
        return Tuple{Int,Float64}[]
    end

    return graph.adj_list[node]
end

# Convert DirectedGraph to adjacency list
function to_adj_list(graph::DirectedGraph)
    adj_list = Dict{Int,Vector{Int}}()

    for node in nodes(graph)
        adj_list[node] = Int[]
        for (neighbor, _) in neighbors(graph, node)
            push!(adj_list[node], neighbor)
        end
    end

    return adj_list
end

# Create inverse graph (for in-neighbors)
function inverse_graph(adj_list::Dict{Int,Vector{Int}})
    inv_adj_list = Dict{Int,Vector{Int}}()

    for node in keys(adj_list)
        if !haskey(inv_adj_list, node)
            inv_adj_list[node] = Int[]
        end

        for neighbor in adj_list[node]
            if !haskey(inv_adj_list, neighbor)
                inv_adj_list[neighbor] = Int[]
            end
            push!(inv_adj_list[neighbor], node)
        end
    end

    return inv_adj_list
end

# Relief implementation (??)
function edge_slack(ranks, u, v)
    max(0, ranks[u] - ranks[v] + 1)
end

function compute_agony(graph, ranks)
    total = 0
    for u in keys(graph)
        for v in graph[u]
            total += edge_slack(ranks, u, v)
        end
    end
    return total
end

function relief_rank(graph, inv_graph)
    # Init nodes at rank 0
    nodes = Set{Int}()
    for (u, neighbors) in graph
        push!(nodes, u)
        for v in neighbors
            push!(nodes, v)
        end
    end

    ranks = Dict(node => 0 for node in nodes)

    # Construct DAG using topological sorting
    in_degree = Dict(node => 0 for node in nodes)

    for (_, neighbors) in graph
        for v in neighbors
            in_degree[v] = get(in_degree, v, 0) + 1
        end
    end

    # Start with nodes with no incoming edges
    queue = Queue{Int}()
    for node in nodes
        if get(in_degree, node, 0) == 0
            enqueue!(queue, node)
        end
    end

    # Perform topological sort and assign initial ranks
    while !isempty(queue)
        u = dequeue!(queue)

        if haskey(graph, u)
            for v in graph[u]
                ranks[v] = max(ranks[v], ranks[u] + 1)
                in_degree[v] -= 1

                if in_degree[v] == 0
                    enqueue!(queue, v)
                end
            end
        end
    end

    # Iteratively improve ranks
    max_iterations = 100
    iteration = 0

    # Calculate initial agony
    current_agony = compute_agony(graph, ranks)

    # Continue until no improvement or max iterations
    previous_agony = typemax(Int)
    while current_agony < previous_agony && iteration < max_iterations
        iteration += 1
        previous_agony = current_agony

        # Find the edge with maximum slack
        max_slack = 0
        max_edge = (0, 0)

        for (u, neighbors) in graph
            for v in neighbors
                slack = edge_slack(ranks, u, v)
                if slack > max_slack
                    max_slack = slack
                    max_edge = (u, v)
                end
            end
        end

        if max_slack <= 0
            break
        end

        p, s = max_edge

        # Apply relief operation
        new_ranks = copy(ranks)
        relief_edges = Set{Tuple{Int,Int}}()

        # Find all nodes reachable from s
        visited = Set{Int}()
        stack = [s]

        while !isempty(stack)
            node = pop!(stack)
            if node in visited
                continue
            end
            push!(visited, node)

            if haskey(graph, node)
                for neighbor in graph[node]
                    if edge_slack(ranks, node, neighbor) == 0
                        push!(stack, neighbor)
                        push!(relief_edges, (node, neighbor))
                    end
                end
            end
        end

        # Apply the minimum slack to all reachable nodes
        for node in visited
            new_ranks[node] += max_slack
        end

        # Update ranks
        ranks = new_ranks

        # Normalize ranks to start from 0
        min_rank = minimum(values(ranks))
        for node in keys(ranks)
            ranks[node] -= min_rank
        end

        # Recalculate agony
        current_agony = compute_agony(graph, ranks)
    end

    return current_agony, ranks
end

# LP implementation
function unweighted_agony_lp(graph::DirectedGraph)
    node_set = Set{Int}()

    # Collect all nodes
    for node in nodes(graph)
        push!(node_set, node)
    end

    # Create node mapping
    node_list = sort(collect(node_set))
    n = length(node_list)

    # Create a mapping from nodeIDs to consecutive integers (1-based)
    node_map = Dict{Int,Int}()
    for (i, node) in enumerate(node_list)
        node_map[node] = i
    end

    # Create edge list for LP
    edge_list = Tuple{Int,Int}[]

    for u in node_list
        for (v, _) in neighbors(graph, u)
            push!(edge_list, (node_map[u], node_map[v]))
        end
    end

    m = length(edge_list)

    # Create optimization model
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Decision variables
    @variable(model, p[1:m] >= 0)  # Edge penalty
    @variable(model, r[1:n] >= 0)  # Node rank

    # Objective: minimize sum of penalties
    @objective(model, Min, sum(p))

    # Constraints
    for (i, e) in enumerate(edge_list)
        u, v = e
        @constraint(model, r[v] - r[u] + p[i] >= 1)
    end

    # Solve model
    optimize!(model)

    agony_value = Int(round(objective_value(model)))
    ranks_array = Int[round(value(r[i])) for i in 1:n]

    # Shift ranks to 0-based
    min_rank = minimum(ranks_array)
    ranks_array .-= min_rank

    # Map ranks back to original nodeIDs
    node_ranks = Dict{Int,Int}()
    for (i, node) in enumerate(node_list)
        node_ranks[node] = ranks_array[i]
    end

    return agony_value, node_ranks
end

# Data I/O
function load_edgelist(filename::String)
    graph = DirectedGraph()

    open(filename, "r") do file
        for line in eachline(file)
            if startswith(line, "#") || isempty(strip(line))
                continue
            end

            parts = split(strip(line))
            if length(parts) >= 2
                source = parse(Int, parts[1])
                target = parse(Int, parts[2])

                weight = length(parts) >= 3 ? parse(Float64, parts[3]) : 1.0
                add_edge!(graph, source, target, weight)
            end
        end
    end

    return graph
end

# Main
function compute_agony(graph::DirectedGraph; method::Symbol=:lp)
    if method == :lp
        println("Using Linear Programming method...")
        agony_value, ranks = unweighted_agony_lp(graph)
    elseif method == :relief
        println("Using Relief method...")
        # Convert to adjacency list format
        adj_list = to_adj_list(graph)
        inv_adj_list = inverse_graph(adj_list)

        agony_value, ranks = relief_rank(adj_list, inv_adj_list)
    else
        throw(ArgumentError("Unknown method: $method. Use :lp or :relief"))
    end

    return agony_value, ranks
end

# If script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("This script implements functions for graph agony.")
    println("Import it in another script.")
end