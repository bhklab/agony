include("agony.jl")

println("Setting up graph...\n")

# Create simple directed graph
graph = DirectedGraph()

# Add nodes
for i in 1:6
    add_node!(graph, i)
end

# Add edges to create hierarchical structure
add_edge!(graph, 1, 2)
add_edge!(graph, 1, 3)
add_edge!(graph, 2, 4)
add_edge!(graph, 3, 4)
add_edge!(graph, 4, 5)
add_edge!(graph, 5, 6)

# Add a backward edge to create agony
add_edge!(graph, 4, 2)

println("Graph structure:")
println("\tAll nodes: ", nodes(graph))

for node in sort(collect(nodes(graph)))
    nbrs = neighbors(graph, node)
    if !isempty(nbrs)
        println("\tNode $node -> ", [n[1] for n in nbrs])
    end
end

# Compute agony using both methods

println("\nLP:")
lp_agony, lp_ranks = compute_agony(graph, method=:lp)
println("\tAgony value: $lp_agony")
println("\tNode ranks: $lp_ranks")


println("\nRelief:")
relief_agony, relief_ranks = compute_agony(graph, method=:relief)
println("\tAgony value: $relief_agony")
println("\tNode ranks: $relief_ranks")
