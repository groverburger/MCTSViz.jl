function state_label(tree::MCTS.MCTSTree, state_id::Integer, show_stats::Bool)
    label = string(tree.s_labels[state_id])
    return show_stats ? "$label (N:$(tree.total_n[state_id]))" : label
end

function action_label(tree::MCTS.MCTSTree, action_id::Integer, show_stats::Bool)
    label = string(tree.a_labels[action_id])
    show_stats || return label
    return "$label (N:$(tree.n[action_id]), Q:$(round(tree.q[action_id], digits = 3)))"
end

function ascii_tree(
    tree::MCTS.MCTSTree,
    next_states::Function,
    root_id::Integer = 1;
    max_depth::Integer = 5,
    show_stats::Bool = true,
)
    1 <= root_id <= length(tree.s_labels) || throw(BoundsError(tree.s_labels, root_id))
    max_depth >= 0 || throw(ArgumentError("max_depth must be nonnegative"))

    lines = [state_label(tree, root_id, show_stats)]
    visited = Set([root_id])

    function append_actions!(state_id::Int, prefix::String, depth::Int)
        depth >= max_depth && return
        actions = tree.child_ids[state_id]
        for (action_index, action_id) in enumerate(actions)
            last_action = action_index == length(actions)
            push!(lines, prefix * (last_action ? "`-- " : "+-- ") *
                         action_label(tree, action_id, show_stats))

            action_prefix = prefix * (last_action ? "    " : "|   ")
            child_states = next_states(state_id, action_id)
            for (state_index, next_state_id) in enumerate(child_states)
                last_state = state_index == length(child_states)
                branch = last_state ? "`-- " : "+-- "
                seen = next_state_id in visited
                push!(lines, action_prefix * branch *
                             state_label(tree, next_state_id, show_stats))
                if !seen
                    push!(visited, next_state_id)
                    state_prefix = action_prefix * (last_state ? "    " : "|   ")
                    append_actions!(next_state_id, state_prefix, depth + 1)
                end
            end
        end
    end

    append_actions!(root_id, "", 0)
    return join(lines, '\n')
end

"""
    mcts_ascii_viz(tree, root_id = 1; max_depth = 5, show_stats = true)

Return the sampled transitions recorded in an MCTS tree as text.
"""
function mcts_ascii_viz(tree::MCTS.MCTSTree, root_id::Integer = 1; kwargs...)
    transitions = recorded_transition_map(tree)
    return ascii_tree(tree, (_state_id, action_id) -> get(transitions, action_id, Int[]),
                      root_id; kwargs...)
end

"""
    mcts_ascii_viz(mdp, mcts_policy, root_id = 1; max_depth = 5, show_stats = true)

Return a text view using the same dominant MDP transitions as [`mcts_viz`](@ref).
"""
function mcts_ascii_viz(mdp, mcts_policy, root_id::Integer = 1; kwargs...)
    tree = mcts_policy.tree
    cache = Dict{Tuple{Int, Int}, Vector{Int}}()
    next_states = (state_id, action_id) -> get!(cache, (state_id, action_id)) do
        dominant_next_state_ids(mdp, tree, state_id, action_id)
    end
    return ascii_tree(tree, next_states, root_id; kwargs...)
end
