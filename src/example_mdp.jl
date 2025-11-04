# Define the state type for grid locations
struct GridWorldState
    x::Int
    y::Int
end

Base.:(==)(s1::GridWorldState, s2::GridWorldState) = s1.x == s2.x && s1.y == s2.y

# Define the MDP type
struct GridWorldMDP <: MDP{GridWorldState, Symbol}
    size_x::Int
    size_y::Int
    reward_states_values::Dict{GridWorldState, Float64}
    hit_wall_reward::Float64
    tprob::Float64
    discount_factor::Float64
end

function GridWorldMDP(; size_x=10, size_y=10,
    reward_states_values=Dict(
        GridWorldState(4,3)=>-10.0,
        GridWorldState(4,6)=>-5.0,
        GridWorldState(9,3)=>10.0,
        GridWorldState(8,8)=>3.0
    ),
    hit_wall_reward=-1.0,
    tprob=0.7,
    discount_factor=0.9
)
    GridWorldMDP(size_x, size_y, reward_states_values, hit_wall_reward, tprob, discount_factor)
end

# State space
function POMDPs.states(mdp::GridWorldMDP)
    states_array = GridWorldState[]
    for x in 1:mdp.size_x
        for y in 1:mdp.size_y
            push!(states_array, GridWorldState(x, y))
        end
    end
    push!(states_array, GridWorldState(-1, -1)) # terminal state
    return states_array
end

POMDPs.isterminal(mdp::GridWorldMDP, s::GridWorldState) = s == GridWorldState(-1, -1)
POMDPs.initialstate(mdp::GridWorldMDP) = Deterministic(GridWorldState(1, 1))

# Actions
POMDPs.actions(mdp::GridWorldMDP) = [:up, :down, :left, :right]

# Transition function
function POMDPs.transition(mdp::GridWorldMDP, s::GridWorldState, a::Symbol)
    if isterminal(mdp, s)
        return SparseCat([s], [1.0])
    end
    if s in keys(mdp.reward_states_values) && mdp.reward_states_values[s] > 0
        return SparseCat([GridWorldState(-1, -1)], [1.0])
    end
    tprob_other = (1 - mdp.tprob) / 3
    new_state_up    = GridWorldState(s.x, min(s.y + 1, mdp.size_y))
    new_state_down  = GridWorldState(s.x, max(s.y - 1, 1))
    new_state_left  = GridWorldState(max(s.x - 1, 1), s.y)
    new_state_right = GridWorldState(min(s.x + 1, mdp.size_x), s.y)
    new_state_vector = [new_state_up, new_state_down, new_state_left, new_state_right]
    t_prob_vector = fill(tprob_other, 4)
    if a == :up
        t_prob_vector[1] = mdp.tprob
    elseif a == :down
        t_prob_vector[2] = mdp.tprob
    elseif a == :left
        t_prob_vector[3] = mdp.tprob
    elseif a == :right
        t_prob_vector[4] = mdp.tprob
    else
        error("Invalid action")
    end
    # Combine probabilities for duplicate states
    for i in 1:4
        for j in (i+1):4
            if new_state_vector[i] == new_state_vector[j]
                t_prob_vector[i] += t_prob_vector[j]
                t_prob_vector[j] = 0.0
            end
        end
    end
    # Remove states with zero probability
    new_state_vector = new_state_vector[t_prob_vector .> 0]
    t_prob_vector = t_prob_vector[t_prob_vector .> 0]
    return SparseCat(new_state_vector, t_prob_vector)
end

# Reward function
function POMDPs.reward(mdp::GridWorldMDP, s::GridWorldState, a::Symbol, sp::GridWorldState)
    if isterminal(mdp, s)
        return 0.0
    end
    if s in keys(mdp.reward_states_values) && mdp.reward_states_values[s] > 0
        return mdp.reward_states_values[s]
    end
    r = 0.0
    if s in keys(mdp.reward_states_values) && mdp.reward_states_values[s] < 0
        r += mdp.reward_states_values[s]
    end
    if s == sp
        r += mdp.hit_wall_reward
    end
    return r
end

function POMDPs.reward(mdp::GridWorldMDP, s::GridWorldState, a::Symbol)
    r = 0.0
    for (sp, p) in transition(mdp, s, a)
        r += p * reward(mdp, s, a, sp)
    end
    return r
end

# Discount function
POMDPs.discount(mdp::GridWorldMDP) = mdp.discount_factor

function example_mdp()
    # Create an instance of the problem
    mdp = GridWorldMDP()

    function callback(planner::MCTS.MCTSPlanner, i::Int64)
        if i == 5
            @info planner.tree
        end
    end

    # Solve using MCTS
    solver = MCTSSolver(;n_iterations=1000, depth=20, exploration_constant=10.0, callback, enable_tree_vis=true)
    mcts_policy = solve(solver, mdp)

    # Example usage
    s = GridWorldState(5, 5)
    println(action(mcts_policy, s))  # Returns the suggested action for state (9, 2)

    mcts_viz(mdp, mcts_policy)
end
