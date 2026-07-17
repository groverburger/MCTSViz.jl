module MCTSViz

using CImGui
using Revise
using Mirage
using POMDPs
using MCTS
using POMDPTools

@kwdef mutable struct TreeNode
    position::Vector{Float64} = [0.0, 0.0]
    velocity::Vector{Float64} = [0.0, 0.0]
    force::Vector{Float64} = [0.0, 0.0]
    text::String = ""
    is_state::Bool = true
    index::Int = 0
    parents::Vector{TreeNode} = []
    children::Vector{TreeNode} = []
    id::Int = 0
end

@kwdef mutable struct VizSettings
    color_code_q_values::Ref{Bool} = Ref(true)
    color_code_n_values::Ref{Bool} = Ref(false)
    show_node_text::Ref{Bool} = Ref(false)
    show_weighted_arrows::Ref{Bool} = Ref(false)
    repulsion_strength::Ref{Float32} = Ref(600.0f0)
    attraction_strength::Ref{Float32} = Ref(0.9f0)
    desired_distance::Float64 = 32.0
end

@kwdef mutable struct VizState
    settings::VizSettings = VizSettings()
    layout_ready::Bool = false
    first_frame::Bool = true
    center_root::Bool = false
    expand_all::Bool = false
    expand_best_path::Bool = false
    collapse_all::Bool = false
    selected_node::Union{Nothing, TreeNode} = nothing
    left_dragged::Bool = false
end

const persisted_settings = Ref(VizSettings())

mutable struct SpatialHashGrid
    cell_size::Float64
    cells::Dict{Tuple{Int, Int}, Vector{TreeNode}}
end

function SpatialHashGrid(cell_size::Float64)
    return SpatialHashGrid(cell_size, Dict{Tuple{Int, Int}, Vector{TreeNode}}())
end

function get_cell_coords(grid::SpatialHashGrid, position::Vector{Float64})
    return (floor(Int, position[1] / grid.cell_size), floor(Int, position[2] / grid.cell_size))
end

function insert!(grid::SpatialHashGrid, node::TreeNode)
    coords = get_cell_coords(grid, node.position)
    if !haskey(grid.cells, coords)
        grid.cells[coords] = []
    end
    push!(grid.cells[coords], node)
end

mutable struct Camera
    pan::Vector{Float64}
    panning::Bool
    zoom::Float64
end

const MATLAB_WINTER_PALETTE = map(t -> (Float32(t[1]), Float32(t[2]), Float32(t[3])), [
    (  0/255,   0/255, 255/255),
    (  0/255,  28/255, 241/255),
    (  0/255,  57/255, 227/255),
    (  0/255,  85/255, 213/255),
    (  0/255, 113/255, 198/255),
    (  0/255, 142/255, 184/255),
    (  0/255, 170/255, 170/255),
    (  0/255, 198/255, 156/255),
    (  0/255, 227/255, 142/255),
    (  0/255, 255/255, 128/255),
])

const MATLAB_HOT_PALETTE = map(t -> (Float32(t[1]), Float32(t[2]), Float32(t[3])), [
    (100/255,  16/255,  12/255),
    (140/255,  20/255,  10/255),
    (180/255,  20/255,   5/255),
    (220/255,  20/255,   0/255),
    (255/255,  80/255,   0/255),
    (255/255, 150/255,   0/255),
    (255/255, 210/255,  40/255),
    (255/255, 245/255, 130/255),
    (255/255, 255/255, 210/255),
    (255/255, 255/255, 255/255),
])

function percentile_intensity(value::Real, sorted_values::AbstractVector{<:Real})
    isfinite(value) || return 0.5
    length(sorted_values) <= 1 && return 0.5

    first_equal = searchsortedfirst(sorted_values, value)
    last_equal = searchsortedlast(sorted_values, value)
    midpoint_rank = ((first_equal - 1) + (last_equal - 1)) / 2
    return clamp(midpoint_rank / (length(sorted_values) - 1), 0.0, 1.0)
end

mutable struct MCTSVizSession
    app::MirageApp
    mdp::Any
    mcts_tree::Any
    root_node::TreeNode
    node_id_counter::Int
    all_nodes::Vector{TreeNode}
    camera::Camera
    expand_levels::Int
    state::VizState
    transition_cache::Dict{Tuple{Int, Int}, Vector{Int}}
end

function request_animation_frame(session::MCTSVizSession, frames::Integer = 1)
    request_frame!(session.app, frames)
    return nothing
end

function recorded_transition_map(tree::MCTS.MCTSTree)
    transitions = Dict{Int, Vector{Int}}()
    for ((visited_action, state_id), _count) in tree._vis_stats
        push!(get!(Vector{Int}, transitions, visited_action), state_id)
    end
    return transitions
end

function dominant_next_state_ids(mdp, tree::MCTS.MCTSTree, state_id::Integer, action_id::Integer)
    distribution = transition(mdp, tree.s_labels[state_id], tree.a_labels[action_id])
    candidates = support(distribution)
    isempty(candidates) && return Int[]

    next_state = first(candidates)
    best_probability = pdf(distribution, next_state)
    for candidate in Iterators.drop(candidates, 1)
        probability = pdf(distribution, candidate)
        if probability > best_probability
            next_state = candidate
            best_probability = probability
        end
    end
    next_state_id = get(tree.state_map, next_state, 0)
    return next_state_id == 0 ? Int[] : [next_state_id]
end

function dominant_next_state_ids(session::MCTSVizSession, state_id::Integer, action_id::Integer)
    key = (Int(state_id), Int(action_id))
    return get!(session.transition_cache, key) do
        dominant_next_state_ids(session.mdp, session.mcts_tree, state_id, action_id)
    end
end

function mcts_viz_frame!(session::MCTSVizSession)
    app = session.app
    viz_state = session.state
    if !viz_state.layout_ready
        dock_layout!(app; center = "Tree View", left = "Settings", left_size = 0.22)
        viz_state.layout_ready = true
    end
    settings_window(session)
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowPadding, (0.0f0, 0.0f0))
    CImGui.Begin("Tree View")
    CImGui.PopStyleVar()
    try
        draw_canvas!(app, :mcts_tree; label = "mcts_tree_canvas") do canvas, viewport
            main_view(canvas, viewport, session)
        end
    finally
        CImGui.End()
    end
    if CImGui.IsMouseClicked(0) || CImGui.IsMouseClicked(1)
        request_animation_frame(session, 10)
    end
    if CImGui.IsMouseReleased(0) || CImGui.IsMouseReleased(1)
        request_animation_frame(session, 10)
    end

    viz_state.first_frame = false
    return nothing
end

"""
    mcts_viz(mdp, mcts_policy; keep_state = true, expand_levels = 2)

Open an interactive view of the tree currently stored in `mcts_policy`. The MDP is
used to connect each action to its most likely successor state; visit counts and
Q-values come from the recorded MCTS tree. Construct the policy with
`enable_tree_vis = true` so tree statistics are available.
"""
function mcts_viz(
    mdp,
    mcts_policy;
    keep_state::Bool = true,
    expand_levels::Int = 2,
    live_reload::Bool = true,
    live_reload_interval::Real = 0.1,
)
    mcts_tree = mcts_policy.tree
    isempty(mcts_tree.s_labels) && throw(ArgumentError("mcts_policy.tree has no states to visualize"))
    app = MirageApp("MCTSViz"; width = 1200, height = 800)
    camera = Camera([0.0, 0.0], false, 1.0)
    settings = keep_state ? persisted_settings[] : VizSettings()
    viz_state = VizState(; settings)

    root_node = TreeNode(text = string(mcts_tree.s_labels[1]), index = 1, id = 1)
    all_nodes = [root_node]
    session = MCTSVizSession(
        app, mdp, mcts_tree, root_node, 1, all_nodes, camera, expand_levels,
        viz_state, Dict{Tuple{Int, Int}, Vector{Int}}(),
    )
    request_frame!(app, 10)

    if live_reload
        run_live!(app; idle_timeout = live_reload_interval) do _app
            Base.invokelatest(mcts_viz_frame!, session)
        end
    else
        run!(app) do _app
            Base.invokelatest(mcts_viz_frame!, session)
        end
    end
    return nothing
end

function main_view(canvas, canvas_viewport, session::MCTSVizSession)
    (; mcts_tree, root_node, all_nodes, camera, expand_levels) = session
    viz_state = session.state
    settings = viz_state.settings
    state_node_map = Dict{Int, TreeNode}(map(n -> n.index => n, filter(n -> n.is_state, all_nodes)))
    n_palette = MATLAB_WINTER_PALETTE
    q_palette = MATLAB_HOT_PALETTE
    sorted_q_values = sort!(Float64[value for value in mcts_tree.q if isfinite(value)])

    min_n_value = Inf
    max_n_value = -Inf
    min_edge_visits = Inf
    max_edge_visits = -Inf

    function edge_visits(parent::TreeNode, child::TreeNode)
        if parent.is_state && !child.is_state
            return mcts_tree.n[child.index]
        elseif !parent.is_state && child.is_state
            return get(mcts_tree._vis_stats, parent.index => child.index, 0)
        else
            return child.is_state ? mcts_tree.total_n[child.index] : mcts_tree.n[child.index]
        end
    end

    for node in all_nodes
        if node.is_state
            n_val = mcts_tree.total_n[node.index]
            min_n_value = min(min_n_value, n_val)
            max_n_value = max(max_n_value, n_val)
        end

        for child in node.children
            visits = edge_visits(node, child)
            min_edge_visits = min(min_edge_visits, visits)
            max_edge_visits = max(max_edge_visits, visits)
        end
    end

    if min_n_value == Inf
        min_n_value = 0.0
        max_n_value = 0.0
    end
    if min_edge_visits == Inf
        min_edge_visits = 0.0
        max_edge_visits = 0.0
    end

    function normalize_range(value, min_value, max_value)
        return max_value == min_value ? 0.5 : clamp((value - min_value) / (max_value - min_value), 0.0, 1.0)
    end

    function rgba_from_palette(intensity, palette; alpha=255)
        color = interpolate_palette(intensity, palette)
        alpha_value = alpha > 1 ? alpha / 255 : alpha
        return (color[1], color[2], color[3], alpha_value)
    end

    function node_label(node::TreeNode)
        if node.is_state
            return string(mcts_tree.s_labels[node.index])
        else
            return string(mcts_tree.a_labels[node.index])
        end
    end

    function node_object(node::TreeNode)
        return node.is_state ? mcts_tree.s_labels[node.index] : mcts_tree.a_labels[node.index]
    end

    function node_visits(node::TreeNode)
        if node.is_state
            return mcts_tree.total_n[node.index]
        else
            return mcts_tree.n[node.index]
        end
    end

    function node_value_text(node::TreeNode)
        return node.is_state ? "" : string(mcts_tree.q[node.index])
    end

    function render_table_row(name, value)
        CImGui.TableNextRow()
        CImGui.TableSetColumnIndex(0)
        CImGui.TextUnformatted(string(name))
        CImGui.TableSetColumnIndex(1)
        CImGui.TextUnformatted(string(value))
        return nothing
    end

    function render_object_table(object)
        table_flags = CImGui.ImGuiTableFlags_Borders | CImGui.ImGuiTableFlags_RowBg
        if CImGui.BeginTable("Object Fields", 2, table_flags)
            try
                render_table_row("type", typeof(object))
                names = fieldnames(typeof(object))
                if isempty(names)
                    render_table_row("value", object)
                else
                    for name in names
                        render_table_row(name, getfield(object, name))
                    end
                end
            finally
                CImGui.EndTable()
            end
        end
        return nothing
    end

    function render_internal_table(node)
        table_flags = CImGui.ImGuiTableFlags_Borders | CImGui.ImGuiTableFlags_RowBg
        if CImGui.BeginTable("Internal Node Details", 2, table_flags)
            try
                render_table_row("node id", node.id)
                render_table_row("tree index", node.index)
                render_table_row("kind", node.is_state ? "state" : "action")
                render_table_row("position", "($(round(node.position[1], digits=2)), $(round(node.position[2], digits=2)))")
            finally
                CImGui.EndTable()
            end
        end
        return nothing
    end

    function render_node_toggle_button(prefix, node)
        if CImGui.Button("Toggle##$(prefix)_$(node.id)")
            toggle_node!(node)
            if viz_state.selected_node === nothing
                CImGui.CloseCurrentPopup()
            end
        end
        CImGui.SameLine()
        CImGui.TextUnformatted("$(node.is_state ? "State" : "Action") #$(node.id): $(node_label(node))")
        return nothing
    end

    function n_color(node::TreeNode; alpha=255)
        intensity = normalize_range(node_visits(node), min_n_value, max_n_value)
        return rgba_from_palette(intensity, n_palette; alpha)
    end

    q_intensity(node::TreeNode) =
        percentile_intensity(mcts_tree.q[node.index], sorted_q_values)

    function edge_color(parent::TreeNode, child::TreeNode; alpha=255)
        min_log_visits = log1p(max(0, min_edge_visits))
        max_log_visits = log1p(max(0, max_edge_visits))
        intensity = normalize_range(log1p(max(0, edge_visits(parent, child))), min_log_visits, max_log_visits)
        return rgba_from_palette(intensity, n_palette; alpha)
    end

    function edge_width(parent::TreeNode, child::TreeNode)
        if !settings.show_weighted_arrows[] || max_edge_visits <= min_edge_visits
            return 1.5
        end
        min_log_visits = log1p(max(0, min_edge_visits))
        max_log_visits = log1p(max(0, max_edge_visits))
        intensity = normalize_range(log1p(max(0, edge_visits(parent, child))), min_log_visits, max_log_visits)
        return 0.75 + 4.75 * intensity
    end

    # Helper functions
    function get_actions_from_state_index(state_index::Int64)
        return (1 <= state_index <= length(mcts_tree.child_ids)) ? mcts_tree.child_ids[state_index] : Int[]
    end

    function next_position(node, index, count)
        if isempty(node.parents)
            angle = 2π * (index - 1) / max(count, 1)
        else
            parent = first(node.parents)
            direction = node.position - parent.position
            base_angle = sum(abs2, direction) > eps() ? atan(direction[2], direction[1]) : 0.0
            fan_angle = count <= 1 ? 0.0 : π / 2
            offset = count <= 1 ? 0.0 : fan_angle * ((index - 1) / (count - 1) - 0.5)
            angle = base_angle + offset
        end
        radius = node.is_state ? 100.0 : 75.0
        return node.position + radius .* [cos(angle), sin(angle)]
    end

    function expand_one_level!(node)
        if !isempty(node.children)
            return false
        end

        if node.is_state
            actions = get_actions_from_state_index(node.index)
            for (a_idx, action) in enumerate(actions)
                session.node_id_counter += 1
                new_node = TreeNode(
                    text = string(mcts_tree.a_labels[action]),
                    is_state = false,
                    index = action,
                    parents = [node],
                    position = next_position(node, a_idx, length(actions)),
                    id = session.node_id_counter
                )
                push!(node.children, new_node)
                push!(all_nodes, new_node)
            end
        else
            @assert !isempty(node.parents) "Action node should have at least one parent"
            parent_state_id = first(node.parents).index
            state_ids = dominant_next_state_ids(session, parent_state_id, node.index)
            for (state_index, state_id) in enumerate(state_ids)
                if haskey(state_node_map, state_id)
                    new_node = state_node_map[state_id]
                    if !(node in new_node.parents)
                        push!(new_node.parents, node)
                    end
                else
                    session.node_id_counter += 1
                    new_node = TreeNode(
                        text = string(mcts_tree.s_labels[state_id]),
                        index = state_id,
                        parents = [node],
                        position = next_position(node, state_index, length(state_ids)),
                        id = session.node_id_counter
                    )
                    state_node_map[state_id] = new_node
                    push!(all_nodes, new_node)
                end
                if !(new_node in node.children)
                    push!(node.children, new_node)
                end
            end
        end
        return true
    end

    function expand_node(node, levels)
        if levels <= 0
            return
        end

        expand_one_level!(node)
        for child in node.children
            expand_node(child, levels - 1)
        end
    end

    function expand_all!(node, visited=Set{Int}())
        if node.id in visited
            return
        end
        push!(visited, node.id)
        expand_one_level!(node)
        for child in copy(node.children)
            expand_all!(child, visited)
        end
    end

    function best_child(node)
        expand_one_level!(node)
        if isempty(node.children)
            return nothing
        end
        if node.is_state
            return node.children[argmax([mcts_tree.q[child.index] for child in node.children])]
        else
            return node.children[argmax([mcts_tree.total_n[child.index] for child in node.children])]
        end
    end

    function expand_best_path!(node, visited=Set{Int}())
        if node.id in visited
            return
        end
        push!(visited, node.id)
        child = best_child(node)
        if child !== nothing
            expand_best_path!(child, visited)
        end
    end

    function collapse_all!()
        for node in all_nodes
            if node !== root_node
                empty!(node.parents)
            end
            empty!(node.children)
        end
        empty!(all_nodes)
        push!(all_nodes, root_node)
        empty!(state_node_map)
        state_node_map[root_node.index] = root_node
        root_node.position .= [0.0, 0.0]
        root_node.velocity .= [0.0, 0.0]
        root_node.force .= [0.0, 0.0]
        viz_state.selected_node = nothing
    end

    function delete_unreachable_nodes(root_node, all_nodes, state_node_map)
        # Find all nodes reachable from the root
        reachable_nodes = Set{TreeNode}()
        q = [root_node]
        push!(reachable_nodes, root_node)

        head = 1
        while head <= length(q)
            curr = q[head]
            head += 1

            for child in curr.children
                if !(child in reachable_nodes)
                    push!(reachable_nodes, child)
                    push!(q, child)
                end
            end
        end

        # Filter all_nodes and state_node_map
        filter!(n -> n in reachable_nodes, all_nodes)

        # Rebuild state_node_map from the filtered all_nodes
        empty!(state_node_map)
        for n in all_nodes
            if n.is_state
                state_node_map[n.index] = n
            end
        end
    end

    function clear_selected_node_if_unreachable!()
        selected_node = viz_state.selected_node
        if selected_node !== nothing && !(selected_node in all_nodes)
            viz_state.selected_node = nothing
            return true
        end
        return false
    end

    function toggle_node!(node)
        if isempty(node.children)
            expand_one_level!(node)
        else
            children_to_process = copy(node.children)
            empty!(node.children)

            for child in children_to_process
                filter!(p -> p.id != node.id, child.parents)
            end

            delete_unreachable_nodes(root_node, all_nodes, state_node_map)
            clear_selected_node_if_unreachable!()
        end
        request_animation_frame(session, 10)
        return nothing
    end

    if viz_state.first_frame
        expand_node(root_node, expand_levels)
    end

    if viz_state.collapse_all
        collapse_all!()
        viz_state.collapse_all = false
        request_animation_frame(session, 30)
    end

    if viz_state.expand_all
        expand_all!(root_node)
        viz_state.expand_all = false
        request_animation_frame(session, 30)
    end

    if viz_state.expand_best_path
        expand_best_path!(root_node)
        viz_state.expand_best_path = false
        request_animation_frame(session, 30)
    end

    # Camera panning
    canvas_size = CImGui.ImVec2(canvas_viewport.width, canvas_viewport.height)
    mouse_rel = canvas_viewport.mouse_rel
    is_hovering_canvas = canvas_viewport.hovered

    function canvas_to_world(mouse, pan, zoom, size)
        centered_mouse = [mouse[1] - size.x / 2, mouse[2] - size.y / 2]
        return (centered_mouse .- pan) ./ zoom
    end

    if viz_state.center_root
        camera.pan .= -root_node.position .* camera.zoom
        viz_state.center_root = false
        request_animation_frame(session, 10)
    end

    wheel_delta = unsafe_load(CImGui.GetIO().MouseWheel)
    if is_hovering_canvas && wheel_delta != 0
        world_pos_before = canvas_to_world(mouse_rel, camera.pan, camera.zoom, canvas_size)
        camera.zoom *= (1.0 + wheel_delta * 0.1)
        camera.zoom = clamp(camera.zoom, 0.1, 10.0)
        world_pos_after = canvas_to_world(mouse_rel, camera.pan, camera.zoom, canvas_size)
        pan_delta = world_pos_before - world_pos_after
        camera.pan .-= pan_delta .* camera.zoom
        request_animation_frame(session, 10)
    end

    left_pressed = CImGui.IsMouseClicked(0)
    left_released = CImGui.IsMouseReleased(0)
    left_pressed && (viz_state.left_dragged = false)

    left_dragging = is_hovering_canvas && CImGui.IsMouseDragging(0, 4.0f0)
    middle_dragging = is_hovering_canvas && CImGui.IsMouseDown(2)
    viz_state.left_dragged |= left_dragging

    if left_dragging || middle_dragging
        if !camera.panning
            camera.panning = true
        end
        mouse_delta = CImGui.GetIO().MouseDelta
        camera.pan .+= [unsafe_load(mouse_delta.x), unsafe_load(mouse_delta.y)]
        request_animation_frame(session, 10)
    else
        camera.panning = false
    end

    # Physics simulation
    function update_physics(nodes, delta_time)
        repulsion_strength = Float64(settings.repulsion_strength[])
        attraction_strength = Float64(settings.attraction_strength[])
        desired_distance = settings.desired_distance
        desired_distance_sq = desired_distance * desired_distance
        damping = 0.85

        for node in nodes
            node.force[1] = 0.0
            node.force[2] = 0.0
        end

        # Build spatial hash grid
        grid_cell_size = desired_distance * 6.0
        grid = SpatialHashGrid(grid_cell_size)
        for node in nodes
            insert!(grid, node)
        end

        # Repulsion
        empty_neighbors = TreeNode[]
        for node_a in nodes
            center_coords = get_cell_coords(grid, node_a.position)
            for i in -1:1
                for j in -1:1
                    neighbor_coords = (center_coords[1] + i, center_coords[2] + j)
                    for node_b in get(grid.cells, neighbor_coords, empty_neighbors)
                        if node_a.id < node_b.id
                            dx = node_a.position[1] - node_b.position[1]
                            dy = node_a.position[2] - node_b.position[2]
                            distance_sq = dx * dx + dy * dy
                            if distance_sq > 1.0 # Avoid extreme forces at very close distances
                                distance = sqrt(distance_sq)
                                force_magnitude = repulsion_strength * desired_distance_sq / distance_sq
                                fx = force_magnitude * dx / distance
                                fy = force_magnitude * dy / distance
                            else
                                # Apply a small fixed force to push nodes apart if they are on top of each other
                                fx = 1.0
                                fy = 0.0
                            end
                            node_a.force[1] += fx
                            node_a.force[2] += fy
                            node_b.force[1] -= fx
                            node_b.force[2] -= fy
                        end
                    end
                end
            end
        end

        # Attraction
        for node in nodes
            if !isempty(node.parents)
                for parent in node.parents
                    fx = attraction_strength * (parent.position[1] - node.position[1])
                    fy = attraction_strength * (parent.position[2] - node.position[2])
                    node.force[1] += fx
                    node.force[2] += fy
                    parent.force[1] -= fx
                    parent.force[2] -= fy
                end
            end
        end

        # Update positions
        max_speed_sq = 0.0
        for node in nodes
            if node.id == 1 # Fix root node
                node.position[1] = 0.0
                node.position[2] = 0.0
                node.velocity[1] = 0.0
                node.velocity[2] = 0.0
                node.force[1] = 0.0
                node.force[2] = 0.0
                continue
            end

            node.velocity[1] = (node.velocity[1] + node.force[1] * delta_time) * damping
            node.velocity[2] = (node.velocity[2] + node.force[2] * delta_time) * damping
            #node.velocity[2] = 0
            node.position[1] += node.velocity[1] * delta_time
            node.position[2] += node.velocity[2] * delta_time
            max_speed_sq = max(max_speed_sq, node.velocity[1] * node.velocity[1] + node.velocity[2] * node.velocity[2])
        end
        return max_speed_sq
    end

    max_speed_sq = 0.0
    for i in 1:6
        max_speed_sq = max(max_speed_sq, update_physics(all_nodes, 3 / 60))
    end
    if max_speed_sq > 0.01
        request_animation_frame(session, 1)
    end

    # Rendering
    Mirage.save()
    Mirage.fillcolor(Mirage.rgba(0, 0, 20, 255))
    Mirage.fillrect(0, 0, canvas.width, canvas.height)
    Mirage.restore()
    Mirage.save()
    Mirage.translate(canvas.width / 2 + camera.pan[1], canvas.height / 2 + camera.pan[2])
    Mirage.scale(camera.zoom, camera.zoom)

    # Draw connections
    function fill_polygon!(points, color)
        Mirage.save()
        try
            Mirage.fillcolor(color)
            Mirage.moveto(points[1]...)
            for point in points[2:end]
                Mirage.lineto(point...)
            end
            Mirage.closepath()
            Mirage.fill()
        finally
            Mirage.restore()
        end
    end

    function draw_arrow(p1, p2, arrowhead_length, arrowhead_angle, node_radius, stroke_width, color)
        if p1 == p2
            return
        end
        dx = p2[1] - p1[1]
        dy = p2[2] - p1[2]
        distance = hypot(dx, dy)
        if distance <= 0.0
            return
        end
        dir_x = dx / distance
        dir_y = dy / distance
        normal_x = -dir_y
        normal_y = dir_x
        tip_x = p2[1] - dir_x * node_radius
        tip_y = p2[2] - dir_y * node_radius
        shaft_half_width = stroke_width / 2

        # End the shaft just inside the arrowhead base. The small overlap avoids a
        # seam, while keeping thick shafts from extending through to the tip.
        base_x = tip_x - arrowhead_length * dir_x
        base_y = tip_y - arrowhead_length * dir_y
        overlap = min(arrowhead_length * 0.5, max(1.0, stroke_width))
        shaft_end_x = base_x + overlap * dir_x
        shaft_end_y = base_y + overlap * dir_y
        arrowhead_half_width = max(arrowhead_length * tan(arrowhead_angle), stroke_width * 1.75)

        fill_polygon!([
            (p1[1] + normal_x * shaft_half_width, p1[2] + normal_y * shaft_half_width),
            (shaft_end_x + normal_x * shaft_half_width, shaft_end_y + normal_y * shaft_half_width),
            (shaft_end_x - normal_x * shaft_half_width, shaft_end_y - normal_y * shaft_half_width),
            (p1[1] - normal_x * shaft_half_width, p1[2] - normal_y * shaft_half_width),
        ], color)

        # Draw arrowhead as a filled triangle so it tracks the shaft color and weight.
        fill_polygon!([
            (tip_x, tip_y),
            (base_x + normal_x * arrowhead_half_width, base_y + normal_y * arrowhead_half_width),
            (base_x - normal_x * arrowhead_half_width, base_y - normal_y * arrowhead_half_width),
        ], color)
    end

    function draw_connections(node, visited=Set())
        if node in visited
            return
        end
        push!(visited, node)
        for child in node.children
            color = settings.color_code_n_values[] ? edge_color(node, child; alpha=230) : Mirage.rgba(255, 255, 255, 90)
            draw_arrow(node.position, child.position, 10.0, pi/6, 24.0, edge_width(node, child), color)
            draw_connections(child, visited)
        end
    end
    Mirage.save()
    draw_connections(root_node)
    Mirage.restore()

    # Draw nodes and handle clicks
    for node in copy(all_nodes)
        Mirage.save()
        
        # Correctly calculate world mouse position considering pan and zoom
        world_mouse_pos = canvas_to_world(mouse_rel, camera.pan, camera.zoom, canvas_size)
        
        # Hit areas use world units, like the rendered geometry, so both grow
        # together as the camera zooms in.
        hit_radius = node.is_state ? 28.0 : 22.0
        is_hovered = (
            is_hovering_canvas &&
            hypot(node.position[1] - world_mouse_pos[1], node.position[2] - world_mouse_pos[2]) <= hit_radius
        )

        if node.is_state
            if settings.color_code_n_values[]
                n_val = mcts_tree.total_n[node.index]
                intensity = normalize_range(n_val, min_n_value, max_n_value)
                Mirage.fillcolor(rgba_from_palette(intensity, n_palette))
            else
                Mirage.fillcolor(Mirage.rgba(0, 0, 200, 255))
            end
        else
            if settings.color_code_q_values[]
                Mirage.fillcolor(rgba_from_palette(q_intensity(node), q_palette))
            else
                Mirage.fillcolor(Mirage.rgba(150, 150, 0, 255))
            end
        end

        if is_hovered && left_released && !viz_state.left_dragged
            toggle_node!(node)
        end

        if is_hovered && CImGui.IsMouseClicked(1)
            viz_state.selected_node = node
            CImGui.OpenPopup("Node Details")
            request_animation_frame(session, 10)
        end
        
        Mirage.translate(node.position...)

        if node.is_state
            # Draw circle for state nodes
            Mirage.circle(24)
            Mirage.fill()
            if is_hovered
                Mirage.save()
                Mirage.strokecolor(Mirage.rgba(255, 255, 255, 255))
                Mirage.strokewidth(3)
                Mirage.circle(24)
                Mirage.stroke()
                Mirage.restore()
            end
        else
            # Draw diamond for action nodes
            local node_size = 24 * 0.75
            Mirage.moveto(0, node_size)
            Mirage.lineto(node_size, 0)
            Mirage.lineto(0, -node_size)
            Mirage.lineto(-node_size, 0)
            Mirage.closepath()
            Mirage.fill()
            Mirage.save()
            Mirage.strokecolor(is_hovered ? Mirage.rgba(255, 255, 255, 255) : Mirage.rgba(255, 255, 255, 120))
            Mirage.strokewidth(is_hovered ? 3 : 1.5)
            Mirage.moveto(0, node_size)
            Mirage.lineto(node_size, 0)
            Mirage.lineto(0, -node_size)
            Mirage.lineto(-node_size, 0)
            Mirage.closepath()
            Mirage.stroke()
            Mirage.restore()
        end

        let
            should_render_text = settings.show_node_text[] || is_hovered
            if camera.zoom < 0.5 && !is_hovered
                should_render_text = false
            end

            if should_render_text
                local text_to_render = node.text
                if node.is_state
                    s_idx = node.index
                    visits = mcts_tree.total_n[s_idx]
                    text_to_render = "$(node.text)
N: $(visits)"
                else
                    a_idx = node.index
                    action = mcts_tree.a_labels[a_idx]
                    visits = mcts_tree.n[a_idx]
                    v_val = round(mcts_tree.q[a_idx], digits=3)
                    text_to_render = "a: $(action)
N: $visits, Q: $v_val"
                end

                Mirage.fillcolor(Mirage.rgba(255, 255, 255, 255))
                Mirage.scale(1 / camera.zoom)
                
                # Estimate text size and center it
                font_size = 16
                lines = split(text_to_render, '
')
                max_width = 0
                for line in lines
                    max_width = max(max_width, length(line))
                end
                text_width = max_width * font_size / 2
                text_height = length(lines) * font_size
                
                if is_hovered
                    Mirage.translate(
                        (world_mouse_pos[1] - node.position[1]) * camera.zoom - text_width / 2,
                        (world_mouse_pos[2] - node.position[2]) * camera.zoom - text_height - 10
                    )
                else
                    # Adjust for multi-line text
                    Mirage.translate(-text_width / 2, -text_height/2 + font_size/2)
                end

                # Render each line of text
                for (i, line) in enumerate(lines)
                    line_width = length(line) * font_size / 2
                    Mirage.save()
                    # Center each line horizontally
                    Mirage.translate(round((text_width - line_width) / 2), round((i-1) * font_size))
                    Mirage.save()
                    Mirage.translate(1, 1)
                    Mirage.fillcolor(Mirage.rgba(0, 0, 0, 255))
                    Mirage.text(string(line))
                    Mirage.restore()
                    Mirage.text(string(line))
                    Mirage.restore()
                end
            end
        end

        Mirage.restore()
    end

    Mirage.restore()

    selected_node = viz_state.selected_node
    if selected_node !== nothing && CImGui.BeginPopup("Node Details")
        try
            CImGui.TextUnformatted(selected_node.is_state ? "State" : "Action")
            CImGui.Separator()
            CImGui.TextUnformatted("N: $(node_visits(selected_node))")
            if !selected_node.is_state
                CImGui.TextUnformatted("Q: $(node_value_text(selected_node))")
            end
            if CImGui.Button("Expand Best Path From Node")
                expand_best_path!(selected_node)
                request_animation_frame(session, 30)
            end
            CImGui.Separator()

            CImGui.TextUnformatted("Fields")
            render_object_table(node_object(selected_node))
            CImGui.Separator()

            CImGui.TextUnformatted("Parents ($(length(selected_node.parents)))")
            if isempty(selected_node.parents)
                CImGui.TextUnformatted("None")
            else
                for parent in selected_node.parents
                    render_node_toggle_button("parent", parent)
                end
            end
            CImGui.Separator()

            CImGui.TextUnformatted("Children ($(length(selected_node.children)))")
            if isempty(selected_node.children)
                if CImGui.Button("Expand##selected_node")
                    toggle_node!(selected_node)
                    if viz_state.selected_node === nothing
                        CImGui.CloseCurrentPopup()
                    end
                end
            else
                for child in selected_node.children
                    render_node_toggle_button("child", child)
                end
            end

            if CImGui.CollapsingHeader("Internal")
                render_internal_table(selected_node)
            end
        finally
            CImGui.EndPopup()
        end
    end

    left_released && (viz_state.left_dragged = false)
    return nothing
end

function settings_window(session::MCTSVizSession)
    viz_state = session.state
    settings = viz_state.settings
    CImGui.Begin("Settings")
    draw_list = CImGui.GetWindowDrawList()
    title_pos = CImGui.GetCursorScreenPos()
    title_color = CImGui.GetColorU32(CImGui.ImVec4(0.92f0, 0.94f0, 0.96f0, 1.0f0))
    CImGui.AddText(draw_list, CImGui.GetFont(), CImGui.GetFontSize() * 1.65, title_pos, title_color, "MCTSViz.jl")
    CImGui.Dummy((0.0f0, CImGui.GetFontSize() * 2.0f0))
    CImGui.Separator()
    CImGui.Spacing()

    CImGui.SeparatorText("Navigation")
    if CImGui.Button("Center Root")
        viz_state.center_root = true
        request_animation_frame(session, 10)
    end
    CImGui.Spacing()

    CImGui.SeparatorText("Tree")
    if CImGui.Button("Expand All")
        viz_state.expand_all = true
        request_animation_frame(session, 30)
    end
    CImGui.SameLine()
    if CImGui.Button("Expand Best Path")
        viz_state.expand_best_path = true
        request_animation_frame(session, 30)
    end
    if CImGui.Button("Collapse All")
        viz_state.collapse_all = true
        request_animation_frame(session, 30)
    end
    CImGui.Spacing()

    CImGui.SeparatorText("Rendering")
    CImGui.Checkbox("Color code Q-values", settings.color_code_q_values)
    CImGui.Checkbox("Color code N-values", settings.color_code_n_values)
    CImGui.Checkbox("Show node text", settings.show_node_text)
    CImGui.Checkbox("Weight arrows by N", settings.show_weighted_arrows)
    CImGui.Spacing()

    CImGui.SeparatorText("Layout Physics")
    CImGui.PushItemWidth(-1)
    CImGui.TextUnformatted("Repulsion")
    if CImGui.SliderFloat("##repulsion", settings.repulsion_strength, 0.0f0, 3000.0f0)
        request_animation_frame(session, 30)
    end
    CImGui.TextUnformatted("Attraction")
    if CImGui.SliderFloat("##attraction", settings.attraction_strength, 0.0f0, 5.0f0)
        request_animation_frame(session, 30)
    end
    CImGui.PopItemWidth()
    CImGui.End()
end

function interpolate_rgb(t::Float64, c1::Tuple, c2::Tuple)::Tuple
    r = (1 - t) * c1[1] + t * c2[1]
    g = (1 - t) * c1[2] + t * c2[2]
    b = (1 - t) * c1[3] + t * c2[3]
    return (r, g, b)
end

function interpolate_palette(t::Float64, colors)::Tuple
    if length(colors) == 1
        return colors[1]
    end

    t_clamped = clamp(t, 0.0, 1.0)

    n = length(colors)
    scaled_t = t_clamped * (n - 1)
    idx = clamp(floor(Int, scaled_t), 0, n - 2)
    local_t = scaled_t - idx

    c1 = colors[idx + 1]
    c2 = colors[idx + 2]
    return interpolate_rgb(local_t, c1, c2)
end

include("./ascii.jl")
include("./example_mdp.jl")
include("./road_trip_mdp.jl")

function (@main)(args::Vector{String})::Cint
    example_mdp()
    return 0
end

export mcts_viz, mcts_ascii_viz, example_mdp, road_trip_example,
       CaliforniaRoadTripMDP, RoadTripState, RoadTripAction, DriveTo,
       VisitLandmark, SleepOvernight, FinishTrip

end # module MCTSViz
