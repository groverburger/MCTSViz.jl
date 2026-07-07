module MCTSViz

using Revise
using GLFW
using CImGui
using LinearAlgebra: normalize
import Mirage
using POMDPs
using MCTS
using POMDPTools

include("MirageImGuiApps.jl")
using .MirageImGuiApps

global application_state::Dict{Symbol, Any} = Dict()
function initialize_application_state()
    global application_state = Dict(
        :mdp => nothing,
        :unfolded_states => Dict(),
        :unfolded_actions => Dict(),
        :first_boot_setup => !isfile("imgui.ini"),
        :first_frame => true,
        :mcts_ascii_tree => "",
        :mcts_visual_tree => nothing,
        :mcts_exploration => Ref{Float32}(1.0f0),
        :mcts_tree => nothing,
        :mcts_plan_result => nothing,
        :desired_distance => 32,
        :canvas_pos => CImGui.ImVec2(0,0),
        :canvas_size => CImGui.ImVec2(0,0),
        :color_code_q_values => Ref(true),
        :color_code_n_values => Ref(false),
        :show_node_text => Ref(true),
        :show_weighted_arrows => Ref(false),
        :physics_repulsion_strength => Ref{Float32}(600.0f0),
        :physics_attraction_strength => Ref{Float32}(0.9f0),
        :center_root_requested => false,
        :expand_all_requested => false,
        :expand_best_path_requested => false,
        :collapse_all_requested => false,
        :selected_node => nothing,
        :show_selected_node_parents => Ref(false),
    )
    return application_state
end

function get_state(key::Symbol)
    global application_state
    @assert haskey(application_state, key) "State key $key does not exist!"
    return application_state[key]
end

function set_state(key::Symbol, value::Any)
    global application_state
    @assert haskey(application_state, key) "State key $key does not exist!"
    if application_state[key] isa Ref
        application_state[key][] = value
    else
        application_state[key] = value
    end
    return application_state[key]
end

const current_app = Ref{Union{Nothing, MirageImGuiApp}}(nothing)
const live_reload_last_error = Ref{Union{Nothing, String}}(nothing)
const live_reload_last_check_time = Ref(0.0)
const live_reload_interval_seconds = Ref(0.1)
requested_animation_frames = 0
function request_animation_frame(frames::Int64 = 1)
    global requested_animation_frames = frames
    if current_app[] !== nothing
        request_frame!(current_app[], frames)
    end
end

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

function get_neighbors(grid::SpatialHashGrid, node::TreeNode)
    neighbors = TreeNode[]
    center_coords = get_cell_coords(grid, node.position)
    for i in -1:1
        for j in -1:1
            neighbor_coords = (center_coords[1] + i, center_coords[2] + j)
            if haskey(grid.cells, neighbor_coords)
                for neighbor_node in grid.cells[neighbor_coords]
                    if neighbor_node.id != node.id
                        push!(neighbors, neighbor_node)
                    end
                end
            end
        end
    end
    return neighbors
end

mutable struct Camera
    pan::Vector{Float64}
    panning::Bool
    zoom::Float64
end

const VISUALIZATION_PALETTE = map(t -> (Float32(t[1]), Float32(t[2]), Float32(t[3])), reverse([
    (  0/255, 174/255, 239/255),  # cyan
    ( 68/255, 206/255,  27/255),  # green
    (187/255, 219/255,  68/255),  # lime
    (247/255, 227/255, 121/255),  # yellow
    (242/255, 161/255,  52/255),  # orange
    (255/255,  69/255,   0/255),  # red-orange
]))

mutable struct MCTSVizSession
    app::MirageImGuiApp
    mcts_tree::Any
    root_node::TreeNode
    node_id_counter::Int
    all_nodes::Vector{TreeNode}
    camera::Camera
    expand_levels::Int
end

function live_reload_frame!(::MirageImGuiApp)
    now = time()
    if now - live_reload_last_check_time[] < live_reload_interval_seconds[]
        return nothing
    end
    live_reload_last_check_time[] = now

    try
        Revise.revise()
        live_reload_last_error[] = nothing
    catch e
        msg = sprint(showerror, e)
        if msg != live_reload_last_error[]
            @warn "Revise failed; fix the error and the live GUI will retry." exception=(e, catch_backtrace())
            live_reload_last_error[] = msg
        end
    end
    return nothing
end

function mcts_viz_frame!(session::MCTSVizSession)
    app = session.app
    settings_window()
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowPadding, (0.0f0, 0.0f0))
    CImGui.Begin("Tree View")
    CImGui.PopStyleVar()
    try
        draw_canvas!(app, :mcts_tree; label = "mcts_tree_canvas") do canvas, viewport
            session.node_id_counter = main_view(
                canvas,
                app.window,
                viewport,
                session.mcts_tree,
                session.root_node,
                session.all_nodes,
                session.camera,
                app.delta_time,
                session.node_id_counter,
                session.expand_levels,
            )
        end
    finally
        CImGui.End()
    end
    if CImGui.IsMouseClicked(0) || CImGui.IsMouseClicked(1)
        request_animation_frame(10)
    end
    if CImGui.IsMouseReleased(0) || CImGui.IsMouseReleased(1)
        request_animation_frame(10)
    end

    set_state(:first_boot_setup, false)
    set_state(:first_frame, false)
    return nothing
end

function mcts_viz(
    mdp,
    mcts_policy;
    keep_state::Bool = true,
    expand_levels::Int = 3,
    live_reload::Bool = true,
    live_reload_interval::Real = 0.1,
)
    mcts_tree = mcts_policy.tree
    app = MirageImGuiApp("MCTSViz"; width = 1200, height = 800)
    current_app[] = app
    camera = Camera([0.0, 0.0], false, 1.0)

    global application_state
    prev_application_state = application_state
    initialize_application_state()
    application_state = (
        keep_state
        ? merge(application_state, prev_application_state)
        : application_state
    )
    set_state(:mdp, mdp)
    set_state(:first_frame, true)
    
    root_node = TreeNode(text = string(mcts_tree.s_labels[1]), index = 1, id = 1)
    all_nodes = [root_node]
    session = MCTSVizSession(app, mcts_tree, root_node, 1, all_nodes, camera, expand_levels)
    live_reload_interval_seconds[] = Float64(live_reload_interval)
    request_frame!(app, 10)

    try
        run!(
            app;
            before_frame! = live_reload ? live_reload_frame! : (app -> nothing),
            idle_timeout = live_reload ? live_reload_interval : nothing,
        ) do app
            Base.invokelatest(mcts_viz_frame!, session)
        end
    catch e
        @error "Error in main loop!" exception=(e, catch_backtrace())
    finally
        current_app[] = nothing
    end
end

function main_view(canvas, window, canvas_viewport, mcts_tree, root_node, all_nodes, camera, delta_time, node_id_counter, expand_levels)
    state_node_map = Dict{Int, TreeNode}(map(n -> n.index => n, filter(n -> n.is_state, all_nodes)))
    n_palette = VISUALIZATION_PALETTE
    q_palette = n_palette

    min_log_q_value = Inf
    max_log_q_value = -Inf
    min_n_value = Inf
    max_n_value = -Inf
    min_arrow_n_value = Inf
    max_arrow_n_value = -Inf
    max_edge_visits = 0

    for node in all_nodes
        if node.is_state
            n_val = mcts_tree.total_n[node.index]
            min_n_value = min(min_n_value, n_val)
            max_n_value = max(max_n_value, n_val)
            min_arrow_n_value = min(min_arrow_n_value, n_val)
            max_arrow_n_value = max(max_arrow_n_value, n_val)
        else
            q_val = signed_log_scale(mcts_tree.q[node.index])
            n_val = mcts_tree.n[node.index]
            min_log_q_value = min(min_log_q_value, q_val)
            max_log_q_value = max(max_log_q_value, q_val)
            min_arrow_n_value = min(min_arrow_n_value, n_val)
            max_arrow_n_value = max(max_arrow_n_value, n_val)
        end

        for child in node.children
            child_visits = child.is_state ? mcts_tree.total_n[child.index] : mcts_tree.n[child.index]
            max_edge_visits = max(max_edge_visits, child_visits)
        end
    end

    if min_log_q_value == Inf
        min_log_q_value = 0.0
        max_log_q_value = 0.0
    end
    if min_n_value == Inf
        min_n_value = 0.0
        max_n_value = 0.0
    end
    if min_arrow_n_value == Inf
        min_arrow_n_value = 0.0
        max_arrow_n_value = 0.0
    end

    function normalize_range(value, min_value, max_value)
        return max_value == min_value ? 0.5 : clamp((value - min_value) / (max_value - min_value), 0.0, 1.0)
    end

    function rgba_from_palette(intensity, palette; alpha=255)
        color = interpolate_palette(intensity, palette)
        return (color[1], color[2], color[3], alpha)
    end

    function node_label(node::TreeNode)
        if node.is_state
            return string(mcts_tree.s_labels[node.index])
        else
            return string(mcts_tree.a_labels[node.index])
        end
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

    function edge_visits(parent::TreeNode, child::TreeNode)
        return child.is_state ? mcts_tree.total_n[child.index] : mcts_tree.n[child.index]
    end

    function n_color(node::TreeNode; alpha=255)
        intensity = normalize_range(node_visits(node), min_arrow_n_value, max_arrow_n_value)
        return rgba_from_palette(intensity, n_palette; alpha)
    end

    function edge_width(parent::TreeNode, child::TreeNode)
        if !get_state(:show_weighted_arrows)[] || max_edge_visits <= 0
            return 1.5
        end
        intensity = log1p(max(0, edge_visits(parent, child))) / log1p(max_edge_visits)
        return 0.75 + 4.25 * intensity
    end

    # Helper functions
    function get_actions_from_state_index(state_index::Int64)
        return (1 <= state_index <= length(mcts_tree.child_ids)) ? mcts_tree.child_ids[state_index] : Int[]
    end

    function get_state_from_action(action_index::Int64)
        return findfirst(x -> x[1] == 1, mcts_tree._vis_stats)[2]
    end

    function find_next_states(state_id::Any, action_id::Any)
        mdp = get_state(:mdp)
        # If no mdp, fallback to the original buggy implementation
        if mdp === nothing
            next_states = Any[]
            if !isempty(mcts_tree._vis_stats)
                for ((said, sid), count) in mcts_tree._vis_stats
                    if said == action_id
                        push!(next_states, sid)
                    end
                end
            end
            return unique(next_states)
        end

        state = mcts_tree.s_labels[state_id]
        action = mcts_tree.a_labels[action_id]

        # Get the distribution of next states
        dist = POMDPs.transition(mdp, state, action)

        # If there are no next states, return empty
        if isempty(support(dist))
            return Int[]
        end

        # Find the most likely next state
        max_p = -1.0
        best_s = nothing
        for s in support(dist)
            p = pdf(dist, s) # Use pdf() to get probability of a state
            if p > max_p
                max_p = p
                best_s = s
            end
        end

        if best_s === nothing
            return Int[]
        end

        # Find the index of the most likely state in the tree's s_labels
        idx = findfirst(isequal(best_s), mcts_tree.s_labels)
        if idx !== nothing
            return [idx]
        else
            return Int[]
        end
    end

    next_position(node, i, i_max) = [cos(i / i_max) * 40.0, sin(i / i_max) * 40.0] + (
        node.position[1] == 0.0 && node.position[2] == 0.0
        ? [0.0, 0.0]
        : node.position + normalize(node.position) * 20
    )

    function expand_one_level!(node)
        if !isempty(node.children)
            return false
        end

        if node.is_state
            actions = get_actions_from_state_index(node.index)
            for (a_idx, action) in enumerate(actions)
                node_id_counter += 1
                new_node = TreeNode(
                    text = string(mcts_tree.a_labels[action]),
                    is_state = false,
                    index = action,
                    parents = [node],
                    position = next_position(node, a_idx, length(actions)),
                    id = node_id_counter
                )
                push!(node.children, new_node)
                push!(all_nodes, new_node)
            end
        else
            @assert !isempty(node.parents) "Action node should have at least one parent"
            states = find_next_states(node.parents[1].index, node.index)
            for state in states
                if haskey(state_node_map, state)
                    new_node = state_node_map[state]
                    if !(node in new_node.parents)
                        push!(new_node.parents, node)
                    end
                else
                    node_id_counter += 1
                    new_node = TreeNode(
                        text = string(mcts_tree.s_labels[state]),
                        index = state,
                        parents = [node],
                        position = next_position(node, 0, 1),
                        id = node_id_counter
                    )
                    state_node_map[state] = new_node
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
        set_state(:selected_node, nothing)
        set_state(:show_selected_node_parents, false)
    end

    if get_state(:first_frame)
        expand_node(root_node, expand_levels)
    end

    if get_state(:collapse_all_requested)
        collapse_all!()
        set_state(:collapse_all_requested, false)
        request_animation_frame(30)
    end

    if get_state(:expand_all_requested)
        expand_all!(root_node)
        set_state(:expand_all_requested, false)
        request_animation_frame(30)
    end

    if get_state(:expand_best_path_requested)
        expand_best_path!(root_node)
        set_state(:expand_best_path_requested, false)
        request_animation_frame(30)
    end

    # Camera panning
    canvas_pos = canvas_viewport.pos
    canvas_size = canvas_viewport.size
    set_state(:canvas_pos, canvas_pos)
    set_state(:canvas_size, canvas_size)
    mx, my = GLFW.GetCursorPos(window)
    is_hovering_canvas = canvas_viewport.hovered

    function screen_to_world(screen, pan, zoom, cpos, csize)
        relative_mouse = [screen[1] - cpos.x - csize.x/2, screen[2] - cpos.y - csize.y/2]
        return (relative_mouse .- pan) ./ zoom
    end

    if get_state(:center_root_requested)
        camera.pan .= -root_node.position .* camera.zoom
        set_state(:center_root_requested, false)
        request_animation_frame(10)
    end

    wheel_delta = unsafe_load(CImGui.GetIO().MouseWheel)
    if is_hovering_canvas && wheel_delta != 0
        mouse_screen = [mx, my]
        world_pos_before = screen_to_world(mouse_screen, camera.pan, camera.zoom, canvas_pos, canvas_size)
        camera.zoom *= (1.0 + wheel_delta * 0.1)
        camera.zoom = clamp(camera.zoom, 0.1, 10.0)
        world_pos_after = screen_to_world(mouse_screen, camera.pan, camera.zoom, canvas_pos, canvas_size)
        pan_delta = world_pos_before - world_pos_after
        camera.pan .-= pan_delta .* camera.zoom
        request_animation_frame(10)
    end

    if is_hovering_canvas && CImGui.IsMouseDown(0) # Right mouse button for panning
        if !camera.panning
            camera.panning = true
        end
        mouse_delta = CImGui.GetIO().MouseDelta
        camera.pan .+= [unsafe_load(mouse_delta.x), unsafe_load(mouse_delta.y)]
        request_animation_frame(10)
    else
        camera.panning = false
    end

    # Physics simulation
    function update_physics(nodes, delta_time)
        repulsion_strength = Float64(get_state(:physics_repulsion_strength)[])
        attraction_strength = Float64(get_state(:physics_attraction_strength)[])
        desired_distance = Float64(get_state(:desired_distance)[])
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
        request_animation_frame(1)
    end

    # Rendering
    Mirage.save()
    Mirage.update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)
    Mirage.fillcolor(Mirage.rgba(0, 0, 20, 255))
    Mirage.fillrect(0, 0, canvas.width, canvas.height)
    Mirage.restore()
    Mirage.save()
    Mirage.update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)
    Mirage.translate(canvas.width / 2 + camera.pan[1], canvas.height / 2 + camera.pan[2])
    Mirage.scale(camera.zoom, camera.zoom)

    # Draw connections
    function draw_arrow(p1, p2, arrowhead_length, arrowhead_angle, node_radius, stroke_width, start_color, end_color, gradient)
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
        p2_adjusted_x = p2[1] - dir_x * node_radius
        p2_adjusted_y = p2[2] - dir_y * node_radius
        
        # Draw the main line
        Mirage.strokewidth(stroke_width)
        if gradient
            segment_count = 12
            for i in 1:segment_count
                t1 = (i - 1) / segment_count
                t2 = i / segment_count
                c = interpolate_rgba(t1, start_color, end_color)
                segment_start_x = p1[1] * (1 - t1) + p2_adjusted_x * t1
                segment_start_y = p1[2] * (1 - t1) + p2_adjusted_y * t1
                segment_end_x = p1[1] * (1 - t2) + p2_adjusted_x * t2
                segment_end_y = p1[2] * (1 - t2) + p2_adjusted_y * t2
                Mirage.strokecolor(c)
                Mirage.moveto(segment_start_x, segment_start_y)
                Mirage.lineto(segment_end_x, segment_end_y)
                Mirage.stroke()
            end
        else
            Mirage.strokecolor(start_color)
            Mirage.moveto(p1[1], p1[2])
            Mirage.lineto(p2_adjusted_x, p2_adjusted_y)
            Mirage.stroke()
        end

        # Calculate arrowhead points
        p3_x = p2_adjusted_x - arrowhead_length * dir_x
        p3_y = p2_adjusted_y - arrowhead_length * dir_y
        
        # Rotate direction vector for arrowhead lines
        # Rotate by +arrowhead_angle
        cos_angle = cos(arrowhead_angle)
        sin_angle = sin(arrowhead_angle)
        arrow_dx = p3_x - p2_adjusted_x
        arrow_dy = p3_y - p2_adjusted_y
        arrow_p1_x = cos_angle * arrow_dx - sin_angle * arrow_dy + p2_adjusted_x
        arrow_p1_y = sin_angle * arrow_dx + cos_angle * arrow_dy + p2_adjusted_y
        
        # Rotate by -arrowhead_angle
        arrow_p2_x = cos_angle * arrow_dx + sin_angle * arrow_dy + p2_adjusted_x
        arrow_p2_y = -sin_angle * arrow_dx + cos_angle * arrow_dy + p2_adjusted_y

        # Draw arrowhead
        Mirage.strokecolor(end_color)
        Mirage.moveto(p2_adjusted_x, p2_adjusted_y)
        Mirage.lineto(arrow_p1_x, arrow_p1_y)
        Mirage.stroke()
        
        Mirage.moveto(p2_adjusted_x, p2_adjusted_y)
        Mirage.lineto(arrow_p2_x, arrow_p2_y)
        Mirage.stroke()
    end

    function draw_connections(node, visited=Set())
        if node in visited
            return
        end
        push!(visited, node)
        color_by_n = get_state(:color_code_n_values)[]
        for child in node.children
            start_color = color_by_n ? n_color(node; alpha=90) : Mirage.rgba(255, 255, 255, 50)
            end_color = color_by_n ? n_color(child; alpha=90) : Mirage.rgba(255, 255, 255, 50)
            draw_arrow(node.position, child.position, 10.0, pi/6, 24.0, edge_width(node, child), start_color, end_color, color_by_n)
            draw_connections(child, visited)
        end
    end
    Mirage.save()
    draw_connections(root_node)
    Mirage.restore()

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

    # Draw nodes and handle clicks
    for node in copy(all_nodes)
        Mirage.save()
        
        # Correctly calculate world mouse position considering pan and zoom
        relative_mouse = [mx - canvas_pos.x - canvas.width/2, my - canvas_pos.y - canvas.height/2]
        world_mouse_pos = (relative_mouse .- camera.pan) ./ camera.zoom
        
        # Make hover radius constant in screen space by scaling it in world space
        is_hovered = (
            is_hovering_canvas &&
            hypot(node.position[1] - world_mouse_pos[1], node.position[2] - world_mouse_pos[2]) <= 24
        )

        if node.is_state
            if get_state(:color_code_n_values)[]
                n_val = mcts_tree.total_n[node.index]
                intensity = normalize_range(n_val, min_n_value, max_n_value)
                Mirage.fillcolor(rgba_from_palette(intensity, n_palette))
            else
                Mirage.fillcolor(Mirage.rgba(0, 0, 200, 255))
            end
        else
            if get_state(:color_code_q_values)[]
                q_val = mcts_tree.q[node.index]
                intensity = normalize_range(signed_log_scale(q_val), min_log_q_value, max_log_q_value)
                #@info (;max_q_value, min_q_value, q_val, intensity)

                Mirage.fillcolor(rgba_from_palette(intensity, q_palette))
            else
                Mirage.fillcolor(Mirage.rgba(150, 150, 0, 255))
            end
        end

        if is_hovered && CImGui.IsMouseClicked(0) #&& !camera.panning
            if isempty(node.children)
                expand_one_level!(node)
            else
                # Sever connections from the clicked node to its children
                children_to_process = copy(node.children)
                empty!(node.children)

                for child in children_to_process
                    filter!(p -> p.id != node.id, child.parents)
                end

                # Garbage collect nodes that are no longer reachable
                delete_unreachable_nodes(root_node, all_nodes, state_node_map)
            end
            request_animation_frame(10)
        end

        if is_hovered && CImGui.IsMouseClicked(1)
            set_state(:selected_node, node)
            set_state(:show_selected_node_parents, false)
            CImGui.OpenPopup("Node Details")
            request_animation_frame(10)
        end
        
        Mirage.translate(node.position...)

        if node.is_state
            # Draw circle for state nodes
            Mirage.circle(24)
            Mirage.fill()
            if node === root_node
                Mirage.save()
                Mirage.scale(1 / camera.zoom)
                Mirage.translate(-6, 8)
                Mirage.save()
                Mirage.translate(1, 1)
                Mirage.fillcolor(Mirage.rgba(0, 0, 0, 255))
                Mirage.text("R")
                Mirage.restore()
                Mirage.fillcolor(Mirage.rgba(255, 255, 255, 255))
                Mirage.text("R")
                Mirage.restore()
            end
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
            if is_hovered
                Mirage.save()
                Mirage.strokecolor(Mirage.rgba(255, 255, 255, 255))
                Mirage.strokewidth(3)
                Mirage.moveto(0, node_size)
                Mirage.lineto(node_size, 0)
                Mirage.lineto(0, -node_size)
                Mirage.lineto(-node_size, 0)
                Mirage.closepath()
                Mirage.stroke()
                Mirage.restore()
            end
        end

        let
            should_render_text = get_state(:show_node_text)[] || is_hovered
            if camera.zoom < 0.5 && !is_hovered
                should_render_text = false
            end

            if should_render_text
                local text_to_render = node.text
                if node.is_state
                    s_idx = node.index
                    state = mcts_tree.s_labels[s_idx]
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

    selected_node = get_state(:selected_node)
    if selected_node !== nothing && CImGui.BeginPopup("Node Details")
        try
            CImGui.TextUnformatted(selected_node.is_state ? "State node" : "Action node")
            CImGui.Separator()
            CImGui.TextUnformatted("Label: $(node_label(selected_node))")
            CImGui.TextUnformatted("Index: $(selected_node.index)")
            CImGui.TextUnformatted("Node id: $(selected_node.id)")
            CImGui.TextUnformatted("Visits N: $(node_visits(selected_node))")
            if !selected_node.is_state
                CImGui.TextUnformatted("Q: $(node_value_text(selected_node))")
            end
            CImGui.TextUnformatted("Position: ($(round(selected_node.position[1], digits=2)), $(round(selected_node.position[2], digits=2)))")
            CImGui.TextUnformatted("Parents: $(length(selected_node.parents))")
            CImGui.TextUnformatted("Children: $(length(selected_node.children))")
            CImGui.Separator()
            if CImGui.Button(get_state(:show_selected_node_parents)[] ? "Hide Parents" : "Show All Parents")
                set_state(:show_selected_node_parents, !get_state(:show_selected_node_parents)[])
                request_animation_frame(10)
            end
            if get_state(:show_selected_node_parents)[]
                if isempty(selected_node.parents)
                    CImGui.TextUnformatted("No parents")
                else
                    for (i, parent) in enumerate(selected_node.parents)
                        CImGui.TextUnformatted("$(i). $(parent.is_state ? "State" : "Action") #$(parent.id): $(node_label(parent))")
                    end
                end
            end
        finally
            CImGui.EndPopup()
        end
    end

    return node_id_counter
end

function settings_window()
    CImGui.Begin("Settings")
    if CImGui.Button("Center Root")
        set_state(:center_root_requested, true)
        request_animation_frame(10)
    end
    if CImGui.Button("Expand All")
        set_state(:expand_all_requested, true)
        request_animation_frame(30)
    end
    if CImGui.Button("Expand Best Path")
        set_state(:expand_best_path_requested, true)
        request_animation_frame(30)
    end
    if CImGui.Button("Collapse All")
        set_state(:collapse_all_requested, true)
        request_animation_frame(30)
    end
    CImGui.Checkbox("Color code Q-values", get_state(:color_code_q_values))
    CImGui.Checkbox("Color code N-values", get_state(:color_code_n_values))
    CImGui.Checkbox("Show node text", get_state(:show_node_text))
    CImGui.Checkbox("Weight arrows by N", get_state(:show_weighted_arrows))
    if CImGui.SliderFloat("Repulsion", get_state(:physics_repulsion_strength), 0.0f0, 3000.0f0)
        request_animation_frame(30)
    end
    if CImGui.SliderFloat("Attraction", get_state(:physics_attraction_strength), 0.0f0, 5.0f0)
        request_animation_frame(30)
    end
    CImGui.End()
end

function signed_log_scale(value::Real)
    return sign(value) * log1p(abs(value))
end

function interpolate_rgb(t::Float64, c1::Tuple, c2::Tuple)::Tuple
    r = (1 - t) * c1[1] + t * c2[1]
    g = (1 - t) * c1[2] + t * c2[2]
    b = (1 - t) * c1[3] + t * c2[3]
    return (r, g, b)
end

function interpolate_rgba(t::Real, c1::Tuple, c2::Tuple)::Tuple
    t_clamped = clamp(Float64(t), 0.0, 1.0)
    r = (1 - t_clamped) * c1[1] + t_clamped * c2[1]
    g = (1 - t_clamped) * c1[2] + t_clamped * c2[2]
    b = (1 - t_clamped) * c1[3] + t_clamped * c2[3]
    a = (1 - t_clamped) * c1[4] + t_clamped * c2[4]
    return (r, g, b, a)
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

function mcts_ascii_viz(tree::MCTS.MCTSTree, root_id::Any = 1; 
                        max_depth::Int = 5, show_stats::Bool = true)
    lines = String[]
    visited_nodes = Set{Any}()

    function find_next_states(tree::MCTS.MCTSTree, action_id::Any)
        next_states = Any[]

        # Use visualization statistics if available
        if !isempty(tree._vis_stats)
            for ((said, sid), count) in tree._vis_stats
                if said == action_id
                    push!(next_states, sid)
                end
            end
        end

        return unique(next_states)
    end

    function format_state_node(state_id::Any, show_stats::Bool)
        state = tree.s_labels[state_id]
        if show_stats
            visits = tree.total_n[state_id]
            return "$(state) (N:$visits)"
        else
            return string(state)
        end
    end

    function format_action_node(action_id::Any, show_stats::Bool)
        action = tree.a_labels[action_id]
        if show_stats
            visits = tree.n[action_id]
            q_val = round(tree.q[action_id], digits=3)
            return "$(action) (N:$visits, Q:$q_val)"
        else
            return string(action)
        end
    end

    function traverse_tree(state_id::Any, prefix::String, is_last::Bool,
                          current_depth::Int, max_depth::Int)
        # Prevent infinite loops and respect depth limit
        if state_id in visited_nodes || current_depth > max_depth
            return
        end

        push!(visited_nodes, state_id)

        # Current node connector
        connector = is_last ? "+-- " : "+-- "
        node_label = format_state_node(state_id, show_stats)
        push!(lines, prefix * connector * node_label)

        # Prepare prefix for children
        child_prefix = prefix * (is_last ? "    " : "|   ")

        # Get action children
        action_children = tree.child_ids[state_id]

        for (i, action_id) in enumerate(action_children)
            is_last_action = (i == length(action_children))

            # Draw action node
            action_connector = is_last_action ? "+-- " : "+-- "
            action_label = format_action_node(action_id, show_stats)
            push!(lines, child_prefix * action_connector * action_label)

            # Prepare prefix for state children of this action
            action_child_prefix = child_prefix * (is_last_action ? "    " : "|   ")

            # Find next states from this action (using transition data if available)
            next_states = find_next_states(tree, action_id)

            for (j, next_state_id) in enumerate(next_states)
                is_last_state = (j == length(next_states))
                traverse_tree(next_state_id, action_child_prefix, is_last_state,
                            current_depth + 1, max_depth)
            end
        end
    end

    # Start with root node label
    root_label = format_state_node(root_id, show_stats)
    push!(lines, root_label)

    # Begin traversal
    traverse_tree(root_id, "", true, 0, max_depth)

    return join(lines, "\n")
end

include("./example_mdp.jl")

function (@main)(args::Vector{String})::Cint
    example_mdp()
    return 0
end

export mcts_viz, mcts_ascii_viz, example_mdp, MirageImGuiApps

end # module MCTSViz
