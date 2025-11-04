module MCTSViz

using Revise
using GLFW
using ModernGL
using CImGui
using LinearAlgebra: normalize
import Mirage
using POMDPs
using MCTS
using POMDPTools

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

requested_animation_frames = 0
function request_animation_frame(frames::Int64 = 1)
    global requested_animation_frames = frames
end

@kwdef mutable struct TreeNode
    position::Vector{Float64} = [0.0, 0.0]
    velocity::Vector{Float64} = [0.0, 0.0]
    force::Vector{Float64} = [0.0, 0.0]
    text::String = ""
    is_state::Bool = true
    index::Int = 0
    parent::Union{TreeNode, Nothing} = nothing
    children::Vector{TreeNode} = []
    id::Int = 0
end

mutable struct Camera
    pan::Vector{Float64}
    panning::Bool
    zoom::Float64
end

function mcts_viz(mdp, mcts_policy; keep_state::Bool = true, expand_levels::Int = 3)
    mcts_tree = mcts_policy.tree

    if !GLFW.Init()
        @error "Failed to initialize GLFW"
        return
    end

    @static if Sys.isapple()
        glsl_version_str = "#version 150"
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 2)
        GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
        GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE)
    else
        glsl_version_str = "#version 130"
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 0)
    end

    window = GLFW.CreateWindow(1200, 800, "MCTSViz")
    @assert window.handle != C_NULL "Could not create a GLFW window 😢"
    GLFW.MakeContextCurrent(window)
    GLFW.SwapInterval(1) # Enable VSync

    Mirage.initialize_render_context()

    imgui_ctx = CImGui.CreateContext()
    io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_NavEnableKeyboard
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable

    try
        if !isdefined(CImGui, :ImGui_ImplGlfw_InitForOpenGL)
            error("ImGui_ImplGlfw_InitForOpenGL not found in CImGui namespace")
        end
        if !isdefined(CImGui, :ImGui_ImplOpenGL3_Init)
            error("ImGui_ImplOpenGL3_Init not found in CImGui namespace")
        end

        if !CImGui.ImGui_ImplGlfw_InitForOpenGL(window.handle, true)
            @error "Failed to initialize ImGui GLFW backend!"
            error("ImGui GLFW Init failed") # Throw to be caught below
        end
        if !CImGui.ImGui_ImplOpenGL3_Init(glsl_version_str)
            @error "Failed to initialize ImGui OpenGL3 backend!"
            error("ImGui OpenGL3 Init failed") # Throw to be caught below
        end
    catch e
        @error "Failed to initialize ImGui GLFW/OpenGL Implementation" exception=(e, catch_backtrace())
        CImGui.DestroyContext(imgui_ctx)
        GLFW.DestroyWindow(window)
        GLFW.Terminate()
        return
    end

    # Setup Dear ImGui style
    CImGui.StyleColorsDark()

    # Initialize camera with zoom
    camera = Camera([0.0, 0.0], false, 1.0)

    # Setup scroll callback for zooming
    scroll_callback = (win, xoffset, yoffset) -> begin
        io = CImGui.GetIO()
        # Only zoom if the mouse is not over an ImGui window
        if !unsafe_load(io.WantCaptureMouse)
            # Get canvas and mouse properties
            canvas_pos = get_state(:canvas_pos)
            canvas_size = get_state(:canvas_size)
            mx, my = GLFW.GetCursorPos(window)
            mouse_screen = [mx, my]

            # Helper to convert screen coordinates to world coordinates
            function screen_to_world(screen, pan, zoom, cpos, csize)
                # Mouse relative to canvas center
                relative_mouse = [screen[1] - cpos.x - csize.x/2, screen[2] - cpos.y - csize.y/2]
                # Account for pan and zoom
                world_pos = (relative_mouse .- pan) ./ zoom
                return world_pos
            end

            world_pos_before = screen_to_world(mouse_screen, camera.pan, camera.zoom, canvas_pos, canvas_size)

            # Update zoom
            camera.zoom *= (1.0 + yoffset * 0.1)
            camera.zoom = clamp(camera.zoom, 0.1, 10.0) # Clamp zoom level

            world_pos_after = screen_to_world(mouse_screen, camera.pan, camera.zoom, canvas_pos, canvas_size)

            # Adjust pan to keep the point under the mouse stationary
            pan_delta = world_pos_before - world_pos_after
            camera.pan .-= pan_delta .* camera.zoom

            request_animation_frame(1) # Request a frame render to show the change
        end
    end
    GLFW.SetScrollCallback(window, scroll_callback)

    dpi = begin
        monitor = GLFW.GetPrimaryMonitor()
        xscale, yscale = GLFW.GetMonitorContentScale(monitor)
        (xscale + yscale) / 2
    end

    # Special case, apple does have high dpi but handles it
    # automatically for us in practice
    if Sys.isapple()
        dpi = 1
    end

    fonts_dir = "fonts"
    fonts = unsafe_load(CImGui.GetIO().Fonts)
    try
        CImGui.AddFontFromFileTTF(fonts, joinpath(fonts_dir, "Roboto-Regular.ttf"), 18 * dpi)
        CImGui.ScaleAllSizes(CImGui.GetStyle(), dpi)
    catch _ end

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
    canvas::Mirage.Canvas = Mirage.create_canvas(100, 100)

    
    root_node = TreeNode(text = string(mcts_tree.s_labels[1]), index = 1, id = 1)
    node_id_counter = 1
    all_nodes = [root_node]

    last_frame_time = time()
    try # Wrap main loop in try/finally for cleanup
        while !GLFW.WindowShouldClose(window)
            current_frame_time = time()
            delta_time = min(1/30, current_frame_time - last_frame_time)
            last_frame_time = current_frame_time

            global requested_animation_frames
            if requested_animation_frames > 0
                GLFW.PollEvents()
                requested_animation_frames -= 1
            else
                GLFW.WaitEvents()
            end

            # Clear the main framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, GLFW.GetFramebufferSize(window)...)
            glClearColor(0.3, 0.3, 0.32, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

            # Start ImGui frame
            CImGui.ImGui_ImplOpenGL3_NewFrame()
            CImGui.ImGui_ImplGlfw_NewFrame()
            CImGui.NewFrame()

            # Set up the dockspace
            viewport = CImGui.GetMainViewport()
            #window_flags = CImGui.ImGuiWindowFlags_MenuBar | CImGui.ImGuiWindowFlags_NoDocking
            window_flags = CImGui.ImGuiWindowFlags_NoTitleBar | CImGui.ImGuiWindowFlags_NoCollapse
            window_flags |= CImGui.ImGuiWindowFlags_NoResize | CImGui.ImGuiWindowFlags_NoMove
            window_flags |= CImGui.ImGuiWindowFlags_NoBringToFrontOnFocus | CImGui.ImGuiWindowFlags_NoNavFocus
            window_flags |= CImGui.ImGuiWindowFlags_NoBackground

            # Position window over entire viewport
            CImGui.SetNextWindowPos(unsafe_load(viewport.Pos))
            CImGui.SetNextWindowSize(unsafe_load(viewport.Size))
            CImGui.SetNextWindowViewport(unsafe_load(viewport.ID))

            # Remove window padding/border
            CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowRounding, 0.0f0)
            CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowBorderSize, 0.0f0)
            CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowPadding, (0.0f0, 0.0f0))

            # Begin the dockspace window
            CImGui.Begin("DockSpace", C_NULL, window_flags)
            CImGui.PopStyleVar(3)

            # Create the actual dockspace
            dockspace_id = CImGui.GetID("MyDockSpace")
            dockspace_flags = CImGui.ImGuiDockNodeFlags_PassthruCentralNode
            dockspace_flags |= CImGui.ImGuiDockNodeFlags_AutoHideTabBar
            CImGui.DockSpace(dockspace_id, (0.0f0, 0.0f0), dockspace_flags)

            settings_window()

            #CImGui.Text("Test: $test")

            node_id_counter = main_view(canvas, window, mcts_tree, root_node, all_nodes, camera, delta_time, node_id_counter, expand_levels)
            #mcts_tree_visual_window(mcts_canvas, window)
            #pomcpow_tree_visual_window(mcts_canvas, window)

            CImGui.End()
            CImGui.Render()
            CImGui.ImGui_ImplOpenGL3_RenderDrawData(CImGui.GetDrawData())

            # Handle multi-viewports / docking if enabled
            if unsafe_load(io.ConfigFlags) & CImGui.ImGuiConfigFlags_ViewportsEnable == CImGui.ImGuiConfigFlags_ViewportsEnable
                backup_current_context = GLFW.GetCurrentContext()
                CImGui.UpdatePlatformWindows()
                CImGui.RenderPlatformWindowsDefault()
                GLFW.MakeContextCurrent(backup_current_context)
            end

            # Buffer animation frames for mouse events. This is a
            # stupid hack, but it works pretty well
            if CImGui.IsMouseClicked(0) || CImGui.IsMouseClicked(1)
                request_animation_frame(10)
            end
            if CImGui.IsMouseReleased(0) || CImGui.IsMouseReleased(1)
                request_animation_frame(10)
            end

            set_state(:first_boot_setup, false)
            set_state(:first_frame, false)
            GLFW.SwapBuffers(window)
            yield()
        end
    catch e
        @error "Error in main loop!" exception=(e, catch_backtrace())
    finally
        println("Cleaning up...")

        Mirage.cleanup_render_context()
        Mirage.destroy!(canvas)
        #Mirage.destroy!(mcts_canvas)

        try
            # Shutdown ImGui platform/renderer backends
            CImGui.ImGui_ImplOpenGL3_Shutdown()
            CImGui.ImGui_ImplGlfw_Shutdown()
        catch e
            @error "Error during ImGui backend shutdown" exception=(e, catch_backtrace())
        end
        CImGui.DestroyContext(imgui_ctx)
        GLFW.DestroyWindow(window)
        GLFW.Terminate()
        println("Cleanup finished.")
    end
end

function main_view(canvas, window, mcts_tree, root_node, all_nodes, camera, delta_time, node_id_counter, expand_levels)
    q_values = values(mcts_tree.q)
    min_q_value = minimum(q_values)
    max_q_value = maximum(q_values)
    max_abs_q = isempty(q_values) ? 0.0 : maximum(abs.(q_values))

    n_values = values(mcts_tree.total_n)
    min_n_value = minimum(n_values)
    max_n_value = maximum(n_values)

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

        # Find the index of the most likely state in the tree\'s s_labels
        idx = findfirst(isequal(best_s), mcts_tree.s_labels)
        if idx !== nothing
            return [idx]
        else
            return Int[]
        end
    end

    function expand_node(node, levels)
        if levels <= 0
            return
        end

        next_position(i, i_max) = [cos(i / i_max) * 40.0, sin(i / i_max) * 40.0] + (
            node.position[1] == 0.0 && node.position[2] == 0.0
            ? [0.0, 0.0]
            : node.position + normalize(node.position) * 20
        )

        if isempty(node.children)
            if node.is_state
                actions = get_actions_from_state_index(node.index)
                for (a_idx, action) in enumerate(actions)
                    node_id_counter += 1
                    new_node = TreeNode(
                        text = string(mcts_tree.a_labels[action]),
                        is_state = false,
                        index = action,
                        parent = node,
                        position = next_position(a_idx, length(actions)),
                        id = node_id_counter
                    )
                    push!(node.children, new_node)
                    push!(all_nodes, new_node)
                    expand_node(new_node, levels - 1)
                end
            else # is action node
                states = find_next_states(node.parent.index, node.index)
                for state in states
                    node_id_counter += 1
                    new_node = TreeNode(
                        text = string(mcts_tree.s_labels[state]),
                        index = state,
                        parent = node,
                        position = next_position(0, 1),
                        id = node_id_counter
                    )
                    push!(node.children, new_node)
                    push!(all_nodes, new_node)
                    expand_node(new_node, levels - 1)
                end
            end
        end
    end

    if get_state(:first_frame)
        expand_node(root_node, expand_levels)
    end

    # Camera panning
    canvas_pos = CImGui.GetItemRectMin()
    canvas_size = CImGui.GetItemRectSize()
    set_state(:canvas_pos, canvas_pos)
    set_state(:canvas_size, canvas_size)
    mx, my = GLFW.GetCursorPos(window)
    io = CImGui.GetIO()
    is_hovering_canvas = mx >= canvas_pos.x && mx <= canvas_pos.x + canvas_size.x && my >= canvas_pos.y && my <= canvas_pos.y + canvas_size.y && !unsafe_load(io.WantCaptureMouse)

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
        repulsion_strength = 600.0
        attraction_strength = 0.9
        damping = 0.85

        for node in nodes
            node.force = [0.0, 0.0]
        end

        # Repulsion
        for i in 1:length(nodes)
            for j in (i+1):length(nodes)
                node_a = nodes[i]
                node_b = nodes[j]
                delta_pos = node_a.position - node_b.position
                distance_sq = sum(delta_pos.^2)
                if distance_sq > 1.0 # Avoid extreme forces at very close distances
                    force_magnitude = repulsion_strength * (get_state(:desired_distance)[]^2) / distance_sq
                    #force_magnitude = repulsion_strength / distance_sq
                    force_vec = force_magnitude * normalize(delta_pos)
                    node_a.force += force_vec
                    node_b.force -= force_vec
                else
                    node_a.force += [1, 0]
                    node_b.force -= [1, 0]
                end
            end
        end

        # Attraction
        for node in nodes
            if node.parent !== nothing
                delta_pos = node.parent.position - node.position
                force = attraction_strength * delta_pos
                node.force += force
                node.parent.force -= force
            end
        end

        # Update positions
        for node in nodes
            if node.id == 1 # Fix root node
                node.position = [0.0, 0.0]
                node.velocity = [0.0, 0.0]
                node.force = [0.0, 0.0]
                continue
            end

            node.velocity += node.force * delta_time
            node.velocity *= damping
            #node.velocity[2] = 0
            node.position += node.velocity * delta_time
        end
    end

    for i in 1:6
        update_physics(all_nodes, 3 / 60)
    end
    request_animation_frame(1)

    # Rendering
    Mirage.resize!(canvas, max(1, Int(trunc(canvas_size.x))), max(1, Int(trunc(canvas_size.y))))
    Mirage.set_canvas(canvas)
    Mirage.save()
    Mirage.fillcolor(Mirage.rgba(0, 0, 20, 255))
    Mirage.fillrect(0, 0, canvas.width, canvas.height)
    Mirage.restore()
    Mirage.save()
    Mirage.update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)
    Mirage.translate(canvas.width / 2 + camera.pan[1], canvas.height / 2 + camera.pan[2])
    Mirage.scale(camera.zoom, camera.zoom)

    # Draw connections
    function draw_connections(node)
        for child in node.children
            Mirage.strokecolor(Mirage.rgba(255, 255, 255, 50))
            Mirage.strokewidth(1.5)
            Mirage.moveto(node.position...)
            Mirage.lineto(child.position...)
            Mirage.stroke()
            draw_connections(child)
        end
    end
    Mirage.save()
    draw_connections(root_node)
    Mirage.restore()

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
                intensity = 0.0
                if max_n_value > 0
                    intensity = (n_val - min_n_value) / (max_n_value - min_n_value)
                end

                rainbow = reverse([
                    #(  0/255,  92/255, 230/255),  # blue
                    (  0/255, 174/255, 239/255),  # cyan
                    #(  0/255, 191/255, 165/255),  # teal green
                    ( 68/255, 206/255,  27/255),  # green
                    (187/255, 219/255,  68/255),  # lime
                    (247/255, 227/255, 121/255),  # yellow
                    (242/255, 161/255,  52/255),  # orange
                    (255/255,  69/255,   0/255),  # red-orange
                ])
                color = interpolate_palette(intensity, map(t -> (Float32(t[1]), Float32(t[2]), Float32(t[3])), rainbow))
                Mirage.fillcolor((color[1], color[2], color[3], 255))
            else
                Mirage.fillcolor((0, 0, 80, 255))
            end
        else
            if get_state(:color_code_q_values)[]
                q_val = mcts_tree.q[node.index]
                intensity = 0.0
                if max_q_value > 0
                    intensity = (q_val - min_q_value) / (max_q_value - min_q_value)
                end
                #@info (;max_q_value, min_q_value, q_val, intensity)

                color = Mirage.rgba(100, 100, 0, 255)
                rainbow = reverse([
                    #(  0/255,  92/255, 230/255),  # blue
                    (  0/255, 174/255, 239/255),  # cyan
                    #(  0/255, 191/255, 165/255),  # teal green
                    ( 68/255, 206/255,  27/255),  # green
                    (187/255, 219/255,  68/255),  # lime
                    (247/255, 227/255, 121/255),  # yellow
                    (242/255, 161/255,  52/255),  # orange
                    (255/255,  69/255,   0/255),  # red-orange
                ])
                color = interpolate_palette(intensity, map(t -> (Float32(t[1]), Float32(t[2]), Float32(t[3])), rainbow))
                Mirage.fillcolor((color[1], color[2], color[3], 255))
            else
                Mirage.fillcolor((100, 100, 0, 255))
            end
        end

        next_position(i, i_max) = [cos(i / i_max) * 40.0, sin(i / i_max) * 40.0] + (
            node.position[1] == 0.0 && node.position[2] == 0.0
            ? [0.0, 0.0]
            : node.position + normalize(node.position) * 20
        )

        if is_hovered && CImGui.IsMouseClicked(0) #&& !camera.panning
            if isempty(node.children)
                if node.is_state
                    actions = get_actions_from_state_index(node.index)
                    for (a_idx, action) in enumerate(actions)
                        node_id_counter += 1
                        new_node = TreeNode(
                            text = string(mcts_tree.a_labels[action]),
                            is_state = false,
                            index = action,
                            parent = node,
                            position = next_position(a_idx, length(actions)),
                            id = node_id_counter
                        )
                        push!(node.children, new_node)
                        push!(all_nodes, new_node)
                    end
                else # is action node
                    states = find_next_states(node.parent.index, node.index)
                    for state in states
                        node_id_counter += 1
                        new_node = TreeNode(
                            text = string(mcts_tree.s_labels[state]),
                            index = state,
                            parent = node,
                            position = next_position(0, 1),
                            id = node_id_counter
                        )
                        push!(node.children, new_node)
                        push!(all_nodes, new_node)
                    end
                end
            else
                function delete_children_recursive(n)
                    for child in n.children
                        delete_children_recursive(child)
                        filter!(x -> x.id != child.id, all_nodes)
                    end
                    empty!(n.children)
                end
                delete_children_recursive(node)
            end
            request_animation_frame(10)
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
                    text_to_render = "$(node.text)\nN: $(visits)"
                else
                    a_idx = node.index
                    action = mcts_tree.a_labels[a_idx]
                    visits = mcts_tree.n[a_idx]
                    v_val = round(mcts_tree.q[a_idx], digits=3)
                    text_to_render = "a: $(action)\nN: $visits, Q: $v_val"
                end

                Mirage.fillcolor(Mirage.rgba(255, 255, 255, 255))
                Mirage.scale(1 / camera.zoom)
                
                # Estimate text size and center it
                font_size = 16
                lines = split(text_to_render, '\n')
                max_width = 0
                for line in lines
                    max_width = max(max_width, length(line))
                end
                text_width = max_width * font_size / 2
                text_height = length(lines) * font_size
                
                if is_hovered
                    Mirage.translate(world_mouse_pos[1] - node.position[1] - text_width / 2, world_mouse_pos[2] - node.position[2] - text_height - 10)
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
    Mirage.set_canvas()

    # Draw canvas to ImGui window
    draw_list = CImGui.GetWindowDrawList()
    CImGui.AddImage(
        draw_list,
        CImGui.ImTextureRef(UInt64(canvas.texture[])),
        CImGui.ImVec2(canvas_pos.x, canvas_pos.y),
        CImGui.ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        CImGui.ImVec2(0, 1),
        CImGui.ImVec2(1, 0)
    )

    return node_id_counter
end

function settings_window()
    CImGui.Begin("Settings")
    CImGui.Checkbox("Color code Q-values", get_state(:color_code_q_values))
    CImGui.Checkbox("Color code N-values", get_state(:color_code_n_values))
    CImGui.Checkbox("Show node text", get_state(:show_node_text))
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

include("./example_mdp.jl")

export mcts_viz, example_mdp

end # module MCTSViz
