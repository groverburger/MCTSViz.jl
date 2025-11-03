module MCTSViz

using Revise
using GLFW
using ModernGL
using CImGui
import Mirage

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
    position::Vector{Number} = [0.0, 0.0]
    text::String = ""
    is_state::Bool = true
    index::Int = 0
    parent_index::Int = 0
end

function main(mcts_tree::MCTS.MCTSTree; keep_state::Bool = true)
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

    set_state(:first_frame, true)
    canvas::Mirage.Canvas = Mirage.create_canvas(100, 100)

    tree_nodes::Vector{TreeNode} = [TreeNode(text = string(mcts_tree.s_labels[1]), index = 1)]
    state_node_map::Dict{Int64, TreeNode} = Dict()
    action_node_map::Dict{Int64, TreeNode} = Dict()

    last_frame_time = time()
    try # Wrap main loop in try/finally for cleanup
        while !GLFW.WindowShouldClose(window)
            current_frame_time = time()
            delta_time = current_frame_time - last_frame_time
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
            window_flags = CImGui.ImGuiWindowFlags_MenuBar | CImGui.ImGuiWindowFlags_NoDocking
            window_flags |= CImGui.ImGuiWindowFlags_NoTitleBar | CImGui.ImGuiWindowFlags_NoCollapse
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

            #CImGui.Text("Test: $test")

            main_view(canvas, window, mcts_tree, tree_nodes)
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

function main_view(canvas, window, mcts_tree, tree_nodes)
    #@info mcts_policy.tree.s_labels
    function get_actions_from_state_index(state_index::Int64)
        #return (;
            #name = mcts_policy.tree.s_labels[state_index],
            #actions = mcts_policy.tree.child_ids[state_index],
        #)
        return mcts_tree.child_ids[state_index]
    end

    function get_actions_from_state(state::GridWorldState)
        return get_actions_from_state_index(mcts_tree.state_map[state])
    end

    function get_state_from_action(action_index::Int64)
        return findfirst(x -> x[1] == 1, mcts_tree._vis_stats)[2]
    end

    if CImGui.IsMouseClicked(0) || CImGui.IsMouseClicked(1)
        request_animation_frame(10)
    end

    viewport = CImGui.GetMainViewport()
    size = unsafe_load(viewport.Size)

    canvas_pos = CImGui.GetItemRectMin()
    canvas_size = CImGui.GetItemRectSize()
    width = canvas_size.x
    height = canvas_size.y

    mx, my = GLFW.GetCursorPos(window)
    mx -= canvas_pos.x
    my -= canvas_pos.y

    Mirage.resize!(
        canvas,
        max(1, Int64(width)), max(1, Int64(height))
    )

    Mirage.set_canvas(canvas)
    Mirage.save()
    Mirage.update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)

    Mirage.translate(width / 2, height / 2)

    for node in tree_nodes
        Mirage.save()
        if node.is_state
            Mirage.fillcolor(Mirage.rgba(0, 0, 80, 255))
        else
            Mirage.fillcolor(Mirage.rgba(155, 155, 0, 255))
        end
        Mirage.translate(node.position...)
        if hypot(node.position[1] + width / 2 - mx, node.position[2] + height / 2 - my) <= 32
            Mirage.fillcolor(Mirage.rgba(0, 0, 180, 255))
            if CImGui.IsMouseClicked(0)
                if node.is_state
                    for action in get_actions_from_state_index(node.index)
                        push!(tree_nodes, TreeNode(
                            text = string(mcts_tree.a_labels[action]),
                            is_state = false,
                            index = action,
                            position = [node.position[1] + 10, node.position[2] + 10]
                        ))
                    end
                else
                    state_index = get_state_from_action(node.index)
                    push!(tree_nodes, TreeNode(
                        text = string(mcts_tree.s_labels[state_index]),
                        index = state_index,
                        position = [node.position[1] + 10, node.position[2] + 10]
                    ))
                end
            end
        end
        Mirage.circle(32)
        Mirage.fill()
        Mirage.fillcolor(Mirage.rgba(255, 255, 255, 255))
        Mirage.text(node.text)
        Mirage.restore()
    end

    Mirage.restore()
    Mirage.set_canvas()

    # Get the window draw list
    draw_list = CImGui.GetWindowDrawList()

    # Use AddImage with the correct syntax
    CImGui.AddImage(
        draw_list,
        CImGui.ImTextureRef(UInt64(canvas.texture[])),
        CImGui.ImVec2(canvas_pos.x, canvas_pos.y),
        CImGui.ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        CImGui.ImVec2(0, 1), # UV coordinates (flipped vertically for OpenGL)
        CImGui.ImVec2(1, 0)
    )
end

using POMDPs
using MCTS
using POMDPTools

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

function test()
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

    #@info findfirst(x -> x == s, values(mcts_policy.tree.s_labels))
    #@info mcts_policy.tree.s_labels
    #@info rand(initialstate(mdp))

    main(mcts_policy.tree)

    #@info mcts_policy.tree.s_labels[1]
    #@info mcts_policy.tree.child_ids[1]
    #@info findfirst(x -> x[1] == 1, mcts_policy.tree._vis_stats)

    #@info get_actions_from_state(s)[1] |> get_state_from_action |> get_actions_from_state_index
end

export main

end # module MCTSViz
