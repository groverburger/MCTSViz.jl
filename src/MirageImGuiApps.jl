module MirageImGuiApps

using GLFW
using ModernGL
using CImGui
import Mirage

export MirageImGuiApp,
    CanvasViewport,
    run!,
    begin_frame!,
    end_frame!,
    destroy!,
    request_frame!,
    stop!,
    get_canvas!,
    resize_canvas!,
    destroy_canvas!,
    draw_canvas!,
    draw_canvas_image!,
    begin_dockspace!,
    end_dockspace!,
    set_scroll_callback!,
    set_key_callback!,
    set_mouse_button_callback!,
    bundled_font_path

mutable struct MirageImGuiApp
    window::GLFW.Window
    imgui_ctx::Ptr{Cvoid}
    canvases::Dict{Symbol, Mirage.Canvas}
    dpi::Float64
    delta_time::Float64
    requested_frames::Int
    running::Bool
    docking::Bool
    clear_color::NTuple{4, Float32}
    callbacks::Vector{Any}
end

struct CanvasViewport
    pos::CImGui.ImVec2
    size::CImGui.ImVec2
    hovered::Bool
    focused::Bool
    active::Bool
    clicked::Bool
    mouse_pos::Tuple{Float64, Float64}
    mouse_rel::Tuple{Float64, Float64}
end

function glsl_version_and_hints!()
    @static if Sys.isapple()
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 2)
        GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
        GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE)
        return "#version 150"
    else
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
        GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 0)
        return "#version 130"
    end
end

function monitor_dpi()
    dpi = 1.0
    try
        monitor = GLFW.GetPrimaryMonitor()
        xscale, yscale = GLFW.GetMonitorContentScale(monitor)
        dpi = (xscale + yscale) / 2
    catch
        dpi = 1.0
    end
    return Sys.isapple() ? 1.0 : dpi
end

function bundled_font_path()
    return normpath(joinpath(@__DIR__, "..", "assets", "fonts", "Roboto-Regular.ttf"))
end

function MirageImGuiApp(
    title::AbstractString;
    width::Integer = 1200,
    height::Integer = 800,
    docking::Bool = true,
    vsync::Bool = true,
    alpha_bits::Integer = 8,
    scale_to_monitor::Bool = true,
    clear_color::NTuple{4, Real} = (0.3, 0.3, 0.32, 1.0),
    font_path::Union{Nothing, AbstractString} = bundled_font_path(),
    font_size::Real = 18,
    scale_style::Bool = true,
    configure_imgui!::Function = app -> nothing,
)
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end

    glsl_version_str = glsl_version_and_hints!()
    GLFW.WindowHint(GLFW.ALPHA_BITS, alpha_bits)
    if scale_to_monitor
        GLFW.WindowHint(0x0002200C, 1) # GLFW_SCALE_TO_MONITOR
    end

    window = GLFW.CreateWindow(width, height, String(title))
    if window.handle == C_NULL
        GLFW.Terminate()
        error("Could not create a GLFW window")
    end

    GLFW.MakeContextCurrent(window)
    GLFW.SwapInterval(vsync ? 1 : 0)

    mirage_initialized = false
    imgui_ctx = C_NULL
    try
        Mirage.initialize_render_context()
        mirage_initialized = true

        imgui_ctx = CImGui.CreateContext()
        io = CImGui.GetIO()
        io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_NavEnableKeyboard
        if docking
            io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable
        end

        if !isdefined(CImGui, :ImGui_ImplGlfw_InitForOpenGL)
            error("ImGui_ImplGlfw_InitForOpenGL not found in CImGui namespace")
        end
        if !isdefined(CImGui, :ImGui_ImplOpenGL3_Init)
            error("ImGui_ImplOpenGL3_Init not found in CImGui namespace")
        end
        if !CImGui.ImGui_ImplGlfw_InitForOpenGL(window.handle, true)
            error("ImGui GLFW backend initialization failed")
        end
        if !CImGui.ImGui_ImplOpenGL3_Init(glsl_version_str)
            error("ImGui OpenGL3 backend initialization failed")
        end

        CImGui.StyleColorsDark()
    catch
        try
            if imgui_ctx != C_NULL
                CImGui.DestroyContext(imgui_ctx)
            end
        catch
        end
        if mirage_initialized
            try
                Mirage.cleanup_render_context()
            catch
            end
        end
        GLFW.DestroyWindow(window)
        GLFW.Terminate()
        rethrow()
    end

    app = MirageImGuiApp(
        window,
        imgui_ctx,
        Dict{Symbol, Mirage.Canvas}(),
        monitor_dpi(),
        0.0,
        0,
        true,
        docking,
        Float32.(clear_color),
        Any[],
    )
    if font_path !== nothing
        CImGui.AddFontFromFileTTF(unsafe_load(CImGui.GetIO().Fonts), String(font_path), font_size * app.dpi)
    end
    if scale_style
        CImGui.ScaleAllSizes(CImGui.GetStyle(), app.dpi)
    end
    configure_imgui!(app)
    return app
end

function request_frame!(app::MirageImGuiApp, frames::Integer = 1)
    app.requested_frames = max(app.requested_frames, Int(frames))
    return app.requested_frames
end

function stop!(app::MirageImGuiApp)
    app.running = false
    GLFW.SetWindowShouldClose(app.window, true)
    return nothing
end

function get_canvas!(app::MirageImGuiApp, key::Symbol = :main; width::Integer = 100, height::Integer = 100)
    return get!(app.canvases, key) do
        Mirage.create_canvas(width, height)
    end
end

function resize_canvas!(canvas::Mirage.Canvas, size::CImGui.ImVec2)
    width = max(1, Int(trunc(size.x)))
    height = max(1, Int(trunc(size.y)))
    Mirage.resize!(canvas, width, height)
    return canvas
end

function destroy_canvas!(app::MirageImGuiApp, key::Symbol)
    if haskey(app.canvases, key)
        Mirage.destroy!(app.canvases[key])
        delete!(app.canvases, key)
    end
    return nothing
end

function draw_canvas_image!(canvas::Mirage.Canvas, pos::CImGui.ImVec2, size::CImGui.ImVec2)
    draw_list = CImGui.GetWindowDrawList()
    CImGui.AddImage(
        draw_list,
        CImGui.ImTextureRef(UInt64(canvas.texture[])),
        CImGui.ImVec2(pos.x, pos.y),
        CImGui.ImVec2(pos.x + size.x, pos.y + size.y),
        CImGui.ImVec2(0, 1),
        CImGui.ImVec2(1, 0),
    )
    return nothing
end

function draw_canvas!(
    render!::Function,
    app::MirageImGuiApp,
    key::Symbol = :main;
    size = nothing,
    label::AbstractString = "##mirage_canvas_$(key)",
    reset_context::Bool = true,
    clear::Bool = true,
    clear_color::NTuple{4, Real} = (0, 0, 0, 0),
)
    canvas = get_canvas!(app, key)
    requested_size = size === nothing ? CImGui.GetContentRegionAvail() : size
    viewport_pos = CImGui.GetCursorScreenPos()

    CImGui.InvisibleButton(String(label), requested_size)
    item_pos = CImGui.GetItemRectMin()
    item_size = CImGui.GetItemRectSize()
    hovered = CImGui.IsItemHovered()
    focused = CImGui.IsItemFocused()
    active = CImGui.IsItemActive()
    clicked = CImGui.IsItemClicked()
    cursor_pos = GLFW.GetCursorPos(app.window)
    mouse_pos = (cursor_pos.x, cursor_pos.y)
    mouse_rel = (mouse_pos[1] - item_pos.x, mouse_pos[2] - item_pos.y)

    resize_canvas!(canvas, item_size)
    viewport = CanvasViewport(item_pos, item_size, hovered, focused, active, clicked, mouse_pos, mouse_rel)

    Mirage.set_canvas(canvas)
    try
        if reset_context
            Mirage.get_context().context_stack = [Mirage.ContextState()]
        end
        if clear
            glViewport(0, 0, canvas.width, canvas.height)
            glClearColor(Float32.(clear_color)...)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        end
        render!(canvas, viewport)
    finally
        Mirage.set_canvas()
    end

    draw_canvas_image!(canvas, item_pos, item_size)
    CImGui.SetCursorScreenPos(CImGui.ImVec2(viewport_pos.x, viewport_pos.y + item_size.y))
    return viewport
end

function begin_dockspace!(app::MirageImGuiApp; id::AbstractString = "MainDockSpace", menu_bar::Bool = false)
    viewport = CImGui.GetMainViewport()
    window_flags = CImGui.ImGuiWindowFlags_NoTitleBar | CImGui.ImGuiWindowFlags_NoCollapse
    if menu_bar
        window_flags |= CImGui.ImGuiWindowFlags_MenuBar
    end
    window_flags |= CImGui.ImGuiWindowFlags_NoResize | CImGui.ImGuiWindowFlags_NoMove
    window_flags |= CImGui.ImGuiWindowFlags_NoBringToFrontOnFocus | CImGui.ImGuiWindowFlags_NoNavFocus
    window_flags |= CImGui.ImGuiWindowFlags_NoBackground

    CImGui.SetNextWindowPos(unsafe_load(viewport.Pos))
    CImGui.SetNextWindowSize(unsafe_load(viewport.Size))
    CImGui.SetNextWindowViewport(unsafe_load(viewport.ID))
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowRounding, 0.0f0)
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowBorderSize, 0.0f0)
    CImGui.PushStyleVar(CImGui.ImGuiStyleVar_WindowPadding, (0.0f0, 0.0f0))
    CImGui.Begin("DockSpace", C_NULL, window_flags)
    CImGui.PopStyleVar(3)

    dockspace_flags = CImGui.ImGuiDockNodeFlags_PassthruCentralNode
    dockspace_flags |= CImGui.ImGuiDockNodeFlags_AutoHideTabBar
    CImGui.DockSpace(CImGui.GetID(String(id)), (0.0f0, 0.0f0), dockspace_flags)
    return nothing
end

function end_dockspace!(::MirageImGuiApp)
    CImGui.End()
    return nothing
end

function begin_frame!(
    app::MirageImGuiApp;
    animate::Bool = false,
    idle_timeout::Union{Nothing, Real} = nothing,
)
    if app.requested_frames > 0 || animate
        GLFW.PollEvents()
        app.requested_frames = max(app.requested_frames - 1, 0)
    elseif idle_timeout === nothing
        GLFW.WaitEvents()
    else
        GLFW.PollEvents()
        sleep(Float64(idle_timeout))
    end

    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, GLFW.GetFramebufferSize(app.window)...)
    glClearColor(app.clear_color...)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

    CImGui.ImGui_ImplOpenGL3_NewFrame()
    CImGui.ImGui_ImplGlfw_NewFrame()
    CImGui.NewFrame()
    return nothing
end

function end_frame!(app::MirageImGuiApp)
    CImGui.Render()
    CImGui.ImGui_ImplOpenGL3_RenderDrawData(CImGui.GetDrawData())

    io = CImGui.GetIO()
    if unsafe_load(io.ConfigFlags) & CImGui.ImGuiConfigFlags_ViewportsEnable == CImGui.ImGuiConfigFlags_ViewportsEnable
        backup_current_context = GLFW.GetCurrentContext()
        CImGui.UpdatePlatformWindows()
        CImGui.RenderPlatformWindowsDefault()
        GLFW.MakeContextCurrent(backup_current_context)
    end

    GLFW.SwapBuffers(app.window)
    yield()
    return nothing
end

function run!(
    frame!::Function,
    app::MirageImGuiApp;
    animate_when::Function = app -> false,
    before_frame!::Function = app -> nothing,
    idle_timeout::Union{Nothing, Real} = nothing,
    menu_bar::Bool = false,
    cleanup!::Function = app -> nothing,
)
    last_frame_time = time()
    try
        while app.running && !GLFW.WindowShouldClose(app.window)
            current_frame_time = time()
            app.delta_time = min(1 / 30, current_frame_time - last_frame_time)
            last_frame_time = current_frame_time

            Base.invokelatest(before_frame!, app)
            animate = animate_when(app)
            begin_frame!(app; animate, idle_timeout)
            if app.docking
                begin_dockspace!(app; menu_bar)
                try
                    Base.invokelatest(frame!, app)
                finally
                    end_dockspace!(app)
                end
            else
                Base.invokelatest(frame!, app)
            end
            end_frame!(app)
        end
    finally
        cleanup!(app)
        destroy!(app)
    end
    return nothing
end

function set_scroll_callback!(callback::Function, app::MirageImGuiApp)
    push!(app.callbacks, callback)
    GLFW.SetScrollCallback(app.window, callback)
    return callback
end

function set_key_callback!(callback::Function, app::MirageImGuiApp)
    push!(app.callbacks, callback)
    GLFW.SetKeyCallback(app.window, callback)
    return callback
end

function set_mouse_button_callback!(callback::Function, app::MirageImGuiApp)
    push!(app.callbacks, callback)
    GLFW.SetMouseButtonCallback(app.window, callback)
    return callback
end

function destroy!(app::MirageImGuiApp)
    app.running = false

    for canvas in values(app.canvases)
        try
            Mirage.destroy!(canvas)
        catch e
            @error "Error while destroying Mirage canvas" exception=(e, catch_backtrace())
        end
    end
    empty!(app.canvases)

    try
        Mirage.cleanup_render_context()
    catch e
        @error "Error during Mirage cleanup" exception=(e, catch_backtrace())
    end

    try
        CImGui.ImGui_ImplOpenGL3_Shutdown()
        CImGui.ImGui_ImplGlfw_Shutdown()
    catch e
        @error "Error during ImGui backend shutdown" exception=(e, catch_backtrace())
    end

    try
        CImGui.DestroyContext(app.imgui_ctx)
    catch e
        @error "Error while destroying ImGui context" exception=(e, catch_backtrace())
    end

    try
        GLFW.DestroyWindow(app.window)
        GLFW.Terminate()
    catch e
        @error "Error during GLFW cleanup" exception=(e, catch_backtrace())
    end

    return nothing
end

end
