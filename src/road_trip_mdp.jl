const MINUTES_PER_HOUR = 60
const ROAD_TRIP_WEEK = 7 * 24 * MINUTES_PER_HOUR
const MAX_AWAKE_TIME = 16 * MINUTES_PER_HOUR

struct RoadTripLandmark
    name::String
    city::Symbol
    duration::Int
    cost::Float64
end

struct RoadTripRoute
    origin::Symbol
    destination::Symbol
    duration::Int
    cost::Float64
end

struct RoadTripState
    city::Symbol
    elapsed::Int
    awake::Int
    visited::UInt16
    finished::Bool
end

Base.:(==)(a::RoadTripState, b::RoadTripState) =
    a.city == b.city && a.elapsed == b.elapsed && a.awake == b.awake &&
    a.visited == b.visited && a.finished == b.finished
Base.isequal(a::RoadTripState, b::RoadTripState) = a == b
Base.hash(s::RoadTripState, h::UInt) =
    hash(s.finished, hash(s.visited, hash(s.awake, hash(s.elapsed, hash(s.city, h)))))

abstract type RoadTripAction end

struct DriveTo <: RoadTripAction
    city::Symbol
end

struct VisitLandmark <: RoadTripAction
    landmark::Symbol
end

struct SleepOvernight <: RoadTripAction end
struct FinishTrip <: RoadTripAction end

struct CaliforniaRoadTripMDP <: MDP{RoadTripState, RoadTripAction}
    start_city::Symbol
    hotel_costs::Dict{Symbol, Float64}
    landmarks::Dict{Symbol, RoadTripLandmark}
    landmark_order::Vector{Symbol}
    routes::Dict{Tuple{Symbol, Symbol}, RoadTripRoute}
    landmark_reward::Float64
    cost_weight::Float64
end

function add_route!(routes, a::Symbol, b::Symbol, hours::Real, cost::Real)
    duration = round(Int, hours * MINUTES_PER_HOUR)
    routes[a, b] = RoadTripRoute(a, b, duration, Float64(cost))
    routes[b, a] = RoadTripRoute(b, a, duration, Float64(cost))
    return routes
end

"""
    CaliforniaRoadTripMDP(; landmark_reward = 100.0, cost_weight = 0.25)

A one-week California road-trip problem. Each new landmark is worth
`landmark_reward`; fuel, admission, parking, and lodging are penalized by
`cost_weight` per dollar. Travelers may be awake for at most 16 hours before
sleeping for eight hours.
"""
function CaliforniaRoadTripMDP(; landmark_reward::Real = 100.0, cost_weight::Real = 0.25)
    hotels = Dict(
        :san_francisco => 180.0,
        :monterey => 145.0,
        :yosemite => 165.0,
        :lake_tahoe => 150.0,
        :los_angeles => 170.0,
        :san_diego => 140.0,
    )

    landmark_specs = [
        (:golden_gate, "Golden Gate Bridge", :san_francisco, 2.0, 0.0),
        (:alcatraz, "Alcatraz", :san_francisco, 4.0, 45.0),
        (:monterey_aquarium, "Monterey Bay Aquarium", :monterey, 4.0, 65.0),
        (:big_sur, "Big Sur", :monterey, 5.0, 10.0),
        (:yosemite_valley, "Yosemite Valley", :yosemite, 7.0, 35.0),
        (:glacier_point, "Glacier Point", :yosemite, 4.0, 0.0),
        (:emerald_bay, "Emerald Bay", :lake_tahoe, 4.0, 10.0),
        (:getty_center, "Getty Center", :los_angeles, 4.0, 25.0),
        (:griffith_observatory, "Griffith Observatory", :los_angeles, 3.0, 10.0),
        (:san_diego_zoo, "San Diego Zoo", :san_diego, 7.0, 74.0),
        (:balboa_park, "Balboa Park", :san_diego, 4.0, 0.0),
    ]
    landmarks = Dict{Symbol, RoadTripLandmark}()
    landmark_order = Symbol[]
    for (id, name, city, hours, cost) in landmark_specs
        landmarks[id] = RoadTripLandmark(name, city, round(Int, hours * MINUTES_PER_HOUR), cost)
        push!(landmark_order, id)
    end

    routes = Dict{Tuple{Symbol, Symbol}, RoadTripRoute}()
    add_route!(routes, :san_francisco, :monterey, 2.0, 35.0)
    add_route!(routes, :san_francisco, :yosemite, 4.0, 65.0)
    add_route!(routes, :san_francisco, :lake_tahoe, 3.5, 60.0)
    add_route!(routes, :monterey, :yosemite, 4.0, 60.0)
    add_route!(routes, :monterey, :los_angeles, 5.5, 90.0)
    add_route!(routes, :yosemite, :lake_tahoe, 3.5, 55.0)
    add_route!(routes, :yosemite, :los_angeles, 5.0, 80.0)
    add_route!(routes, :los_angeles, :san_diego, 2.5, 40.0)

    return CaliforniaRoadTripMDP(
        :san_francisco, hotels, landmarks, landmark_order, routes,
        Float64(landmark_reward), Float64(cost_weight),
    )
end

function landmark_bit(mdp::CaliforniaRoadTripMDP, id::Symbol)
    index = findfirst(==(id), mdp.landmark_order)
    isnothing(index) && throw(KeyError(id))
    return UInt16(1) << (index - 1)
end

has_visited(mdp::CaliforniaRoadTripMDP, state::RoadTripState, id::Symbol) =
    !iszero(state.visited & landmark_bit(mdp, id))

function fits_schedule(state::RoadTripState, duration::Integer)
    return state.elapsed + duration <= ROAD_TRIP_WEEK &&
           state.awake + duration <= MAX_AWAKE_TIME
end

POMDPs.initialstate(mdp::CaliforniaRoadTripMDP) =
    Deterministic(RoadTripState(mdp.start_city, 0, 0, 0x0000, false))

POMDPs.isterminal(::CaliforniaRoadTripMDP, state::RoadTripState) =
    state.finished || state.elapsed >= ROAD_TRIP_WEEK

POMDPs.discount(::CaliforniaRoadTripMDP) = 1.0

function POMDPs.actions(mdp::CaliforniaRoadTripMDP, state::RoadTripState)
    isterminal(mdp, state) && return RoadTripAction[]
    available = RoadTripAction[]

    destinations = sort!([destination for ((origin, destination), _) in mdp.routes
                          if origin == state.city]; by = string)
    for destination in destinations
        route = mdp.routes[state.city, destination]
        fits_schedule(state, route.duration) && push!(available, DriveTo(destination))
    end

    for id in mdp.landmark_order
        landmark = mdp.landmarks[id]
        if landmark.city == state.city && !has_visited(mdp, state, id) &&
           fits_schedule(state, landmark.duration)
            push!(available, VisitLandmark(id))
        end
    end

    state.elapsed + 8 * MINUTES_PER_HOUR <= ROAD_TRIP_WEEK &&
        push!(available, SleepOvernight())
    push!(available, FinishTrip())
    return available
end

function POMDPs.transition(mdp::CaliforniaRoadTripMDP, state::RoadTripState, action::DriveTo)
    route = mdp.routes[state.city, action.city]
    fits_schedule(state, route.duration) || throw(ArgumentError("drive does not fit the schedule"))
    return Deterministic(RoadTripState(
        action.city, state.elapsed + route.duration, state.awake + route.duration,
        state.visited, false,
    ))
end

function POMDPs.transition(mdp::CaliforniaRoadTripMDP, state::RoadTripState, action::VisitLandmark)
    landmark = mdp.landmarks[action.landmark]
    landmark.city == state.city || throw(ArgumentError("landmark is in another city"))
    has_visited(mdp, state, action.landmark) && throw(ArgumentError("landmark was already visited"))
    fits_schedule(state, landmark.duration) || throw(ArgumentError("visit does not fit the schedule"))
    return Deterministic(RoadTripState(
        state.city, state.elapsed + landmark.duration, state.awake + landmark.duration,
        state.visited | landmark_bit(mdp, action.landmark), false,
    ))
end

function POMDPs.transition(::CaliforniaRoadTripMDP, state::RoadTripState, ::SleepOvernight)
    return Deterministic(RoadTripState(
        state.city, state.elapsed + 8 * MINUTES_PER_HOUR, 0, state.visited, false,
    ))
end

POMDPs.transition(::CaliforniaRoadTripMDP, state::RoadTripState, ::FinishTrip) =
    Deterministic(RoadTripState(state.city, state.elapsed, state.awake, state.visited, true))

function POMDPs.reward(mdp::CaliforniaRoadTripMDP, state::RoadTripState, action::DriveTo)
    return -mdp.cost_weight * mdp.routes[state.city, action.city].cost
end

function POMDPs.reward(mdp::CaliforniaRoadTripMDP, state::RoadTripState, action::VisitLandmark)
    landmark = mdp.landmarks[action.landmark]
    return (has_visited(mdp, state, action.landmark) ? 0.0 : mdp.landmark_reward) -
           mdp.cost_weight * landmark.cost
end

POMDPs.reward(mdp::CaliforniaRoadTripMDP, state::RoadTripState, ::SleepOvernight) =
    -mdp.cost_weight * mdp.hotel_costs[state.city]
POMDPs.reward(::CaliforniaRoadTripMDP, ::RoadTripState, ::FinishTrip) = 0.0

POMDPs.reward(mdp::CaliforniaRoadTripMDP, state::RoadTripState,
              action::RoadTripAction, ::RoadTripState) = reward(mdp, state, action)

function Base.show(io::IO, state::RoadTripState)
    day = state.elapsed ÷ (24 * MINUTES_PER_HOUR) + 1
    hour = (state.elapsed % (24 * MINUTES_PER_HOUR)) / MINUTES_PER_HOUR
    visited = count_ones(state.visited)
    print(io, titlecase(replace(string(state.city), '_' => ' ')),
          " | day ", min(day, 7), ", ", round(hour; digits = 1), "h | ",
          visited, " visited", state.finished ? " | finished" : "")
end

Base.show(io::IO, action::DriveTo) =
    print(io, "Drive to ", titlecase(replace(string(action.city), '_' => ' ')))
Base.show(io::IO, action::VisitLandmark) =
    print(io, "Visit ", titlecase(replace(string(action.landmark), '_' => ' ')))
Base.show(io::IO, ::SleepOvernight) = print(io, "Sleep overnight")
Base.show(io::IO, ::FinishTrip) = print(io, "Finish trip")

struct RoadTripRolloutPolicy <: Policy
    mdp::CaliforniaRoadTripMDP
end

function nearest_unvisited_city(mdp::CaliforniaRoadTripMDP, state::RoadTripState)
    targets = Set(
        landmark.city for (id, landmark) in mdp.landmarks
        if !has_visited(mdp, state, id)
    )
    isempty(targets) && return nothing

    distances = Dict(city => Inf for city in keys(mdp.hotel_costs))
    first_hops = Dict{Symbol, Symbol}()
    distances[state.city] = 0.0
    remaining = Set(keys(mdp.hotel_costs))

    while !isempty(remaining)
        city = argmin(candidate -> distances[candidate], remaining)
        delete!(remaining, city)
        isinf(distances[city]) && break
        city in targets && return get(first_hops, city, city)

        for ((origin, destination), route) in mdp.routes
            origin == city || continue
            candidate_distance = distances[city] + route.duration
            if candidate_distance < distances[destination]
                distances[destination] = candidate_distance
                first_hops[destination] = city == state.city ? destination : first_hops[city]
            end
        end
    end
    return nothing
end

function POMDPs.action(policy::RoadTripRolloutPolicy, state::RoadTripState)
    mdp = policy.mdp
    available = actions(mdp, state)
    isempty(available) && return FinishTrip()

    visits = filter(action -> action isa VisitLandmark, available)
    if !isempty(visits)
        return argmin(action -> mdp.landmarks[action.landmark].duration, visits)
    end

    local_landmark_waiting = any(
        landmark.city == state.city && !has_visited(mdp, state, id)
        for (id, landmark) in mdp.landmarks
    )
    sleep = findfirst(action -> action isa SleepOvernight, available)
    if local_landmark_waiting && !isnothing(sleep)
        return available[sleep]
    end

    destination = nearest_unvisited_city(mdp, state)
    if !isnothing(destination) && destination != state.city
        drive = findfirst(
            action -> action isa DriveTo && action.city == destination,
            available,
        )
        !isnothing(drive) && return available[drive]
        !isnothing(sleep) && return available[sleep]
    end

    return FinishTrip()
end

"""
    road_trip_example()

Plan a one-week California road trip, populate an MCTS tree from the initial state,
and open it in MCTSViz.
"""
function road_trip_example()
    mdp = CaliforniaRoadTripMDP()
    solver = MCTSSolver(;
        n_iterations = 10_000,
        depth = 40,
        exploration_constant = 100.0,
        estimate_value = RolloutEstimator(RoadTripRolloutPolicy(mdp); max_depth = 40),
        enable_tree_vis = true,
    )
    policy = solve(solver, mdp)
    start = rand(initialstate(mdp))
    println("Suggested first action: ", action(policy, start))
    mcts_viz(mdp, policy)
end
