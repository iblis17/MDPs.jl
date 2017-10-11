
rand_reward(::Type{T}, dims...) where {T<:AbstractFloat} = 1 .- 2*rand(T, dims...)

rand_reward(::Type{T}, dims...) where {T<:Integer} = rand(T, dims...)


sum_columns_to_one(a::Array) = a ./ sum(a, 1)


# replace all elements of `a` with 0 where `mask` is `false`
function zero_mask!(a::AbstractArray{T,3}, mask::AbstractArray{Bool,2}) where T
    all(sum(mask, 1) .> 0) ||
        error("All columns of mask must have at least one 'true' value.")
    size(mask) == size(a)[1:2] ||
        error("Mask size must be `states`×`states`.")
    for k = 1:size(a)[3]
        setindex!(sub(a, :, :, k), zero(T), find(!mask))
    end
end

function zero_mask!(a::AbstractArray{T,3}, mask::AbstractArray{Bool,3}) where T
    all(sum(mask, 1) .> 0) ||
        error("All columns of mask must have at least one 'true' value.")
    size(mask) == size(a) ||
        error("Mask size must be `states`×`states`×`actions`.")
    setindex!(a, zero(T), find(!mask))
end


function random(
    ::Type{P},
    ::Type{R},
    states,
    actions,
    mask::Nullable{Array{Bool,N}},
) where {P<:AbstractFloat,R<:Real,N}
    transition = rand(P, states, states, actions)
    isnull(mask) ? nothing : zero_mask!(transition, get(mask))
    transition = sum_columns_to_one(transition)
    reward = rand_reward(R, states, actions)
    (transition, reward)
end

random(::Type{P}, ::Type{R}, states, actions, mask::Array{Bool}) where {P,R} =
    random(P, R, states, actions, Nullable(mask))

random(::Type{P}, ::Type{R}, states, actions) where {P,R} =
    random(P, R, states, actions, Nullable{Array{Bool,1}}())

random(states, actions, mask::Nullable{Array{Bool,N}}) where {N} =
    random(Float64, Float64, states, actions, mask)

random(states, actions, mask::Array{Bool}) =
 random(states, actions, Nullable(mask))

random(states, actions) = random(states, actions, Nullable{Array{Bool,1}}())
