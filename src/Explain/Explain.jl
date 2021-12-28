module Explain

using ..Utils
using DataFrames
using CSV

export
explain_greedy

# structure for use in heap
struct Candidate
    features::Array
    prob::Float64
end

function Base.isless(c1::Candidate, c2::Candidate)
    return c1.prob < c2.prob
end

function explain_greedy(ep_func, ep_params, dec_bdry, inst, label, max_features, k, encoder=nothing, cats=[], verbose=true, csv_file=nothing, prb_file=nothing, log_file=nothing)
    full_ep = 0
    full_log_mar = 0
    if encoder === nothing
        full_ep = ep_func([inst],ep_params,true,nothing)[1]
        full_log_mar = marginals([inst],ep_params[1])[1]
    else
        full_ep = ep_func(encoder([inst],cats),ep_params,true,nothing)[1]
        full_log_mar = marginals(encoder([inst],cats),ep_params[1])[1]
    end

    predicted_label = (full_ep >= dec_bdry)+0
    if csv_file !== nothing
        println(csv_file)
        CSV.write(csv_file, DataFrame(transpose(convert(Array{Union{Missing,Int}},inst)),:auto), header=false)
    end
    if prb_file !== nothing
        println(prb_file)
        CSV.write(prb_file, DataFrame(hcat([[label,full_ep],[predicted_label,full_log_mar]]...),:auto), header=false)
    end

    logger = nothing
    if log_file !== nothing
        println(log_file)
        logger = open(log_file, "a")
        write(logger, "Inst:\nActual label: $(label)\nExp Pred: $(full_ep)\nPredicted Label: $(predicted_label)\n")
    end
    if(verbose)
        println("Actual label: $(label)")
        println("Exp pred: $(full_ep)")
        println("Predicted label: $(predicted_label)")
        println()
    end

    best_cand = nothing
    best_log_mar = 0
    best_ep = (predicted_label == 1) ? -999 : 999
    empty_inst = deepcopy(inst)
    empty_inst .= missing
    top_k = [empty_inst]

    total_timed = @timed begin
        for i in 1:max_features
            level_timed = @timed begin
                if(verbose)
                    println("Looking at subsets of length $(i)...")
                end
                if logger !== nothing
                    write(logger, "LEVEL $(i)\n")
                end
                expanded_set = Set()
                expanded_arr = [] #TODO: optimize to one alloc
                for cand in top_k
                    expanded = expand_all(cand, inst)
                    for subset in expanded
                        if !(subset in expanded_set)
                            push!(expanded_set, subset)
                            append!(expanded_arr, [subset])
                        end
                    end
                end
                num_expanded = size(expanded_arr)[1]
                expanded_ep = nothing
                expanded_mars = nothing
                if encoder === nothing
                    expanded_ep = ep_func(expanded_arr, ep_params, true, logger)
                    expanded_mars = marginals(expanded_arr, ep_params[1])
                else
                    expanded_ep = ep_func(encoder(expanded_arr,cats), ep_params, true, logger)
                    expanded_mars = marginals(encoder(expanded_arr,cats), ep_params[1])
                end
                expanded_cands = Array{Tuple{Candidate,Float64}}(undef,num_expanded)
                for j in 1:num_expanded
                    expanded_cands[j] = (Candidate(expanded_arr[j], expanded_ep[j]),expanded_mars[j])
                end
                if (predicted_label == 1)
                    sort!(expanded_cands, by = x -> (x[1].prob, x[2]),rev=true)
                else
                    sort!(expanded_cands, by = x -> (-x[1].prob,x[2]), rev=true)
                end
                top_k = []
                for j in 1:k
                    if j > num_expanded
                        break
                    end
                    append!(top_k, [expanded_cands[j][1].features])
                end
                best_level_ep = expanded_cands[1][1].prob
                best_level_mar = expanded_cands[1][2]
                if (predicted_label == 1 && best_level_ep > best_ep) || (predicted_label == 0 && best_level_ep < best_ep)
                    best_ep = best_level_ep
                    best_log_mar = best_level_mar
                    best_cand = expanded_cands[1][1].features
                elseif (best_level_ep == best_ep && best_level_mar > best_log_mar)
                    best_log_mar = best_level_mar
                    best_cand = expanded_cands[1][1].features
                end
                if csv_file !== nothing
                    CSV.write(csv_file, DataFrame(transpose(convert(Array{Union{Missing,Int}},best_cand)),:auto), append=true)
                end
                if prb_file !== nothing
                    CSV.write(prb_file, DataFrame(reshape([best_ep,best_log_mar],1,2),:auto), append=true)
                end
            end #level_timed
            if logger !== nothing
                write(logger, "level time:\n$(get_timed_string(level_timed))\n")
            end
        end
    end #total_timed
    if logger !== nothing
        write(logger, "total time:\n$(get_timed_string(total_timed))")
    end
    if logger !== nothing
        close(logger)
    end
    return best_cand
end

function expand_all(x, instance)
    ret = []
    num_features = size(instance)[1]
    for i in 1:num_features
        if ismissing(x[i])
            newx = copy(x)
            newx[i] = instance[i]
            append!(ret, [newx])
        end
    end
    return ret
end

end#module