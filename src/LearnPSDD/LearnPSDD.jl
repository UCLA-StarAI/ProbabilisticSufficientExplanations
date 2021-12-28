module LearnPSDD

using LinearAlgebra
using LogicCircuits
using Statistics
using Random
using DataFrames
using ProbabilisticCircuits

export learn_circuit_miss

function heuristic_loss_miss(circuit::LogicCircuit, train_x; 
        pick_edge,
        pick_var,
        iter)
    
    train_x_impute = median_impute(train_x);
        
    candidates, scope = split_candidates(circuit)
    # values, flows = marginal_flows(circuit, train_x)
    values, flows = satisfies_flows(circuit, train_x_impute)
    if pick_edge == "eFlow"
        edge, flow = ProbabilisticCircuits.eFlow(values, flows, candidates)
    elseif pick_edge == "eRand"
        edge = ProbabilisticCircuits.eRand(candidates)
    else
        error("Heuristics $pick_edge to pick edge is undefined.")
    end
    
    or, and = edge
    lits = collect(Set{Lit}(scope[and]))
    vars =  Var.(intersect(filter(l -> l > 0, lits), - filter(l -> l < 0, lits)))
    
    if pick_var == "vMI"
        var, score = ProbabilisticCircuits.vMI(values, flows, edge, vars, train_x_impute)
    elseif pick_var == "vRand"
        var = ProbabilisticCircuits.vRand(vars)
    else
        error("Heuristics $pick_var to pick variable is undefined.")
    end
    return (or, and), var
end
    

"""
Learn structure of a single structured decomposable circuit
"""
function learn_circuit_miss(train_x;
        pick_edge="eFlow", 
        pick_var="vRand", 
        depth=1, 
        pseudocount=1.0,
        sanity_check=true,
        seed=nothing,
        return_vtree=true,
        outdir::String,
        valid_x=nothing,
        maxiter=100,
        init_epochs=10,
        struct_step=10,
        batch_size=1000,
        save_freq=200,
        max_circuit_nodes=50000,
        epoch_printer::Function=println,
        )

    # output dir
    if !isdir(outdir)
        mkpath(outdir)
    end

    if seed !== nothing
        Random.seed!(seed)
    end

    # Initial Structure
    train_x_impute = median_impute(train_x)
    println(typeof(train_x_impute))
    println(size(train_x_impute))
    pc, vtree = learn_chow_liu_tree_circuit(train_x_impute)

    # structure_update
    iter = 0
    loss(circuit) = heuristic_loss_miss(circuit, train_x; pick_edge=pick_edge, pick_var=pick_var, iter)

    pc_split_step(circuit) = begin
        c::ProbCircuit, = split_step(circuit; loss=loss, depth=depth, sanity_check=sanity_check)
        estimate_parameters_em(c, train_x; pseudocount=pseudocount)
        return c, missing
    end
   
    log_per_iter(circuit) = begin
        iter += 1
        if iter % 10 == 0
            ll = MAR(circuit, train_x);
            println("Iteration $iter/$maxiter. LogLikelihood = $(mean(ll)); nodes = $(num_nodes(circuit)); edges =  $(num_edges(circuit)); params = $(num_parameters(circuit))")
        end

        if iter % save_freq == 0
            lls = MAR(circuit, train_x);
            epoch_printer("Train lls; mean = $(mean(lls)), std = $(std(lls))");
            if nothing !=valid_x
                lls = MAR(circuit, valid_x);
                epoch_printer("valid lls; mean = $(mean(lls)), std = $(std(lls))");
            end
            save_circuit(outdir * "epoch_$(iter).psdd", circuit, vtree)
        end

        if num_nodes(circuit) > max_circuit_nodes
            epoch_printer("Stopping early, circuit size above max threshold $(max_circuit_nodes). Current size = $(num_nodes(circuit))");
            return true;
        end

        false
    end
    log_per_iter(pc)
    pc = struct_learn(pc; 
        primitives=[pc_split_step], kwargs=Dict(pc_split_step=>()), 
        maxiter=maxiter, stop=log_per_iter)

    if return_vtree
        pc, vtree
    else
        pc
    end
end

function median_impute(X, train=nothing; pseudocount=1.0)
    if isnothing(train)
        train=X
    end

    X_impute = deepcopy(X)
    for f = 1:size(X)[2]
        mask_y = ismissing.(train[:,f])
        cur_median = median(train[:,f][.!(mask_y)] )
        mask_x = ismissing.(X[:,f])
        X_impute[mask_x,f] .= Bool(cur_median .>= 0.5)
    end
    X_impute = DataFrame(BitArray( convert(Matrix,X_impute)))
    return X_impute
end

end