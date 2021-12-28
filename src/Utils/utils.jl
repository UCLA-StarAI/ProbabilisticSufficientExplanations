module Utils

using ..LearnPSDD
using LogicCircuits
using ProbabilisticCircuits
using DataStructures
using CSV
using Statistics
using XGBoost
using JSON
using DataFrames
using LinearAlgebra

export
load_pc,
load_mnist_data,
load_adult_data,
load_adult3_data,
load_data,
learn_mnist_psdd,
get_num_cats,
one_hot,
mar_prob,
get_encoder_and_cats,
load_forest,
get_paths,
get_best_pred_ub,
batch_marginal_threaded,
exp_pred_batch_threaded,
get_timed_string,
get_ep_func,
sample_sdp,
gen_sdp_lb_csv_xgb,
actual_sdp,
marginals

function load_pc(dataset)
    circuit_path = ""
    num_features = 0
    if (dataset == "adult")
        circuit_path = "circuits/adult_2/adult.psdd"
        num_features = 130 # one hot features
    elseif (dataset == "mnist")
        circuit_path = "circuits/mnist-3-5/mnist35.psdd"
        num_features = 784
    elseif (dataset == "adult3")
        circuit_path = "circuits/adult3/adult3.psdd"
        num_features = 76
    end
    # pc = load_prob_circuit(circuit_path)
    # pc = ParamBitCircuit(pc,DataFrame(missings(Bool,1,num_features)))
    pc = Base.read(circuit_path, ProbCircuit)
    return pc
end

function load_mnist_data()
    path = "data/MNIST-3-5/"
    x_train = convert.(Bool,CSV.read(path*"train_images.csv", DataFrame; header=false))
    y_train = CSV.read(path*"train_labels.csv", DataFrame; header=false)
    x_test = convert.(Bool,CSV.read(path*"test_images.csv", DataFrame; header=false))
    y_test = CSV.read(path*"test_labels.csv", DataFrame; header=false)
    return x_train, x_test, y_train, y_test
end

function load_adult_data()
    path = "data/adult2/"
    x_train = CSV.read(path*"x_train.csv"; header=false)
    y_train = CSV.read(path*"y_train.csv"; header=false)
    x_test = CSV.read(path*"x_test.csv"; header=false)
    y_test = CSV.read(path*"y_test.csv"; header=false)
    return x_train, x_test, y_train, y_test
end

function load_adult3_data()
    path = "data/adult3/"
    x_train = CSV.read(path*"x_train.csv"; header=false)
    y_train = CSV.read(path*"y_train.csv"; header=false)
    x_test = CSV.read(path*"x_test.csv"; header=false)
    y_test = CSV.read(path*"y_test.csv"; header=false)
    return x_train, x_test, y_train, y_test
end

function load_data(dataset)
    if dataset == "mnist"
        return load_mnist_data()
    elseif dataset == "adult"
        return load_adult_data()
    elseif dataset == "adult3"
        return load_adult3_data()
    end
    return nothing
end

function learn_mnist_psdd()
    folder = "./circuits/mnist-3-5/"
    mkpath(folder)

    max_epochs = 1000
    max_circuit_nodes = 20000
    println("Training circuit with max_epochs=$(max_epochs), max_size=$(max_circuit_nodes)");
    
    x_train, x_test, y_train, y_test = load_mnist_data()
    pc, vtree = learn_psdd_miss(x_train, outdir=folder, max_epochs=max_epochs, max_circuit_nodes=max_circuit_nodes);

    save_circuit(folder * "mnist35.psdd", pc, vtree)
end

function get_num_cats(dataset)
    ret = []
    if dataset == "adult"
        ret = [5, 9, 5, 16, 4, 7, 15, 6, 5, 2, 5, 5, 4, 42]
    elseif dataset == "adult3"
        ret = [5, 7, 16, 7, 14, 6, 5, 2, 5, 5, 4]
    end
    return ret
end

function one_hot(x, cats)
    n = size(x)[1]
    num_cats = size(cats)[1]
    starts = Int.(zeros(num_cats+1))
    starts[1] = 1
    for i in 2:num_cats+1
        starts[i] = starts[i-1] + cats[i-1]
    end
    ret = Array{Array{Union{Missing,Bool}}}(undef,n)
    Threads.@threads for i in 1:n
        ret[i] = one_hot_single(x[i], starts)
    end
    return ret
end

function one_hot_single(x, cat_starts)
    num_features = size(x)[1]
    tot = cat_starts[end]-1
    ret = Array{Union{Missing,Bool}}(undef,tot)
    for i in 1:num_features
        if !ismissing(x[i])
            ret[cat_starts[i]:cat_starts[i+1]-1] .= 0
            ret[cat_starts[i]+x[i]] = 1
        end
    end
    return ret
end

function mar_prob_onehot(x, pc)
    prob = exp.(log_proba(pc, XData(transpose(hcat(x)))))
    return prob[1]
end

function mar_prob(x, pc, encoder)
    if (encoder === nothing)
        return mar_prob_onehot(x, pc)
    end
    return mar_prob_onehot(encoder(x),pc)
end

function get_encoder_and_cats(dataset)
    enc = nothing
    if (dataset == "adult" || dataset == "adult3")
        enc = one_hot
    elseif (dataset == "mnist")
        enc = nothing
    end
    cats = get_num_cats(dataset)
    return enc,cats
end

"""
TREE STUFF
"""

struct Path
    vars::Dict{Integer, Integer}
    weight::AbstractFloat
end

function load_forest(dataset)
    bst_json_file = ""
    if (dataset == "adult")
        bst_json_file = "classifiers/adult/xgboost.json"
    elseif (dataset == "mnist")
        bst_json_file = "classifiers/MNIST-3-5/xgboost.json"
    elseif (dataset == "adult3")
        bst_json_file = "classifiers/adult3/xgboost.json"
    end
    dmp = JSON.parsefile(bst_json_file)
    return dmp
end

function get_paths(dataset)
    paths = Array{Path,1}()
    forest = load_forest(dataset)
    for tree in forest
        append!(paths, get_paths_for_tree(tree))
    end
    return paths
end

function get_paths_for_tree(tree)
    paths = Array{Path,1}()

    nodes = Stack{Dict{String,Any}}()
    push!(nodes,tree)
    temp_paths = Stack{Dict{Integer,Integer}}()
    push!(temp_paths,Dict{Integer,Integer}())

    while (!isempty(nodes))
        curr_node = pop!(nodes)
        curr_path = pop!(temp_paths)

        if (haskey(curr_node, "leaf"))
            final_path = Path(curr_path,curr_node["leaf"])
            append!(paths, [final_path])
        else
            split_feature = curr_node["split"]
            split_feature_num = parse(Int64, split_feature[2:length(split_feature)]) + 1
            yes_child = curr_node["yes"]
            no_child = curr_node["no"]
            children = curr_node["children"]
            for child in children
                child_id = child["nodeid"]
                temp = copy(curr_path)
                if (child_id == yes_child)
                    temp[split_feature_num] = 0
                else
                    temp[split_feature_num] = 1
                end
                push!(nodes, child)
                push!(temp_paths, temp)
            end
        end
    end
    return paths
end

function get_best_pred_ub(inst, forest, label)
    best_pred_ub = 0
    for tree in forest
        best = (label == 1) ? -999 : 999
        paths = get_paths_for_tree(tree)
        for path in paths
            # if path agree with inst update best
            pairs = path.vars
            cont = false
            for p in pairs
                key = p[1]
                val = p[2]
                if !ismissing(inst[key]) && inst[key] != val
                    cont = true
                    break
                end
            end
            if cont
                continue
            end
            if (label == 1 && path.weight > best) || (label == 0 && path.weight < best)
                best = path.weight
            end
        end
        best_pred_ub += best
    end
    return best_pred_ub
end

function batch_marginal_threaded!(res, x, pc, batch_size=50000)
    n = size(x)[1]
    num_batches = cld(n, batch_size) # ceiling division
    Threads.@threads for i in 1:num_batches
        idx1 = (i-1)*batch_size+1
        idx2 = min(i*batch_size, n)
        res[idx1:idx2] .= marginal(pc, DataFrame(x[idx1:idx2,:], :auto))
    end
end

function get_weights_and_mat(x, paths)
    leaves = size(paths)[1]
    num_features = size(x)[1]
    mat = missings(Bool,leaves+1,num_features)
    mat[1,:].=x
    # mat .= transpose(x)
    weights = Float32.(zeros(leaves))
    i = 1
    for path in paths
        mat[i+1,:].=x
        pairs = path.vars
        is_zero = false
        for p in pairs
            key = p[1]
            val = p[2]
            if !ismissing(mat[i+1,key]) && mat[i+1,key] != val
                is_zero = true
                break
            end
            mat[i+1,key] = val
        end
        if (!is_zero)
            weights[i] = path.weight
            i += 1
        end
    end
    return weights[1:i-1], mat[1:i,:]
end

function exp_pred_batch_threaded(x, ep_params, verbose=false, logger=nothing)
    pc, paths = ep_params
    n = size(x)[1]
    num_features = size(x[1])[1]
    leaves = size(paths)[1]

    traverse_timed = @timed begin
        weights = Array{Array{Float32,1},1}(undef,n)
        mats = Array{Array{Union{Missing, Bool},2},1}(undef,n)
        mat = missings(Bool,n*(leaves+1),num_features)
        Threads.@threads for i in 1:n
            w_single, mat_single = get_weights_and_mat(x[i], paths)
            weights[i] = w_single
            mats[i] = mat_single
        end
        midxs = Int32.(ones(n+1))
        for i in 1:n
            midxs[i+1] = midxs[i] + size(mats[i])[1]
        end
        Threads.@threads for i in 1:n
            mat[midxs[i]:midxs[i+1]-1,:].=mats[i]
        end
    end

    mar_timed = @timed begin
        log_probs = Float32.(zeros(midxs[end]-1))
        mar_batch_size = 50000
        batch_marginal_threaded!(log_probs, mat[1:midxs[end]-1,:], pc, mar_batch_size)

        exp_preds = Array{Float64,1}(undef,n)
        Threads.@threads for i in 1:n
            curr_probs = log_probs[midxs[i]:midxs[i+1]-1]
            exp_preds[i] = dot(weights[i], exp.(curr_probs[2:end].-curr_probs[1]))
        end
    end
    if logger !== nothing
        write(logger, "num exp preds: $(size(x)[1])\n")
        write(logger, "num marginals: $(midxs[end]-1)\n")
        write(logger, "marginals batch size: $(mar_batch_size)\n")
        write(logger, "forest traverse time:\n$(get_timed_string(traverse_timed))")
        write(logger, "marginals pc time:\n$(get_timed_string(mar_timed))")
    end
    return exp_preds
end

function get_timed_string(tpl)
    return "time:$(tpl.time)\nbytes:$(tpl.bytes)\ngctime:$(tpl.gctime)\ngcstats:$(tpl.gcstats)\n"
end

function get_ep_func(classifier)
    if classifier == "xgb"
        return exp_pred_batch_threaded
    end
end

function sample_sdp(inst, label, ep_func, ep_params, dec_bdry, num_samples=10000)
    pc = ep_params[1]
    data = DataFrame(reshape(inst, 1, :))
    samples, lls = sample(pc, num_samples, data)
    samples = samples[:,1,:]
    samples_arr = Array{Array{Union{Bool,Missing},1},1}(undef,num_samples)
    for i in 1:num_samples
        samples_arr[i] = convert(Array{Union{Bool,Missing},1},samples[i,:])
    end
    preds = ep_func(samples_arr, ep_params, true, nothing)
    classes = (preds.>=dec_bdry).+0
    if label == 0
        classes = 1 .- classes
    end
    sdp = sum(classes)/num_samples
    sample_var = sum((classes.-sdp).^2)/(num_samples-1)
    t_95 = 1.960
    t_90 = 1.645
    conf_range = t_95*sqrt(sample_var)/sqrt(num_samples)
    return sdp, conf_range
end

function gen_sdp_lb_csv_xgb(dataset, folder)
    pc = load_pc(dataset)
    forest = load_forest(dataset)
    paths = get_paths(dataset)
    enc,cats = get_encoder_and_cats(dataset)
    if folder[end] != '/'
        folder = folder*"/"
    end
    exps = allowmissing(CSV.read("exp/$(dataset)/$(folder)exps.csv"; header=false))[2:end,:]
    n = size(exps)[1]
    probs = CSV.read("exp/$(dataset)/$(folder)probs.csv"; header=false)[1]
    label = (probs[2]>=0)+0
    probs = probs[3:end]
    sdps = Float64.(zeros(n,3))
    Threads.@threads for i in 1:n
        inst = convert(Array{Union{Int,Missing}},exps[i,:])
        if enc === nothing
            inst = convert(Array{Union{Bool,Missing}},inst)
        else
            inst = enc([inst],cats)[1]
        end
        sdp, conf_range = sample_sdp(inst, label, exp_pred_batch_threaded, [pc,paths], 0,1000)
        sdps[i,1] = probs[i]/get_best_pred_ub(inst, forest, label)
        sdps[i,2] = sdp
        sdps[i,3] = conf_range
    end
    CSV.write("exp/$(dataset)/$(folder)sdps.csv",DataFrame(sdps),header=false)
    return sdps
end

function actual_sdp(x,pc,paths,bst,label=nothing)
    if label === nothing
        label = exp_pred_batch_threaded([x], [pc, paths])[1]
    end
    num_features = size(x)[1]
    num_missing = 0
    missing_cols = []
    for i in 1:num_features
        if ismissing(x[i])
            num_missing += 1
            append!(missing_cols,i)
        end
    end
    if num_missing == 0
        return 1
    end
    n = 2^num_missing
    mat = missings(Bool, n+1, num_features)
    Threads.@threads for i in 1:num_features
        if !ismissing(x[i])
            mat[:,i].=x[i]
        end
    end
    Threads.@threads for k in 1:num_missing
        mcol = missing_cols[k]
        num_con = 2^(k-1)
        counter = 0
        val = false
        for i in 1:n
            mat[i,mcol] = val
            counter += 1
            if counter == num_con
                val = !val
                counter = 0
            end
        end
    end
    log_probs = Float32.(zeros(n+1))
    batch_marginal_threaded!(log_probs, mat, pc)
    preds = predict(bst, convert(Array{Int,2},mat[1:end-1,:]))
    classes = (preds.>=0).+0
    if label < 0
        classes = 1 .- classes
    end
    sdp = dot(classes, exp.(log_probs[1:end-1].-log_probs[end]))
    equals1 = true
    for c in classes
        if c == 0
            equals1 = false
            break
        end
    end
    return Float64(sdp), equals1
end

function marginals(x,pc)
    n = size(x)[1]
    n_features = size(x[1])[1]
    mar_df = missings(Bool,n,n_features)
    for i in 1:n
        mar_df[i,:].=x[i]
    end
    mars = marginal(pc, DataFrame(mar_df, :auto))
    return mars
end

end#module