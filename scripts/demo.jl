using Revise
using ProbabilisticSufficientExplanations
using Dates
using DataFrames

function run_suff_exp(dataset, classifier, ind, max_features, k)
    curtime = Dates.now()

    ep_params = []
    pbc = load_pc(dataset)
    dec_bdry = 0
    if classifier == "xgb"
        paths = get_paths(dataset)
        ep_params = [pbc, paths]
        dec_bdry = 0
    elseif classifier == "logreg"
        w,b = load_weights(dataset)
        ep_params = [pbc,w,b]
        dec_bdry = 0.5
    else
        println("invalid classifier")
        return
    end
    encoder,cats = get_encoder_and_cats(dataset)
    ep_func = get_ep_func(classifier)

    x_train, x_test, y_train, y_test = load_data(dataset)

    inst = allowmissing(Array(x_test[ind,:]))
    label = y_test[!,1][ind]

    exp_base_dir = "./exp/$(dataset)"
    if !isdir(exp_base_dir)
        mkdir(exp_base_dir)
    end
    dirname = replace("$(classifier)_$(ind)_$(max_features)_$(k)_$(curtime)",":"=>"-")
    exp_dir = "$(exp_base_dir)/$(dirname)"
    if !isdir(exp_dir)
        mkdir(exp_dir)
    end
    
    csv_name = "exps.csv"
    prb_name = "probs.csv"
    log_name = "logfile.log"
    logger = open("$(exp_dir)/$(log_name)","w")
    write(logger, "$(curtime)\nthreads: $(Threads.nthreads())\nparams:\nx_test idx $(ind)\nmax_features $(max_features)\nbeam_size $(k)\n")
    close(logger)

    ProbabilisticSufficientExplanations.Explain.explain_greedy(ep_func,ep_params,dec_bdry,inst,label,max_features,k,encoder,cats,true,"$(exp_dir)/$(csv_name)","$(exp_dir)/$(prb_name)","$(exp_dir)/$(log_name)")
end