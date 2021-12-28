# Probabilistic Sufficient Explanations

Code for the ICJAI-21 paper [Probabilistic Sufficient Explanations](http://starai.cs.ucla.edu/papers/WangIJCAI21.pdf)

To generate explanations, first start julia using
```
julia -i --project=. --threads=[num_threads] scripts/demo.jl
```
Then run the function `run_suff_exp` to generate explanations. For example, `run_suff_exp("mnist","xgb",17,30,5)` will explain mnist test instance 17 with max 30 features and beam size 5 usign xgboost classifier. Outputs will be saved in the `exp` folder.

Explanations can be visualized using the script located at `python/mnist_exp_vis.py`. For example, from the `python` directory, run `python3 mnist_exp_vis.py ../exp/mnist/xgb_17_10_5_2021-12-27T17-53-47.049`. 