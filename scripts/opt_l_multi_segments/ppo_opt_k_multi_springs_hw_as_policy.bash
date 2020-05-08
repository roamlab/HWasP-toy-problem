#!/bin/bash
export datetime=`date '+%Y_%m_%d_%H_%M_%S'`
export n_terminals=4
export n_runs_each_terminal=2

function run_exps_in_terminal()
{
    for (( run_id=0; run_id<$n_runs_each_terminal; run_id++ ))
    do
        python launchers/train/opt_k_multi_springs/ppo_opt_k_multi_springs_hw_as_policy.py --seed=$(( $1*$n_terminals+$run_id )) --exp_id=$datetime
    done
}

export -f run_exps_in_terminal

for (( terminal_id=0; terminal_id<$(( $n_terminals-1 )); terminal_id++ ))
do
    xterm -hold -e "run_exps_in_terminal $terminal_id" & # "&" means do not wait until finish, run in parallel
    sleep 3
done

xterm -hold -e "run_exps_in_terminal $(( $n_terminals-1 ))" # do not use "&" for the last one