import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse
import shared_params.params_opt_k as params
# import shared_params.params_opt_l as params

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ppo_log_list', help='a list of ppo log dirs, seperate by comma')
    parser.add_argument('cmaes_ppo_log_list', help='a list of cmaes log dirs, seperate by comma')

    args = parser.parse_args()

    ppo_batch_size = params.ppo_train_kwargs['batch_size']
    cmaes_batch_size = params.ppo_inner_train_kwargs['n_epochs'] * params.ppo_inner_train_kwargs['batch_size'] * params.cmaes_options['popsize']

    ppo_log_list = [(item)for item in args.ppo_log_list.split(',')]
    cmaes_ppo_log_list = [(item)for item in args.cmaes_ppo_log_list.split(',')]

    plt.figure()

    color_dict = matplotlib.colors.TABLEAU_COLORS

    for i, log_path in enumerate(ppo_log_list):
        color = list(color_dict.keys())[i]
        csv_paths = glob.glob('**/'+log_path+'/**/*.csv', recursive=True)

        ppo_avg_discounted_return = pd.DataFrame(columns = None)
        for j, csv_path in enumerate(csv_paths):
            csv_df = pd.read_csv(csv_path)
            ppo_avg_discounted_return.insert(j, 'AverageDiscountedReturn{}'.format(j), csv_df['AverageDiscountedReturn'])

        mean = ppo_avg_discounted_return.mean(numeric_only=True, axis=1)
        plt.plot(np.array(range(ppo_avg_discounted_return.shape[0])) * ppo_batch_size, ppo_avg_discounted_return, color, alpha=0.1)
        plt.plot(np.array(range(mean.shape[0])) * ppo_batch_size, mean, color, alpha=1.0)
        del ppo_avg_discounted_return
    
    for k, log_path in enumerate(cmaes_ppo_log_list):
        color = list(color_dict.keys())[i+k+1]
        csv_paths = glob.glob('**/'+log_path+'/**/*.csv', recursive=True)

        cmaes_avg_discounted_return = pd.DataFrame(columns = None)
        for j, csv_path in enumerate(csv_paths):
            csv_df = pd.read_csv(csv_path)
            cmaes_avg_discounted_return.insert(j, 'AverageDiscountedReturn{}'.format(j), csv_df['AverageDiscountedReturn'])

        mean = cmaes_avg_discounted_return.mean(numeric_only=True, axis=1)
        plt.plot(np.array(range(cmaes_avg_discounted_return.shape[0])) * cmaes_batch_size, cmaes_avg_discounted_return, color, alpha=0.1)
        plt.plot(np.array(range(mean.shape[0])) * cmaes_batch_size, mean, color, alpha=1.0)

        del cmaes_avg_discounted_return

    plt.xlabel('Environment steps')
    plt.ylabel('Average discounted return')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.show()