import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loglist', nargs='+', help='a list of garage log dirs, seperate by space')
    args = parser.parse_args()

    plt.figure()

    color_dict = matplotlib.colors.TABLEAU_COLORS

    for i, log_path in enumerate(args.loglist):
        color = list(color_dict.keys())[i]
        csv_paths =  glob.glob('**/'+log_path+'/**/*.csv', recursive=True)

        avg_discounted_return = pd.DataFrame(columns = None)
        for j, csv_path in enumerate(csv_paths):
            csv_df = pd.read_csv(csv_path)
            avg_discounted_return.insert(j, 'AverageDiscountedReturn{}'.format(j), csv_df['AverageDiscountedReturn'])

        mean = avg_discounted_return.mean(numeric_only=True, axis=1)
        plt.plot(avg_discounted_return, color, alpha=0.1)
        plt.plot(mean, color,alpha=1.0)
        plt.xlabel('Number of policy updates')
        plt.ylabel('Average discounted return')
        del avg_discounted_return
    
    plt.show()