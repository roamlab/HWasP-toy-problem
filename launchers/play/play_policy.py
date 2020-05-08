"""Simulates pre-learned policy."""
import argparse
import sys
import numpy as np
import joblib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
import tensorflow as tf

from garage.sampler.utils import rollout
import matplotlib.pyplot as plt


def query_yes_no(question, default='yes'):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--max_path_length',
                        type=int,
                        default=500,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    parser.add_argument('--deterministic', help='use the mean action or stochastic action', action='store_true')
    args = parser.parse_args()
    print(args)
    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.compat.v1.Session():
    #     [rest of the code]
    with tf.compat.v1.Session() as sess:
        data = joblib.load(args.file)
        policy = data['algo'].policy
        env = data['env']
        while True:
            path = rollout(env,
                           policy,
                           max_path_length=args.max_path_length,
                           animated=True,
                           speedup=args.speedup,
                           deterministic=args.deterministic)

            plt.figure()
            plt.title('observations')
            plt.xlabel('time steps')
            plt.plot(range(args.max_path_length), path['observations'])

            plt.figure()
            plt.title('actions')
            plt.xlabel('time steps')
            plt.plot(range(args.max_path_length), path['actions'])

            plt.figure()
            plt.title('rewards')
            plt.xlabel('time steps')
            plt.plot(range(args.max_path_length), path['rewards'])
            
            plt.show(block=False)

            print('episode reward: ')
            print(np.sum(path['rewards']))

            print('agent_infos: ')
            for key in path['agent_infos']:
                print(key, ': ')
                print(path['agent_infos'][key][-1])
            if not query_yes_no('Continue simulation?'):
                break