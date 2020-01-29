#!/usr/bin/env python3

import argparse

import joblib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf

from garage.misc.console import query_yes_no
from garage.sampler.utils import rollout

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the .pkl file')
    parser.add_argument(
        '--max_path_length',
        type=int,
        default=500,
        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.compat.v1.Session():
    #     [rest of the code]
    with tf.compat.v1.Session() as sess:
        data = joblib.load(args.file)
        policy = data['algo'].policy
        env = data['env']
        while True:
            path = rollout(
                env,
                policy,
                max_path_length=args.max_path_length,
                animated=True,
                speedup=args.speedup,
                deterministic=True)
            plt.figure()
            plt.title('observations')
            plt.xlabel('iteration')
            plt.plot(range(args.max_path_length), path['observations'])

            plt.figure()
            plt.title('actions')
            plt.xlabel('iteration')
            plt.plot(range(args.max_path_length), path['actions'])
            plt.show(block=False)

            print('agent_infos: ')
            for key in path['agent_infos']:
                print(key, ': ')
                print(path['agent_infos'][key][-1])

            if not query_yes_no('Continue simulation?'):
                break
