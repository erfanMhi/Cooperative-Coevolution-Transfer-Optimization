
import numpy as np
import lhsmdu
import argparse
import os

from pprint import pprint
from utils.data_manipulators import *


def get_args():
    parser = argparse.ArgumentParser(description='CoOperative CoEvolution Transfer Optimization Algorithm for Solving Multi-location Inventory Planning with Lateral Transshipments')

    parser.add_argument('-t', dest='tsamples', default=10,
                        type=int, nargs='?',
                        help='Number of testing data samples')
                    
    # parser.add_argument('-q', dest='matrix_num', default='a',
    #                     type=str, nargs='?',
    #                     help='T^0_H matrix selector for section b')

    return parser.parse_args()

def main():

  ################# Preconfigurations ###################
  args = get_args()

  models_path = 'models'
  source_models = os.path.join(models_path, 'source_models')

  if os.path.exists(source_models):
    source_models = Tools.load_from_file(source_models)
  else:
    raise FileNotFoundError('Source models not exist in the {} path'.format(source_models))

  tsamples = args.tsamples # number of testing data samples
  mc_samples = 30

  n = 4
  D_mean = 500* lhsmdu.sample(tsamples, n).T + 100 # generate problem dataset with demands ranging from 100 to 600
  X_data = D_mean.T
  c = 10
  h = np.ones((1,4))
  p = 50*h


  ################# Evolutionary Algorithm ###################


if __name__ == '__main__':
  main()