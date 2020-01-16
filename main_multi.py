
import argparse
import os
import queue

import multiprocessing as mp
# import SharedArray as sa
import numpy as np


from copy import deepcopy
from time import time
from pprint import pprint
from utils.data_manipulators import *
from evolution.operators import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel
from evolution.chromosome import *


class EAProcess(mp.Process):
  def __init__(self, dims, psize, gen, problem, shared_queue, 
              shared_array, t_lock, list_lock, return_list, transfer_interval=2):
      super(EAProcess, self).__init__()
      self.dims = dims
      self.psize = psize
      print('hi')
      self.gen = gen
      self.problem = problem
      self.shared_queue = shared_queue
      self.shared_array = shared_array
      # self.shared_lock = shared_lock
      self.t_lock = t_lock
      self.list_lock = list_lock
      self.transfer_interval = transfer_interval
      self.reinitialize()
      self.return_list = return_list

  def reinitialize(self):

    self.fitness_hist = np.zeros((self.gen, self.psize))
    self.fitness_time = np.zeros((self.gen))

    init_func = lambda n: np.round(np.random.rand(n))
    self.pop = get_pop_init(self.psize, self.dims, init_func)

  def _ea(self):
    
    start = time()

    for i in range(self.psize): self.pop[i].fitness_calc(self.problem)

    self.bestfitness = np.max(self.pop).fitness
    self.fitness = Chromosome.fitness_to_numpy(self.pop)
    self.fitness_hist[0, :] = self.fitness

    self.fitness_time[0] = start - time()


    
    for i in range(1, self.gen):
      start = time()

      if i%self.transfer_interval == 0 and i//self.transfer_interval == 1:
        print('transfer start')
        self.t_lock.release()

      
      if i%self.transfer_interval == 0:
        recieved_pops = None
        try:
            while True:
              if recieved_pops is None:
                recieved_pops = list(self.shared_queue.get(block=True))
              else:
                recieved_pops += list(self.shared_queue.get(block=False))
            
        except queue.Empty:
          print('Queue is empty now')
        print('recieved_pops: ', len(recieved_pops))
        self.pop = total_selection_pop(np.concatenate((self.pop, recieved_pops)), self.psize)

      offsprings = total_crossover(self.pop)

      for j in range(self.psize): offsprings[j].mutation(1/self.dims)

      # Fitness Calculation
      cfitness = np.zeros(self.psize)
      for j in range(self.psize): 
        cfitness[j] = offsprings[j].fitness_calc(self.problem)


      self.pop, self.fitness = total_selection(np.concatenate((self.pop, offsprings)),
                                          np.concatenate((self.fitness, cfitness)), self.psize)

      self.fitness_hist[i, :] = self.fitness

      if self.fitness[0] > self.bestfitness:
          self.bestfitness = self.fitness[0]

      print('Generation %d best fitness = %f' % (i, self.bestfitness))

      self.list_lock.acquire()
      self.shared_array[:] = Chromosome.genes_to_list(self.pop)
      self.list_lock.release()

      self.fitness_time[i] = time() - start

      print('Shared Array is now available')

    self.return_list.append([self.fitness_time, self.fitness_hist])      
    


  def run(self):

       # When target array is prepared it will be unlocked
      print ('called run method in process: %s' %self.name)
      self._ea()
      return


class TransferProcess(mp.Process):
  def __init__(self, dims, problem, mutation_strength,
              sample_size, sub_sample_size, src_models,
              shared_queue, shared_array, t_lock,
              list_lock, transfer_interval=2):
      super(TransferProcess, self).__init__()
      self.dims = dims
      self.problem = problem
      self.src_models = src_models
      self.mutation_strength = mutation_strength
      self.sample_size = sample_size
      self.sub_sample_size = sub_sample_size
      self.shared_queue = shared_queue
      self.shared_array = shared_array
      # self.shared_lock = shared_lock
      self.t_lock = t_lock
      self.list_lock = list_lock
      self.transfer_interval = transfer_interval
      self.reinitialize()
  
  def reinitialize(self):

    # self.fitness_hist = np.zeros((self.gen, self.psize))
    # self.fitness_time = np.zeros((self.gen))

    dims_s2 = len(self.src_models)+1
    self.second_specie = StrategyChromosome(dims_s2)

  def _transfer_ea(self):
    prev_samples = None
    genes_differ = None

    target_model = ProbabilisticModel(modelType='umd')

    self.list_lock.acquire()
    target_array = np.array(self.shared_array[:])
    self.list_lock.release()

    target_model.buildModel(target_array)

    _, sampled_offsprings, prev_samples = \
                        self.second_specie.fitness_calc(self.problem, self.src_models, target_model, self.sample_size,
                                          self.sub_sample_size, mutation_vec=genes_differ, prev_samples=deepcopy(prev_samples),
                                          efficient_version=True)

    self.shared_queue.put(sampled_offsprings)

    while True:
      offspring = deepcopy(self.second_specie)

      genes_differ = offspring.mutation(self.mutation_strength, 0, 1)

      target_model = ProbabilisticModel(modelType='umd')

      self.list_lock.acquire()
      target_array = np.array(self.shared_array[:])
      self.list_lock.release()

      target_model.buildModel(target_array)

      _, sampled_offsprings, prev_samples_tmp = \
                    offspring.fitness_calc(self.problem, self.src_models, target_model, self.sample_size,
                                      self.sub_sample_size, mutation_vec=genes_differ, prev_samples=deepcopy(prev_samples),
                                      efficient_version=True)

      self.shared_queue.put(sampled_offsprings)
      
      self.second_specie, self.mutation_strength, is_off_selected = selection_adoption(self.second_specie, offspring, self.mutation_strength)

      if is_off_selected:
        prev_samples = prev_samples_tmp
    # second_species_gen_num += 1
    # while True:



  def run(self):

    self.t_lock.acquire()
    print ('called run method in process: %s' %self.name)
    self._transfer_ea()
    return

def get_args():
  parser = argparse.ArgumentParser(description='CoOperative CoEvolution Transfer Optimization Algorithm for Solving Multi-location Inventory Planning with Lateral Transshipments')


  parser.add_argument('--stop_condition',  default=True, 
                      type=bool, nargs='?',
                      help="Stop after i number of iteraction if fitness didn't changed")

  parser.add_argument('--reps', default=1,
                      type=int, nargs='?',
                      help='Number of repetition')

  parser.add_argument('--delta', default=2,
                      type=int, nargs='?',
                      help='Step for switiching between transfer optimization and evolutionary operations')
  
  # parser.add_argument('--buildmodel', default=True,
  #                     type=bool, nargs='?',
  #                     help='Should we build source models?')

  parser.add_argument('--src_version', default='v1',
                      type=str, nargs='?',
                      help='What version of source models should be used?')

  parser.add_argument('--s1_psize', default=50,
                      type=int, nargs='?',
                      help='Population size for the first species?')
  
  # parser.add_argument('--s2_psize', default=20,
  #                     type=int, nargs='?',
  #                     help='Population size for the second species?')

  parser.add_argument('--sample_size', default=50,
                      type=int, nargs='?',
                      help='Number of samples generated from each AlphaChromosome?')

  parser.add_argument('--sub_sample_size', default=50,
                      type=int, nargs='?',
                      help='How many samples should we take from sample_size number of samples generated?')               
  
  # parser.add_argument('-v', dest='version', default='v1',
  #                   type=str, nargs='?',
  #                   help='What version should be executed?')

  parser.add_argument('--mutation_strength', default=1,
                  type=int, nargs='?',
                  help='The same step-size which we use in evolution strategy')
  
  parser.add_argument('--injection_type', default='elite',
                type=str, nargs='?',
                help='What method do you want to use for injection of species 2 to species 1?')

  parser.add_argument('--to_repititon_num', default=1,
              type=int, nargs='?',
              help='How many time should we repeat the transferring step in evolution strategy?')
  
  parser.add_argument('--selection_version', default='v1',
              type=str, nargs='?',
              help='What selection version should we use in evolution strategy E(1 + 1)?')

  parser.add_argument('-c', default=2,
                  type=int, nargs='?',
                  help='Parameter of E(1 + 1) algorithm selection')

  parser.add_argument('--efficient_version', default=False,
                        type=bool, nargs='?',
                        help='Efficient version of evaluation strategy version?')

  parser.add_argument('--transfer_repeat_num', default=None,
                      type=int, nargs='?',
                      help='''  Number of times transfer optimization should be run.
                       if it is None, it will be repeated in every delta iteration''')


  # parser.add_argument('-q', dest='matrix_num', default='a',
  #                     type=str, nargs='?',
  #                     help='T^0_H matrix selector for section b')

  return parser.parse_args()

def main_multi(args):

  # constants
  models_path = 'models'
  source_models_path = os.path.join(models_path, 'knapsack_source_models')
  knapsack_problem_path = 'problems/knapsack'

  dims = 1000
  psize = args.s1_psize
  mutation_strength = args.mutation_strength
  reps = args.reps
  transfer_interval = args.delta
  sub_sample_size = args.sub_sample_size
  sample_size = args.sample_size
  gen = 100

  # Loading Target Problem
  target_problem = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))

  # Loading Source Models
  src_models = Tools.load_from_file(source_models_path + '_{}'.format(args.src_version))

  main_m = mp.Manager()
  return_list = main_m.list()
  for i in range(reps):
    # Shared Variables
    m = mp.Manager()
    shared_queue = m.Queue()
    shared_array = m.list([[0 for j in range(dims)] for i in range(psize)])
    # prep_lock = m.Lock() # This lock is used for starting transfer learning
    # prep_lock.acquire()
    list_lock = m.Lock() # \\ for synchronozing read & write of the list
    # q_lock = m.Lock() # \\ for synchronozing put & get of the queue
    transfer_lock = m.Lock() # \\ will synchronize the transfer_interval for EAProcess
    transfer_lock.acquire()


    ea_process = EAProcess(dims, psize, gen, target_problem, shared_queue,
                          shared_array, transfer_lock, list_lock, return_list,
                          transfer_interval=transfer_interval)
    
    
    tr_process = TransferProcess(dims, target_problem, mutation_strength,
                                sample_size, sub_sample_size, src_models,
                                shared_queue, shared_array, transfer_lock,
                                list_lock, transfer_interval=transfer_interval) 

    ea_process.start()
    tr_process.start()

    ea_process.join()
    tr_process.terminate()
    tr_process.join()
  
  Tools.save_to_file(args.save_path, return_list[:])


if __name__ == '__main__':
  args = get_args()
  main_multi(args)
  
