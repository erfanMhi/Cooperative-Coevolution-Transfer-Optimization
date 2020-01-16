
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
from utils.double_pole_physics import PoledCart
from utils.neural_network import Net


class EAProcess(mp.Process):
  def __init__(self, dims, psize, gen, shared_queue, 
              shared_array, t_lock, list_lock, return_list,
              solution_found, transfer_interval=2, muc=10,
              mum=10, s_len=0.8):
      super(EAProcess, self).__init__()
      self.dims = dims
      self.psize = psize
      self.gen = gen
      self.shared_queue = shared_queue
      self.shared_array = shared_array
      # self.shared_lock = shared_lock
      self.t_lock = t_lock
      self.list_lock = list_lock
      self.transfer_interval = transfer_interval
      self.muc = muc
      self.mum = mum
      self.s_len = s_len
      self.reinitialize()
      self.return_list = return_list
      self.solution_found = solution_found

  def reinitialize(self):

    self.fitness_hist = np.zeros((self.gen, self.psize))
    self.fitness_time = np.zeros((self.gen))

    self.cart = PoledCart(self.s_len)


    n_input = 6
    n_hidden = 10
    n_output = 1
    self.net = Net(n_input, n_hidden, n_output)
    self.n_vars = self.net.nVariables

    init_func = lambda n: 12 * np.random.rand(n) - 6
    self.pop = get_pop_init(self.psize, self.n_vars, init_func, p_type='double_pole')

  def _ea(self):
    
    start = time()

    func_eval_num = 0

    cfitness = np.zeros(self.psize)
    for j in range(self.psize): 
      cfitness[j] = self.pop[j].fitness_calc(self.net, self.cart, self.s_len)
      if not self.solution_found.value:
        func_eval_num += 1
      if cfitness[j] - 2000 > -0.0001:
        self.solution_found.value = True



    self.bestfitness = np.max(self.pop).fitness
    self.fitness = cfitness
    self.fitness_hist[0, :] = self.fitness

    self.fitness_time[0] = start - time()


    
    for i in range(1, self.gen):
      start = time()

      # Notifying transfer algorithm to start its work 
      if i%self.transfer_interval == 0 and i//self.transfer_interval == 1:
        print('transfer start')
        self.t_lock.release()

      print('EA self.solution_found.value: ', self.solution_found.value)
      ################ Recieving the solutions which have been found by transfer algorithm ################
      if i%self.transfer_interval == 0:
        recieved_pops = None
        try:
            while True:
              if recieved_pops is None:
                recieved_pops = list(self.shared_queue.get(block=True, timeout=300))
              else:
                recieved_pops += list(self.shared_queue.get(block=False))
            
        except queue.Empty:
          print('Queue is empty now')

        if recieved_pops is not None:
          print('recieved_pops: ', len(recieved_pops))
          self.pop = total_selection_pop(np.concatenate((self.pop, recieved_pops)), self.psize)


      ################## Crossover & Mutation ##################
      randlist = np.random.permutation(self.psize)
      offsprings = np.ndarray(self.psize, dtype=object)


      for j in range(0, self.psize, 2):
            offsprings[j] = ChromosomePole(self.n_vars)
            offsprings[j+1] = ChromosomePole(self.n_vars)
            p1 = randlist[j]
            p2 = randlist[j+1]
            offsprings[j].genes, offsprings[j+1].genes = \
              sbx_crossover(self.pop[p1], self.pop[p2], self.muc, self.n_vars)
            offsprings[j].mutation(self.mum, self.n_vars)
            offsprings[j+1].mutation(self.mum, self.n_vars)

      ############### Fitness Calculation ###############
      cfitness = np.zeros(self.psize)
      for j in range(self.psize): 
        cfitness[j] = offsprings[j].fitness_calc(self.net, self.cart, self.s_len)
        if not self.solution_found.value:
          func_eval_num += 1
        if cfitness[j] - 2000 > -0.0001:
          self.solution_found.value = True



      self.pop, self.fitness = total_selection(np.concatenate((self.pop, offsprings)),
                                          np.concatenate((self.fitness, cfitness)), self.psize)

      self.fitness_hist[i, :] = self.fitness

      if self.fitness[0] > self.bestfitness:
          self.bestfitness = self.fitness[0]

      print('Generation %d best fitness = %f' % (i, self.bestfitness))

      if self.solution_found.value:
        break

      self.list_lock.acquire()
      self.shared_array[:] = Chromosome.genes_to_list(self.pop)
      self.list_lock.release()

      self.fitness_time[i] = time() - start

      print('Shared Array is now available')

    self.return_list.append([self.fitness_time, self.fitness_hist, func_eval_num])
    print('Evolutionary Process Has Been Finished')
    


  def run(self):

       # When target array is prepared it will be unlocked
      print ('called run method in process: %s' %self.name)
      self._ea()
      return


class TransferProcess(mp.Process):
  def __init__(self, dims, mutation_strength,
              sample_size, sub_sample_size, src_models,
              shared_queue, shared_array, t_lock,
              list_lock, solution_found, return_list,
              transfer_interval=2, muc=10, mum=10, s_len=0.8):
      super(TransferProcess, self).__init__()
      self.dims = dims
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
      self.muc = muc
      self.mum = mum
      self.s_len = s_len
      self.solution_found = solution_found
      self.return_list = return_list
      self.reinitialize()
  
  def reinitialize(self):

    # self.fitness_hist = np.zeros((self.gen, self.psize))
    # self.fitness_time = np.zeros((self.gen))

    dims_s2 = len(self.src_models)+1
    self.second_specie = StrategyChromosome(dims_s2)

    self.cart = PoledCart(self.s_len)

    n_input = 6
    n_hidden = 10
    n_output = 1
    self.net = Net(n_input, n_hidden, n_output)
    self.n_vars = self.net.nVariables

    # init_func = lambda n: 12 * np.random.rand(n) - 6
    # self.pop = get_pop_init(self.psize, self.n_vars, init_func, p_type='double_pole')


  def _transfer_ea(self):
    prev_samples = None
    genes_differ = None

    target_model = ProbabilisticModel(modelType='umd')

    func_eval_num = 0

    self.list_lock.acquire()
    target_array = np.array(self.shared_array[:])
    self.list_lock.release()

    target_model.buildModel(target_array)

    fitness, first_specie_offsprings, eval_num = self.second_specie. \
                            fitness_calc_pole(self.net, self.cart, self.s_len, self.src_models,
                                            target_model, self.sample_size, solution_found=self.solution_found,
                                            mutation_vec=None, prev_samples=None, efficient_version=False)
    
    func_eval_num += eval_num

    self.shared_queue.put(first_specie_offsprings)

    while True:
      offspring = deepcopy(self.second_specie)

      genes_differ = offspring.mutation(self.mutation_strength, 0, 1)

      target_model = ProbabilisticModel(modelType='umd')

      self.list_lock.acquire()
      target_array = np.array(self.shared_array[:])
      self.list_lock.release()

      target_model.buildModel(target_array)

      fitness, first_specie_offsprings, eval_num = self.second_specie. \
                              fitness_calc_pole(self.net, self.cart, self.s_len, self.src_models,
                                              target_model, self.sample_size, mutation_vec=None, solution_found=self.solution_found,
                                              prev_samples=None, efficient_version=False)

      func_eval_num += eval_num
      if self.solution_found.value:
        break
      self.shared_queue.put(first_specie_offsprings)
      
      self.second_specie, self.mutation_strength, is_off_selected = selection_adoption(self.second_specie, offspring, self.mutation_strength)

    # second_species_gen_num += 1
    # while True:
    self.return_list.append([func_eval_num])

    print('Transfer Process is finished') 



  def run(self):

    self.t_lock.acquire()
    print ('called run method in process: %s' %self.name)
    print('value of solution found: {}'.format(self.solution_found.value))
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


  parser.add_argument('--save_path', default='',
                    type=str, nargs='?',
                    help='Where should we save the output?')

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

  s_poles_length = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775]

  # constants
  models_path = 'models'
  source_models_path = 'models/pole_models/src_model'
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
  target_pole_len = 0.8

  # Loading Source Models
  src_models = []
  for i, s_len in enumerate(s_poles_length):
    if os.path.isfile(source_models_path + '_{}.pkl'.format(s_len)):
      src_model = Tools.load_from_file(source_models_path + '_{}'.format(s_len))
      src_models.append(src_model)
      print('---------------------- {} source model loaded---------------------'.format(i+1))

  
  main_m = mp.Manager()
  return_list = main_m.list()
  return_list_transfer = main_m.list()
  sol_found = []
  for i in range(reps):
    print('#################### Repetition {} ####################'.format(i+1))
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
    solution_found = m.Value('i', False)
    ea_process = EAProcess(dims, psize, gen, shared_queue,
                          shared_array, transfer_lock, list_lock, return_list,
                          solution_found, transfer_interval=transfer_interval,
                          muc=10, mum=10, s_len=target_pole_len)
    
    
    tr_process = TransferProcess(dims, mutation_strength,
                                sample_size, sub_sample_size, src_models,
                                shared_queue, shared_array, transfer_lock,
                                list_lock, solution_found, return_list_transfer,
                                transfer_interval=transfer_interval, muc=10, mum=10,
                                s_len=target_pole_len) 
    
    ea_process.start()
    tr_process.start()

    ea_process.join()
    tr_process.join()
    sol_found.append(solution_found.value)

  save_path = args.save_path + '_outcome_transfer_pole'
  Tools.save_to_file(save_path, return_list[:])
  save_path = args.save_path + 'outcome_ea_pole'
  Tools.save_to_file(save_path, return_list_transfer[:])
  save_path = args.save_path + 'soluton_found_out'
  Tools.save_to_file(save_path, sol_found)
  func_eval = 0
  sol_found_num = 0
  for i in range(reps):
    if sol_found[i] and (return_list[i][2] + return_list_transfer[i][0] <= 10000):
      sol_found_num += 1
      func_eval += return_list[i][2] + return_list_transfer[i][0]


  print('Function Evaluations: {}'.format(func_eval/sol_found_num))
  print('Solutions found: {}/{}'.format(sol_found_num, reps))
  
  # Tools.save_to_file(args.save_path, return_list[:])


if __name__ == '__main__':
  args = get_args()
  main_multi(args)
  
