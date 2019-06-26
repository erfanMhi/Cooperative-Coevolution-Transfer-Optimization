
import numpy as np
import lhsmdu
import argparse
import os

from copy import deepcopy
from time import time
from pprint import pprint
from utils.data_manipulators import *
from evolution.operators import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel
from evolution.chromosome import *


def evolutionary_algorithm(problem, dims, psize=100, gen=100, src_models=[], stop_condition=True):
  
  fitness_hist = np.zeros((gen, psize))

  init_func = lambda n: np.round(np.random.rand(n))
  pop = get_pop_init(psize, dims, init_func)

  for i in range(psize): pop[i].fitness_calc(problem)

  bestfitness = np.max(pop).fitness
  fitness = Chromosome.fitness_to_numpy(pop)
  fitness_hist[0, :] = fitness

  counter = 0 # Generation Repetition without fitness improvement counter
  for i in range(1, gen):

      # Crossover & Mutation
      offsprings = total_crossover(pop)
      for j in range(psize): offsprings[j].mutation(1/dims)
      
      # Fitness Calculation
      cfitness = np.zeros(psize)
      for j in range(psize): 
        cfitness[j] = offsprings[j].fitness_calc(problem)

      # Selection
      pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                 np.concatenate((fitness, cfitness)), psize)

      fitness_hist[i, :] = fitness
      print('Generation %d best fitness = %f' % (i, bestfitness))


      if fitness[0] > bestfitness:
          bestfitness = fitness[0]
          counter = 0
      else:
          counter += 1
      if counter == 20 and stop_condition:
          fitness_hist[i:, :] = fitness[0]
          break;
  best_sol = pop[0];
  model = ProbabilisticModel('umd')
  print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
  model.buildModel(Chromosome.genes_to_numpy(pop))
  src_models.append(model)
  return src_models, best_sol, fitness_hist


def transfer_ea(problem, dims, reps, trans, psize=50, gen=100, src_models=[]):
  # load probabilistic models

  if trans['transfer'] and (not src_models):
      raise ValueError('No probabilistic models stored for transfer optimization.')

  init_func = lambda n: np.round(np.random.rand(n))
  fitness_hist = np.zeros([reps, gen, psize])
  alpha = list()

  for rep in range(reps):
      alpha_rep = []
      pop = get_pop_init(psize, dims, init_func)
      for i in range(psize): pop[i].fitness_calc(problem)

      bestfitness = np.max(pop).fitness
      fitness = Chromosome.fitness_to_numpy(pop)
      fitness_hist[rep, 0, :] = fitness

      print('Generation 0 best fitness = %f' % bestfitness)
      for i in range(1, gen):
          if trans['transfer'] and i % trans['delta'] == 0:
              mixModel = MixtureModel(src_models)
              mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'umd')
              mixModel.EMstacking()
              mixModel.mutate()
              offsprings = mixModel.sample(psize)
              offsprings = np.array([Chromosome(offspring) for offspring in offsprings])
              alpha_rep = np.concatenate((alpha_rep, mixModel.alpha), axis=0)
              print('Mixture coefficients: %s' % np.array(mixModel.alpha))
          else:
              # Crossover & Mutation
              offsprings = total_crossover(pop)
              for j in range(psize): offsprings[j].mutation(1/dims)
              

            
          # Fitness Calculation
          cfitness = np.zeros(psize)
          for j in range(psize): 
            cfitness[j] = offsprings[j].fitness_calc(problem)

          # Selection
          pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                    np.concatenate((fitness, cfitness)), psize)

          bestfitness = fitness[0]
          fitness_hist[rep, i, :] = fitness
          print('Generation %d best fitness = %f' % (i, bestfitness))

      alpha.append(alpha_rep)

  return fitness_hist, alpha

# def transfer_cc(problem, dims, reps, trans, psize=50, gen=100, src_models=[]):

#   target_set = []
#   species = []
#   fitness_hist = np.zeros([reps, gen, psize])
  
#   ALPHA_SPECIES_PSIZE = 50

#   init_func = lambda n: np.round(np.random.rand(n))
#   decision_species = get_pop_init(psize, dims, init_func)
#   alpha_species = get_pop_init(ALPHA_SPECIES_PSIZE, len(src_models))
#   last_alpha_species = get_pop_init(ALPHA_SPECIES_PSIZE, 1)

#   species = [decision_species, alpha_species, last_alpha_species]
#   # species_index = list(range(NUM_SPECIES))
#   # last_index_added = species_index[-1]
  
#   # Init with random a representative for each species
#   representatives = [np.random.choice(species[i]) for i in range(len(species))]
#   # best_fitness_history = [None] * IMPROVMENT_LENGTH
  
#   g = 0
#   while g < gen:
#       # Initialize a container for the next generation representatives
#       next_repr = [None] * len(species)
#       for (i, pop), j in zip(enumerate(species), species_index):
#           # Vary the species individuals
#           offsprings = total_crossover(pop)
#           for j in range(len(pop)): offsprings[j].mutation(1/dims)
          
#           # Get the representatives excluding the current species
#           r = representatives[:i] + representatives[i+1:]
#           for ind in pop:
#               # Evaluate and set the individual fitness
#               ind.fitness.values = toolbox.evaluate([ind] + r, target_set)
          
#           pop, fitness = total_selection(np.concatenate((pop, offsprings)),
#                           np.concatenate((fitness, cfitness)), len(pop))
#           # Select the individuals
#           species[i] = pop # Tournament selection
#           next_repr[i] = np.max(pop)  # Best selection
          
#           g += 1
      
#       representatives = next_repr
      
      # Keep representatives fitness for stagnation detection
      # best_fitness_history.pop(0)
      # best_fitness_history.append(representatives[0].fitness.values[0])

def transfer_cc_v1(problem, dims, reps, trans,
                   s1_psize=50, s2_psize=20, gen=100,
                   sample_size=50, sub_sample_size=50,
                   src_models=[]):
  start = time()
  if trans['transfer'] and (not src_models):
    raise ValueError('No probabilistic models stored for transfer optimization.')

  delta = trans['delta']

  init_func_s1 = lambda n: np.round(np.random.rand(n))
  
  fitness_hist_s1 = np.ndarray([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
  fitness_hist_s2 = np.ndarray([reps, int(gen/delta), s2_psize], dtype=object)
  time_hist_s1 = np.zeros([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
  dims_s2 = len(src_models)+1

  best_chrom = None # Best Chromosome to inject to the first species from second species
  # Init with random a representative for each species
  # representatives = [np.random.choice(species[i]) for i in range(len(species))]
  for rep in range(reps):
      print('------------------------- Repetition {} ---------------------------'.format(rep))
      first_species = get_pop_init(s1_psize, dims, init_func_s1) # For choosing decision params
      second_species = get_pop_init_s2(s2_psize, dims_s2) # For choosing alpha params
      for i in range(s1_psize):
        first_species[i].fitness_calc(problem)
        time_hist_s1[rep, 0, i] = time()

      bestfitness = np.max(first_species).fitness
      fitness = Chromosome.fitness_to_numpy(first_species)
      s2_fitness = None
      fitness_hist_s1[rep, 0, :] = first_species
      print('Generation %d best fitness of first species = %f' % (0, bestfitness))
      for g in range(1, gen):
        # Initialize a container for the next generation representatives
        if trans['transfer'] and g % delta == 0:
            if g/delta != 1:
              offsprings = total_crossover_s2(second_species)
              for j in range(s2_psize): offsprings[j].mutation(1/dims_s2)
            else:
              offsprings = second_species

            target_model = ProbabilisticModel(modelType='umd')
            target_model.buildModel(Chromosome.genes_to_numpy(first_species))

            s2_cfitness = np.zeros(s2_psize)
            best_chrom = Chromosome(dims)
            print('start fitness o ina')
            for i in range(s2_psize):
              
              s2_cfitness[i], best_offspring = offsprings[i].fitness_calc(problem, src_models,
                                                                          target_model, sample_size,
                                                                          sub_sample_size)
              if best_chrom < best_offspring: # Saving best Chromosome for future injection
                best_chrom = best_offspring
            print('end fitness o ina')
            if g/delta != 1:
              second_species, s2_fitness = total_selection(np.concatenate((second_species, offsprings)),
                                np.concatenate((s2_fitness, s2_cfitness)), s2_psize)
            else:
              second_species, s2_fitness = total_selection(offsprings, s2_cfitness, s2_psize)
            # Replacing the best chromosome found by sampling from second species with the worst chromosome of first species
            first_species[-1] = best_chrom 

            best_fitness_s2 = s2_fitness[0]
            fitness_hist_s2[rep, int(g/delta)-1, :] = second_species
            print('Generation %d: Best Fitness of Second Species: %s' % (g, best_fitness_s2))
            print('Best Alpha generation {}: best fitness of second species = {}'.format(g, second_species[0].genes))
        else:
            # Crossover & Mutation
            offsprings = total_crossover(first_species)
            for j in range(s1_psize): offsprings[j].mutation(1/dims)
              
            # Fitness Calculation
            cfitness = np.zeros(s1_psize)
            for j in range(s1_psize): 
              cfitness[j] = offsprings[j].fitness_calc(problem)
              time_hist_s1[rep, int(np.ceil(g/delta*(delta-1))), i] = time()

            # Selection
            first_species, fitness = total_selection(np.concatenate((first_species, offsprings)),
                                      np.concatenate((fitness, cfitness)), s1_psize)

            bestfitness = fitness[0]
            fitness_hist_s1[rep, int(np.ceil(g/delta*(delta-1))), :] = first_species
            print('Generation %d best fitness of first species= %f' % (g, bestfitness))
  print('Finished')
  return fitness_hist_s1, fitness_hist_s2, (time_hist_s1 - start)
      

def transfer_cc_v2(problem, dims, reps, trans,
                   s1_psize=50, s2_psize=1, gen=100,
                   sample_size=50, sub_sample_size=50,
                   mutation_strength=1, src_models=[]):
  start = time()
  if trans['transfer'] and (not src_models):
    raise ValueError('No probabilistic models stored for transfer optimization.')

  delta = trans['delta']

  init_func_s1 = lambda n: np.round(np.random.rand(n))
  
  fitness_hist_s1 = np.ndarray([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
  fitness_hist_s2 = np.ndarray([reps, int(gen/delta), s2_psize], dtype=object)
  time_hist_s1 = np.zeros([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
  mutation_strength_hist = np.zeros([reps, int(gen/delta), s2_psize])
  
  dims_s2 = len(src_models)+1

  best_chrom = None # Best Chromosome to inject to the first species from second species

  for rep in range(reps):
      print('------------------------- Repetition {} ---------------------------'.format(rep))
      first_species = get_pop_init(s1_psize, dims, init_func_s1) # For choosing decision params
      # second_species = get_pop_init_s2(s2_psize, dims_s2) # For choosing alpha params
      second_specie = StrategyChromosome(dims_s2)
      for i in range(s1_psize):
        first_species[i].fitness_calc(problem)
        time_hist_s1[rep, 0, i] = time()

      bestfitness = np.max(first_species).fitness
      fitness = Chromosome.fitness_to_numpy(first_species)
      s2_fitness = None
      fitness_hist_s1[rep, 0, :] = first_species
      print('Generation %d best fitness of first species = %f' % (0, bestfitness))
      for g in range(1, gen):
        # Initialize a container for the next generation representatives
        if trans['transfer'] and g % delta == 0:
          ################# Add Evolution Strategy #####################
            
            if g/delta != 1:
              print('Test Mutation: ')
              offspring = deepcopy(second_specie)
              print ('Offspring genes before mutation: {}'.format(offspring.genes))
              offspring.mutation(mutation_strength, 0, 1)
              print ('Offspring genes after mutation: {}'.format(offspring.genes))
              print('Test Finished')
              # offsprings = total_crossover_s2(second_species)
              # for j in range(s2_psize): offsprings[j].mutation(1/dims_s2)
            else:
              offspring = second_specie

            target_model = ProbabilisticModel(modelType='umd')
            target_model.buildModel(Chromosome.genes_to_numpy(first_species))

            print('start fitness o ina')
            # for i in range(s2_psize):
              
            _, best_chrom = offspring.fitness_calc(problem, src_models, target_model,
                                                          sample_size, sub_sample_size)
              # if best_chrom < best_offspring: # Saving best Chromosome for future injection
              #   best_chrom = best_offspring
            print('end fitness o ina')
            if g/delta != 1:
              second_specie, mutation_strength = selection_adoption(second_specie, offspring, mutation_strength)

            # Replacing the best chromosome found by sampling from second species with the worst chromosome of first species
            first_species[-1] = best_chrom 

            fitness_hist_s2[rep, int(g/delta)-1, :] = second_specie
            mutation_strength_hist[rep, int(g/delta)-1, :]  = mutation_strength
            print('Generation %d: Best Fitness of Second Species: %s' % (g, second_specie.fitness))
            print('Best Alpha generation {}: best fitness of second species = {}'.format(g, second_specie.genes))
        else:
            # Crossover & Mutation
            offsprings = total_crossover(first_species)
            for j in range(s1_psize): offsprings[j].mutation(1/dims)
              
            # Fitness Calculation
            cfitness = np.zeros(s1_psize)
            for j in range(s1_psize): 
              cfitness[j] = offsprings[j].fitness_calc(problem)
              time_hist_s1[rep, int(np.ceil(g/delta*(delta-1))), i] = time()

            # Selection
            first_species, fitness = total_selection(np.concatenate((first_species, offsprings)),
                                      np.concatenate((fitness, cfitness)), s1_psize)

            bestfitness = fitness[0]
            fitness_hist_s1[rep, int(np.ceil(g/delta*(delta-1))), :] = first_species
            print('Generation %d best fitness of first species= %f' % (g, bestfitness))
  print('Finished')
  return fitness_hist_s1, fitness_hist_s2, mutation_strength_hist, (time_hist_s1 - start)


def get_args():
  parser = argparse.ArgumentParser(description='CoOperative CoEvolution Transfer Optimization Algorithm for Solving Multi-location Inventory Planning with Lateral Transshipments')

  parser.add_argument('-t', dest='tsamples', default=10,
                      type=int, nargs='?',
                      help='Number of testing data samples')

  parser.add_argument('--stop_condition',  default=True, 
                      type=bool, nargs='?',
                      help="Stop after i number of iteraction if fitness didn't changed")
  parser.add_argument('--reps', default=1,
                      type=int, nargs='?',
                      help='Number of repetition')
  parser.add_argument('--transfer', default=True,
                      type=bool, nargs='?',
                      help='Should we use transfer optimization?')
  parser.add_argument('--delta', default=2,
                      type=int, nargs='?',
                      help='Step for switiching between transfer optimization and evolutionary operations')
  parser.add_argument('--buildmodel', default=True,
                      type=bool, nargs='?',
                      help='Should we build source models?')

  parser.add_argument('--src_version', default='v1',
                      type=str, nargs='?',
                      help='What version of source models should be used?')

  parser.add_argument('--s1_psize', default=50,
                      type=int, nargs='?',
                      help='Population size for the first species?')
  
  parser.add_argument('--s2_psize', default=20,
                      type=int, nargs='?',
                      help='Population size for the second species?')

  parser.add_argument('--sample_size', default=50,
                      type=int, nargs='?',
                      help='Number of samples generated from each AlphaChromosome?')

  parser.add_argument('--sub_sample_size', default=50,
                      type=int, nargs='?',
                      help='How many samples should we take from sample_size number of samples generated?')               
  
  parser.add_argument('-v', dest='version', default='v1',
                    type=str, nargs='?',
                    help='What version should be executed?')

  parser.add_argument('--mutation_strength', default=1,
                  type=int, nargs='?',
                  help='The same step-size which we use in evolution strategy')
  

  # parser.add_argument('-q', dest='matrix_num', default='a',
  #                     type=str, nargs='?',
  #                     help='T^0_H matrix selector for section b')

  return parser.parse_args()

def check_args(args):
  if args.sample_size < args.sub_sample_size:
    raise ValueError('sub_sample_size has greater value than sample_size')


def main(args=False):

  ################# Preconfigurations ##############
  if args is False:
    args = get_args()

  check_args(args)
  models_path = 'models'
  source_models_path = os.path.join(models_path, 'knapsack_source_models')
  knapsack_problem_path = 'problems/knapsack'
  src_models = None

  target_problem = None
  src_problems = []
  # Loading Problems Data
  if args.src_version == 'v1':
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    KP_wc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_rk'))
    KP_sc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_rk'))
    KP_uc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_rk'))
    src_problems = [KP_uc_rk, KP_sc_rk, KP_wc_rk, KP_sc_ak]
    target_problem = KP_uc_ak
  elif args.src_version == 'v2':
    src_problem_set = [(40, 'KP_wc_ak'), (320, 'KP_wc_rk'), (320, 'KP_sc_rk'), (320, 'KP_uc_rk')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))

    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_rk'))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_sc_ak
  else:
    print('Source problems version is not correct {}'.format(args.src_version))

  print("All source problems & target problems are loaded: length = {}".format(len(src_problems)))
  ################# Evolutionary Algorithm ###############
  src_models = []


  if args.buildmodel:
    # build source probabilistic models

    now = time()
    for problem in src_problems:
      src_models, _, _ = evolutionary_algorithm(problem, 1000, src_models=src_models, stop_condition=args.stop_condition)
      # src_models, _, _ = evolutionary_algorithm(KP_sc_rk, 1000, src_models=src_models, stop_condition=args.stop_condition)
      # src_models, _, _ = evolutionary_algorithm(KP_wc_rk, 1000, src_models=src_models, stop_condition=args.stop_condition)
      # src_models, _, _ = evolutionary_algorithm(KP_sc_ak, 1000, src_models=src_models, stop_condition=args.stop_condition)

    Tools.save_to_file(source_models_path + '_{}'.format(args.src_version), src_models)
    print('Building models took {} minutes'.format(str((time()-now)/60)))
  else:
    try:
      src_models = Tools.load_from_file(source_models_path + '_{}'.format(args.src_version))
    except FileNotFoundError:
      print('Source models not exist in the {} path'.format(source_models_path))

  #AMTEA solving KP_wc_ak
  reps = args.reps

  trans = {}
  trans['transfer'] = args.transfer
  trans['delta'] = args.delta
  if args.version == 'v1':
    return transfer_cc_v1(target_problem, 1000, reps, trans, s1_psize=args.s1_psize,
                           s2_psize=args.s2_psize, gen=100, sample_size=args.sample_size,
                           sub_sample_size=args.sub_sample_size, src_models=src_models)
  elif args.version == 'v2':
      return transfer_cc_v2(target_problem, 1000, reps, trans, s1_psize=args.s1_psize,
                           s2_psize=1, gen=100, sample_size=args.sample_size,
                           sub_sample_size=args.sub_sample_size, src_models=src_models, 
                           mutation_strength=args.mutation_strength)                 
  elif args.version == 'to':
    return transfer_ea(target_problem, 1000, reps, trans, psize=args.s1_psize, src_models=src_models)
  elif args.version == 'ea':
    return evolutionary_algorithm(target_problem, 1000, src_models=src_models, stop_condition=args.stop_condition)
  else:
    raise ValueError('Version which you entered is not right')


  # tsamples = args.tsamples # number of testing data samples
  # mc_samples = 30

  # n = 4
  # D_mean = 500* lhsmdu.sample(tsamples, n).T + 100 # generate problem dataset with demands ranging from 100 to 600
  # X_data = D_mean.T
  # c = 10
  # h = np.ones((1,4))
  # p = 50*h




if __name__ == '__main__':
  main()