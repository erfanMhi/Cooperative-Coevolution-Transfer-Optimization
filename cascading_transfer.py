
import numpy as np
# import lhsmdu
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
from utils.double_pole_physics import PoledCart
from utils.neural_network import Net




def evolutionary_algorithm(sLen, psize=100, gen=100, muc=10, mum=10, stop_condition=True, create_model=True):
  
  src_model = None

  fitness_hist = np.zeros((gen, psize))
  fitness_time = np.zeros((gen))

  cart = PoledCart(sLen)


  n_input = 6
  n_hidden = 10
  n_output = 1
  net = Net(n_input, n_hidden, n_output)
  n_vars = net.nVariables

  init_func = lambda n: 12 * np.random.rand(n) - 6
  pop = get_pop_init(psize, n_vars, init_func, p_type='double_pole')
  start = time()

  for j in range(psize):
    pop[j].fitness_calc(net, cart, sLen)

  
  bestfitness = np.max(pop).fitness
  fitness = Chromosome.fitness_to_numpy(pop)
  fitness_hist[0, :] = fitness

  fitness_time[0] = start - time()
  counter = 0 # Generation Repetition without fitness improvement counter
  for i in range(1, gen):
      start = time()
      randlist = np.random.permutation(psize)
      offsprings = np.ndarray(psize, dtype=object)

      # Crossover & Mutation
      for j in range(0, psize, 2):
            offsprings[j] = ChromosomePole(n_vars)
            offsprings[j+1] = ChromosomePole(n_vars)
            p1 = randlist[j]
            p2 = randlist[j+1]
            offsprings[j].genes, offsprings[j+1].genes = sbx_crossover(pop[p1], pop[p2], muc, n_vars)
            offsprings[j].mutation(mum, n_vars)
            offsprings[j+1].mutation(mum, n_vars)

      
      # Fitness Calculation
      cfitness = np.zeros(psize)
      for j in range(psize):
          # print(pop[j].genes)
          cfitness[j] = offsprings[j].fitness_calc(net, cart, sLen)

      # Selection
      pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                 np.concatenate((fitness, cfitness)), psize)

      fitness_hist[i, :] = fitness

      if fitness[0] > bestfitness:
          bestfitness = fitness[0]
          counter = 0
      else:
          counter += 1

      print('Generation %d best fitness = %f' % (i, bestfitness))
      if fitness[0] - 2000 > -0.0001 and stop_condition:
          print('Solution found!')
          fitness_hist[i:, :] = fitness[0]
          break

      fitness_time[i] = time() - start

  best_sol = pop[0]
  if create_model and fitness_hist[-1, 0] - 2000 > -0.0001:
    model = ProbabilisticModel('mvarnorm')
    print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
    model.buildModel(Chromosome.genes_to_numpy(pop))
    print("Model built successfully!")
    src_model = model
  elif not create_model:
    print("Evolutionary algorithm didn't reach the criteria!")
    # src_models.append(model)
  
  return src_model, best_sol, fitness_hist, fitness_time


def transfer_ea(sLen, src_models, psize=100, gen=100,
                 muc=10, mum=10, reps=1, delta=2,
                 build_model=True):

  

  if not src_models:
      raise ValueError('No probabilistic models stored for transfer optimization.')

  init_func = lambda n: 12 * np.random.rand(n) - 6
  
  fitness_hist = np.zeros([reps, gen, psize])
  fitness_time = np.zeros((reps, gen,))
  alpha = list()

  cart = PoledCart(sLen)

  n_input = 6
  n_hidden = 10
  n_output = 1
  net = Net(n_input, n_hidden, n_output)
  n_vars = net.nVariables

  pop = None
  func_eval_nums = []
  sols_found = []
  for rep in range(reps):
      alpha_rep = []

      pop = get_pop_init(psize, n_vars, init_func, p_type='double_pole')
      start = time()
      for j in range(psize):
        pop[j].fitness_calc(net, cart, sLen)

      bestfitness = np.max(pop).fitness
      fitness = Chromosome.fitness_to_numpy(pop)
      fitness_hist[rep, 0, :] = fitness
      fitness_time[rep, 0] = time() - start
      print('Generation 0 best fitness = %f' % bestfitness)
      solution_found = None
      func_eval_num = None
      for i in range(1, gen):
          start = time()
          if i % delta == 0:
              mixModel = MixtureModel(src_models)
              mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'mvarnorm')
              mixModel.EMstacking()
              mixModel.mutate()
              offsprings = mixModel.sample(psize)
              offsprings = np.array([ChromosomePole(offspring) for offspring in offsprings])
              alpha_rep = np.concatenate((alpha_rep, mixModel.alpha), axis=0)
              print('Mixture coefficients: %s' % np.array(mixModel.alpha))
          else:
              # Crossover & Mutation
              randlist = np.random.permutation(psize)
              offsprings = np.ndarray(psize, dtype=object)
              for j in range(0, psize, 2):
                  offsprings[j] = ChromosomePole(n_vars)
                  offsprings[j+1] = ChromosomePole(n_vars)
                  p1 = randlist[j]
                  p2 = randlist[j+1]
                  offsprings[j].genes, offsprings[j+1].genes = sbx_crossover(pop[p1], pop[p2], muc, n_vars)
                  offsprings[j].mutation(mum, n_vars)
                  offsprings[j+1].mutation(mum, n_vars)
              

            
          # Fitness Calculation
          cfitness = np.zeros(psize)
          for j in range(psize): 
            cfitness[j] = offsprings[j].fitness_calc(net, cart, sLen)
            if cfitness[j] - 2000 > -0.0001:
              func_eval_num = (i*psize + j+1)
              solution_found = True


          # Selection
          pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                    np.concatenate((fitness, cfitness)), psize)

          
          fitness_hist[rep, i, :] = fitness
          fitness_time[rep, i] = time() - start

          if fitness[0] > bestfitness:
            bestfitness = fitness[0]

          
          print('Generation %d best fitness = %f' % (i, bestfitness))
          print(fitness[0])
          if fitness[0] - 2000 > -0.0001:
              print('Solution found!')
              fitness_hist[rep, i:, :] = fitness[0]
              func_eval_nums.append(func_eval_num)
              sols_found.append(solution_found)
              break

      alpha.append(alpha_rep)
  
  model = None
  print('fitness_hist: ', fitness_hist[0, -1, 0])
  if build_model and fitness_hist[0, -1, 0] - 2000 > -0.0001:
    model = ProbabilisticModel('mvarnorm')
    print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
    model.buildModel(Chromosome.genes_to_numpy(pop))
    print("Model built successfully!")
    # src_model = model
  else:
    print("Evolutionary algorithm didn't reach the criteria!")
  
  if build_model:
    return fitness_hist[0, ...], alpha, fitness_time[0, ...], model
  else:
    return fitness_hist, alpha, fitness_time, sols_found, func_eval_nums

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
                   injection_type='elite', src_models=[]):

  if trans['transfer'] and (not src_models):
    raise ValueError('No probabilistic models stored for transfer optimization.')

  delta = trans['delta']

  init_func_s1 = lambda n: np.round(np.random.rand(n))
  
  fitness_hist_s1 = np.ndarray([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
  fitness_hist_s2 = np.ndarray([reps, int(gen/delta), s2_psize], dtype=object)
  time_hist_s1 = np.zeros([reps, int((gen/delta * (delta-1)) + gen%delta) + 1], dtype=object)
  dims_s2 = len(src_models)+1

  best_chrom = None # Best Chromosome to inject to the first species from second species
  # Init with random a representative for each species
  # representatives = [np.random.choice(species[i]) for i in range(len(species))]
  for rep in range(reps):
      print('------------------------- Repetition {} ---------------------------'.format(rep))
      first_species = get_pop_init(s1_psize, dims, init_func_s1) # For choosing decision params
      second_species = get_pop_init_s2(s2_psize, dims_s2) # For choosing alpha params
      start = time()
      
      for i in range(s1_psize):
        first_species[i].fitness_calc(problem)

      bestfitness = np.max(first_species).fitness
      fitness = Chromosome.fitness_to_numpy(first_species)
      s2_fitness = None
      fitness_hist_s1[rep, 0, :] = first_species
      time_hist_s1[rep, 0] = time() - start
      print('Generation %d best fitness of first species = %f' % (0, bestfitness))
      start = time()
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
            sampled_offsprings = np.ndarray(s2_psize*sub_sample_size, dtype=object)

            for i in range(s2_psize):
              s2_cfitness[i], offsprings = offsprings[i].fitness_calc(problem, src_models,
                                                                      target_model, sample_size,
                                                                      sub_sample_size)
              sampled_offsprings[:i*sub_sample_size] = offsprings

            # Injecting elite chromosomes to first species
            if injection_type == 'elite':
              first_species[-1] == np.max(sampled_offsprings)
            elif injection_type == 'full':
              first_species = total_selection_pop(np.concatenate((first_species, sampled_offsprings)), s1_psize)

            # Selecting elite chromosome from second species
            if g/delta != 1:
              second_species, s2_fitness = total_selection(np.concatenate((second_species, offsprings)),
                                np.concatenate((s2_fitness, s2_cfitness)), s2_psize)
            else:
              second_species, s2_fitness = total_selection(offsprings, s2_cfitness, s2_psize)
            # Replacing the best chromosome found by sampling from second species with the worst chromosome of first species
            # first_species[-1] = best_chrom 

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

            # Selection
            first_species, fitness = total_selection(np.concatenate((first_species, offsprings)),
                                      np.concatenate((fitness, cfitness)), s1_psize)

            bestfitness = fitness[0]
            fitness_hist_s1[rep, int(np.ceil(g/delta*(delta-1))), :] = first_species
            time_hist_s1[rep, int(np.ceil(g/delta*(delta-1)))] = time() - start
            print('Generation %d best fitness of first species= %f' % (g, bestfitness))
            start = time()
  print('Finished')
  return fitness_hist_s1, fitness_hist_s2, time_hist_s1
      

def transfer_cc_v2(problem, dims, reps, trans,
                   s1_psize=50, s2_psize=1, gen=100,
                   sample_size=50, sub_sample_size=50,
                   mutation_strength=1, injection_type='full', 
                   to_repititon_num=1, selection_version='v1', 
                   c=2, src_models=[], efficient_version=False, 
                   transfer_repeat_num=None):


  
  if trans['transfer'] and (not src_models):
    raise ValueError('No probabilistic models stored for transfer optimization.')

  delta = trans['delta']

  init_func_s1 = lambda n: np.round(np.random.rand(n))
  
  if transfer_repeat_num is None:
    transfer_repeat_num = float('inf') # repeat in all iterations
    fitness_hist_s1 = np.ndarray([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
    fitness_hist_s2 = np.ndarray([reps, int(gen/delta), s2_psize], dtype=object)
    time_hist_s1 = np.zeros([reps, int((gen/delta * (delta-1)) + gen%delta) + 1, s1_psize], dtype=object)
    mutation_strength_hist = np.zeros([reps, int(gen/delta), s2_psize])
  else:
    fitness_hist_s1 = np.ndarray([reps, gen-transfer_repeat_num, s1_psize], dtype=object)
    fitness_hist_s2 = np.ndarray([reps, transfer_repeat_num, s2_psize], dtype=object)
    time_hist_s1 = np.zeros([reps, gen-transfer_repeat_num, s1_psize], dtype=object)
    mutation_strength_hist = np.zeros([reps, transfer_repeat_num, s2_psize])
  dims_s2 = len(src_models)+1

  best_chrom = None # Best Chromosome to inject to the first species from second species
  
  ms_value = mutation_strength
  for rep in range(reps):
      print('------------------------- Repetition {} ---------------------------'.format(rep))
      first_species = get_pop_init(s1_psize, dims, init_func_s1) # For choosing decision params
      # second_species = get_pop_init_s2(s2_psize, dims_s2) # For choosing alpha params
      start = time()
      second_specie = StrategyChromosome(dims_s2)
      for i in range(s1_psize):
        first_species[i].fitness_calc(problem)

      second_species_gen_num = 0 # used in selection version 2 for calculating the g
      second_species_gen_success_num = 0 # used in selection version 2 for calculating the g
      ea_counter = 0
      mutation_strength = ms_value
      bestfitness = np.max(first_species).fitness
      fitness = Chromosome.fitness_to_numpy(first_species)
      s2_fitness = None
      fitness_hist_s1[rep, 0, :] = first_species
      time_hist_s1[rep, 0] = time() - start
      print('Generation %d best fitness of first species = %f' % (0, bestfitness))
      start = time()
      prev_samples = None
      genes_differ = None
      for g in range(1, gen):
        # Initialize a container for the next generation representatives
        if trans['transfer'] and (g % delta == 0) and (g/delta <= transfer_repeat_num):
          ################# Add Evolution Strategy #####################
            for tg in range(to_repititon_num):
              if g/delta != 1 or tg != 0:
                print('Test Mutation: ')
                offspring = deepcopy(second_specie)
                second_species_gen_num += 1
                print ('Offspring genes before mutation: {}'.format(offspring.genes/np.sum(offspring.genes)*50))
                genes_differ = offspring.mutation(mutation_strength, 0, 1)
                print ('Offspring genes after mutation: {}'.format(offspring.genes/np.sum(offspring.genes)*50))
                print ('Genes Difference: {}'.format(genes_differ))
                print ('Genes Difference sum: {}'.format(np.sum(genes_differ)))
                print('Test Finished')
                # offsprings = total_crossover_s2(second_species)
                # for j in range(s2_psize): offsprings[j].mutation(1/dims_s2)
              else:
                offspring = second_specie

              target_model = ProbabilisticModel(modelType='umd')
              target_model.buildModel(Chromosome.genes_to_numpy(first_species))

              # print('start fitness o ina')
              # for i in range(s2_psize):

              if efficient_version:
                _, sampled_offsprings, prev_samples_tmp = offspring.fitness_calc(problem, src_models, target_model, sample_size,
                                                                sub_sample_size, mutation_vec=genes_differ, prev_samples=deepcopy(prev_samples),
                                                                efficient_version=True)
              else:
                _, sampled_offsprings = offspring.fitness_calc(problem, src_models, target_model,
                                                            sample_size, sub_sample_size)
                # if best_chrom < best_offspring: # Saving best Chromosome for future injection
                #   best_chrom = best_offspring
              # print('end fitness o ina')

                # print('hereee')
              is_off_selected = False
              if g/delta != 1 or tg != 0:
                if selection_version == 'v1':
                  second_specie, mutation_strength, is_off_selected = selection_adoption(second_specie, offspring, mutation_strength)
                elif selection_version == 'v2':
                  second_specie, mutation_strength, second_species_gen_success_num = selection_adoption_v2(second_specie, offspring, mutation_strength,
                                                                                                          second_species_gen_num, second_species_gen_success_num, c=c)
                else:
                  raise ValueError('selection_version value is wrong')

              if efficient_version and (is_off_selected or (g/delta == 1 and tg == 0)):
                prev_samples = prev_samples_tmp
              # Replacing the best chromosome found by sampling from second species with the worst chromosome of first species
              if injection_type == 'elite':
                  first_species[-1] == np.max(sampled_offsprings)
              elif injection_type == 'full':
                  first_species = total_selection_pop(np.concatenate((first_species, sampled_offsprings)), s1_psize)
                  

              fitness_hist_s2[rep, int(g/delta)-1, :] = second_specie
              mutation_strength_hist[rep, int(g/delta)-1, :]  = mutation_strength
              print('Generation %d: Best Fitness of Second Species: %s' % (g, second_specie.fitness))
              print('Best Alpha generation {}: best fitness of second species = {}'.format(g, second_specie.genes))
              print('generation {}: mutation strength = {}'.format(g, mutation_strength))
        else:
            # Crossover & Mutation
            offsprings = total_crossover(first_species)
            for j in range(s1_psize): offsprings[j].mutation(1/dims)
              
            # Fitness Calculation
            cfitness = np.zeros(s1_psize)
            for j in range(s1_psize): 
              cfitness[j] = offsprings[j].fitness_calc(problem)

            # Selection
            first_species, fitness = total_selection(np.concatenate((first_species, offsprings)),
                                      np.concatenate((fitness, cfitness)), s1_psize)

            bestfitness = fitness[0]
            fitness_hist_s1[rep, ea_counter, :] = first_species
            time_hist_s1[rep, ea_counter] = time() - start
            print('Generation %d best fitness of first species= %f' % (g, bestfitness))
            start = time()
            ea_counter += 1

  print('Finished')
  return fitness_hist_s1, fitness_hist_s2, mutation_strength_hist, time_hist_s1


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

def check_args(args):
  if args.sample_size < args.sub_sample_size:
    raise ValueError('sub_sample_size has greater value than sample_size')



def main_source(s_poles_length, target_pole_len,
           gen=100, src_save_dir='models/pole_models/src_model'):
  # src_models = np.ndarray(len(s_poles_length), dtype=object)
  src_models = []
  src_model = None
  if os.path.isfile(src_save_dir + '_{}.pkl'.format(s_poles_length[0])):
    src_model = Tools.load_from_file(src_save_dir + '_{}'.format(s_poles_length[0]))
  else:
    src_model, _, _, _ = evolutionary_algorithm(
                                s_poles_length[0], psize=100, gen=gen, 
                                muc=20, mum=20, stop_condition=True, 
                                create_model=True
                                )
    Tools.save_to_file(src_save_dir + '_{}'.format(s_poles_length[0]), src_model)
  src_models.append(src_model)
  for i, s_len in enumerate(s_poles_length[1:]):
    if os.path.isfile(src_save_dir + '_{}.pkl'.format(s_len)):
        src_model = Tools.load_from_file(src_save_dir + '_{}'.format(s_len))
        src_models.append(src_model)
        print('---------------------- {} source model loaded---------------------'.format(i))
    else:
      while(True):
        print('-------------- S_Length: {} ------------'.format(s_len))
        _, _, _, src_model = transfer_ea(s_len, src_models, psize=100, gen=gen, 
                                muc=20, mum=20, reps=1, build_model=True)
        
        if src_model is not None:
          Tools.save_to_file(src_save_dir + '_{}'.format(s_len), src_model)
          src_models.append(src_model)
          print('---------------------- {} source model created ---------------------'.format(i))
          break

  fitness_hist, alpha, fitness_time, sols_found, func_eval_nums =  \
        transfer_ea(target_pole_len, src_models, psize=100, gen=100, 
                muc=20, mum=20, reps=50, build_model=False)

  Tools.save_to_file('transfer_pole_outcome', [fitness_hist, alpha, fitness_time, sols_found, func_eval_nums])
  print('Function Evaluations: {}'.format(np.mean(func_eval_nums)))
  print('Solutions found: {}/{}'.format(np.sum(sols_found), 50))


  

  
  

if __name__ == '__main__':
  src_poles_length = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.79]
  target_pole_len = 0.8
  main_source(src_poles_length, target_pole_len, gen=10000)