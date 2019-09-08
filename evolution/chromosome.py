from time import time
from evolution.individual import *
# from evolution.operators import kid_generator
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel

from copy import deepcopy

class Chromosome(Individual):
  def __init__(self, n, init_func=np.random.rand):
    super().__init__(n, init_func=init_func)

  def mutation(self, mprob):
    mask = np.random.rand(self.genes.shape[0]) < mprob
    self.genes[mask] = np.abs(1 - self.genes[mask])


  def fitness_calc(self, problem): # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
    if self.fitness >= 0:
      return self.fitness
    weights = problem['w']
    profits = problem['p']
    ratios = profits/weights
    selected_items = self.genes == 1
    total_weight = np.sum(weights[selected_items])
    total_profit = np.sum(profits[selected_items])
    
    if total_weight > problem['cap']: # Repair solution
        selections = np.sum(selected_items)
        r_index = np.zeros((selections,2))
        counter = 0

        for j in range(len(self.genes)):
            if selected_items[j] == 1:
                r_index[counter,0] = ratios[j]
                r_index[counter,1] = int(j)
                counter = counter + 1
            if counter >= selections:
                break

        r_index = r_index[r_index[:,0].argsort()[::-1]]
        # print(List)
        counter = selections-1
        while total_weight > problem['cap']:
            l = int(r_index[counter,1])
            selected_items[l] = 0 
            total_weight = total_weight - weights[l]
            total_profit = total_profit - profits[l]
            counter = counter - 1

    self.fitness = total_profit
    return self.fitness

class AlphaChromosome(Individual):
  def __init__(self, n, init_func=np.random.rand):
      super().__init__(n, init_func=init_func)

  def mutation(self, mprob, mtype='normal'):
    mask = np.random.rand(self.genes.shape[0]) < mprob
    if mtype=='normal':
      self.genes[mask] = self.genes[mask] + np.random.normal(0, .25, size=self.genes[mask].shape)
      self.genes[self.genes > 1] = 1
      self.genes[self.genes < 0] = 0
      if np.sum(self.genes) == 0:
        self.genes[-1] = 1
      # print('mutation: ', self.genes)
    else:
      self.genes[mask] = np.abs(1 - self.genes[mask])

      
  
  def fitness_calc(self, problem, src_models, target_model, sample_size, sub_sample_size): # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
    start = time()
    normalized_alpha = self.genes/np.sum(self.genes)
    mixModel = MixtureModel(src_models, alpha=normalized_alpha)
    mixModel.add_target_model(target_model)
    # mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'umd')
    # mixModel.EMstacking()
    # mixModel.mutate()
    print('sample start')
    offsprings = mixModel.sample(sample_size)
    print('sample end')
    print('selecting start')
    idx = np.random.randint(sample_size, size=sub_sample_size)
    offsprings = offsprings[idx] # Creating sub_samples of samples
    print('selecting end')
    offsprings = np.array([Chromosome(offspring) for offspring in offsprings])
    sfitness = np.zeros(sub_sample_size)
    print('fitness_calc start')
    for i in range(sub_sample_size): 
      sfitness[i] = offsprings[i].fitness_calc(problem)
    print('fitness_calc end') 
    self.fitness = np.mean(sfitness)
    self.fitness_calc_time = time() - start
    # best_offspring = np.max(offsprings)
    return self.fitness, offsprings
  
class StrategyChromosome(Individual):
  def __init__(self, n, init_func=np.random.rand):
    super().__init__(n, init_func=init_func)

  def mutation(self, mute_strength, genes_min_bounds=None, genes_max_bounds=None, genes_num=None):
    if genes_num==None:
        genes_num = len(self.genes)
    tmp_genes = deepcopy(self.genes)
    self.genes = self.genes + mute_strength * np.random.randn(genes_num)
    self.genes = np.clip(self.genes, genes_min_bounds, genes_max_bounds)
    if np.sum(self.genes) == 0:
      self.genes[-1] = 1
      print('yes')
    return self.genes/np.sum(self.genes) - tmp_genes/np.sum(tmp_genes) # needed for fitness evaluation step

    

  def fitness_calc(self, problem, src_models, target_model, sample_size,
                   sub_sample_size, mutation_vec=None, prev_samples=None,
                   efficient_version=False): # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
    start = time()
    if not efficient_version or (mutation_vec is None):
      normalized_alpha = self.genes/np.sum(self.genes)
    else:
      normalized_alpha = np.clip(mutation_vec, 0, None)
    mixModel = MixtureModel(src_models, alpha=normalized_alpha)
    mixModel.add_target_model(target_model)
    # mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'umd')
    # mixModel.EMstacking()
    # mixModel.mutate()
    # print('sample start')
    if efficient_version:
      offsprings = mixModel.sample_dic(sample_size)
      flat_offsprings = np.array([])
      is_prev = (prev_samples is not None) and (mutation_vec is not None)
      if is_prev:
        # removing samples
        removing_samples = np.clip(np.ceil(mutation_vec*sample_size).astype(int), None, 0)
        print('Removing Samples: {}'.format(removing_samples))
        for i in range(len(removing_samples)):
          if removing_samples[i]!=0:
            r_num = len(prev_samples[i]) + removing_samples[i]
<<<<<<< HEAD
=======
            print(r_num)
>>>>>>> 1dc5e31e42a64fab5a0a57e0ce69d8ae1b8903db
            if r_num!=0:
              prev_samples[i] = np.random.choice(prev_samples[i], r_num, replace=False)
            else:
              prev_samples[i] = None

        
        # adding sapmles
        for i in range(len(offsprings)):
          if offsprings[i] is not None:
            if prev_samples[i] is None:
              prev_samples[i] = offsprings[i]
            else:
              offspring_add = [Chromosome(offspring) for offspring in offsprings[i]]
              flat_offsprings = np.append(flat_offsprings, offspring_add)
              prev_samples[i] = np.append(prev_samples[i], offspring_add, axis=0)
        offsprings = prev_samples        
      self.fitness = 0
      count = 0
      # fitness calc
      for i in range(len(offsprings)):
        if offsprings[i] is not None:
          if not is_prev:
            offspring_add = np.array([Chromosome(offspring) for offspring in offsprings[i]])
            flat_offsprings = np.append(flat_offsprings, offspring_add)
            offsprings[i] = offspring_add
          for j in range(len(offsprings[i])):
            self.fitness += offsprings[i][j].fitness_calc(problem)
            count += 1
      print('number of all samples: {}'.format(count))
      self.fitness /= count

      return self.fitness, flat_offsprings, offsprings
    else:
      offsprings = mixModel.sample(sample_size)
      idx = np.random.randint(sample_size, size=sub_sample_size)
      offsprings = offsprings[idx] # Creating sub_samples of samples
      # print('selecting end')
      offsprings = np.array([Chromosome(offspring) for offspring in offsprings])
      sfitness = np.zeros(sub_sample_size)
      # print('fitness_calc start')
      for i in range(sub_sample_size): 
        sfitness[i] = offsprings[i].fitness_calc(problem)
      # print('fitness_calc end') 
      self.fitness = np.mean(sfitness)
      # print('sample end')
      # print('selecting start')

      self.fitness_calc_time = time() - start
      # best_offspring = np.max(offsprings)
      return self.fitness, offsprings