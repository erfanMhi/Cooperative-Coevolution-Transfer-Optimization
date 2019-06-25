from time import time
from evolution.individual import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel

class Chromosome(Individual):
  def __init__(self, n, init_func=np.random.rand):
    super().__init__(n, init_func=init_func)

  def mutation(self, mprob):
    mask = np.random.rand(self.genes.shape[0]) < mprob
    self.genes[mask] = np.abs(1 - self.genes[mask])
  
  def fitness_calc(self, problem): # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
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
    best_offspring = np.max(offsprings)
    return self.fitness, best_offspring
  
class StrategyChromosome:
  def __init__(self, n, init_func=np.random.rand):
    super().__init__(n, init_func=init_func)