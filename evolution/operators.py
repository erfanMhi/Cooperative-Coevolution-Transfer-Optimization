import numpy as np
from abc import ABC, abstractmethod
from evolution.chromosome import Chromosome, AlphaChromosome

def crossover(p1, p2, alpha=np.random.rand(), ctype='uniform'):
  print('alpha: ', alpha)
  if ctype=='convex':
    return Chromosome(alpha*p1.genes+(1-alpha)*p2.genes)
  # elif stype=='uniform':
  #   return max(np.random.choice(self.pop,k,False),key=lambda c:c.fitness)
  else:
    raise ValueError('Type of crossover which you entered is wrong')

def total_crossover(pop):
  dims = len(pop[0].genes)
  psize = len(pop)
  parent1 = pop[np.random.permutation(psize)]
  parent2 = pop[np.random.permutation(psize)]
  offsprings = np.ndarray(psize, dtype=object)
  tmp = np.random.rand(psize, dims)
  p1_selection = tmp >= 0.5
  p2_selection = tmp < 0.5
  for i in range(psize):
    offsprings[i] = Chromosome(dims)
    offsprings[i].genes[p1_selection[i]] = parent1[i].genes[p1_selection[i]]
    offsprings[i].genes[p2_selection[i]] = parent2[i].genes[p2_selection[i]]
  return offsprings

def total_crossover_s2(pop):
  dims = len(pop[0].genes)
  psize = len(pop)
  parent1 = pop[np.random.permutation(psize)]
  parent2 = pop[np.random.permutation(psize)]
  offsprings = np.ndarray(psize, dtype=object)
  tmp = np.random.rand(psize, dims)
  p1_selection = tmp >= 0.5
  p2_selection = tmp < 0.5
  for i in range(psize):
    offsprings[i] = AlphaChromosome(dims)
    offsprings[i].genes[p1_selection[i]] = parent1[i].genes[p1_selection[i]]
    offsprings[i].genes[p2_selection[i]] = parent2[i].genes[p2_selection[i]]
  return offsprings

def total_selection(pop, fitnesses, psize):
  index = np.argsort(-fitnesses)  # default argsort is ascending
  return pop[index[:psize]], fitnesses[index[:psize]]

def selection(pop, stype='roulette'):
  if stype=='roulette':
    sum_fit = np.sum([ch.fitness for ch in pop])
    pick = np.random.uniform(0, sum_fit)
    current = 0
    for chromosome in pop:
        current += chromosome.fitness
        if current > pick:
            return chromosome
  elif stype=='tournament':
      return max(np.random.choice(self.pop,k,False),key=lambda c:c.fitness)
  else:
    raise ValueError('Type of selection which you entered is wrong')


def get_pop_init(n, gn, init_func=np.random.rand):
  """[Initial Population Generator]
  
  Arguments:
      n {[int]} -- [Number of members of population]
      gn {[int]} -- [Number of individual's genes]
      init_func {[function]} -- the function which initialize each chromosome
  
  Returns:
      [np.ndarray] -- [Array of chromosomes]
  """
  return np.array([Chromosome(gn, init_func) for _ in range(n)])

def get_pop_init_s2(n, gn, init_func=np.random.rand):
  """[Initial Population Generator]
  
  Arguments:
      n {[int]} -- [Number of members of population]
      gn {[int]} -- [Number of individual's genes]
      init_func {[function]} -- the function which initialize each chromosome
  
  Returns:
      [np.ndarray] -- [Array of chromosomes]
  """
  return np.array([AlphaChromosome(gn, init_func) for _ in range(n)])
