import numpy as np
from abc import ABC, abstractmethod
from evolution.chromosome import Chromosome, AlphaChromosome, ChromosomePole


def kid_generator(parent, mute_strength, genes_min_bounds=None, genes_max_bounds=None, genes_num=None):
    # no crossover, only mutation
    if genes_num==None:
      genes_num = len(parent)
    offspring = parent + mute_strength * np.random.randn(genes_num)
    offspring = np.clip(offspring, genes_min_bounds, genes_max_bounds)
    return offspring



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

def total_selection_pop(pop, psize):
  index = np.argsort(-pop)
  return pop[index[:psize]]
  
def selection(pop, stype='roulette', k=1):
  if stype=='roulette':
    sum_fit = np.sum([ch.fitness for ch in pop])
    pick = np.random.uniform(0, sum_fit)
    current = 0
    for chromosome in pop:
        current += chromosome.fitness
        if current > pick:
            return chromosome
  elif stype=='tournament':
      return max(np.random.choice(pop, k, False),key=lambda c:c.fitness)
  else:
    raise ValueError('Type of selection which you entered is wrong')


def get_pop_init(n, gn, init_func=np.random.rand, p_type='knapsack'):
  """[Initial Population Generator]
  
  Arguments:
      n {[int]} -- [Number of members of population]
      gn {[int]} -- [Number of individual's genes]
      init_func {[function]} -- the function which initialize each chromosome
  
  Returns:
      [np.ndarray] -- [Array of chromosomes]
  """
  if p_type == 'knapsack':
    return np.array([Chromosome(gn, init_func) for _ in range(n)])
  elif p_type == 'double_pole':
    return np.array([ChromosomePole(gn, init_func) for _ in range(n)])

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



def selection_adoption(parent, offspring, mute_strength, genes_num=None):
  if genes_num==None:
    genes_num = len(parent)

  fp = parent.fitness
  fk = offspring.fitness
  p_target = 1/5
  if fp < fk:     # kid better than parent
      parent = offspring
      ps = 1.     # kid win -> ps = 1 (successful offspring)
  else:
      ps = 0.
  # adjust global mutation strength
  mute_strength *= np.exp(1/np.sqrt(genes_num+1) * (ps - p_target))

  return parent, mute_strength, fp < fk

def selection_adoption_v2(parent, offspring, mute_strength, gen, success_gen, c=2, genes_num=None):
  if genes_num==None:
    genes_num = len(parent)

  fp = parent.fitness
  fk = offspring.fitness
  p_target = 1/5
  if fp < fk:     # kid better than parent
      parent = offspring
      ps = 1.     # kid win -> ps = 1 (successful offspring)
  else:
      ps = 0.
  success_gen = success_gen+ps
  print('success_gen is {}'.format(success_gen))
  print('gen is {}'.format(gen))
  g = (success_gen)/gen
  print('g is {}'.format(g))
  c_sq = c**2
  if g > p_target:
    mute_strength = mute_strength/c_sq
  else:
    mute_strength = c_sq*mute_strength
  return parent, mute_strength, success_gen, fp < fk


def sbx_crossover(p1, p2, muc, dims):
    child1 = np.zeros(dims)
    child2 = np.zeros(dims)
    randlist = np.random.rand(dims)
    for i in range(dims):
        if randlist[i] <= 0.5:
            k = (2*randlist[i])**(1/(muc + 1))
        else:
            k = (1/(2*(1 - randlist[i])))**(1/(muc + 1))
        child1[i] = 0.5*(((1 + k)*p1.genes[i]) + (1 - k)*p2.genes[i])
        child2[i] = 0.5*(((1 - k)*p1.genes[i]) + (1 + k)*p2.genes[i])
        if np.random.rand() < 0.5:
            tmp = child1[i]
            child1[i] = child2[i]
            child2[i] = tmp
    return child1, child2

  