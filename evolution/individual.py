import numpy as np
from functools import total_ordering
from abc import ABC, abstractmethod

@total_ordering
class Individual(ABC):

  # genes_to_numpy = np.frompyfunc(lambda c: c.genes.astype(int), 1, 1)
  # fitness_to_numpy = np.frompyfunc(lambda c: c.fitness.astype(np.float64), 1, 1)

  genes_to_numpy = lambda pop: np.array(list(map(lambda c: c.genes, pop)))
  fitness_to_numpy = lambda pop: np.array(list(map(lambda c: c.fitness, pop)))

  def __init__(self, n, init_func=np.random.rand):
    if isinstance(n, int):
      self.genes = init_func(n)
    elif isinstance(n, np.ndarray):
      self.genes = n
    else:
      raise ValueError('The input parameter n, is not valid')
    
    self.fitness = float('-inf')

  @abstractmethod
  def mutation(self):
    pass


  @abstractmethod
  def fitness_calc(self):
    pass

  def __lt__(self, other):
    return self.fitness < other.fitness

  def __eq__(self, other):
    return self.fitness == other.fitness

  def __neg__(self):
    return -self.fitness

  @staticmethod
  def chromosome_to_numpy(ch):
    return ch.genes