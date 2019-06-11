import numpy as np
from abc import ABC, abstractmethod

class Individual(ABC):

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


