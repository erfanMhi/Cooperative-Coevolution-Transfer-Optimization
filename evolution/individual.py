import numpy as np
from abc import ABC, abstractmethod

class Individual(ABC):

  def __init__(self, n):
    self.genes = np.zeros(n)

  @abstractmethod
  def mutation(self):
    pass


  @abstractmethod
  def fitness_calc(self):
    pass


