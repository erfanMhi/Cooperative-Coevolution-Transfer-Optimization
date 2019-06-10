import numpy as np
from abc import ABC, abstractmethod
from evolution.chromosome import Chromosome

def crossover(p1, p2, alpha=np.random.rand(), ctype='convex'):
  print('alpha: ', alpha)
  if ctype=='convex':
    return Chromosome(alpha*p1.genes+(1-alpha)*p2.genes)


def selection():
  pass