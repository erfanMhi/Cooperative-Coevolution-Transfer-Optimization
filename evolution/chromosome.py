from evolution.individual import *

class Chromosome(Individual):
  def __init__(self, n):
    super(Individual, self).__init__(n)

  def mutation(self):
    pass
  
  def fitness_calc(self):
    pass