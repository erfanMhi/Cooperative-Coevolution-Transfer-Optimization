from evolution.individual import *

class Chromosome(Individual):
  def __init__(self, n, init_func=np.random.rand):
    super().__init__(n, init_func=init_func)

  def mutation(self, mprob):
    if np.random.rand() < mprob:
      self.genes = np.abs(1 - self.genes)
  
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