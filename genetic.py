from chromosome import Chromosome
import gutils
import numpy as np

class Genetic():  
  def __init__(self, var_min, var_max, min_dist, size_pop=60, 
               num_it=80, cross_rate=0.95, mut_rate=0.2, verbose=True):
    self.size_pop = size_pop
    self.num_it =  num_it
    self.cross_rate = cross_rate
    self.mut_rate = mut_rate
    self.var_min = var_min
    self.var_max = var_max
    self.min_dist = min_dist
    self.verbose = verbose
    self.best_global = None

  def init_pop(self):
    self.pop = [Chromosome() for _ in range(self.size_pop)]
    for p in self.pop:
      p.init_chr(self.var_min, self.var_max, self.min_dist)

  def evaluate_pop(self):
    i = 1
    for p in self.pop:
      self.fn_fitness(p)
      i += 1
    
    self.sort_pop = sorted(self.pop, key=lambda p: p.fitness, reverse=True)
    
    if self.best_global == None or self.best_global.fitness < self.sort_pop[0].fitness:
      self.best_global = self.sort_pop[0].copy()

  def select(self):
    idx = np.random.randint(0, self.size_pop, 2)
    candidates = [self.pop[id] for id in idx]
    return candidates[0] if candidates[0].fitness > candidates[1].fitness else candidates[1]
  
  def crossover(self):
    father = self.select()
    mother = self.select()
    
    if np.random.random() <= self.cross_rate:
      offspring1 = Chromosome()
      offspring1.init(self.var_min, self.var_max, self.min_dist)
      offspring2 = Chromosome()
      offspring2.init(self.var_min, self.var_max, self.min_dist)

      point1 = np.random.randint(0, father.snr_v.size)
      offspring1.set_snr(father.snr_v[:point1], mother.snr_v[point1:], point1)
      offspring2.set_snr(mother.snr_v[:point1], father.snr_v[point1:], point1)

      point2 = np.random.randint(0, father.tp_v.size)
      offspring1.set_tp(father.tp_v[:point2], mother.tp_v[point2:], point2)
      offspring2.set_tp(mother.tp_v[:point2], father.tp_v[point2:], point2)

      point3 = np.random.randint(0, father.sf_v.size)
      offspring1.set_sf(father.sf_v[:point3], mother.sf_v[point3:], point3)
      offspring2.set_sf(mother.sf_v[:point3], father.sf_v[point3:], point3)

      point4 = np.random.randint(0, len(father.rules01))
      offspring1.set_rules01(father.rules01[:point4], mother.rules01[point4:])
      offspring2.set_rules01(mother.rules01[:point4], father.rules01[point4:])

      point5 = np.random.randint(0, len(father.rules02))
      offspring1.set_rules02(father.rules02[:point5], mother.rules02[point5:])
      offspring2.set_rules02(mother.rules02[:point5], father.rules02[point5:])

      offspring1.set_solution()
      offspring2.set_solution()
      return offspring1, offspring2
    else:
      offspring1 = father.copy()
      offspring2 = mother.copy()
      return offspring1, offspring2

  def mutate(self, chr):
    if np.random.random() <= self.mut_rate:
      gene = 0
      while gene == 0 or gene == len(chr.snr_v) - 1:
        gene = np.random.randint(0, len(chr.snr_v))
      chr.change_snr(gene)

      gene = 0
      while gene == 0 or gene == len(chr.tp_v) - 1:
        gene = np.random.randint(0, len(chr.tp_v))
      chr.change_tp(gene)

      gene = 0
      while gene == 0 or gene == len(chr.sf_v) - 1:
        gene = np.random.randint(0, len(chr.sf_v))
      chr.change_sf(gene)

      gene = 0
      while gene == 0 or gene == len(chr.rules01) - 1:
        gene = np.random.randint(0, len(chr.rules01))
      chr.change_rules01(gene)

      gene = 0
      while gene == 0 or gene == len(chr.rules02) - 1:
        gene = np.random.randint(0, len(chr.rules02))
      chr.change_rules02(gene)
    
    chr.set_solution()

  def solution_in_pop(self, sol):
    for i in range(len(self.pop)):
      if self.pop[i].fitness == sol.fitness:
        return i
    return -1

  def execute(self):
    self.init_pop()
    self.evaluate_pop()
    self.global_bests = [self.best_global]
    self.iter_bests = [self.sort_pop[0]]

    if self.verbose:
      print(f'Best Solution Iteration 0:')
      self.sort_pop[0].print()
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    gutils.log(self.sort_pop[0])

    for i in range(self.num_it):
      new_pop = []

      while len(new_pop) < self.size_pop:
        offspring1, offspring2 = self.crossover()

        self.mutate(offspring1)
        self.mutate(offspring2)

        new_pop.append(offspring1)
        new_pop.append(offspring2)
      
      self.pop.clear()
      self.sort_pop.clear()
      self.pop = new_pop

      self.evaluate_pop()
      self.global_bests.append(self.best_global)
      self.iter_bests.append(self.sort_pop[0])
      
      if self.verbose:
        print(f'Best Iteration {i+1}:')
        self.sort_pop[0].print()
        print(f'Best Global {i+1}:')
        self.best_global.print()
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
      
      gutils.log(self.sort_pop[0])

      # Elitism
      if self.solution_in_pop(self.best_global) == -1:
        idx = self.solution_in_pop(self.sort_pop[-1])
        if idx != -1:
          self.pop[idx] = self.best_global.copy()

    if self.verbose:
      print(f'Best Solution:')
      self.best_global.print()
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    gutils.log_i_bests(self.iter_bests)
    gutils.log_g_bests(self.global_bests)

    for g in self.global_bests:
      g.print()

    for i in self.iter_bests:
      i.print()

  def print(self):
    print(f'size_pop={self.size_pop}')
    print(f'num_it={self.num_it}')
    print(f'cross_rate={self.cross_rate}')
    print(f'cross_rate={self.mut_rate}')
    print(f'var_min={self.var_min}')
    print(f'var_min={self.var_max}')
    print(f'min_dist={self.min_dist}')
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
  
  def print_pop(self):
    print('*********************************************************************************************')
    for i in range(self.size_pop):
      print(f'Indivíduo {i+1}')
      self.pop[i].print()
      print('*********************************************************************************************')

  def print_spop(self):
    self.sort_pop = sorted(self.pop, key=lambda p: p.fitness, reverse=True)
    print('*********************************************************************************************')
    for i in range(self.size_pop):
      print(f'Indivíduo {i+1}')
      self.sort_pop[i].print()
      print('*********************************************************************************************')

  def fn_fitness(self, p):
    gutils.fill_fll(p.vertexes, p.rules01, p.rules02)
    gutils.simulate_range()
    pdr, energy = gutils.calc_range()
    p.pdr = pdr
    p.energy = energy
    p.fitness = pdr

if __name__ ==  '__main__':
  g = Genetic([-5.5, 2, 7], [27.8, 14, 12], [1, 1, 1])
  g.print()
  g.execute()