#!/usr/bin/python3
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import gutils

n_v = 12
snr_min, tp_min, sf_min = -5.5, 2.0, 7.0
xl_snr = [snr_min for _ in range(n_v)]
xl_tp = [tp_min for _ in range(n_v)]
xl_sf = [sf_min for _ in range(n_v)]
xl = np.concatenate((xl_snr, xl_tp, xl_sf))

snr_max, tp_max, sf_max = 27.8, 14.0, 12.0
xu_snr = [snr_max for _ in range(n_v)]
xu_tp = [tp_max for _ in range(n_v)]
xu_sf = [sf_max for _ in range(n_v)]
xu = np.concatenate((xu_snr, xu_tp, xu_sf))

min_d = 1
pop_size = 80
n_var = 36
n_obj = 2

def generate_vertexes(min_val, max_val, min_dist):
  coords = np.zeros(12)
  
  coords[0] = min_val
  coords[11] = max_val

  coords[1:11] = np.random.uniform(min_val, max_val, 10)
  coords.sort()

  while coords[2] - coords[0] < min_dist:
    coords[2] += np.random.uniform(0.05, 0.25)
    coords[2] = min(max_val, coords[2])

  while coords[5] - coords[3] < min_dist:
    coords[5] += np.random.uniform(0.05, 0.25)
    coords[5] = min(max_val, coords[5])  
  
  while coords[8] - coords[6] < min_dist:
    coords[8] += np.random.uniform(0.05, 0.25)
    coords[8] = min(max_val, coords[8])  
    
  while coords[11] - coords[9] < min_dist:
    coords[9] -= np.random.uniform(0.05, 0.25)
    coords[9] = max(min_val, coords[9])
  
  coords[3] = np.random.uniform(coords[0], coords[2])
  coords[6] = np.random.uniform(coords[3], coords[5])
  coords[9] = np.random.uniform(coords[6], coords[8])

  return coords

def swap(a, b):
  aux = a
  a = b
  b = aux
  return a, b

def check_v(v, min_val, max_val, min_dist):
  v[0] = min_val
  v[11] = max_val

  v[0:3].sort()
  while v[2] - v[0] < min_dist:
    v[2] += np.random.uniform(0, 0.25)
    v[2] = min(max_val, v[2])
    v[0:3].sort()
  
  v[3:6].sort()
  while v[5] - v[3] < min_dist:
    v[5] += np.random.uniform(0, 0.25)
    v[5] = min(max_val, v[5])
    v[3:6].sort()

  v[6:9].sort()
  while v[8] - v[6] < min_dist:
    v[8] += np.random.uniform(0, 0.25)
    v[8] = min(max_val, v[8])
    v[6:9].sort()

  v[9:12].sort()
  while v[11] - v[9] < min_dist:
    v[9] -= np.random.uniform(0, 0.25)
    v[9] = max(min_val, v[9])
    v[9:12].sort()
  
  if v[3] > v[2]:
    v[3], v[2] = swap(v[3], v[2])
  
  if v[6] > v[5]:
    v[6], v[5] = swap(v[6], v[5])
  
  if v[9] > v[8]:
    v[9], v[8] = swap(v[9], v[8])

class FuzzyProblem(Problem):
  def __init__(self):
    self.num_run = 0
    super().__init__(n_var=n_var,
                      n_obj=n_obj,
                      n_constr=0,
                      xl=xl,
                      xu=xu,
                      elementwise_evaluation=True)

  def _check_x(self, x):
    snr = x[0:12]
    tp = x[12:24]
    sf = x[24:36]

    check_v(snr, snr_min, snr_max, min_d)
    check_v(tp, tp_min, tp_max, min_d)
    check_v(sf, sf_min, sf_max, min_d)

  def _simulate(self, x):
    results = []
    for i in x:      
      gutils.fill_fll2(i)
      gutils.simulate_range()
      pdr, energy = gutils.calc_range()
      results.append(np.array([-pdr, energy]))

    return np.array(results)

  def _evaluate(self, x, out, *args, **kwargs):
    if self.num_run > 0:
      for i in x:
        self._check_x(i)
    
    out['F'] = [self._simulate(x)]
    
    self.num_run += 1

def execute():
  X = []
  for i in range(pop_size):
    snr = generate_vertexes(snr_min, snr_max, min_d)
    tp = generate_vertexes(tp_min, tp_max, min_d)
    sf = generate_vertexes(sf_min, sf_max, min_d)
    X.append(np.concatenate((snr, tp, sf)))

  problem = FuzzyProblem()
  algorithm = NSGA2(pop_size=pop_size, 
                    mutation=PolynomialMutation(prob=0.2),
                    crossover=TwoPointCrossover(prob=1.0),
                    eliminate_duplicates=True,
                    sampling=np.array(X))

  res = minimize(problem,
                algorithm,
                ('n_gen', 90),
                save_history=True,
                verbose=True)

  print('Solutions founded:')
  for solution in res.F:
    print(solution)

  X = res.X
  print('X', X)

  plot = Scatter()
  plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
  plot.add(res.F, facecolor="none", edgecolor="red")
  plot.show()

if __name__ == '__main__':
  execute()