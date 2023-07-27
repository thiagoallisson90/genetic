import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

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

pop_size = 100

def generate_vertexes(min_val, max_val, min_dist):
  coords = np.zeros(12)
  
  coords[0] = min_val
  coords[11] = max_val

  flag = True
  while flag or coords[2] - coords[0] < min_dist:
    coords[1:3] = np.random.uniform(coords[0], max_val, 2)
    coords[0:3].sort()
    flag = False

  flag = True
  while flag or coords[5] - coords[3] < min_dist:
    coords[3] = np.random.uniform(coords[0], coords[2])
    coords[4:6] = np.random.uniform(coords[3], max_val, 2)
    coords[3:6].sort()
    flag = False
  
  if coords[5] <= coords[2]:
    coords[5] = coords[2] + np.random.uniform(0.05, 0.25)
    coords[5] = min(max_val, coords[5])
  
  flag = True
  while flag or coords[8] - coords[6] < min_dist:
    coords[6] = np.random.uniform(coords[0], coords[5])
    coords[7:9] = np.random.uniform(coords[6], max_val, 2)
    coords[6:9].sort()
    flag = False

  if coords[3] >= coords[6]:
    coords[3] = coords[6] - np.random.uniform(0.05, 0.25)
    coords[3] = max(min_val, coords[3])

  flag = True
  while flag or coords[11] - coords[9] < min_dist:
    coords[9] = np.random.uniform(coords[0], coords[8])
    coords[10] = np.random.uniform(coords[9], max_val)
    coords[9:12].sort()
    flag = False
  
  if coords[8] <= coords[5]:
    coords[8] = coords[5] + np.random.uniform(0.05, 0.25)
    coords[8] = min(max_val, coords[8])

  if coords[6] >= coords[9]:
    coords[6] = coords[9] - np.random.uniform(0.05, 0.25)
    coords[6] = max(min_val, coords[6])

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
  
  if v[3] > v[2]:
    v[3], v[2] = swap(v[3], v[2])
  
  if v[5] <= v[2]:
    v[5] = v[2] + np.random.uniform(0.05, 0.25)
    v[5] =  min(max_val, v[5])

  v[6:9].sort()
  while v[8] - v[6] < min_dist:
    v[8] += np.random.uniform(0, 0.25)
    v[8] = min(max_val, v[8])
    v[6:9].sort()
  
  if v[5] < v[6]:
    v[5], v[6] = swap(v[5], v[6])
  
  if v[3] >= v[6]:
    v[3] = v[6] - np.random.uniform(0.05, 0.25)
    v[3] =  max(min_val, v[3])
  
  if v[8] <= v[5]:
    v[8] = v[5] + np.random.uniform(0.05, 0.25)
    v[8] = min(max_val, v[8])
  
  v[9:12].sort()
  while v[11] - v[9] < min_dist:
    v[9] -= np.random.uniform(0, 0.25)
    v[9] = max(min_val, v[9])
    v[9:12].sort()
  
  if v[6] >= v[9]:
    v[6] = v[9] - np.random.uniform(0.05, 0.25)
    v[6] = max(min_val, v[6])

class FuzzyProblem(Problem):
  def __init__(self):
    X = []
    for i in range(pop_size):
      snr = generate_vertexes(snr_min, snr_max, min_d)
      tp = generate_vertexes(tp_min, tp_max, min_d)
      sf = generate_vertexes(sf_min, sf_max, min_d)
      X.append(np.concatenate((snr, tp, sf)))

    super().__init__(n_var=36,
                      n_obj=2,
                      n_constr=0,
                      xl=xl,
                      xu=xu,
                      elementwise_evaluation=True,
                      X=X)

  def _check_x(self, x):
    snr = x[0:12]
    tp = x[12:24]
    sf = x[24:36]

    check_v(snr, snr_min, snr_max, min_d)
    check_v(tp, tp_min, tp_max, min_d)
    check_v(sf, sf_min, sf_max, min_d)

  def _evaluate(self, x, out, *args, **kwargs):
    self._check_x(x)
    f1 = np.random.uniform(0.1, 5.0)
    f2 = -np.random.uniform(1, 136)
    out["F"] = [f1, f2]

def execute():
  problem = FuzzyProblem()
  algorithm = NSGA2(pop_size=100, 
                    mutation=get_mutation('real_pm', prob=0.2),
                    crossover=get_crossover('real_two_point', prob=0.95),
                    eliminate_duplicates=True)

  termination = get_termination('n_gen', 100)
  res = minimize(problem,
                algorithm,
                termination,
                seed=1,
                verbose=True)

  print("Soluções encontradas:")
  for solution in res.X:
    print(solution)

if __name__ == '__main__':
  pass
  