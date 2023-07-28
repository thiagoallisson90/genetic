import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from fuzzy_problem import SingleFuzzyProblem

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

n_var = 36
n_obj = 1
min_d = 1
pop_size = 80

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

def main():
  X = []
  for i in range(pop_size):
    snr = generate_vertexes(snr_min, snr_max, min_d)
    tp = generate_vertexes(tp_min, tp_max, min_d)
    sf = generate_vertexes(sf_min, sf_max, min_d)
    X.append(np.concatenate((snr, tp, sf)))

  problem = SingleFuzzyProblem(xl, xu)

  algorithm = GA(
      pop_size=pop_size,
      mutation=PolynomialMutation(prob=0.2),
      crossover=TwoPointCrossover(prob=1.0),
      eliminate_duplicates=True,
      sampling=np.array(X))

  res = minimize(problem,
                algorithm,
                ('n_gen', 90),
                save_history=True,
                verbose=False)

  print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

  plt.plot(range(1, len(res.F)+1), res.F)
  plt.xlabel('Iterações')
  plt.ylabel('Fitness')
  plt.title('Gráfico de Barras')
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
  main()