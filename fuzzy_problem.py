import numpy as np
from pymoo.core.problem import Problem

import gutils

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

snr_min, tp_min, sf_min = -5.5, 2.0, 7.0
snr_max, tp_max, sf_max = 27.8, 14.0, 12.0
min_d = 1

class SingleFuzzyProblem(Problem):
  def __init__(self, xl, xu):
    self.is_init = True
    super().__init__(n_var=36,
                      n_obj=1,
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
      pdr, _ = gutils.calc_range()
      results.append(np.array([-pdr]))

    return np.array(results)

  def _evaluate(self, x, out, *args, **kwargs):
    if self.is_init == False:
      for i in x:
        self._check_x(i)
    else:
      self.is_init = False
    
    out['F'] = [self._simulate(x)]
