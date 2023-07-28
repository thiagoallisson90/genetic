import numpy as np

class Chromosome2():  
  def init_chr(self, var_min, var_max, min_dist):
    self.var_min = var_min
    self.var_max = var_max
    self.min_dist = min_dist

    self.snr_v = self.gen_vertexes(var_min[0], var_max[0], min_dist[0])
    self.tp_v = self.gen_vertexes(var_min[1], var_max[1], min_dist[1])
    self.sf_v = self.gen_vertexes(var_min[2], var_max[2], min_dist[2])
    self.vertexes = \
      np.concatenate((self.snr_v, self.tp_v, self.sf_v))
    self.pdr = -np.inf
    self.energy = np.inf
    self.fitness = -np.inf

  def init(self, var_min, var_max, min_dist):
    self.var_min = var_min
    self.var_max = var_max
    self.min_dist = min_dist
    self.pdr = -np.inf
    self.energy = np.inf
    self.fitness = -np.inf

  def swap(self, a, b):
    aux = a
    a = b
    b = aux
    return a, b

  def adjust_a(self, v, var_max, min_dist, ref):
    v.sort()
    if v[0] > ref[-1]:
      v[0], ref[-1] = self.swap(v[0], ref[-1])

    v.sort()
    while v[-1] - v[0] < min_dist:
      v[-1] += np.random.uniform(0, 0.25)
      v[-1] = min(var_max, v[-1])
    v.sort()

  def adjust_m(self, v, var_max, min_dist):
    v.sort()
    while v[-1] - v[0] < min_dist:
      v[-1] += np.random.uniform(0, 0.25)
      v[-1] = min(var_max, v[-1])
    v.sort()
  
  def adjust_b(self, v, var_max, min_dist, ref):
    v.sort()
    if v[-1] < ref[0]:
      v[-1], ref[0] = self.swap(v[-1], ref[0])

    v.sort()
    while v[-1] - v[0] < min_dist:
      v[-1] += np.random.uniform(0, 0.25)
      v[-1] = min(var_max, v[-1])
    
    v.sort()

  def out_snr(self):
    if self.snr_v[5] <= self.snr_v[2]:
      self.snr_v[5] = self.snr_v[2] + np.random.uniform(0.05, 0.25)
      self.snr_v[5] = min(self.var_max[0], self.snr_v[5])
    
    if self.snr_v[3] >= self.snr_v[6]:
      self.snr_v[3] = self.snr_v[6] - np.random.uniform(0.05, 0.25)
      self.snr_v[3] = max(self.var_min[0], self.snr_v[3])

  def out_tp(self):
    if self.tp_v[5] <= self.tp_v[2]:
      self.tp_v[5] = self.tp_v[2] + np.random.uniform(0.05, 0.25)
      self.tp_v[5] = min(self.var_max[1], self.tp_v[5])
    
    if self.tp_v[3] >= self.tp_v[6]:
      self.tp_v[3] = self.tp_v[6] - np.random.uniform(0.05, 0.25)
      self.tp_v[3] = max(self.var_min[1], self.tp_v[3])

  def out_sf(self):
    if self.sf_v[5] <= self.sf_v[2]:
      self.sf_v[5] = self.sf_v[2] + np.random.uniform(0.05, 0.25)
      self.sf_v[5] = min(self.var_max[2], self.sf_v[5])
    
    if self.sf_v[3] >= self.sf_v[6]:
      self.sf_v[3] = self.sf_v[6] - np.random.uniform(0.05, 0.25)
      self.sf_v[3] = max(self.var_min[2], self.sf_v[3])

  def set_snr(self, _snr1, _snr2, point):
    snr1 = [i for i in _snr1]
    snr2 = [i for i in _snr2]
    self.snr_v = np.concatenate((snr1, snr2))

    if point == 3 or point == 6:
      if point == 3:
        self.adjust_a(self.snr_v[3:6], self.var_max[0], self.min_dist[0], self.snr_v[0:3])
        self.out_snr()
      if point == 6:
        self.adjust_a(self.snr_v[6:9], self.var_max[0], self.min_dist[0], self.snr_v[3:6])

    if point == 1 or point == 4 or point == 7:
      if point == 1:
        self.adjust_m(self.snr_v[0:3], self.var_max[0], self.min_dist[0])
      if point == 4:
        self.adjust_m(self.snr_v[3:6], self.var_max[0], self.min_dist[0])
        self.out_snr()
    
    if point == 2 or point == 5:
      if point == 2:
        self.adjust_b(self.snr_v[0:3], self.var_max[0], self.min_dist[0], self.snr_v[3:6])
      if point == 5:
        self.adjust_b(self.snr_v[3:6], self.var_max[0], self.min_dist[0], self.snr_v[6:9])
        self.out_snr()

  def set_tp(self, _tp1, _tp2, point):
    tp1 = [i for i in _tp1]
    tp2 = [i for i in _tp2]
    self.tp_v = np.concatenate((tp1, tp2))

    if point == 3 or point == 6:
      if point == 3:
        self.adjust_a(self.tp_v[3:6], self.var_max[1], self.min_dist[1], self.tp_v[0:3])
        self.out_tp()
      if point == 6:
        self.adjust_a(self.tp_v[6:9], self.var_max[1], self.min_dist[1], self.tp_v[3:6])

    if point == 1 or point == 4 or point == 7:
      if point == 1:
        self.adjust_m(self.tp_v[0:3], self.var_max[1], self.min_dist[1])
      if point == 4:
        self.adjust_m(self.tp_v[3:6], self.var_max[1], self.min_dist[1])
        self.out_tp()
    
    if point == 2 or point == 5:
      if point == 2:
        self.adjust_b(self.tp_v[0:3], self.var_max[1], self.min_dist[1], self.tp_v[3:6])
      if point == 5:
        self.adjust_b(self.tp_v[3:6], self.var_max[1], self.min_dist[1], self.tp_v[6:9])
        self.out_tp()
  
  def set_sf(self, _sf1, _sf2, point):
    sf1 = [i for i in _sf1]
    sf2 = [i for i in _sf2]
    self.sf_v = np.concatenate((sf1, sf2))

    if point == 3 or point == 6:
      if point == 3:
        self.adjust_a(self.sf_v[3:6], self.var_max[2], self.min_dist[2], self.sf_v[0:3])
        self.out_sf()
      if point == 6:
        self.adjust_a(self.sf_v[6:9], self.var_max[2], self.min_dist[2], self.sf_v[3:6])

    if point == 1 or point == 4 or point == 7:
      if point == 1:
        self.adjust_m(self.sf_v[0:3], self.var_max[2], self.min_dist[2])
      if point == 4:
        self.adjust_m(self.sf_v[3:6], self.var_max[2], self.min_dist[2])
        self.out_sf()
    
    if point == 2 or point == 5:
      if point == 2:
        self.adjust_b(self.sf_v[0:3], self.var_max[2], self.min_dist[2], self.sf_v[3:6])
      if point == 5:
        self.adjust_b(self.sf_v[3:6], self.var_max[2], self.min_dist[2], self.sf_v[6:9])
        self.out_sf()
  
  def change_a(self, v, gene, var_min, min_dist, refs):
    v.sort()
    v[gene] = np.random.uniform(refs[0], refs[-1])
    while v[-1] - v[0] < min_dist:
      v[0] -= np.random.uniform(0, 0.25) 
      v[0] = max(var_min, v[0])
    
    v.sort()

  def change_m(self, v, gene, var_max, min_dist):
    v.sort()
    v[gene] = np.random.uniform(v[0], v[-1])
    while v[-1] - v[0] < min_dist:
      v[-1] += np.random.uniform(0, 0.25)
      v[-1] = min(var_max, v[-1])
    
    v.sort()

  def change_b(self, v, gene, var_max, min_dist, refs):
    v.sort()
    v[gene] = np.random.uniform(refs[0], refs[-1])
    while v[-1] - v[0] < min_dist:
      v[-1] += np.random.uniform(0, 0.25)
      v[-1] = min(var_max, v[-1])

    v.sort()

  def change_snr(self, gene):
    if gene == 1 or gene == 4 or gene == 7:
      if gene == 1:
        self.change_m(self.snr_v[0:3], 1, self.var_max[0], self.min_dist[0])
      if gene == 4:
        self.change_m(self.snr_v[3:6], 1, self.var_max[0], self.min_dist[0])
        self.out_snr()
      if gene == 7:
        self.change_m(self.snr_v[6:9], 1, self.var_max[0], self.min_dist[0])
      
    if gene == 3 or gene == 6:
      if gene == 3:
        self.change_a(self.snr_v[3:6], 0, self.var_min[0], self.min_dist[0], self.snr_v[0:3])
        self.out_snr()
      if gene == 6:
        self.change_a(self.snr_v[6:9], 0, self.var_min[0], self.min_dist[0], self.snr_v[0:6])
    
    if gene == 2 or gene == 5:
      if gene == 2:
        self.change_b(self.snr_v[0:3], 2, self.var_max[0], self.min_dist[0], self.snr_v[3:9])
      if gene == 5:
        self.change_b(self.snr_v[3:6], 2, self.var_max[0], self.min_dist[0], self.snr_v[6:9])
        self.out_snr()

  def change_tp(self, gene):
    if gene == 1 or gene == 4 or gene == 7:
      if gene == 1:
        self.change_m(self.tp_v[0:3], 1, self.var_max[1], self.min_dist[1])
      if gene == 4:
        self.change_m(self.tp_v[3:6], 1, self.var_max[1], self.min_dist[1])
        self.out_tp()
      if gene == 7:
        self.change_m(self.tp_v[6:9], 1, self.var_max[1], self.min_dist[1])
      
    if gene == 3 or gene == 6:
      if gene == 3:
        self.change_a(self.tp_v[3:6], 0, self.var_min[1], self.min_dist[1], self.tp_v[0:3])
        self.out_tp()
      if gene == 6:
        self.change_a(self.tp_v[6:9], 0, self.var_min[1], self.min_dist[1], self.tp_v[0:6])
    
    if gene == 2 or gene == 5:
      if gene == 2:
        self.change_b(self.tp_v[0:3], 2, self.var_max[1], self.min_dist[1], self.tp_v[3:9])
      if gene == 5:
        self.change_b(self.tp_v[3:6], 2, self.var_max[1], self.min_dist[1], self.tp_v[6:9])
        self.out_tp()

  def change_sf(self, gene):
    if gene == 1 or gene == 4 or gene == 7:
      if gene == 1:
        self.change_m(self.sf_v[0:3], 1, self.var_max[2], self.min_dist[2])
      if gene == 4:
        self.change_m(self.sf_v[3:6], 1, self.var_max[2], self.min_dist[2])
        self.out_sf()
      if gene == 7:
        self.change_m(self.sf_v[6:9], 1, self.var_max[2], self.min_dist[2])
      
    if gene == 3 or gene == 6:
      if gene == 3:
        self.change_a(self.sf_v[3:6], 0, self.var_min[2], self.min_dist[2], self.sf_v[0:3])
        self.out_sf()
      if gene == 6:
        self.change_a(self.sf_v[6:9], 0, self.var_min[2], self.min_dist[2], self.sf_v[0:6])
    
    if gene == 2 or gene == 5:
      if gene == 2:
        self.change_b(self.sf_v[0:3], 2, self.var_max[2], self.min_dist[2], self.sf_v[3:9])
      if gene == 5:
        self.change_b(self.sf_v[3:6], 2, self.var_max[2], self.min_dist[2], self.sf_v[6:9])
        self.out_sf()

  def set_solution(self):
    self.vertexes = np.concatenate((self.snr_v, self.tp_v, self.sf_v))

  def gen_vertexes(self, min_val, max_val, min_dist):
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
  
  def copy(self):
    chr = Chromosome()
    chr.var_min = [i for i in self.var_min]
    chr.var_max = [i for i in self.var_max]
    chr.min_dist = [i for i in self.min_dist]
    chr.snr_v = np.array([i for i in self.snr_v])
    chr.tp_v = np.array([i for i in self.tp_v])
    chr.sf_v = np.array([i for i in self.sf_v])
    chr.vertexes = np.concatenate((chr.snr_v, chr.tp_v, chr.sf_v))
    chr.pdr = self.pdr
    chr.energy = self.energy
    chr.fitness = self.fitness
    
    return chr

  def print(self):
    print(f'vertexes={self.vertexes};\nfitness={self.fitness}.')
  
  def __str__(self):
    return f'vertexes={self.vertexes};fitness={self.fitness}.'

if __name__ == '__main__':
 c = Chromosome2()
 c.init_chr([-5.5, 2, 7], [27.8, 14, 12], [1, 1, 1])
 print(c)