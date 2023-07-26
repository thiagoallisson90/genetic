# AG-Fuzzy ADR LoRaWAN
- AG to generate database and rules for SIFs related to ADR in NS-3 simulator

## Input Vars:
1. SNR: [-5.5, 27.8] dBm; 3 sets: {poor, acceptable, good}

## Output Vars:
1. TP: [2.0, 14.0] dB; 3 sets: {low, average, high}
2. SF: [7, 12]; 3 sets: {low, average, high}

## Features
- Chromosome: 33 genes, particioned in five parts
  - 9 genes for SNR sets
  - 9 genes for TP sets
  - 9 genes for SF sets
  - 3 genes for SIF between SNR and TP
  - 3 genes for SIF between SNR and SF
- Selection: Tournament
- Crossover Rate: 0.95
- Mutation Rate: 0.2
- Elistism with k = 1
- Crossover with five points: SNR, TP, SF, Rules1 (SNR - TP), Rules2 (SNR - SF)
- Mutation applied for the five parts of chromosome

## Exemple:
```python
if __name__ ==  '__main__':
  g = Genetic([-5.5, 2, 7], [27.8, 14, 12], [1, 1, 1])
  g.print()
  g.execute()