# AG para gerar base de dados e regras de SIFs para ADR no NS-3

## Vars de Entrada:
1. SNR: [-5.5, 27.8] dBm; 3 conjuntos

## Vars de Saída:
1. TP: [2.0, 14.0] dB; 3 conjuntos
2. SF: [7, 12]; 3 conjuntos

## Exemplo:
```python
if __name__ ==  '__main__':
  g = Genetic([-5.5, 2, 7], [27.8, 14, 12], [1, 1, 1])
  g.print()
  g.execute()