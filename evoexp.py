from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api

import json

from deap import base
from deap import creator
from deap import tools

import numpy as np

# NASBENCH_TFRECORD = './data/nasbench_full.tfrecord'
NASBENCH_TFRECORD = './data/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

MAX_EDGES = 9
MAX_OPERATIONS = 5
OPERATIONS = [CONV1X1, CONV3X3, MAXPOOL3X3]

# Load the data
nasbench = api.NASBench(NASBENCH_TFRECORD)


def eval_model(encoded, decoder):  
  model_spec = decoder(encoded)
  fitness = 0
  try:
    data = nasbench.query(model_spec)
    fitness = data['test_accuracy']
  except api.OutOfDomainError:
    print("Invalid solution")

  return fitness


def init_model_binary():
  # INPUT + [CONV1X1]5 + [CONV3X3]5 + [MAXPOOL3X3]5 + OUTPUT
  _ops = 2 + MAX_OPERATIONS * len(OPERATIONS)
  _prob = MAX_EDGES / (_ops ** 2)
  matrix = np.random.choice([0, 1], size=(_ops, _ops), p=[(1 - _prob), _prob])
  return matrix.flatten().tolist()


def decode_individual_binary(encoded):
  pass


def init_model_incep():
  # TODO a model may have less than MAX_OPERATIONS
  _ops = 2 + MAX_OPERATIONS
  _prob = MAX_EDGES / (_ops * (_ops - 1) / 2)
  matrix = np.random.choice([0, 1], size=(_ops, _ops), p=[(1 - _prob), _prob])
  matrix = np.triu(matrix, k=2)
  ops = [INPUT] + np.random.choice(OPERATIONS, size=MAX_OPERATIONS).tolist() + [OUTPUT]
  return [matrix, ops]


def decode_model_incep(encoded):
  [matrix, ops] = encoded[0]
  model_spec = api.ModelSpec(
      matrix=matrix,
      ops=ops)
  return model_spec


def mutate_model_incep(encoded, indpb):
  [matrix, ops] = encoded[0]
  # TODO mutate the matrix
  # TODO improve the mutation of the ops
  for row in range(0, matrix.shape[0]):
    for col in range(row + 1, matrix.shape[1]):
      if np.random.rand() < indpb:
        # bit flip
        matrix[row, col] = 0 if matrix[row, col] else 1


def cxSinglePoint(ind1, ind2):
  # TODO improve crossover
  _ops2 = ind2[0][1].copy()
  ind2[0][1] = ind1[0][1].copy()
  ind1[0][1] = _ops2


def print_stats(pop):
  fits = [ind.fitness.values[0] for ind in pop]
  length = len(pop)
  mean = sum(fits) / length
  sum2 = sum(x*x for x in fits)
  std = abs(sum2 / length - mean**2)**0.5        
  print("  Min %s" % min(fits))
  print("  Max %s" % max(fits))
  print("  Avg %s" % mean)
  print("  Std %s" % std)


def ea_opt(argv):
  del argv  # Unused
  CXPB, MUTPB = 0.5, 0.05

  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()
  # Attribute generator 
  toolbox.register("init_model", init_model_incep)
  # Structure initializers
  toolbox.register("individual", tools.initRepeat, creator.Individual, 
      toolbox.init_model, 1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", eval_model, decoder=decode_model_incep)
  # algorithm operations
  toolbox.register("mate", cxSinglePoint)
  toolbox.register("mutate", mutate_model_incep, indpb=MUTPB)
  toolbox.register("select", tools.selTournament, tournsize=3)  

  pop = toolbox.population(n=4)
  fitnesses = list(map(toolbox.evaluate, pop))
  for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = (fit,)

  fits = [ind.fitness.values[0] for ind in pop]

  print_stats(pop)

  # Variable keeping track of the number of generations
  g = 0
  # Begin the evolution
  while max(fits) < 1 and g < 2:
    # A new generation
    g = g + 1
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
      if np.random.rand() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
      toolbox.mutate(mutant)
      del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = (fit,)

    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    print_stats(pop)

  print("END")
  best_ind = tools.selBest(pop, 1)
  print(best_ind)


if __name__ == '__main__':
  app.run(ea_opt)
