from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api

import json

NASBENCH_TFRECORD = './data/nasbench_full.tfrecord'
# NASBENCH_TFRECORD = './data/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

MODEL_DIR = './models/'

def single_model(argv):
  del argv  # Unused

  # Load the data from file (this will take some time)
  nasbench = api.NASBench(NASBENCH_TFRECORD)

  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

  random_estimate = nasbench.random_estimate(model_spec, MODEL_DIR)
  print(random_estimate)
  # Query this model from dataset, returns a dictionary containing the metrics
  # associated with this model.
  # print('Querying an Inception-like model.')
  data = nasbench.query(model_spec)
  print(data)


def random_models(argv):
  del argv  # Unused
  LIMIT = 100
  OUTPUT = './experiments/random_sampling.json'

  # Load the data from file (this will take some time)
  nasbench = api.NASBench(NASBENCH_TFRECORD)

  npEnc = api._NumpyEncoder()
  for index, unique_hash in enumerate(nasbench.hash_iterator()):
    if index >= LIMIT:
      break
    fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
        unique_hash)
    model_spec = api.ModelSpec(
        matrix=fixed_metrics['module_adjacency'],
        ops=fixed_metrics['module_operations'])
    random_estimate = nasbench.random_estimate(model_spec, MODEL_DIR)
    data = nasbench.query(model_spec)
    merge = {**fixed_metrics, **computed_metrics}
    merge['module_adjacency'] = npEnc.default(fixed_metrics['module_adjacency'])
    merge['random_sampling_time'] = random_estimate['prediction_time']
    merge['random_samples'] = random_estimate['evaluation_results']
    merge['train_accuracy'] = data['train_accuracy']
    merge['test_accuracy'] = data['test_accuracy']
    merge['validation_accuracy'] = data['validation_accuracy']
    print(index, merge)
    
    with open(OUTPUT, 'a') as f:
      f.write(json.dumps(merge) + '\n')
    

if __name__ == '__main__':
  app.run(single_model)
