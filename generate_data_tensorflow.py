import sys
import os

# this fixes a problem with openmp https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# add path to core
sys.path.append("core/")

import matplotlib.pyplot as plt
import io_utils as io
import nn_utils as nn
from geometry import Geometry
from resampled_geometry import ResampledGeometry
from data_container import DataContainer
import numpy as np

model = 'data/output_sphere.vtp'
soln = io.read_geo(model).GetOutput()
soln_array, _, p_array = io.get_all_arrays(soln)
pressures, velocities = io.gather_pressures_velocities(soln_array)
geometry = Geometry(p_array)

for t in pressures:
    pressures[t] = pressures[t] / 1333.2

rgeo = ResampledGeometry(geometry, 5, remove_caps = True)
nodes, edges, lengths, inlet_node, outlet_nodes = rgeo.generate_nodes()

times = [t for t in pressures]
times.sort()
gpressures, gvelocities, areas = rgeo.generate_fields(pressures, velocities, soln_array['area'], times)

#%%

nodes, edges, _, inlet_node, outlet_nodes = rgeo.generate_nodes()

#%%
ntimes = len(times)
# cells = np.repeat(edges[None, :, :], ntimes, axis = 0)
# mesh_pos = np.repeat(nodes[None, :, :], ntimes, axis = 0)
cells = np.repeat(edges[None, :, :], 1, axis = 0)
mesh_pos = np.repeat(nodes[None, :, :], 1, axis = 0)
nnodes = nodes.shape[0]
node_type = np.zeros(nnodes)
node_type[inlet_node] = 4
node_type[outlet_nodes] = 5

# for ind in range(0, nnodes):
#     if node_type[ind] == 0:
#         count = np.count_nonzero(edges == ind)
#         if count > 2:
#             # +7 because we start numbering the junctions from 10
#             node_type[ind] = 7
            
# node_type = np.repeat(node_type[None, :], ntimes, axis = 0)
node_type = np.repeat(node_type[None, :], 1, axis = 0)
node_type = np.expand_dims(node_type, axis = 2)

flowrate = np.zeros((0, nnodes))
pressure = np.zeros((0, nnodes))
area = np.zeros((0, nnodes))

for t in times:
    flowrate = np.concatenate((flowrate, gvelocities[t].transpose()), axis = 0)
    pressure = np.concatenate((pressure, gpressures[t].transpose()), axis = 0)
    # area = np.concatenate((area, areas.transpose()), axis = 0)
    
flowrate = np.expand_dims(flowrate, axis = 2)
pressure = np.expand_dims(pressure, axis = 2)
area = np.expand_dims(areas, axis = 0)
fp_state = np.concatenate((pressure, flowrate), axis=2)
#%%

# import tensorflow as tf

# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def serialize_example(f0,f1,f2,f3,f4,f5):
#   """
#   Creates a tf.train.Example message ready to be written to a file.
#   """
#   # Create a dictionary mapping the feature name to the tf.train.Example-compatible
#   # data type.
#   feature = {
#       'cells': _bytes_feature(tf.io.serialize_tensor(f0)),
#       'mesh_pos': _bytes_feature(tf.io.serialize_tensor(f1)),
#       'node_type': _bytes_feature(tf.io.serialize_tensor(f2)),
#       'flowrate': _bytes_feature(tf.io.serialize_tensor(f3)),
#       'pressure': _bytes_feature(tf.io.serialize_tensor(f4)),
#       'area': _bytes_feature(tf.io.serialize_tensor(f5))
#   }

#   # Create a Features message using tf.train.Example.

#   example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#   return example_proto.SerializeToString()

# def tf_serialize_example(f0,f1,f2,f3,f4,f5):
#   tf_string = tf.py_function(
#     serialize_example,
#     (f0,f1,f2,f3,f4,f5),  # Pass these args to the above function.
#     tf.string)      # The return type is `tf.string`.
#   return tf.reshape(tf_string, ()) # The result is a scalar.

# feature_dataset = tf.data.Dataset.from_tensor_slices((cells.astype(np.int32), 
#                                                       mesh_pos.astype(np.float32), 
#                                                       node_type.astype(np.int32), 
#                                                       flowrate.astype(np.float32), 
#                                                       pressure.astype(np.float32), 
#                                                       area.astype(np.float32)))
# serialized_features_dataset = feature_dataset.map(tf_serialize_example)

# filename = 'train.tfrecord'
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(serialized_features_dataset)

#%% 
import tensorflow as tf

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(f0,f1,f2,f3,f4):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'cells': _bytes_feature(f0.numpy().tobytes()),
      'mesh_pos': _bytes_feature(f1.numpy().tobytes()),
      'node_type': _bytes_feature(f2.numpy().tobytes()),
      'pf_state': _bytes_feature(f3.numpy().tobytes()),
      'area': _bytes_feature(f4.numpy().tobytes())
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def tf_serialize_example(f0,f1,f2,f3,f4):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3,f4),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar.

def flatten(tens):
    return tf.reshape(tens, (1, tf.size(tens)))

feature_dataset = tf.data.Dataset.from_tensor_slices((flatten(cells.astype(np.int32)), 
                                                      flatten(mesh_pos.astype(np.float32)), 
                                                      flatten(node_type.astype(np.int32)), 
                                                      flatten(fp_state.astype(np.float32)), 
                                                      flatten(area.astype(np.float32))))
serialized_features_dataset = feature_dataset.map(tf_serialize_example)

filename = 'train.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
#%%
# filenames = [filename]
# raw_dataset = tf.data.TFRecordDataset(filenames)

# for raw_record in raw_dataset.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(raw_record.numpy())
#   print(example)
  
  
#   result = {}
#   # example.features.feature is the dictionary
#   for key, feature in example.features.feature.items():
#     # The values are the Feature objects which contain a `kind` which contains:
#     # one of three fields: bytes_list, float_list, int64_list
    
#       kind = feature.WhichOneof('kind')
#       result[key] = np.array(getattr(feature, kind).value)
    
#     result
    
