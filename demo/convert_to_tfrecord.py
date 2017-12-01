# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Create a TFRecord-based dataset from the UCI Census dataset available here:
https://archive.ics.uci.edu/ml/datasets/Census+Income

Datasets obtainable via:
- wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Reference for converting data to TFRecords:
- https://stackoverflow.com/questions/41402332/tensorflow-create-a-tfrecords-file-from-csv
"""

import pandas
import tensorflow as tf


DATA_FILE = "/tmp/adult.data"
OUTPUT_FILE = "/tmp/adult.tfrecords"

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'label'
]

numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                      'capital-loss', 'hours-per-week']

df = pandas.read_csv(DATA_FILE).values
with tf.python_io.TFRecordWriter(OUTPUT_FILE) as writer:
  for row in df:
    example = tf.train.Example()
    for col_name, val in zip(column_names, row):
      if col_name in numerical_features:
        example.features.feature[col_name].int64_list.value.append(val)
      else:
        example.features.feature[col_name].bytes_list.value.append(val.strip())
    writer.write(example.SerializeToString())
