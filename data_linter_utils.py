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
"""Data Linter utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from feature_statistics_pb2 import FeatureNameStatistics as Stats

_BYTES_STRING = '__BYTES_VALUE__'


def _get_type_feature_stats(stats, type_):
  """Returns stats of features that have a specified type.

  Args:
    stats: A DatasetFeatureStatistics proto
    type_: A FeatureNameStatistics.Type value

  Returns:
    A list of FeatureNameStatistics for features that have the specified type.
  """
  return [f for f in stats.features if f.type == type_]


def get_string_features(stats):
  """Returns names of string features (that are not bytes)."""
  str_feature_stats = _get_type_feature_stats(stats, Stats.STRING)
  str_features = {f.name for f in str_feature_stats}
  for stats in str_feature_stats:
    buckets = stats.string_stats.rank_histogram.buckets
    n_bytes_buckets = sum(bucket.label == _BYTES_STRING for bucket in buckets)
    if len(buckets) == n_bytes_buckets:
      str_features.remove(stats.name)
  return str_features


def get_bytes_features(stats):
  """Returns names of bytes features and string features with no encoding."""
  bytes_features = {f.name for f in _get_type_feature_stats(stats, Stats.BYTES)}
  str_features = {f.name for f in _get_type_feature_stats(stats, Stats.STRING)}
  bytes_string_features = str_features - get_string_features(stats)
  return bytes_features | bytes_string_features


def get_float_features(stats):
  """Returns names of float features."""
  return {f.name for f in _get_type_feature_stats(stats, Stats.FLOAT)}


def get_int_features(stats):
  """Returns names of int features."""
  return {f.name for f in _get_type_feature_stats(stats, Stats.INT)}


def get_numeric_features(stats):
  """Returns names of both int and float features."""
  return get_int_features(stats) | get_float_features(stats)


def get_stats(feature):
  """Returns the type-specific statistics associated with a feature.

  Args:
    feature: A `FeatureNameStatistics` proto.

  Returns:
    The value of the `FeatureNameStatistics.stats` field.
  """
  return getattr(feature, feature.WhichOneof('stats'))


def get_feature(example, feature_name):
  """Returns the type and value list of a `tf.Feature`."""
  feature = example.features.feature.get(feature_name)
  if not feature:
    return None, []
  kind = feature.WhichOneof('kind')
  if kind is None:
    return None, []
  return kind, getattr(feature, kind).value


def is_empty(vals):
  if not vals:
    return True
  if issubclass(type(vals[0]), basestring):
    return sum(map(len, vals)) == 0
  return np.isnan(vals).all()


class _NaN(object):
  """NaN sentinel that equals itself."""

  def __eq__(self, other):
    return isinstance(other, type(self))
_nan = _NaN()  #  pylint: disable=g-wrong-blank-lines


def example_tuplizer(feature_names, denan=False):
  """Returns a function that converts a `tf.Example` to a standard format.

  A tuplized `tf.Example` will equal another tuplized `tf.Example` iff they
  have the same the values for the provided `feature_names` (unless there are
  `float('nans')` and `denan=False`).

  Args:
    feature_names: A list of names of features to tuplize.
    denan: Whether to replace `float('nan')` with a sentinel value that
           equals itself.

  Returns:
    A function that formats a `tf.Example` as
    (('feature1', (val1, val2, ..., valN)), ..., ('featureN', (val1, ...)))
  """

  def _denan(v):
    if not denan or not isinstance(v, float):
      return v
    return v if not np.isnan(v) else _nan

  def tuplizer(example):
    return tuple((name, tuple(map(_denan, get_feature(example, name)[1])))
                 for name in feature_names)
  return tuplizer


def get_zscore(x, mean, std_dev):
  if std_dev == 0:
    return 0
  return abs(x - mean) / std_dev
