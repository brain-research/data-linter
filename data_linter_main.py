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
"""DataLinter standalone binary.

The DataLinter binary runs the default set of linters against the
data+stats, and writes out the results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import apache_beam as beam

import data_linter
import example_pb2
from feature_statistics_pb2 import DatasetFeatureStatisticsList
import linters


# Flags.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path',
                    help='path to directory containing your '
                    'TFRecord-encoded dataset.')
parser.add_argument('--stats_path',
                    help='path where stats are stored.')
parser.add_argument('--results_path',
                    help='where DataLinter results are stored.',
                    default='/tmp/datalinter/results/lint_results.bin')
args = parser.parse_args()

# Some linters are currently disabled due to a bug.
DEFAULT_STATS_LINTERS = [  # These linters require dataset statistics.
    linters.CircularDomainDetector,
    linters.DateTimeAsStringDetector,
    linters.DuplicateExampleDetector,
#    linters.EnumDetector,
#    linters.IntAsFloatDetector,
    linters.NonNormalNumericFeatureDetector,
    linters.NumberAsStringDetector,
    linters.TailedDistributionDetector,
    linters.TokenizableStringDetector,
    linters.UncommonListLengthDetector,
#    linters.UncommonSignDetector,
    linters.ZipCodeAsNumberDetector,
]
DEFAULT_LINTERS = [
    linters.EmptyExampleDetector,
]


def main():
  if not os.path.exists(args.stats_path):
    raise ValueError('Error: stats path does not seem to exist (%s)' %
                     args.stats_path)

  stats = _read_feature_stats(args.stats_path)

  run_linters = [stats_linter(stats) for stats_linter in DEFAULT_STATS_LINTERS]
  run_linters.extend([linter() for linter in DEFAULT_LINTERS])
  datalinter = data_linter.DataLinter(run_linters, args.results_path)

  _ensure_directory_exists(args.results_path)
  with beam.Pipeline() as p:
    _ = (
        p
        | _make_dataset_reader(args.dataset_path,
                               beam.coders.ProtoCoder(example_pb2.Example))
        | 'LintData' >> datalinter)


def _ensure_directory_exists(path):
  directory_path = os.path.dirname(path)
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)


def _make_dataset_reader(dataset_path, example_coder):
  """Returns the appropriate reader for the dataset.

  Args:
    dataset_path: The path (or glob) to the dataset files.
    example_coder: A `ProtoCoder` for `tf.Example`s

  Returns:
    A `LabeledPTransform` that yields a `PCollection` of the
    `tf.Example`s in the dataset.
  """
  reader = beam.io.tfrecordio.ReadFromTFRecord(dataset_path,
                                               coder=example_coder)
  return 'ReadExamples' >> reader


def _read_feature_stats(stats_path):
  with open(stats_path) as fin:
    summaries = DatasetFeatureStatisticsList()
    summaries.ParseFromString(fin.read())
    return summaries.datasets[0]


if __name__ == '__main__':
  main()
