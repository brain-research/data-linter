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
"""DataLinter runs a set of lint detectors and writes out results.

The idea of the DataLinter is as a unit test suite for data. Accordingly,
it runs a battery of `LintDetector`s against the examples and/or user-provided
information (e.g., stats, a schema).

Each `LintDetector` yields at most one `LintResult` proto which summarizes
the findings of that detector. The collection of `LintResult`s is
written as a table in which keys are `LintDetector` names and values
are their `LintResult`s.

For efficiency [via parallelism], DataLinter uses Apache Beam to process
examples. Each `LintDetector` must, therefore, return a `PCollection`
containing its `LintResult`. These `PCollections` are then merged and
written out.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import apache_beam as beam
import lint_result_pb2

class DataLinter(beam.PTransform):
  """A wrapper for running a pipeline from examples to linters to results."""

  def __init__(self, linters, results_path):
    """Initializes a `DataLinter`.

    Args:
      linters: An `Iterable` containing `LintDetector`s to run.
        Must not contain duplicates.
      results_path: A string specifying the file path to which to write
        the results.
    """
    self._linters = set(linters)
    self._results_path = results_path

  def expand(self, examples):
    """Runs the linters on the data and writes out the results.

    The order in which the linters run is unspecified.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Examples`.

    Returns:
      A pipeline containing the `DataLinter` `PTransform`s.
    """
    coders = (beam.coders.coders.StrUtf8Coder(),
              beam.coders.coders.ProtoCoder(lint_result_pb2.LintResult))
    return (
        [examples | linter for linter in self._linters if linter.should_run()]
        | 'MergeResults' >> beam.Flatten()
        | 'DropEmpty' >> beam.Filter(lambda (_, r): r and len(r.warnings))
        | 'ToDict' >> beam.combiners.ToDict()
        | 'WriteResults' >> beam.io.textio.WriteToText(
            self._results_path,
            coder=beam.coders.coders.PickleCoder(),
            shard_name_template=''))
