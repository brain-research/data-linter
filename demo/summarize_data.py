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
Generate summary statistics using Facets.

Facets must be installed. See: https://github.com/PAIR-code/facets

In particular, see the Python code here:
https://github.com/PAIR-code/facets/tree/master/facets_overview/python
"""
from feature_statistics_generator import ProtoFromTfRecordFiles

DATASET_PATH = "/tmp/adult.tfrecords"
OUTPUT_PATH = "/tmp/adult_summary.bin"
DATASET_NAME = "uci_census"

result = ProtoFromTfRecordFiles(
    [{"name": DATASET_NAME, "path": DATASET_PATH}],
    max_entries=1000000)
with open(OUTPUT_PATH, "w") as fout:
  fout.write(result.SerializeToString())
