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
"""Explains the results of data linting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pydoc

import lint_explorer
import explanations


parser = argparse.ArgumentParser()
parser.add_argument('--results_path',
                    help='path to the Data Linter results.',
                    default='/tmp/datalinter/results/lint_results.bin')
parser.add_argument('--page_results',
                    help='whether to page the results.',
                    default=True)
args = parser.parse_args()


def main():
  lint_results = lint_explorer.load_lint_results(args.results_path)
  suppressed_warnings = lint_explorer.suppress_warnings(lint_results)
  disp_results = lint_explorer.format_results(lint_results, suppressed_warnings)
  if args.page_results:
    pydoc.pager(disp_results)
  else:
    print(disp_results)


if __name__ == '__main__':
  main()
