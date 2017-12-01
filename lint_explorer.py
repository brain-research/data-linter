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
"""Utilities for understanding linter warnings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import lint_result_pb2
import linters
import explanations


DESC_ATTR = 'DESCRIPTION'
TRIGGER_TMPL = '* {}'
SUMMARY_TMPL = 'The following linter(s) triggered on your dataset:\n{}'
CONGRATS = 'Congratulations, DataLinter found no issues with your data!'


def load_lint_results(results_path):
  """Loads LintResult protos."""
  with open(results_path) as fin:
    return pickle.load(fin)


def suppress_warnings(results):
  """Suppresses warnings based on implications between linters.

  Args:
    results: a dict of linter_name: LintResults

  Returns:
    A dict of linter_name: {warnings} representing the warnings to suppress for
    that linter.
  """
  suppressed = {}

  def _suppress(general_linter, specific_linter):
    """Suppress the warnings of the more general linter."""
    general_linter_name = general_linter.__name__
    specific_linter_name = specific_linter.__name__
    r_gen = results.get(general_linter_name)
    r_sp = results.get(specific_linter_name)
    if r_gen and r_sp:
      suppressed[general_linter_name] = (
          set(r_gen.warnings) & set(r_sp.warnings))

  # DateTimeAsString is more specific than NumberAsString
  _suppress(linters.NumberAsStringDetector,
            linters.DateTimeAsStringDetector)

  # IntAsFloat is uninformative when it's already known to be an enum
  _suppress(linters.IntAsFloatDetector, linters.EnumDetector)
  _suppress(linters.EnumDetector, linters.ZipCodeAsNumberDetector)

  # enums aren't numbers
  _suppress(linters.NonNormalNumericFeatureDetector, linters.EnumDetector)
  _suppress(linters.TailedDistributionDetector, linters.EnumDetector)

  return suppressed


def format_results(results, linter_suppress={}):  # pylint: disable=dangerous-default-value
  """Returns a pretty formatted string for a collection of LintResults.

  Args:
    results: a dict of linter names to their respective LintResults protos
    linter_suppress: a dict from linter name to sets of suppressed warnings

  Returns:
    A string suitable for display to a user.
  """
  triggered = {l for l, r in results.items()
               if len(r.warnings) - len(linter_suppress.get(l, {})) > 0}
  if not triggered:
    return CONGRATS

  triggered_list = sorted(triggered)
  triggered_str = '\n'.join(map(TRIGGER_TMPL.format, triggered_list))

  result_strs = [SUMMARY_TMPL.format(triggered_str)]
  for linter_name in triggered_list:
    result = results[linter_name]
    linter = getattr(linters, linter_name)
    hr = '=' * 80
    lines = [hr, linter_name, hr]
    if hasattr(linter, DESC_ATTR):
      lines.extend([getattr(linter, DESC_ATTR).strip(), '-----'])
    suppress = linter_suppress.get(linter_name, set())
    wstrs = linter.format_warnings(result, suppress=suppress)
    lines.extend(wstrs)
    lines.append(hr)
    result_strs.append(u'\n'.join(lines))
  return '\n\n\n'.join(result_strs)
