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
"""Implementations of LintDetectors.

Naming conventions:
num_x: `x` is numeric (as opposed to some other type)
n_x: The count of `x`.
freq_x: A ratio of counts (e.g., n_some_subset_of_x / n_x)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import itertools
import re

import apache_beam as beam
import dateutil.parser
import numpy as np
import scipy.stats
import six

import lint_result_pb2
import data_linter_utils as utils


class LintDetector(beam.PTransform):
  """Base class for lint detectors.

  To create a new `LintDetector`, one must implement the `_lint` method.
  By default, the only data provided to a `LintDetector` object is the
  `PCollection` of examples. To use other data (e.g., statistics),
  one must implement a constructor that stores these data.
  Note that because the `_lint` method allows returning a bare `LintResult`,
  there is no requirement to use the examples and the associated `Pipeline`.
  """

  N_LINT_SAMPLES = 2

  def should_run(self):
    """Returns whether the linter should run."""
    return True

  def __eq__(self, other):
    return isinstance(other, type(self))  # only run each linter once

  @classmethod
  def _make_result(cls, **kwargs):
    return lint_result_pb2.LintResult(linter_name=cls.__name__, **kwargs)

  def _lint(self, examples):
    """Performs linting and returns the result.

    This must be implemented by `LintDetector` subclasses.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      If this linter has results, this method must return either a `LintResult`
      or a `PTransform` that yields a `PCollection` containing exactly one.
      Otherwise, this function may return None, an empty `PCollection`, or a
      `LintResult` with an empty `warnings` list.
    """
    raise NotImplementedError()

  def expand(self, examples):
    """Implements the interface required by `PTransform`.

    Args:
      examples: A `PTransform` that yields a `PCollection` of tf.Examples.

    Returns:
      A `PTransform` that yields a `PCollection` containing at most one tuple in
      which the first element is the `LintDetector` name and the second is the
      `LintResult`.
    """

    result = self._lint(examples)
    if not isinstance(result,
                      (beam.pvalue.PCollection, beam.transforms.PTransform)):
      result_pcoll = beam.Create([result] if result else [])
      result = examples.pipeline | 'Materialize' >> result_pcoll
    return result | 'PairWithName' >> beam.Map(
        lambda r: (type(self).__name__, r))


class DateTimeAsStringDetector(LintDetector):
  """Detects datetime-like objects encoded as strings."""

  _NEAR_FUTURE_YEAR = datetime.datetime.today().year + 100
  _EPOCH_YEAR = 1970

  @classmethod
  def _string_is_datetime(cls, maybe_dt):
    try:
      dateutil.parser.parse(maybe_dt)
      try:
        # `dateutil.parser.parse` will treat small numbers as year/month/day.
        # We don't want to flag bare numbers unless they're timestamps
        # or potential recent years.
        float_dt = float(maybe_dt)
        if float_dt > 1e8:
          return True  # it might be a unix timestamp
        elif float_dt >= cls._EPOCH_YEAR and float_dt <= cls._NEAR_FUTURE_YEAR:
          return True  # it might be a bare year
        return False
      except ValueError:
        return True
    except (ValueError, OverflowError):
      return False

  def __init__(self, stats):
    """Constructs a `DateTimeAsStringDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto.
    """
    super(DateTimeAsStringDetector, self).__init__()
    self._stats = stats

  def should_run(self):
    return bool(utils.get_string_features(self._stats))

  def _lint(self, examples):
    """Returns the result of the `DateTimeAsStringDetector` linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_sample: [{ strings=[vals..] } for each warning]
    """
    result = self._make_result()
    string_features = utils.get_string_features(self._stats)
    lint_samples = collections.defaultdict(set)

    for feature in self._stats.features:
      if feature.name not in string_features:
        continue
      str_stats = feature.string_stats
      n_samples = str_stats.common_stats.num_non_missing
      if n_samples == 0:
        continue
      num_date_parsable = 0
      for bucket in str_stats.rank_histogram.buckets:
        if self._string_is_datetime(bucket.label):
          num_date_parsable += bucket.sample_count
          samples = lint_samples[feature.name]
          if len(samples) < self.N_LINT_SAMPLES:
            samples.add(bucket.label)

      if num_date_parsable / n_samples > 0.5:
        result.warnings.append(feature.name)
        result.lint_samples.add(strings=lint_samples[feature.name])

    return result


class TokenizableStringDetector(LintDetector):
  """Detects long strings which may need tokenization."""

  def __init__(self, stats, length_threshold=30,
               enum_threshold=20):
    """Constructs a `TokenizableStringDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto.
      length_threshold: Maximum length of a string before which it is flagged
                        as potentially needing tokenization.
      enum_threshold: Number of unique strings above which
                      feature will be considered as non-enum.
    """
    super(TokenizableStringDetector, self).__init__()
    self._stats = stats
    self._length_threshold = length_threshold
    self._enum_threshold = enum_threshold

  def should_run(self):
    return bool(utils.get_string_features(self._stats))

  def _lint(self, examples):
    """Returns the result of the `TokenizableStringDetector` linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_samples: [{ strings=[vals..] } for each warning]
    """
    result = self._make_result()
    string_features = utils.get_string_features(self._stats)
    for feature in self._stats.features:
      if feature.name not in string_features:
        continue
      str_stats = feature.string_stats
      if (str_stats.avg_length > self._length_threshold and
          str_stats.unique > self._enum_threshold):
        result.warnings.append(feature.name)
        samples = [bucket.label for bucket in str_stats.rank_histogram.buckets
                   if len(bucket.label) > self._length_threshold]
        result.lint_samples.add(strings=samples[:self.N_LINT_SAMPLES])

    return result


class ZipCodeAsNumberDetector(LintDetector):
  """Detects numeric features that may actually be zip codes."""

  _ZIP_RE = re.compile(r'([\W_]|\b)zip(code|[\W_]|\b)')

  def __init__(self, stats):
    """Constructs a `ZipCodeAsNumberDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto.
    """
    super(ZipCodeAsNumberDetector, self).__init__()
    self._stats = stats

  def should_run(self):
    return bool(utils.get_numeric_features(self._stats))

  def _lint(self, examples):
    """Returns the result of the `ZipCodeAsNumberDetector` linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_samples: None
    """
    result = self._make_result()
    numeric_features = utils.get_numeric_features(self._stats)
    for feature in self._stats.features:
      if (feature.name in numeric_features and
          self._ZIP_RE.search(feature.name.lower())):
        result.warnings.append(feature.name)
    return result


class NumberAsStringDetector(LintDetector):
  """Detects numbers encoded as strings."""

  def __init__(self, stats, non_num_tol=0.5):
    """Constructs a `NumberAsStringDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      non_num_tol: Proportion of non-number characters to tolerate when treating
                   a string as numeric. Increase this to flag more potential
                   numeric strings. 0.5 is a reasonable choice since it permits
                   formats like currency ($N.NN) and percents (1%) while passing
                   actual strings.
    """
    super(NumberAsStringDetector, self).__init__()
    self._stats = stats
    self._non_num_tol = non_num_tol

  def should_run(self):
    return bool(utils.get_string_features(self._stats))

  def _lint(self, examples):
    """Returns the result of the `NumberAsStringDetector` linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_samples: [{ strings=[vals..] } for each warning]
    """
    result = self._make_result()
    string_features = utils.get_string_features(self._stats)
    lint_samples = collections.defaultdict(set)

    for feature in self._stats.features:
      if feature.name not in string_features:
        continue
      str_stats = feature.string_stats
      n_samples = str_stats.common_stats.num_non_missing
      if n_samples == 0:
        continue
      num_numeric = 0
      for bucket in str_stats.rank_histogram.buckets:
        try:
          nums_only = re.sub(r'\D', '', bucket.label)
          if len(nums_only) / len(bucket.label) >= 1 - self._non_num_tol:
            num_numeric += bucket.sample_count
            samples = lint_samples[feature.name]
            if len(samples) < self.N_LINT_SAMPLES:
              samples.add(bucket.label)
        except (ValueError, ZeroDivisionError):
          pass

      if num_numeric / n_samples > 0.5:
        result.warnings.append(feature.name)
        result.lint_samples.add(strings=lint_samples[feature.name])

    return result


class NonNormalNumericFeatureDetector(LintDetector):
  """Detects numeric features that are scaled differently from the rest."""

  IGNORE_FEATURE_NAMES = {'lat', 'lon', 'latitude', 'longitude', 'id'}
  _TYPICAL_STATS_ID = '_typical_'
  WARNING_FMT = '{}:{}'  # feature_name:stat_types

  def __init__(self, stats, max_deviance=2, trim_proportion=0.1):
    """Constructs a `NonNormalNumericFeatureDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      max_deviance: The z-score of a feature's mean/standard deviation from
                    the trimmed version of the statistic above which the
                    feature is considered as needing rescaling. Set this to
                    a higher value to be more tolerant of widely varying scales.
      trim_proportion: The fraction of feature means/stds to trim from either
                       end of the empirical distributions when computing the
                       trimmed statistics. For example, trim_proportion=0.1
                       means that the top/bottom 10% of values will be ignored.
                       The median represents a trim_proportion of >= 0.5.
    """
    super(NonNormalNumericFeatureDetector, self).__init__()
    self._stats = stats
    self._max_deviance = max_deviance
    self._trim_proportion = trim_proportion

  def should_run(self):
    return any(feature_name.lower() not in self.IGNORE_FEATURE_NAMES
               for feature_name in utils.get_numeric_features(self._stats))

  def _get_trimmed_stats(self, values):
    values.sort()
    trimmed_values = scipy.stats.trimboth(values, self._trim_proportion)
    return trimmed_values.mean(), trimmed_values.std()

  def _lint(self, examples):
    """Returns the result of the `NonNormalNumericFeatureDetector` linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_sample: [
          stats: {mean, std_dev}  # for a "typical" numeric feature
          stats: {mean, std_dev, min, max}  # for each flagged feature
        ]
    """
    result = self._make_result()
    numeric_features = utils.get_numeric_features(self._stats)
    numeric_feature_stats = []
    feature_means = []
    feature_std_devs = []
    for feature_stats in self._stats.features:
      if (feature_stats.name not in numeric_features
          or feature_stats.name in self.IGNORE_FEATURE_NAMES):
        continue
      numeric_feature_stats.append(feature_stats)
      num_stats = feature_stats.num_stats
      feature_means.append(num_stats.mean)
      feature_std_devs.append(num_stats.std_dev)

    means_trimmed_mean, means_trimmed_std = self._get_trimmed_stats(
        feature_means)
    std_devs_trimmed_mean, std_devs_trimmed_std = self._get_trimmed_stats(
        feature_std_devs)

    typical_stats = lint_result_pb2.Statistics(
        id=self._TYPICAL_STATS_ID,
        mean=means_trimmed_mean, std_dev=std_devs_trimmed_mean)
    result.lint_samples.add(stats=[typical_stats])

    for feature_stats in numeric_feature_stats:
      num_stats = feature_stats.num_stats
      mean_deviance = utils.get_zscore(
          num_stats.mean, means_trimmed_mean, means_trimmed_std)
      std_dev_deviance = utils.get_zscore(
          num_stats.std_dev, std_devs_trimmed_mean, std_devs_trimmed_std)
      warnings = []
      if mean_deviance > self._max_deviance:
        warnings.append('mean')
      if std_dev_deviance > self._max_deviance:
        warnings.append('std_dev')
      if warnings:
        result.warnings.append(
            self.WARNING_FMT.format(feature_stats.name, ','.join(warnings)))
        result.lint_samples.add(stats=[lint_result_pb2.Statistics(
            id=feature_stats.name,
            mean=num_stats.mean, std_dev=num_stats.std_dev,
            min=num_stats.min, max=num_stats.max)])

    return result


class UniqueValueCountsDetector(LintDetector):
  """A base class for `LintDetector`s that use the counts of unique values.

  A subclass of `UniqueValueCountsDetector` must provide a `_counted_features`
  property and implement the `_check_feature` method.
  Optionally, a `_count_transformer` property may be specified to transform
  the raw feature-value counts.

  A `UniqueValueCountsDetector` will run when it has features to count, as
  specified by `_counted_features`.

  Produces a `LintResult` such that `zip(warnings, lint_samples)` yields
  [(feat1_warning, feat1_samples), (feat2_warning, feat2_samples), ...].
  """

  @property
  def _counted_features(self):
    """Returns an iterable of unique feature names with values to count."""
    raise NotImplementedError()

  @property
  def _count_transformer(self):
    """Returns a `PTransform` that modifies the raw feature-value counts.

    The `PTransform` will receive as its pipeline input a `PCollection`
    containing entries of the format ((feature_name, feature_val), count) and
    must produce a `PCollection` containing entries of the same format.
    """
    raise NotImplementedError()

  def _check_feature(self, feature_w_val_counts):
    """Checks the feature-value counts for lint.

    Args:
      feature_w_val_counts: A tuple of the format (feature_name, counts_dict)
                            where `counts_dict` is a dict of item counts.

    Returns:
      Either a tuple of the format (warning, lint_sample) where
      `warning` is a non-empty string and `lint_sample` is a `LintSample`.
      or None if there are no warnings for this feature.
    """
    raise NotImplementedError()

  def should_run(self):
    return bool(self._counted_features)

  def _flatten_feature_vals(self, feature_vals):
    feature, vals = feature_vals
    return [(feature, v) for v in vals]

  def _shift_key(self, feature_val_w_counts):
    (feature, val), counts = feature_val_w_counts
    return feature, (val, counts)

  def _val_counts_as_dict(self, feature_val_counts):
    feature, val_counts = feature_val_counts
    return feature, dict(val_counts)

  def _to_result(self, warning_samples):
    if warning_samples:
      warnings, samples = zip(*warning_samples)
      return self._make_result(warnings=warnings, lint_samples=samples)

  def _lint(self, examples):
    feature_val_w_counts = (
        examples
        | 'Tuplize' >> beam.FlatMap(
            utils.example_tuplizer(self._counted_features))
        | 'FlattenFeatureVals' >> beam.FlatMap(self._flatten_feature_vals)
        | 'CountFeatureVals' >> beam.combiners.Count.PerElement())

    if hasattr(self, '_count_transformer'):
      feature_val_w_counts |= 'TransformCounts' >> self._count_transformer

    return (
        feature_val_w_counts
        | 'PairValWithCount' >> beam.Map(self._shift_key)
        | 'GroupByFeature' >> beam.GroupByKey()
        | 'ValCountsToDict' >> beam.Map(self._val_counts_as_dict)
        | 'GenResults' >> beam.Map(self._check_feature)
        | 'DropUnwarned' >> beam.Filter(bool)
        | 'AsList' >> beam.combiners.ToList()
        | 'ToResult' >> beam.Map(self._to_result))


class EnumDetector(UniqueValueCountsDetector):
  """Detects categorical features."""

  N_LINT_SAMPLES = 4

  def __init__(self, stats, enum_threshold=20, ignore_strings=True):
    """Constructs a `EnumDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      enum_threshold: Number of unique values above which a feature will be
                      regarded as real valued rather than as an enum.
      ignore_strings: Whether to assume that strings are already enums.
    """
    super(EnumDetector, self).__init__()
    self._stats = stats
    self._enum_threshold = enum_threshold
    self._ignore_strings = ignore_strings
    self._numeric_features = utils.get_numeric_features(self._stats)

  @property
  def _counted_features(self):
    checked_features = self._numeric_features
    if not self._ignore_strings:
      checked_features.update(utils.get_string_features(self._stats))
    return checked_features

  def _check_feature(self, feature_w_val_counts):
    """Returns the result of the `EnumDetector` linter.

    Args:
      feature_w_val_counts: A tuple of the format (feature_name, counts)
                            where `counts` is a dict containing the number of
                            times each unique feature value occurs.

    Returns:
      Either a tuple of the format (warning, lint_sample) where
        warning: feature_name
        lint_sample: LintSample(strings|nums=[val1, ...])
      or None if there are no warnings for the feature.
    """
    feature, counts = feature_w_val_counts
    if len(counts) >= self._enum_threshold:
      return None
    samp_vals = itertools.islice(iter(counts), self.N_LINT_SAMPLES)
    if feature not in self._numeric_features:
      samp_strs = [six.text_type(s).encode('utf8') for s in samp_vals]
      samples = lint_result_pb2.LintSample(strings=samp_strs)
    else:
      samples = lint_result_pb2.LintSample(nums=samp_vals)
    return feature, samples


class IntAsFloatDetector(UniqueValueCountsDetector):
  """Detects a (non-categorical) integral feature encoded as a float."""

  def __init__(self, stats, int_threshold=0.95):
    """Constructs an `IntAsFloatDetector` linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      int_threshold: Fraction of examples that must be integral for the
                     feature to be considered integral.
    """
    super(IntAsFloatDetector, self).__init__()
    self._stats = stats
    self._int_threshold = int_threshold

  @property
  def _counted_features(self):
    return utils.get_float_features(self._stats)

  @property
  def _count_transformer(self):
    return (
        'DropNaN' >> beam.Filter(lambda (f_v, _): not np.isnan(f_v[1]))
        | 'IsIntegral' >> beam.Map(
            lambda (f_v, c): ((f_v[0], f_v[1] % 1 == 0), c))
        | 'Count' >> beam.CombinePerKey(sum))

  def _check_feature(self, feature_w_intp_counts):
    """Returns the result of the `IntAsFloatDetector` linter.

    Args:
      feature_w_intp_counts: A tuple of the format (feature_name, intp_counts)
                             where `intp_counts` is a dictionary with boolean
                             keys representing integral or not integral and
                             values of the count of non-missing
                             integral/non-integral values taken by the feature.

    Returns:
      Either a tuple of the format (warning, lint_sample) where
        warning: feature_name
        lint_sample: LintSample(nums=[num_non_missing, num_integral])
      or None if there are no warnings for the feature.
    """
    feature, intp_counts = feature_w_intp_counts
    num_present = sum(six.itervalues(intp_counts))
    int_count = intp_counts.get(True, 0)
    if int_count / num_present >= self._int_threshold:
      sample = lint_result_pb2.LintSample(nums=[num_present, int_count])
      return feature, sample
    return None


class UncommonSignDetector(UniqueValueCountsDetector):
  """Detects numeric features with values that uncommonly take a certain sign.

  Flags numeric features with values that have most, but not all, of their
  signs in a particular domain, where domain is defined as
  {positive, negative, zero, nan}.

  The motivating example is of a custom placeholder value of -999 that's the
  only negative value taken by the feature.
  """

  _SIGN_TO_STR = {1: 'positive', -1: 'negative', 0: 'zero'}

  def __init__(self, stats, domain_freq_threshold=0.05):
    """Constructs a UncommonSignDetector linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      domain_freq_threshold: The minimum fraction of the time that a feature's
                             unique values must have a particular domain for it
                             to be considered not-unusual.
    """
    super(UncommonSignDetector, self).__init__()
    self._stats = stats
    self._domain_freq_threshold = domain_freq_threshold

  @property
  def _counted_features(self):
    return utils.get_numeric_features(self._stats)

  @property
  def _count_transformer(self):
    return (
        'ToSigns' >> beam.Map(
            lambda (f_v, _): (f_v[0], np.sign(f_v[1])))
        | 'CountSigns' >> beam.combiners.Count.PerElement())

  def _check_feature(self, feature_sign_counts):
    """Returns the result of the UncommonSignDetector linter.

    Args:
      feature_sign_counts: A tuple of the format (feature_name, sign_counts)
                           where `sign_counts` is a dict from sign strings
                           (+/-/0/nan) to the number of unique values with
                           that sign.

    Returns:
      A tuple of the format (warnings, lint_sample) where
        warnings: [feature_name]
        lint_sample: [
          [nums=[n_unique_vals, n_with_sign1, ...]
           strings=[uncommon_sign1, uncommon_sign2, ...]]
          for each feature
        ]
    """
    feature_name, sign_counts = feature_sign_counts
    num_stats = next(stats for stats in self._stats.features
                     if stats.name == feature_name).num_stats

    n_unique = sum(six.itervalues(sign_counts))
    uncommon_sign_counts = {}
    for sign, count in six.iteritems(sign_counts):
      # For 0 and NaN, the type (as opposed to token) count will always be
      # either 0 or 1. It's not obvious how to threshold 1/N for being uncommon
      # so the token counts are used instead.
      if sign == 0:
        count = num_stats.num_zeros
      elif sign == float('nan'):
        common_stats = num_stats.common_stats
        count = common_stats.tot_num_values - common_stats.num_non_missing
      sign_freq = count / n_unique
      if sign_freq > 0 and sign_freq <= self._domain_freq_threshold:
        uncommon_sign_counts[sign] = count

    if uncommon_sign_counts:
      sample = lint_result_pb2.LintSample(nums=[n_unique])
      for sign, count in six.iteritems(uncommon_sign_counts):
        sample.strings.append(self._SIGN_TO_STR.get(sign, str(sign)))
        sample.nums.append(count)
      return feature_name, sample
    return None


class DuplicateExampleDetector(LintDetector):
  """Detects duplicated examples."""

  N_LINT_SAMPLES = 10

  def __init__(self, stats):
    """Constructs a DuplicateExampleDetector linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
    """
    super(DuplicateExampleDetector, self).__init__()
    self._stats = stats

  def _to_result(self, _, n_duplicates, samples):
    warning = [str(n_duplicates)] if n_duplicates else []
    return self._make_result(warnings=warning, lint_samples=[samples])

  def _lint(self, examples):
    """Returns the `PTransform` for the DuplicateExampleDetector linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `PTransform` that yields a `LintResult` of the format
        warnings: [num_duplicates]
        lint_sample: [ features: [sample duplicates...] ]
    """
    feature_names = sorted(f.name for f in self._stats.features)
    tuplize = utils.example_tuplizer(feature_names, denan=True)

    duplicates = (
        examples
        | 'Tuplize' >> beam.Map(lambda x: (tuplize(x), x))
        | 'CollectDuplicates' >> beam.GroupByKey()
        | 'ExamplesToList' >> beam.Map(
            lambda (example_tuple, examples): (example_tuple, list(examples)))
        | 'FilterDuplicates' >> beam.Filter(
            lambda (_, examples): len(examples) > 1))

    samples = (
        duplicates
        | 'TakeExamples' >> beam.Map(lambda (_, examples): examples[0])
        | 'Sample' >> beam.combiners.Sample.FixedSizeGlobally(
            self.N_LINT_SAMPLES)
        | 'ToSample' >> beam.Map(
            lambda x: lint_result_pb2.LintSample(examples=x)))

    n_duplicates = (
        duplicates
        | 'CountDuplicates' >> beam.Map(lambda (_, examples): len(examples))
        | 'ExcessCounts' >> beam.Map(lambda x: x - 1)
        | 'Total' >> beam.CombineGlobally(sum))

    return (
        # this is effectively a `Flatten` but with deterministic argument order
        examples.pipeline
        | 'SyncSideInputs' >> beam.Create([None])
        | 'ToResult' >> beam.Map(self._to_result,
                                 beam.pvalue.AsSingleton(n_duplicates),
                                 beam.pvalue.AsSingleton(samples)))


class EmptyExampleDetector(LintDetector):
  """Detects examples that contain only missing/empty values."""

  def _example_is_empty(self, example):
    features = example.features.feature.values()
    kinds = [feature.WhichOneof('kind') for feature in features]
    vals = (getattr(feature, kind).value
            for feature, kind in zip(features, kinds) if kind)
    return all(utils.is_empty(val) for val in vals)

  def _lint(self, examples):
    """Returns the `PTransform` for the EmptyExampleDetector linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `PTransform` that yields a `LintResult` of the format
        warnings: [num empties]
        lint_sample: None
    """
    n_empties = (
        examples
        | 'DetectEmpties' >> beam.Map(self._example_is_empty)
        | 'Count' >> beam.CombineGlobally(sum)
        | 'NoZero' >> beam.Filter(bool)
        | 'ToResult' >> beam.Map(
            lambda w: self._make_result(warnings=[str(w)])))
    return n_empties


class UncommonListLengthDetector(LintDetector):
  """Detects list features that have an unusual number of elements."""

  def __init__(self, stats, dropoff_threshold=0.85):
    """Constructs a UncommonListLengthDetector linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      dropoff_threshold: Relative drop between most and second most common
                         lengths to be considered suspicious.
    """
    super(UncommonListLengthDetector, self).__init__()
    self._stats = stats
    self._dropoff_threshold = dropoff_threshold

    self._variable_length_features = set()
    for feature in self._stats.features:
      common_stats = utils.get_stats(feature).common_stats
      if common_stats.min_num_values != common_stats.max_num_values:
        self._variable_length_features.add(feature.name)

  def should_run(self):
    return bool(self._variable_length_features)

  def _lint(self, examples):
    """Returns the result of the UncommonListLengthDetector linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_sample: [
          nums: [num_samples]       # total number of samples for feature
          stats: {count, min, max}  # total/min/max samples in top bucket
          stats: {count, min, max}  # total/min/max samples in outlier bucket
          for each flagged feature
        ]
    """
    result = self._make_result()

    for feature in self._stats.features:
      if feature.name not in self._variable_length_features:
        continue
      common_stats = utils.get_stats(feature).common_stats
      n_samples = common_stats.num_non_missing
      n_values_histogram = common_stats.num_values_histogram

      unique_bucket_counts = collections.defaultdict(int)
      for b in n_values_histogram.buckets:
        unique_bucket_counts[b.low_value, b.high_value] += b.sample_count

      top_two_buckets_w_counts = sorted(unique_bucket_counts.items(),
                                        key=lambda x: -x[1])[:2]
      most_common_bucket_count, second_most_common_bucket_count = [
          count for _, count in top_two_buckets_w_counts]
      dropoff = ((most_common_bucket_count - second_most_common_bucket_count) /
                 most_common_bucket_count)

      if dropoff >= self._dropoff_threshold:
        result.warnings.append(feature.name)
        top2_lohi = [lint_result_pb2.Statistics(count=n, min=lo, max=hi)
                     for (lo, hi), n in top_two_buckets_w_counts]
        result.lint_samples.add(stats=top2_lohi, nums=[n_samples])

    return result


class TailedDistributionDetector(LintDetector):
  """Detects numeric features with tailed distributions.

  The primary goal of this linter is to detect custom missing/placeholder
  values (like -999). It also may find obvious statistical outliers.
  """

  _MIN = 'min'
  _MAX = 'max'

  def __init__(self, stats, z_score_threshold=0.5):
    """Constructs a TailedDistributionDetector linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
      z_score_threshold: The z-score of the min/max-trimmed mean (using the
                         un-trimmed standard deviation) above which a feature's
                         distribution will be considered tailed.
    """
    super(TailedDistributionDetector, self).__init__()
    self._stats = stats
    self._z_score_threshold = z_score_threshold

    self.numeric_features = utils.get_numeric_features(self._stats)
    self.feature_num_stats = {}
    for feature in self._stats.features:
      if feature.name not in self.numeric_features:
        continue
      self.feature_num_stats[feature.name] = feature.num_stats

  def should_run(self):
    return bool(self.numeric_features)

  def _flatten_feature_vals(self, sel_features):
    def _flattener(example):
      feature_values = []
      for feature in sel_features:
        values = utils.get_feature(example, feature)[1]
        feature_values.extend((feature, value) for value in values
                              if np.isfinite(value))
      return feature_values
    return _flattener

  def _to_result(self, feature_trimmed_means):
    result = self._make_result()
    for feature, trimmed_means in feature_trimmed_means:
      ([min_trimmed_mean], [max_trimmed_mean]) = trimmed_means
      stats = self.feature_num_stats[feature]
      z_min = utils.get_zscore(min_trimmed_mean, stats.mean, stats.std_dev)
      z_max = utils.get_zscore(max_trimmed_mean, stats.mean, stats.std_dev)
      outlying = {}
      if z_min > self._z_score_threshold:
        outlying[self._MIN] = stats.min
      if z_max > self._z_score_threshold:
        outlying[self._MAX] = stats.max
      if outlying:
        result.warnings.append(feature)
        result.lint_samples.add(stats=[
            lint_result_pb2.Statistics(id=','.join(outlying), **outlying)])
    return result

  def _make_trimmed_averager(self, extremum):
    pipeline_branch = 'Trim' + extremum.capitalize()

    def _value_is_non_extremal(feature_value):
      feature, value = feature_value
      return value != getattr(self.feature_num_stats[feature], extremum)

    return (
        pipeline_branch >> beam.Filter(_value_is_non_extremal)
        | pipeline_branch + 'Mean' >> beam.combiners.Mean.PerKey())

  def _lint(self, examples):
    """Returns the result of the TailedDistributionDetector linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `PTransform` that yields a `LintResult` of the format
        warnings: [feature names]
        lint_samples: [
          [stats: {min: feature_min if outlying, max: feature_max if outlying}]
          for each warning
        ]
    """

    feature_values = (
        examples
        | 'FlattenFeatureValue' >> beam.FlatMap(
            self._flatten_feature_vals(self.numeric_features)))

    feature_min_trimmed_mean = (
        feature_values | self._make_trimmed_averager(self._MIN))
    feature_max_trimmed_mean = (
        feature_values | self._make_trimmed_averager(self._MAX))

    return (
        (feature_min_trimmed_mean, feature_max_trimmed_mean)
        | 'MergeTrimmedMeans' >> beam.CoGroupByKey()
        | 'AsList' >> beam.combiners.ToList()
        | 'ToResult' >> beam.Map(self._to_result))


class CircularDomainDetector(LintDetector):
  """Detects features with values that wrap around (e.g., time, angle)."""

  CIRCULAR_NAME_RES = map(re.compile, [
      r'deg([\W_]|\b)', r'(wind.*|^)degrees?$', r'rad([\W_]|ian|\b)',   # degree
      r'(month|week|day|time|hour|min(ute)?|sec(ond)?)[\W_]?o[f\W_]',   # x of y
      r'^(week|day|hour|month|(milli|micro)?sec((ond)?s?)|minutes?)$',  # times
      r'([\W_]|\b)(lat|lon)([\W_]|\b|\w*?itude)',                       # latlon
      r'([\W_]|\b)angle([\W_]|\b)', 'heading', 'rotation'])
      # r'dir([\W_]|ection)']) this one is flaky

  def __init__(self, stats):
    """Constructs a CircularDomainDetector linter.

    Args:
      stats: A `DatasetFeatureStatisticsList` proto describing the examples.
    """
    super(CircularDomainDetector, self).__init__()
    self._stats = stats

  def should_run(self):
    return bool(utils.get_numeric_features(self._stats))

  def _name_is_suspicious(self, name):
    canonical_name = name.lower()
    return any(circular_name_re.search(canonical_name)
               for circular_name_re in self.CIRCULAR_NAME_RES)

  def _lint(self, examples):
    """Returns the result of the CircularDomainDetector linter.

    Args:
      examples: A `PTransform` that yields a `PCollection` of `tf.Example`s.

    Returns:
      A `LintResult` of the format
        warnings: [feature names]
        lint_sample: None
    """
    result = self._make_result()
    numeric_features = utils.get_numeric_features(self._stats)
    for feature in self._stats.features:
      name = feature.name
      if name in numeric_features and self._name_is_suspicious(name):
        result.warnings.append(name)
    return result
