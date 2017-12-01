# Data Linter

## Summary

This code accompanies the
[NIPS 2017 ML Systems Workshop](http://learningsys.org/nips17/) paper/poster,
"The Data Linter: Lightweight, Automated Sanity Checking for ML Data Sets."

The Data Linter identifies potential issues (lints) in your ML training data.

# Using the Data Linter

## Prerequisites

You'll need the following installed to use the Data Linter:

1. Python
2. [Apache Beam](https://beam.apache.org/)
3. [TensorFlow](https://www.tensorflow.org/)
4. [Facets](https://github.com/PAIR-code/facets)

## Data Linter Demo

The easiest way to see how to use the Data Linter is to follow the demo
instructions found in `demo/README.md`.

## Running the Data Linter

Running the Data Linter requires the following steps:

1. Encoding your data in TFRecord format.
2. Generating summary statistics for those data, using Facets.
3. Running the Data Linter.
4. Using the Lint Explorer to produce the lint results.

### Creating Data in the TFRecord Format

To see how to convert CSV files to the TFRecord format, look at the example code
in `demo/convert_to_tfrecord.py`.

### Summarizing Your Data Using Facets

To see how to generate summary statistics for your data, see the example code in
`demo/summarize_data.py`.

### Executing the Data Linter

Once you have both the data and summary statistics, you can run the Data Linter
as such:

```shell
python data_linter_main.py --dataset_path PATH_TO_TFRECORDS \
  --stats_path PATH_TO_FACETS_SUMMARIES --results_path PATH_FOR_SAVING_RESULTS
```

For example, if you follow the instructions in the demo folder, you'll invoke
the Data Linter like this:

```shell
python data_linter_main.py --dataset_path /tmp/adult.tfrecords \
  --stats_path /tmp/adult_summary.bin \
  --results_path /tmp/datalinter/results/lint_results.bin
```

### Viewing Results with the Lint Explorer

After the Data Linter is done examining your data, you can view the results
using this command:

```shell
python lint_explorer_main.py --results_path PATH_TO_RESULTS
```

For example:

```shell
python lint_explorer_main.py --results_path \
  /tmp/datalinter/results/lint_results.bin
```

# Notes

The code makes use of
[Google's protobuf format](https://developers.google.com/protocol-buffers/).
The protos are defined in `protos/`.

To make it easier to run the code, we include protobuf definitions from
[TensorFlow](https://www.tensorflow.org/) and
[Facets](https://github.com/PAIR-code/facets) in this distribution.

# Support

This is not an official Google project. This project will not be supported or
maintained, and we will not accept any pull requests.

# Authors

The Data Linter was created by Nick Hynes (nhynes@berkeley.edu) during an
internship at Google with Michael Terry (michaelterry@google.com).
