# Data Linter Demo

This document walks through the process of using the Data Linter with the [UCI
Census dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income).

## Step 1: Install the necessary software

You'll need to have the following software installed:

1. Python
2. [Apache Beam](https://beam.apache.org/)
3. [TensorFlow](https://www.tensorflow.org/)
4. [Facets](https://github.com/PAIR-code/facets)

## Step 2: Download the dataset

Download the dataset at this location:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

For example:

```shell
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
```

## Step 3: Convert the data to TFRecords

1. Open `convert_to_tfrecord.py` (in this directory).
2. Set the `DATA_FILE` variable to the location of `adult.data` (which you just
   downloaded).
3. Set the `OUTPUT_FILE` variable to the location of where you would like to
   store the dataset once converted to TFRecords.
4. Run `convert_to_tfrecord.py` (e.g., `python convert_to_tfrecord.py`).

## Step 4: Create summary statistics

1. Open `summarize_data.py` (in this directory).
2. Set `DATASET_PATH` to the location of `OUTPUT_FILE` in the previous step.
3. Set `OUTPUT_PATH` to the location of where you wish to store the summary
   statistics.
4. Run `summarize_data.py` (e.g., `python summarize_data.py`).

## Step 5: Run the Data Linter

Switch back to the parent folder and run the Data Linter as such:

```shell
python data_linter_main.py --dataset_path PATH_TO_TFRECORDS \
  --stats_path PATH_TO_FACETS_SUMMARIES --results_path PATH_FOR_SAVING_RESULTS
```

For example, if you used the defaults in the example files, you'd run it like
this:

```shell
python data_linter_main.py --dataset_path /tmp/adult.tfrecords \
  --stats_path /tmp/adult_summary.bin \
  --results_path /tmp/datalinter/results/lint_results.bin
```

## Step 6: Run the Lint Explorer

After the Data Linter is done running, you can run the Lint Explorer to view the
results:

```shell
python lint_explorer_main.py --results_path PATH_TO_RESULTS
```

For example:

```shell
python lint_explorer_main.py --results_path \
  /tmp/datalinter/results/lint_results.bin
```
