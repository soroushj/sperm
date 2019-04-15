# Train

```shell
./train.py <model> <label> <iters> [<config>]
```

Trains the model and saves the results.

- **`model:`** model name; file '*model*.py' should be found in 'models' dir
- **`label:`** label to train for; one of:
  - **a:** acrosome
  - **h:** head
  - **t:** tail
  - **v:** vacuole
- **`iters:`** number of train iterations
- **`config:`** optional train config; one of:
  - **0:** oversampling on, augmentation on (default)
  - **1:** oversampling on, augmentation off
  - **2:** oversampling off, augmentation on
  - **3:** oversampling off, augmentation off

## Results

Results will be saved as CSV files in 'results' dir. Naming pattern is '*model*.*label*.*config*.csv'.

## Checkpoints

Best checkpoints—in terms of validation loss, accuracy, f-1.0 and f-0.5 score—will be saved in 'checkpoints' dir. Naming pattern is '*model*.*label*.*config*.*iter*.h5'.

## Training on Intel AI DevCloud

Request access at [https://software.intel.com/en-us/ai/devcloud](https://software.intel.com/en-us/ai/devcloud).

### Requirements

All requirements are pre-installed, except for OpenCV. Install OpenCV using this command:

```shell
echo "pip install --user opencv-python-headless" | qsub
```

### Submitting Jos

Sample jobs can be found in 'jobs' and 'jobs-kfold' dirs. Submit jobs using the `qsub` command.

Submit a file:

```shell
qsub job.sh
```

Submit multiple files:

```shell
for j in *.sh; do qsub $j; done
```

Submit a command:

```shell
echo "my command" | qsub
```

View jobs status:

```shell
qstat
```

# Train (k-fold)

```shell
./train-kfold.py <model> <label> <iters> [<config> [<k>]]
```

Trains the model and saves the results.

- **`model:`** model name; file '*model*.py' should be found in 'models' dir
- **`label:`** label to train for; one of:
  - **a:** acrosome
  - **h:** head
  - **t:** tail
  - **v:** vacuole
- **`iters:`** number of train iterations
- **`config:`** optional train config; one of:
  - **0:** oversampling on, augmentation on (default)
  - **1:** oversampling on, augmentation off
  - **2:** oversampling off, augmentation on
  - **3:** oversampling off, augmentation off
- **`k:`** the fold (0 <= k <= 4)

## Results

Results will be saved as CSV files in 'results' dir. Naming pattern is '*model*.*label*.*config*.*k*.csv'.

## Checkpoints

Best checkpoints—in terms of validation loss, accuracy, f-1.0 and f-0.5 score—will be saved in 'checkpoints' dir. Naming pattern is '*model*.*label*.*config*.*k*.*iter*.h5'.

# Model Summary

```shell
./summary.py <model>
```

Prints model architecture summary.

- **`model:`** model name; file '*model*.py' should be found in 'models' dir
