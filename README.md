# Train

```
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

# Training on Intel AI DevCloud

Sample jobs can be found in 'jobs' dir. Submit jobs using the `qsub` command.

# Model Summary

```
./summary.py <model>
```

Prints model architecture summary.

- **`model:`** model name; file '*model*.py' should be found in 'models' dir
