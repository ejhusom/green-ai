[Documentation - Home](../../index.md)

[Overview of pipeline](../03_pipeline.md)

# Stage 2: clean


This stage cleans the data for unwanted variables and samples, in addition to
encoding the target variable if the case of a classification task.


## Parameters

- `clean.target`: Name of target variable. Must correspond to the name of the
  column that contains this variable in your .csv-files.
- `clean.classification`: `True` or `False`. Specify whether we are dealing
  with a classification task or not. If set to `True`, the various classes will
  be encoded to numbers.
- `clean.onehot_encode_target`: `True` or `False`. Specify whether we want to
  one-hot encode the target variable or not. One-hot encoding is only needed
  for classification tasks with more than two classes. Without one-hot
  encoding, the various classes will be encoded as numbers on a continuous
  scale. This usually gives a wrong impression of how the classes relate to
  each other, for example by implying that some classes are closer than
  others, while in fact they are actually just different categories. One-hot
  encoding means that instead of encoding classes as numbers, each class will
  get its own output vector that indicate whether a sample is a part of that
  class or not. In some cases we might not want to use one-hot encoding even
  though we have a multi-class classification task, for example if the
  categories correspond to size (small, medium, large). In that case the
  classes might actually be well represented as numbers on a continuous scale.
- `clean.combine_files`: `True` or `False`. Specify whether all files should be
  combined into one before subsequences are extracted from the data. If we have
  multiple files containing time series data, there is usually a gap in time
  between the different files. In this case it does not make sense to
  combine the files, because when subsequences are extracted from the data,
  some subsequences will be made across this time gap. However, if the data
  files contain one, continuous time series, the files can be combined. If your
  data only resides in one file, this parameter will have no effect.
- `clean.percentage_zeros_threshold`: Fraction from 0.0 to 1.0. If a variable
  has a fraction of zero-values higher than this threshold, it will be removed.
  If this parameter is set to 1.0, only the variables that only contains zeros
  will be removed.
- `clean.input_max_correlation_threshold`: Fraction from 0.0 to 1.0. If two
  variables have a correlation higher than this threshold, one of them will be
  removed. This parameter gives you a way of reducing the number of variables (which
  in turn reduces computational costs and model complexity). However, keep in
  mind that the model performance often can be reduced by setting this
  parameter lower than 1.0. If set to 1.0, only variables that are perfectly
  correlated will be affected.

## Processing

During this stage, the following operations will be performed:

- Some variables/columns might be removed from the data set, based on the
  following criteria:
    - If a variable is constant, it will be removed because it does not
      contribute any valuable information.
    - If a variables
- If we are dealing with a classification task (in which case the parameter
  `clean.classification` must be set to `True` in `params.yaml`, the target
  variable will be encoded to a number.

Previous stage: [profile](01_profile.md)

Next stage: [featurize](03_featurize.md)
