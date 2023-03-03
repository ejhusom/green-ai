[Documentation - Home](../../index.md)

[Overview of pipeline](../03_pipeline.md)

# Stage 1: profile

The profile-stage creates a profile report of your data set. The report will
contain information on the number of variables, number of samples, statistical
properties of all the variables and correlation between variables, among other
things. The report may also generate warnings, for example if there are some
variables that contain a large amount of null-values, or if a variable stays
constant all the time. These warnings are used to clean the data in the next
stage of the pipeline.

The profile report is saved to the file `assets/profile/profile.html`, which
can be opened in a browser to give a comprehensive view of the properties of
your data set.

## Parameters

No parameters are mandatory in this stage, but the following parameters can
be specified for this stage in `params.yaml`:

- `profile.dataset` (optional): Name of subfolder containing data set inside `assets/data/raw/`.

Example: Place files in `assets/data/raw/[NAME OF DATA SET]/`, and specify the
parameter like so:

```
profile:
    dataset: [NAME OF DATASET]
```

It the parameter is left empty, data files must be placed directly in
`assets/data/raw/`. See
[Quickstart](02_quickstart.md)
for more info on how to add data to your project.


Next stage: [clean](02_clean.md)
