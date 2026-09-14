# Dataset protocols

These READMEs are the source of truth for the three datasets used by the local
MMDetection and SR-TOD experiments:

- [Varroa](varroa/README.md)
- [LEVIR-Ship](levir_ship/README.md)
- [TinyPerson](tinyperson/README.md)

Do not create a training config from a dataset filename alone. Copy the dataset
root, annotation paths, split seed, preprocessing, and evaluator behavior from
the relevant README into the experiment manifest.
