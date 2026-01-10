# Jan's Spellchecking Experiments with transformers
Currently included:
* Dataset generation (sentence with errors -> corrected sentence) from wikipedia edits dataset
* Fine-tuning Roberta for misspelling detection as token classification task
* Interactive web-app using fine-tuned checkpoint to detect errors in user input


# Ideas
* Does model learn sentence-level biases from the dataset, i.e there's always exactly one token to be flagged (or: probs should add up to 1)? Does it tell us sth about ability of attention layers to "coordinate" with each other.
* Put the model and the demo online
* Distributed training: two gpus on same ec2, two gpus on different ec2s

