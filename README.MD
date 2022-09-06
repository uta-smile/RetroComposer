# Environment
We include the comprehensive packages and versions of our environment in the requirements.txt file.
Note that not all packages in the file are necessary.
Here are some important packages and their versions:

```
rdkit-pypi                    2021.9.5.1
torch                         1.7.1+cu110
torch-geometric               1.7.2
torch-scatter                 2.0.6
torch-sparse                  0.6.9
```

# Stage 1
In the stage 1, we conduct retrosynthesis by composing templates and applying the templates to target molecules.


## Preprocessing
We need first extract reaction templates, and decompose each template into 
product smarts and reactant smarts which are later canonicalized to be used
as token. Run the following script to preprocesss the USPTO-50K dataset:
```
python extract_templates.py   
```

After the program ends, there will be four json files in the data path 
(data/USPTO50K). Please carefully check the scripts and code if you 
can not find these json files:
```
data/USPTO50K/templates_cano_train.json
data/USPTO50K/templates_test.json
data/USPTO50K/templates_train.json
data/USPTO50K/templates_valid.json
```

Then we generate training data for stage 1:
```
python prepare_mol_graph.py --retro   
```


## Training

We may train in single process mode or multi-process mode, which is much faster. We train models in multi process mode to report results.


### Single Process (slow)

```
python run_retro.py  --device 1

# for with types setting
python run_retro.py  --device 1  --typed
```

### Multi Process (fast)

```
python run_retro.py  --device 1  --multiprocess

# for with types setting
python run_retro.py  --device 1  --multiprocess --typed 
```

## Testing
Testing can also be done in single or multi process mode.

### Single Process (slow)

```
python run_retro.py  --device 1 --input_model_file model.pt --test_only

# for with types setting
python run_retro.py  --device 1 --input_model_file model.pt --test_only --typed
```

### Multi Process (fast)

```
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  

# for with types setting
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  --typed
```

### Run testing on validation and training splits:
```
# run validation split
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  --eval_split valid 

# run training split
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  --eval_split train

# for with types setting
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  --eval_split valid --typed 
python run_retro.py  --device 1 --multiprocess  --input_model_file model.pt --test_only  --eval_split train --typed
```

You can find three json files in the log directory:
```
logs/USPTO50K/uspto50k/beam_result_test.json
logs/USPTO50K/uspto50k/beam_result_train.json
logs/USPTO50K/uspto50k/beam_result_valid.json

# for with types setting
logs/USPTO50K/uspto50k_typed/beam_result_test.json
logs/USPTO50K/uspto50k_typed/beam_result_train.json
logs/USPTO50K/uspto50k_typed/beam_result_valid.json
```

# Stage 2
In the stage 2, we will train a ranking modeling to scoring the predicted reactants.

## Preprocessing
We first preprocess data for the ranking model.

```
python prepare_mol_graph.py

# for with types setting
python prepare_mol_graph.py --typed
```

## Training

```
python run_ranking.py  --device 1 --multiprocess

# for with types setting
python run_ranking.py  --device 1 --multiprocess --typed 
```

## Testing:

```
python run_ranking.py  --device 1 --test_only --input_model_file model.pt

# for with types setting
python run_ranking.py  --device 1 --test_only --input_model_file model.pt --typed 
```

