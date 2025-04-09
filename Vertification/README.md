# For evaluation

## Requirements

This code package was developed and tested with Python 3.7.6. Make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Usage
```
python --dataset {the dataset name} --gpu 0 --v {if v is added, the model will automatically use dataset_enhanced dataset} --log_file {dataset.log} {other hyperparameters can be adjusted as needed}
```



**Please make sure that sentence-bert have been downloaded in ../model.**
