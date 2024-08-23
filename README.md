# SimCLR_Pytorch
Only `train_simclr_cifar10.py` and `logistic_regression.py` are written by myself. The rest of this repository is forked from [SimCLR](https://github.com/sthalles/SimCLR). To run the script, you can refer to the README of [SimCLR](https://github.com/sthalles/SimCLR). Also, you can simply run my script by executing:
```sh
python train_simclr_cifar10.py
```

After obtaining the checkpoints, run:
```sh
python logistic_regression.py
```
to perform logistic regression.

## Hint
Remember to change the dataset path in `train_simclr_cifar10.py`.