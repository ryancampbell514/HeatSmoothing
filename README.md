# HeatSmoothing

## Deterministic Averaging Neural Netowrks

Using a variational method to deterministically average DNNs.

The code is tested with python3 and PyTorch v1.5.0 (CUDA version 10.2). See https://pytorch.org/get-started/locally/ for installation details.

Then clone this repository:

```
git clone https://github.com/ryancampbell514/HeatSmoothing.git
cd HeatSmoothing
```

### CIFAR-10 Experiments

To begin, run `cd cifar10`.

#### Training

Train the base model, Cohen model, and Salman model by running
```
./run.sh
```
from the command line with the correct script selected on line 42.

To train our averaged model, run
```
python train_ours.py --data-dir 'PATH TO CIFAR-10 DATASET' --init-model-dir 'DIRECTORY OF THE TRAINED BASE MODEL' --pth-name 'best.pth.tar'
```
from the command line.

Alternatively, the four pretrained models can be downloaded [here](https://drive.google.com/file/d/1p0TXoOeQfvkgXkHqaXY7YAmjhRdmN-S8/view?usp=sharing).

#### Certification

To certify the baseline and our adveraged models, cd into `certify` and run the following from the command line,
```
python certify.py --data-dir 'WHERE THE DATA IS STORED' --model-dir 'MODEL DIRECTORY' --pth-name 'MODEL PATH.pth.tar'
```
For the Cohen and Salman models, run
```
python certify.py --data-dir 'WHERE THE DATA IS STORED' --model-dir 'MODEL DIRECTORY' --pth-name 'MODEL PATH.pth.tar' --is-cohen
```
Using the resulting .pkl dataframes, make the certification plot using the code provided in the notebook `figs/cert_plots.ipynb`.

#### Attacking

To attack our baseline and our averaged models, cd into `attack` and run the following from the command line
```
python run-attack.py --data-dir 'WHERE THE DATA IS STORED' --model-dir 'MODEL DIRECTORY' --pth-name 'MODEL PATH.pth.tar' --criterion 'top1' --attack 'DDN or PGD'
```
To attack the Cohen and Salman models, run
```
python run-attack.py --data-dir 'WHERE THE DATA IS STORED' --model-dir 'MODEL DIRECTORY' --pth-name 'MODEL PATH.pth.tar' --criterion 'cohen' --attack 'DDN or PGD'
```
Using the resulting .npz adversarial distances, make the certification plot using the code provided in the notebook `figs/adv_plots.ipynb`.

### ImageNet-1k Experiments

Now, run `cd imagenet`.

#### Training

To train a base model, simply execute
```
./run.sh
```
from the command line. This basic training code is modified from [Train ImageNet in 18 minutes](https://github.com/cybertronai/imagenet18). This fast ImageNet training is obtained by training on smaller images. If you run locally, you may need to download the special ImageNet dataset yourself from [here](https://s3.amazonaws.com/yaroslavvb2/data/imagenet18.tar).

Using this initial model, train the deterministic averaged model by running
```
./run_ours.sh
```

Alternatively, you can download the pretrained version of these two models, along with the pretrained Cohen and Salman models [here](https://drive.google.com/file/d/1Gvt6zNAnAAZaOiPWcCc_CzkFJV3k-_GL/view?usp=sharing).

#### Certification

Run `cd certify`. To certify the baseline model and our averaged model by running
```
python certify.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --std 0.25 --rule 'top5'
```
For the pretrained Cohen and Salman models, the model loading code is slightly different. For the Cohen model, run
```
python certify-cohen.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --std 0.25 --rule 'top5' --is-cohen
```
and for the Salman model, run
```
python certify-salman.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --std 0.25 --rule 'top5' --is-cohen
```

Using the resulting .pkl dataframes, make the certification plot using the code provided in the notebook `figs/cert_plots.ipynb`.

#### Attacking

Run `cd attack`. To attack  the baseline model and our averaged model by running
```
python run-attack.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --attack 'DDN or PGD'
```
For the Cohen model, run
```
python run-attack-cohen.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --attack 'DDN or PGD'
```
For the Salman model, run
```
python run-attack-salman.py --datadir 'WHERE DATA IS STORED' --model-path 'MODEL PATH.pth.tar' --attack 'DDN or PGD'
```
Using the resulting .npz adversarial distances, make the certification plot using the code provided in the notebook `figs/adv_plots.ipynb`.
