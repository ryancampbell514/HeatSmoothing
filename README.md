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

Begin by entering the CIFAR-10 directory,
```
cd cifar10
```

Next, train the base model, Cohen model, and Salman model by running
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

### ImageNet-1k Experiments

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
