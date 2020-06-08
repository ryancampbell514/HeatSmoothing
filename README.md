# HeatSmoothing

## Deterministic Averaging Neural Netowrks

Using a variational method to deterministically average DNNs.

The code is tested with python3 and PyTorch v1.5.0 (CUDA version 10.2). See https://pytorch.org/get-started/locally/ for installation details.

Then clone this repository:

```
git clone https://github.com/ryancampbell514/GaussianNets.git
cd GaussianNets
```

### CIFAR-10 Experiments

### ImageNet-1k Experiments

To train a base model, simply execute
```
./run.sh
```
from the command line. This basic training code is modified from [Train ImageNet in 18 minutes](https://github.com/cybertronai/imagenet18). This fast ImageNet training is obtained by training on smaller images. If you run locally, you may need to download imagenet yourself from [here](https://s3.amazonaws.com/yaroslavvb2/data/imagenet18.tar).

Using this initial model, train the deterministic averaged model by running
```
./run_ours.sh
```

Download the pretrained Cohen RandomizedSmoothing models [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view). Download the pretrained Salman SmoothingAdversarial models [here](https://drive.google.com/file/d/1GH7OeKUzOGiouKhendC-501pLYNRvU8c/view).
