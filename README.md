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

Alternatively, you can download the pretrained version of these two models [here](https://drive.google.com/file/d/1_rRoT8so6-s9yQbl6pb5X0t4sqbnOpU0/view?usp=sharing) and [here](https://drive.google.com/file/d/1X6LdoeLZg2PD1GrHB88slVXwTovttYUo/view?usp=sharing). Next, download the pretrained Cohen RandomizedSmoothing models [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view). Download the pretrained Salman SmoothingAdversarial models [here](https://drive.google.com/file/d/1GH7OeKUzOGiouKhendC-501pLYNRvU8c/view).

