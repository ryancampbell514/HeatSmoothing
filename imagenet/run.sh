#!/bin/bash

# This script is an example for training ImageNet models
# nproc_per_node is the number of GPUs you want to use

# Set path to ImageNet data storage
IMGDATA='/mnt/data/scratch/data/imagenet'

TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR

# The script assumes you have a local scratch drive
SCRATCH='/mnt/data/scratch/'$USER-'runs/'$TIMESTAMP

NAME='baserun-'$TIMESTAMP
DIR='runs/'$NAME

# mkdir -p $IMGDATA
mkdir -p $SCRATCH
chmod g+rwx $SCRATCH # so that others can delete this folder if we kill the experiment and forget to
mkdir -p $DIR
mv $DIR  $SCRATCH/$NAME

ln -s $SCRATCH/$NAME $DIR

ulimit -n 4096

python -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  python ./train.py $IMGDATA \
  --workers=4 \
  --init-bn0 \
  --logdir $DIR --distributed \
  --phases "[{'ep': 0, 'sz': 128, 'bs': 128, 'trndir': '-sz/160'}, {'ep': (0, 8), 'lr': (0.25, 0.5)}, {'ep': (8, 15), 'lr': (0.5, 0.0625)}, {'ep': 15, 'sz': 224, 'bs': 48, 'trndir': '-sz/320', 'min_scale': 0.087}, {'ep': (15, 25), 'lr': (0.1, 0.01)}, {'ep': (25, 28), 'lr': (0.01, 0.001)}, {'ep': 28, 'sz': 288, 'bs': 32, 'min_scale': 0.5, 'rect_val': True}, {'ep': (28, 29), 'lr': (0.000625, 0.0000625)}]" --skip-auto-shutdown 

# If you have GPUs with more working memory you can double the batch-size and lr values
# above to train faster
#
# On four GTX 1080 Ti's, this script takes about 33 hours
# Without regularization (setting tikhonov to zero) it'll take roughly 20 hours

rm $DIR
mv $SCRATCH/$NAME $DIR
