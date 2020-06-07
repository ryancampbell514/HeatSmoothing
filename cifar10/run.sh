#!/bin/sh

# Shell script to launch PyTorch model testing experiments.
# This is a template to use as a starting point for creating
# your own script files.

# Instructions for use:
# Make sure the paths are correct and execute from the command
# line using:
# $ ./yourscript.sh
# You will have to change file permissions before you can
# execute it:
# $ chmod +x yourscript.sh
# To automate the execution of multiple scipts use the
# jobdispatcher.py tool.

MODEL='ResNet34'

# Setup
TIMESTAMP=`date +%y-%m-%dT%H%M%S`  # Use this in LOGDIR
DATASET='cifar10'   # Use the dataset name in LOGDIR
DATADIR='/home/campus/oberman-lab/data/cifar10'  # Shared data file store

BASELOG='./logs/'$DATASET/$MODEL
LOGDIR=$BASELOG/$TIMESTAMP
SCRATCH='/mnt/data/scratch/'$USER-'runs/'$TIMESTAMP  # During training write to a local drive, not a network drive

mkdir -p $DATADIR
mkdir -p $SCRATCH
chmod g+rwx $SCRATCH # so that others can delete this folder if we kill the experiment and forget to
mkdir -p $BASELOG

ln -s $SCRATCH $LOGDIR


# If you want to specify which GPU to run on,
# prepend the following with
#CUDA_VISIBLE_DEVICES=<id> \
# or alternately, from the command line issue
# $ export CUDA_VISIBLE_DEVICES=<id>
# to make only that GPU visible
python -u ./train_salman.py \
    --bn \
    --lr 0.1 \
    --lr-schedule '[[0,1],[60,0.2],[120,0.04],[160,0.008]]'\
    --cutout 0 \
    --epochs 200 \
    --std 0.1 \
    --test-batch-size 100 \
    --model $MODEL \
    --dataset $DATASET \
    --datadir $DATADIR \
    --logdir $LOGDIR \
    | tee $LOGDIR/log.out 2>&1 # Write stdout directly to log.out.
                               # If you don't want to see the output, replace
                               # '| tee' with '>'
                               # (with '>', if you want to see results in real time,
                               # use tail -f on the logfile)

rm $LOGDIR
mv $SCRATCH $LOGDIR
