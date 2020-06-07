"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Training template for DNN computer vision research in PyTorch')
parser.add_argument('--datadir', type=str, required=True, default='/home/campus/oberman-lab/data/cifar10', metavar='DIR',
        help='data storage directory')
parser.add_argument('--dataset', type=str,help='dataset (default: "cifar10")',
        default='cifar10', metavar='DS',
        choices=['cifar10','cifar100', 'TinyImageNet','Fashion','mnist','svhn'])
parser.add_argument('--greyscale', type=bool, default=False)
parser.add_argument('--std', type=float, default=None, metavar='SD',
        help = 'standard deviation of the added gaussian noise')
parser.add_argument('--num-samples',type=int,default=10, metavar='NS',
        help='number of Gaussian samples to draw')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--logdir', type=str, default=None,metavar='DIR',
        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed (default: int(time.time()) )')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
parser.add_argument('--valid-size', type=float, default=None, metavar='V',
        help='size of the validation set (if needed)')
parser.add_argument('--train-pen', type=bool, default=False, metavar='TP',
        help='whether or not to train the penalization term on the validation set')
parser.add_argument('--retrain', type=bool, default=False, metavar='RET',
        help='whether or not to continue training a pre-trained model (default: False)')
parser.add_argument('--model-dir', type=str, 
        default=None, 
        metavar='DIR', help='Directory where model is saved')
parser.add_argument('--pth-name', type=str, default='best.pth.tar')
parser.add_argument('--parallel', action='store_true', dest='parallel',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--strict', action='store_true', dest='strict',
        help='only allow exact matches to model keys during loading')
parser.add_argument('--adv-train',action='store_true',
        help = "Whether or not to adversarially train")
parser.add_argument('--AT-steps', type=int, default=10,
        help='number of PGD steps')
parser.add_argument('--AT-ball', type=float, default=0.5,
        help='Max norm of perturbation')

parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--filename', type=str, default='data.pkl')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet34',
        help='Model architecture (default: ResNet34)')
group1.add_argument('--classes',type=int, default=10,
        help='How many classes the model has')
group1.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--cutout',type=int, default=0, metavar='N',
        help = 'Cutout size, if data loader supports cutout (default: 0)')
group1.add_argument('--add-gaussian', type=bool, default=False, metavar='AG',
        help = 'whether or not to add gaissian noise to the images (default: False)')
group1.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
group1.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
group1.set_defaults(bn=False)
group1.add_argument('--last-layer-nonlinear',
        action='store_true', default=False)
group1.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
group1.add_argument('--no-bias',action='store_false', dest='bias',
       help = "Don't use biases")
group1.set_defaults(bias=False)
group1.add_argument('--kernel-size',type=int, default=3, metavar='K',
        help='convolution kernel size (default: 3)')
group1.add_argument('--model-args',type=str,
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')


group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]', help='List of epochs and multiplier '
        'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.008]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
       help='SGD momentum parameter (default: 0.9)')


group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--decay',type=float, default=5e-4, metavar='L',
        help='Lagrange multiplier for weight decay (sum '
        'parameters squared) (default: 5e-4)')
group2.add_argument('--norm', type=str, default='L2', metavar='norm')
group2.add_argument('--fd-order', type=str, choices=['O1','O2'], default='O1',
        help='accuracy of finite differences (default: O1)')
group2.add_argument('--random-labels', type=bool, default=False, metavar='RL',
        help='whether or not to train with random labels')

#group3 = parser.add_argument_group('Loss hyperparameters')
#group3.add_argument('--loss-function', type=int, default=None, metavar='LOSS',
#        help='0 - XEnt, 1 - New/Unbiased probability loss, 2 - Chris Loss, 3 - Use Distances, 4 - CW Loss',
#        )
#group3.add_argument('--cross-entropy', type=bool, default=False, metavar='XENT',
#        help='Whether or not to use cross-entropy as the loss function.')
#group3.add_argument('--loss-weight', type=float, default=None, metavar='W',
#        help='Weight for the new loss function, creates a convex combination'
#             ' of pmax and pcorrect. Defaults to 0, which gives CrossEntropy.')
#group3.add_argument('--tik', type=float, default=None, metavar='TIK',
#        help='The Tikhanov regularization constant (Optimal = 0.1).')
#group3.add_argument('--pen-const', type=float, default=None, metavar='CONST',
#        help='regularization constant (Optimal = 0.1).')
#group3.add_argument('--use-unbiased-loss', type=bool, default=False, metavar='SM',
#        help='use new unbiased softmax loss?')
#group3.add_argument('--use-distances', type=bool, default=False, metavar='DIST',
#        help='train with distances?')

#group4 = parser.add_argument_group('Attack Hyperparameters')
#group4.add_argument('--num-images', type=int, default=1000, metavar='NI',
#        help='number of images to consider')
#group4.add_argument('--save-images', action='store_true', default=False,
#        help='save perturbed images to a npy file (default: False)')
#group4.add_argument('--random-subset', action='store_true',
#        default=False, help='use random subset of test images (default: False)')
