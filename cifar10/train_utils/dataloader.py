import os
import warnings
import numpy as np
import torch as th

#import random
import torchnet as tnt
import torchvision 
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST 
import torch.utils.data.sampler as sampler
from math import floor, ceil

def cutout(mask_size,channels=3):
    if channels>1:
        mask_color=tuple([0]*channels)
    else:
        mask_color=0

    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    if mask_size >0:
        def _cutout(image):
            image = np.asarray(image).copy()

            if channels >1:
                h, w = image.shape[:2]
            else:
                h, w = image.shape

            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            square = image[ymin:ymax, xmin:xmax]
            avg = np.mean(square, axis=(0,1))
            image[ymin:ymax, xmin:xmax] = avg
            if channels==1:
                image = image[:,:,None]
            return image
    else:
        def _cutout(image):
            return image

    return _cutout

def optim_cutout(model, loss_func, images, labels, cutout):
    (Nb,cols,l,l) = images.shape
    #Ix = torch.arange(Nb)
    output = model(images)
    loss = loss_func(output, labels)
    # compute grad loss w.r.t. x
    grad_loss = th.autograd.grad(loss.mean(), images)[0]
    g = grad_loss.mean(dim=1)  # will be of shape (Nb,l,l)

    # the 'center' indices (x coordinate, y coordinate)
    idx = th.tensor([[l/2,l/2] for i in range(Nb)])

    while l > cutout:
        l = int(l/2)
        bi = th.zeros(Nb,4,l,l)
        for i in range(Nb):
            x = int(idx[i,0])
            y = int(idx[i,1])
            bi[i] = th.tensor([g[i,y-l:y,x-l:x].tolist(), g[i,y-l:y,x:x+l].tolist(), 
                    g[i,y:y+l,x-l:x].tolist(), g[i,y:y+l,x:x+l].tolist()])
        region = bi.reshape(Nb,4,-1).mean(dim=-1).argmax(dim=-1)  # per image, which of the 4 bisections has the highest average gradient

        idx[region==0,0] /= 2
        idx[region==0,1] /= 2

        idx[region==1,0] *= 1.5
        idx[region==1,1] /= 2

        idx[region==2,0] /= 2
        idx[region==2,1] *= 1.5

        idx[region==3,0] *= 1.5
        idx[region==3,1] *= 1.5

        # ensure index doesnt leave the image
        idx[idx > 32] = 32-l

    l = int(l/2)
    idx = idx.int()

    # cutout about half of the images in the batch
    #cut_cond = th.randint(0,2,(Nb,)) == 1
    for i in range(Nb):
        #if cut_cond[i] == 1:
        images[i, :, (idx[i,1]-l):(idx[i,1]+l), (idx[i,0]-l):(idx[i,0]+l)] = 1.0

    return images, idx

def mnist_cutout(images, cutout):
    """ Take in a batch of MNIST images,
        apply a random cutout  """

    cut_m = int(cutout/2)
    cut_p = cutout - cut_m

    Nb = images.shape[0]
    Ix = th.arange(Nb)
    #imshape = images.shape[1:4]

    images = np.asarray(images.cpu())
    for i in range(Nb):
        x_c = th.randint(0,28,(1,)).item()
        x_l = x_c - cut_m
        if x_l < 0:
            x_l = 0
        x_u = x_c + cut_p
        if x_u > 28:
            x_u = 28

        y_c = th.randint(0,28,(1,)).item()
        y_l = y_c - cut_m
        if y_l < 0:
            y_l = 0
        y_u = y_c + cut_p
        if y_u > 28:
            y_u = 28

        images[i,:,y_l:y_u,x_l:x_u] = np.mean(images[i,:,y_l:y_u,x_l:x_u])

    images = th.from_numpy(images)
    return images

def mnist(datadir, training_transforms=[], mode='train', transform=False, **kwargs):
    assert mode in ['train', 'test']

    #if len(training_transforms)>0 or transform is True:
    #    warnings.warn('mnist dataloader does not transform images')

    train=mode=='train'

    #if mode in ['train','test']:
    #    if train and transform:
    #        tlist = [transforms.RandomCrop(28,padding=4),
    #                 transforms.RandomHorizontalFlip(),
    #                *training_transforms,
    #                 transforms.ToTensor()]
    #    else:
    #        tlist = [transforms.ToTensor()]
    #transform = transforms.Compose(tlist)
    ds = MNIST(root=os.path.join(datadir,'mnist'), download=True, train=train) #, transform=transform)

    data = getattr(ds, 'train_data' if train else 'test_data')
    labels = getattr(ds, 'train_labels' if train else 'test_labels')

    augment = tnt.transform.compose([lambda x: x[None,:,:].float() /255.0])

    tds = tnt.dataset.TensorDataset([data, labels])
    tds = tds.transform({0:augment})

    tds = tds.parallel(**kwargs)


    tds.Nsamples= 60000 if mode=='train' else 10000
    tds.classes = 10
    tds.image_shape = (1, 28, 28)

    return tds


def TinyImageNet(datadir, training_transforms = [], mode='train', transform=True, **kwargs):
    assert mode in ['train', 'test', 'val']

    dataset_dir = os.path.join(datadir,'TinyImageNet')

    pathexist = os.path.isdir(dataset_dir)
    if not pathexist:
        raise ValueError('download TinyImageNet from "http://cs231n.stanford.edu/tiny-imagenet-200.zip" '
                ' and unzip to %s'%sdatset_dir)

    train=mode=='train'
    if train and transform:
        d = os.path.join(dataset_dir,mode)
        train_dataset = torchvision.datasets.ImageFolder(d, transforms.Compose([
                        transforms.RandomCrop(64,padding=8),
                        transforms.RandomHorizontalFlip(),
                        *training_transforms,
                        transforms.ToTensor()]))

        ds = th.utils.data.DataLoader(train_dataset, **kwargs)


    else:
        d = os.path.join(dataset_dir,mode)
        test_dataset = torchvision.datasets.ImageFolder(d,
                transforms.Compose([transforms.ToTensor()]))
        ds  = th.utils.data.DataLoader(test_dataset,  **kwargs)


    if mode=='train':
        ds.Nsamples = 100000
    else:
        ds.Nsamples = 10000 # both test and validation sets have 10000 images

    ds.classes = 200
    ds.image_shape = (3, 64, 64)

    return ds

def Fashion(datadir, training_transforms = [], mode='train', transform=True, **kwargs):
    assert mode in ['train', 'test']

    train=mode=='train'
    if train and transform:
        tlist = [transforms.RandomCrop(28,padding=4),
                 transforms.RandomHorizontalFlip(),
                *training_transforms,
                 transforms.ToTensor()]
    else:
        tlist = [transforms.ToTensor()]


    transform = transforms.Compose(tlist)
    root = os.path.join(datadir,'FashionMNIST')

    ds = FashionMNIST(root, download=True, train=train, transform=transform)



    dataloader = th.utils.data.DataLoader(ds, **kwargs)

    dataloader.Nsamples = 60000 if train else 10000
    dataloader.classes = 10
    dataloader.image_shape = (1, 28, 28)

    return dataloader


def cifar10(datadir, rand_lbls=False, greyscale=False, training_transforms=[], mode='train', transform=True, subset=None, **kwargs):

    assert mode in ['train', 'test', 'val']

    train=mode=='train'

    root = os.path.join(datadir,'cifar10')
    gtransf = [transforms.Grayscale()] if greyscale else []

    if mode in ['train','test']:
        if train and transform:
            tlist = [*gtransf,
                     transforms.RandomCrop(32,padding=4),
                     transforms.RandomHorizontalFlip(),
                     *training_transforms,
                     transforms.ToTensor()]
        else:
            tlist = [*gtransf,
                     transforms.ToTensor()]

        transform = transforms.Compose(tlist)

        ds = CIFAR10(root, download=True, train=train, transform=transform)
        sample_size = 50000 if train else 10000
        if rand_lbls:
            # replace the labels in tha dataset with random ones
            # to measure Rademacher complexity
            ds.targets = th.randint(0,2,(sample_size,)).long()

    else:
        filename = 'cifar10.1'

        label_filename = 'cifar10.1_v6_labels.npy'
        imagedata_filename = 'cifar10.1_v6_data.npy'

        label_filepath = os.path.join(root, label_filename)
        imagedata_filepath = os.path.join(root, imagedata_filename)

        try:
            labels = np.load(label_filepath)
            data = np.load(imagedata_filepath)
        except FileNotFoundError as e:
            raise type(e)('Download CIFAR10.1 .npy files from https://github.com/modestyachts/CIFAR-10.1 '
                  'and place in %s'%root)

        ds = tnt.dataset.TensorDataset([data, labels])
        augment = transforms.ToTensor()
        ltrans  = lambda x: np.array(x, dtype=np.int_)
        ds = ds.transform({0:augment, 1:ltrans})
        sample_size=2000


    # check if we're looking at a range
    if isinstance(subset, range):
        indices = np.arange(subset.start, subset.stop)
    elif isinstance(subset, tuple) and len(subset)==2:
        indices = np.arange(subset[0], subset[1])
    elif isinstance(subset, np.ndarray):
        indices = subset
    elif isinstance(subset, float):
        if (subset > 0. and subset < 1.):
            num_samples = floor(subset  * sample_size)
            assert num_samples >0
            indices = np.random.choice(sample_size, num_samples)
        else:
            raise ValueError('subset fraction must be between 0 and 1')
    elif subset is not None:
        raise ValueError('Invalid subset parameter.')

    if subset:

        # according to Pytorch docs shuffle cannot be true if we are using a sampler
        # so we're going to turn it off in case that it's on
        kwargs['shuffle'] = False

        dataloader = th.utils.data.DataLoader(ds,
                sampler=sampler.SubsetRandomSampler(indices), **kwargs)
        dataloader.Nsamples = indices.size

    else:
        dataloader = th.utils.data.DataLoader(ds, **kwargs)
        if mode=='train':
            dataloader.Nsamples = 50000
        elif mode=='test':
            dataloader.Nsamples = 10000
        else:
            dataloader.Nsamples = 2000

    dataloader.classes = 10
    dataloader.image_shape = (3, 32, 32)

    return dataloader


def cifar100(datadir, training_transforms = [], mode='train', transform=True, **kwargs):
    assert mode in ['train', 'test']

    train=mode=='train'
    if train and transform:
        tlist = [transforms.RandomCrop(32,padding=4),
                 transforms.RandomHorizontalFlip(),
                *training_transforms,
                 transforms.ToTensor()]
    else:
        tlist = [transforms.ToTensor()]
    transform = transforms.Compose(tlist)

    root = os.path.join(datadir,'cifar100')

    ds = CIFAR100(root, download=True, train=train,transform=transform)

    dataloader = th.utils.data.DataLoader(ds,
                      **kwargs)
    dataloader.Nsamples = 50000 if train else 10000
    dataloader.classes = 100
    dataloader.image_shape = (3, 32, 32)

    return dataloader
