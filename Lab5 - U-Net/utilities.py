import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

# Step(1) Fixing Seed
# Set seed for Python's random number generator
random.seed(27)
# Set seed for NumPy's random number generator
np.random.seed(27)
# Set seed for PyTorch's random number generator
torch.manual_seed(27)
torch.cuda.manual_seed_all(27)  # If using CUDA


def generate_random_data(height, width, count):
    x, y = zip(*[generate_img_and_mask(height, width) for i in range(0, count)])

    X = np.asarray(x) * 255
    X = X.repeat(3, axis=1).transpose([0, 2, 3, 1]).astype(np.uint8)
    Y = np.asarray(y)

    return X, Y


def generate_img_and_mask(height, width):
    shape = (height, width)

    triangle_location = get_random_location(*shape)
    circle_location1 = get_random_location(*shape, zoom=0.7)
    circle_location2 = get_random_location(*shape, zoom=0.5)
    mesh_location = get_random_location(*shape)
    square_location = get_random_location(*shape, zoom=0.8)
    plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=bool)
    arr = add_triangle(arr, *triangle_location)
    arr = add_circle(arr, *circle_location1)
    arr = add_circle(arr, *circle_location2, fill=True)
    arr = add_mesh_square(arr, *mesh_location)
    arr = add_filled_square(arr, *square_location)
    arr = add_plus(arr, *plus_location)
    arr = np.reshape(arr, (1, height, width)).astype(np.float32)

    # Create target masks
    masks = np.asarray([
        add_filled_square(np.zeros(shape, dtype=bool), *square_location),
        add_circle(np.zeros(shape, dtype=bool), *circle_location2, fill=True),
        add_triangle(np.zeros(shape, dtype=bool), *triangle_location),
        add_circle(np.zeros(shape, dtype=bool), *circle_location1),
         add_filled_square(np.zeros(shape, dtype=bool), *mesh_location),
        # add_mesh_square(np.zeros(shape, dtype=bool), *mesh_location),
        add_plus(np.zeros(shape, dtype=bool), *plus_location)
    ]).astype(np.float32)

    return arr, masks


def add_square(arr, x, y, size):
    s = int(size / 2)
    arr[x-s,y-s:y+s] = True
    arr[x+s,y-s:y+s] = True
    arr[x-s:x+s,y-s] = True
    arr[x-s:x+s,y+s] = True

    return arr


def add_filled_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, yy > y - s, yy < y + s]))


def logical_and(arrays):
    new_array = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        new_array = np.logical_and(new_array, a)

    return new_array


def add_mesh_square(arr, x, y, size):
    s = int(size / 2)

    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]

    return np.logical_or(arr, logical_and([xx > x - s, xx < x + s, xx % 2 == 1, yy > y - s, yy < y + s, yy % 2 == 1]))


def add_triangle(arr, x, y, size):
    s = int(size / 2)

    triangle = np.tril(np.ones((size, size), dtype=bool))

    arr[x-s:x-s+triangle.shape[0],y-s:y-s+triangle.shape[1]] = triangle

    return arr


def add_circle(arr, x, y, size, fill=False):
    xx, yy = np.mgrid[:arr.shape[0], :arr.shape[1]]
    circle = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    new_arr = np.logical_or(arr, np.logical_and(circle < size, circle >= size * 0.7 if not fill else True))

    return new_arr


def add_plus(arr, x, y, size):
    s = int(size / 2)
    arr[x-1:x+1,y-s:y+s] = True
    arr[x-s:x+s,y-1:y+1] = True

    return arr


def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)


def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])


def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))


def plot_errors(results_dict, title,x_label,y_label,ax=None):
    # markers = itertools.cycle(('+', 'x', 'o'))

    if ax is None:
      fig, ax = plt.subplots()


    ax.set_title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        # ax.plot(result, marker=next(markers), label=label)
        ax.plot(result, label=label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend(loc=3, bbox_to_anchor=(1, 0))

    if ax is None:
      plt.show()


def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def generate_images_and_masks_then_plot():
    # Generate some random images
    input_images, target_masks = generate_random_data(192, 192, count=3)

    for x in [input_images, target_masks]:
        print(x.shape)
        print(x.min(), x.max())

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [masks_to_colorimg(x) for x in target_masks]

    # Left: Input image (black and white), Right: Target mask (6ch)
    plot_side_by_side([input_images_rgb, target_masks_rgb])


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]


def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    train_set = SimDataset(100, transform = trans)
    val_set = SimDataset(25, transform = trans)
    test_set = SimDataset(25, transform = trans)
    image_datasets = {
        'train': train_set, 'val': val_set, 'test': test_set
    }

    batch_size = 25

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return dataloaders


def dice_loss(pred, target, smooth=1.):
    '''
     The Dice coefficient D between two sets 𝐴 and 𝐵 is defined as:
     D= (2×∣A∩B∣)/ (∣A∣+∣B∣)
     ∣A∩B∣: total no of pixels in pred,gold that has +ve
    '''
    pred = pred.contiguous() # contiguous() is a method that is used to ensure that the tensor is stored in a contiguous block of memory.
    target = target.contiguous()  #torch.Size([25, 6, 192, 192])

    intersection = (pred * target).sum(dim=2).sum(dim=2)  # Sumation of Both Width & Height

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def jaccard_index(pred, target, smooth=1.0):
    '''
    Jaccard Index (IoU) between two sets 𝐴 and 𝐵 is defined as:
    J(A, B) = 1 - (∣A∩B∣ / ∣A∪B∣)
    Where:
    ∣A∩B∣: Intersection of sets A and B
    ∣A∪B∣: Union of sets A and B
    '''
    pred = pred.contiguous() 
    target = target.contiguous() 

    intersection = (pred * target).sum(dim=2).sum(dim=2)  
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection

    IOU = ((intersection + smooth) / (union + smooth))
    
    return 1- IOU.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    # Binary Cress Entropy
    # In PyTorch, binary_cross_entropy_with_logits is a loss function that combines a sigmoid activation function and binary cross-entropy loss.
    # However, it doesn't explicitly apply the sigmoid function to the input. Instead, it expects the input to be logits, which are the raw outputs of a model without applying any activation function.
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    # Custom Loss function that combines bce & dice losses
    # Binary Cross-Entropy (BCE) Loss: BCE loss aims to minimize the difference between the predicted probability distribution and the ground truth binary labels.
    # It penalizes deviations from the true binary labels, typically encouraging the model to output probabilities that align well with the ground truth.
    # Dice Loss: Dice loss aims to maximize the overlap between the predicted segmentation mask and the ground truth mask.
    # It penalizes deviations from the true segmentation mask, typically encouraging the model to produce segmentations that align well with the ground truth boundaries.
    loss = bce * bce_weight + dice * (1 - bce_weight)

    jac_index=jaccard_index(pred, target)


    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['jaccrod_index']+=jac_index.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler,bce_weight=0.5, num_epochs=25):
    dataloaders = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    loss_per_epoch=[]
    bce_loss_per_epoch=[]
    dice_loss_per_epoch=[]
    jacord_index_per_epoch=[]

    loss_per_val_epoch=[]
    bce_loss_per_val_epoch=[]
    dice_loss_per_val_epoch=[]
    jacord_index_per_val_epoch=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train by passing train condition to set_grad_enabled :D
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics,bce_weight=bce_weight)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # optimzation custom loss function :D
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              loss_per_epoch.append(epoch_loss)
              bce_loss_per_epoch.append(metrics['bce']/epoch_samples)
              dice_loss_per_epoch.append(metrics['dice']/epoch_samples)
              jacord_index_per_epoch.append(metrics['jaccrod_index']/epoch_samples)

            elif phase=="val":
              loss_per_val_epoch.append(epoch_loss)
              bce_loss_per_val_epoch.append(metrics['bce']/epoch_samples)
              dice_loss_per_val_epoch.append(metrics['dice']/epoch_samples)
              jacord_index_per_val_epoch.append(metrics['jaccrod_index']/epoch_samples)


            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model,dataloaders['test'],loss_per_epoch,bce_loss_per_epoch,dice_loss_per_epoch,jacord_index_per_epoch,loss_per_val_epoch,bce_loss_per_val_epoch,dice_loss_per_val_epoch,jacord_index_per_val_epoch



def compute_test_loss(pred,target):
    pred = F.sigmoid(pred)
    # Dice Loss
    # dice_score = (1- dice_loss(pred, target)).cpu().numpy() * target.size(0)
    dice_score = (1- dice_loss(pred, target)).cpu().numpy()

    # jaccord Index
    jaccord_index= jaccard_index(pred, target).cpu().numpy()


    return dice_score,jaccord_index

def test_model(model,test_data_loader):
    print("Testing Model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_idx=0

    dice_score=0
    dice_loss=0
    jaccord_index=0

    for inputs, labels in test_data_loader:
        print(f'{batch_idx}/{len(test_data_loader)}')
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            dice_score_i,jaccord_index_i = compute_test_loss(outputs, labels)

            dice_score+=dice_score_i
            jaccord_index+=jaccord_index_i
            dice_loss+=(1-dice_score_i)
    
    # Average Dice Score
    dice_score = dice_score/len(test_data_loader)
    dice_loss = dice_loss/len(test_data_loader)
    jaccord_index = jaccord_index/len(test_data_loader)

    print(f'Test: DiceLoss : {dice_loss} Dice Score: {dice_score} Jaccord Index: {jaccord_index}') 
    return

def run(UNet,lr=1e-4,step_size=30, gamma=0.1,num_epochs=60,bce_weight=0.5,test=False):
    num_class = 6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # The StepLR scheduler decreases the learning rate of the optimizer by a factor (gamma) at specified intervals (step_size).
    # Here, the learning rate will be decreased by a factor of 0.1 every 30 epochs.
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model,test_dataloader,loss_per_epoch,bce_loss_per_epoch,dice_loss_per_epoch,jacord_index_per_epoch,loss_per_val_epoch,bce_loss_per_val_epoch,dice_loss_per_val_epoch,jacord_index_per_val_epoch = train_model(model, optimizer_ft, exp_lr_scheduler, bce_weight=bce_weight, num_epochs=num_epochs)
    print("Done Training")


    if test:
      print("Testting Model")
      model.eval()  # Set model to the evaluation mode
      trans = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
      ])

      # Test DataSet
      test_model(model,test_dataloader)

      # Create another simulation dataset for test
      test_dataset = SimDataset(3, transform = trans)
      test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

      # Get the first batch
      inputs, labels = next(iter(test_loader))
      inputs = inputs.to(device)
      labels = labels.to(device)

      # Predict
      pred = model(inputs)
      # The loss functions include the sigmoid function.
      pred = F.sigmoid(pred)
      # TODO: Computing Dice Score & Jaccard Index
      # Dice Score
      dice_sample_loss=dice_loss(pred, labels)
      dice_sample_score = 1-dice_sample_loss
      # jaccord Index
      jaccord_index= jaccard_index(pred, labels)
      print(f'Test(Sample): DiceLoss : {dice_sample_loss} Dice Score: {dice_sample_score} Jaccord Index: {jaccord_index}') 

      pred = pred.data.cpu().numpy()


      # Change channel-order and make 3 channels for matplot
      input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

      # Map each channel (i.e. class) to each color
      target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
      pred_rgb = [masks_to_colorimg(x) for x in pred]

      plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

      return
    return loss_per_epoch,bce_loss_per_epoch,dice_loss_per_epoch,jacord_index_per_epoch,loss_per_val_epoch,bce_loss_per_val_epoch,dice_loss_per_val_epoch,jacord_index_per_val_epoch