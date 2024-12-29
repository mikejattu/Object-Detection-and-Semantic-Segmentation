import numpy as np
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import transforms
import torchvision
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import subprocess
import sys
from tqdm import tqdm
import torchvision.transforms as T
import os


def install(package):
    """
    Description: Install the required packages
    Input: package - the package to be installed
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def install_requirements():
    """
    Description: Install all the required packages
    """
    import os
    print("Installing requirements...")
    install("Ultralytics")

def detect_and_segment(images):
    """
    Description: Detect and segment the images
    Input: images - the images to be detected and segmented
    Output: pred_class - the predicted classes
            pred_bboxes - the predicted bounding boxes
            pred_seg - the predicted segmentation masks
    """
    # Ensure the required packages are installed
    install_requirements()
    from ultralytics import YOLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        'detection': "best_model-detect.pt",
        'segmentation': "best_model-seg.pt"
        }

    N = images.shape[0]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    model_detect = YOLO(params['detection'])
    #model_segment = YOLO("best-seg2.pt")
    for i in range(N):
        img = images[i].reshape(64, 64, 3)
        # Run inference
        results = model_detect(img)
        # Extract classification labels and bounding boxes
        classifications = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Sort based on confidence if more than 2 detections
        if len(classifications) > 2:
            # confidences
            confidences = results[0].boxes.conf.cpu().numpy()
            # sort the indices based on confidence
            sorted_indices = confidences.argsort()
            # limiting to the highest 2
            sorted_indices = sorted_indices[-2:]
            classifications = classifications[sorted_indices]
            sorted_boxes = boxes[sorted_indices]
        # Sort based on classification
        sorted_indices = classifications.argsort()[:2]  # Limit to 2
        sorted_classes = classifications[sorted_indices]
        sorted_boxes = boxes[sorted_indices]
        
        # Pad if fewer than 2 detections
        if len(sorted_classes) < 2:
            sorted_classes = np.pad(sorted_classes, (0, 2 - len(sorted_classes)), constant_values=-1)  # -1 as placeholder
            sorted_boxes = np.pad(sorted_boxes, ((0, 2 - len(sorted_boxes)), (0, 0)), constant_values=0)
        
        # Store labels and boxes in output arrays
        pred_class[i] = sorted_classes.astype(np.int32)
        pred_bboxes[i] = sorted_boxes
        
        # Generate segmentation masks
        unet_model = UNET(in_channels=3, out_channels=11)
        unet_model.load_state_dict(torch.load(params['segmentation'], map_location=device))
        unet_model.to(device)
        unet_model.eval()

        # Convert the image to a tensor 
        img_tensor = torch.tensor(2.0*img/np.amax(img)-1.0,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
        img_tensor = img_tensor.to(device)
        # Predict the segmentation mask
        with torch.no_grad():
            pred_mask = unet_model(img_tensor)

        # Convert the predicted mask to numpy array
        pred_mask = pred_mask.squeeze().argmax(dim=0).cpu().numpy()
        pred_seg[i] = pred_mask.flatten()

    return pred_class, pred_bboxes, pred_seg

class MNISTDataset(torch.utils.data.Dataset):
    """
    Description: A custom dataset class for the MNIST dataset
    """
    def __init__(self, npz_path, transform=None):
        self.data = np.load(npz_path)
        self.images = self.data['images']
        self.labels = self.data['labels']
        self.bboxes = self.data['bboxes']
        self.semantic_masks = self.data['semantic_masks']
        self.instance_masks = self.data['instance_masks']
        self.transform = transform or transforms.ToTensor()
        
    def __len__(self):
        """
        Description: Return the length of the dataset
        """
        return len(self.images)

    def convert_bbox(self, bbox):
        """
        Description: Convert the bounding box to the required format for YOLO
        """
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        # Normalize the coordinates
        x_center /= 64
        y_center /= 64
        width = x_max - x_min
        height = y_max - y_min
        # Normalize the width and height
        width /= 64
        height /= 64
        return [x_center, y_center, width, height]
    
    def __getitem__(self, idx):
        """
        Description: Get the item at the specified index
        """
        image = self.images[idx].reshape((64, 64, 3))
        label = list(self.labels[idx])
        bbox = [self.convert_bbox(self.bboxes[idx][0]), self.convert_bbox(self.bboxes[idx][1])]
        semantic_mask = self.semantic_masks[idx].reshape((64, 64))
        instance_mask = self.instance_masks[idx].reshape((64, 64))

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        image = transforms.Resize((640, 640)) # Resize the image to 640x640

        target = {
            'boxes': bbox,
            'labels': torch.tensor([label]),
            'image_id': torch.tensor([idx]),
            'semantic_mask': semantic_mask,
            'instance_mask': instance_mask
        }

        return image, target

def save_dataset(dataset, root_dir='datasets/train', name='train', type='detection'):
    """
    Description: Save the dataset to the folder according to YOLO format
    """
    images_dir = os.path.join(root_dir, 'images')
    labels_dir = os.path.join(root_dir, 'labels')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Saving dataset " + name):
        image, target = dataset[idx]

        img_path = os.path.join(images_dir, f'img_{idx}.png')
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(img_path)

        label_path = os.path.join(labels_dir, f'img_{idx}.txt')
        bbox = target['boxes']
        box1 = bbox[0]
        box2 = bbox[1]
        labels = target['labels'][0]
        with open(label_path, 'w') as f:
            f.write(f"{labels[0]} {box1[0]} {box1[1]} {box1[2]} {box1[3]}\n")
            f.write(f"{labels[1]} {box2[0]} {box2[1]} {box2[2]} {box2[3]}")

def train_yolo11():
    """
    Description: Train the YOLO model for detection
    """
    from ultralytics import YOLO
    # Initialize the model
    model = YOLO('yolo11n.pt')

    # Load the dataset
    dataset_train = MNISTDataset('train.npz', transform=transforms.ToTensor())
    save_dataset(dataset_train, root_dir='datasets/train', name='train', type = "detection")
    dataset_val = MNISTDataset('valid.npz', transform=transforms.ToTensor())
    save_dataset(dataset_val, root_dir='datasets/valid', name='valid', type = "detection")
    # Train the model
    model.train(data='MNISTDataset.yaml', epochs=30, batch=8, imgsz=640, device=0)

"""
The following classes are made with reference to the .ipynb notebook provided in the class
"""
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            #Sine(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            #Sine(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )    
        return expand    

class CustomDataset(Dataset):
    # from skimage.io import imread
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    def __getitem__(self, index):
        image = None
        #image = imread(self.image_paths[index])
        t_image = self.transforms_image(image)
        # Load the mask and convert RGB values to class labels
        #mask = imread(self.target_paths[index])  # Load the mask as RGB
        # Convert mask to torch tensor of type long
        mask = torch.tensor(mask, dtype=torch.long)
        return t_image, mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)
    
def train_unet(train_loader, optimizer, num_epochs=6, device="cuda"):
    """
    Description: A simplified training loop for the U-Net model.
    
    Arguments:
    - train_loader (DataLoader): DataLoader for the training data.
    - optimizer (Optimizer): Optimizer to use for training.
    - num_epochs (int): Number of epochs to train.
    - device (str): Device for training ('cuda' or 'cpu').
    """
    # Move model to the correct device
    from torch.optim.lr_scheduler import CosineAnnealingLR
    # Training parameters
    unet_model = UNET(3,3).to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    for epoch in range(num_epochs):
        # Training
        unet_model.train()  # Set network to training mode
        train_loss = 0
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (data, target) in train_progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = unet_model(data)
            # Compute loss
            loss = F.cross_entropy(output, target)
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            # Accumulate training loss
            train_loss += loss.item()
            # Update the progress bar with the current batch loss
            train_progress_bar.set_postfix(batch_loss=loss.item())
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
    torch.save(unet_model.state_dict(), "best_model-seg.pt")

def save_images_from_npz(npz_file, image_dir, mask_dir):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Iterate over the data and save images and masks
    for i, (image, mask) in enumerate(zip(data['images'], data['instance_masks'])):
        image_path = os.path.join(image_dir, f'img{i:04d}.png')
        mask_path = os.path.join(mask_dir, f'img{i:04d}.png')
        
        # Convert arrays to images and save
        Image.fromarray(image.reshape(64, 64, 3)).save(image_path)
        Image.fromarray(mask.reshape(64, 64)).save(mask_path)


