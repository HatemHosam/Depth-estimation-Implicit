import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import convnext_tiny
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm

class DepthMapDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform_img=None, transform_depth=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform_img = transform_img
        self.transform_depth = transform_depth
        self.filenames = [f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.filenames[idx])
        depth_path = os.path.join(self.depth_dir, self.filenames[idx].replace('.jpg','.npy'))

        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_image = self.transform_img(rgb_image)
        
        depth_image = np.load(depth_path).astype(np.float32)
        #depth_image = Image.fromarray(depth_image, mode = 'L')
        #depth_image = transform_depth(depth_image)
        depth_image = cv2.resize(depth_image, dsize=(30,40))
        depth_image = torch.from_numpy(depth_image)

        return rgb_image, depth_image

transform_img = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

transform_depth = transforms.Compose([
    #transforms.Resize((15, 20)),
    transforms.ToTensor(),
])

class CustomConvNextTiny(nn.Module):
    def __init__(self, original_model):
        super(CustomConvNextTiny, self).__init__()
        # Adjust indices based on actual model architecture
        self.initial_layers = nn.Sequential(*list(original_model.children())[0][:5], 
                                            list(original_model.children())[0][5][:6])

        # Add a new convolutional layer
        self.extra_conv = nn.Conv2d(in_channels=384, out_channels=1, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.extra_conv(x)
        return x
if __name__ == '__main__':       
    train_dataset = DepthMapDataset('/data/i5O/nyudepthv2_data/train/image/', '/data/i5O/nyudepthv2_data/train/depth/', transform_img, transform_depth)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory= True)
    
    val_dataset = DepthMapDataset('/data/i5O/nyudepthv2_data/val/image/', '/data/i5O/nyudepthv2_data/val/depth/', transform_img, transform_depth)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0,  pin_memory= True)
    # Load ConvNext-tiny pre-trained model
    base_model = convnext_tiny(pretrained=True)
    
    
    
    # Initialize the custom model
    model = CustomConvNextTiny(base_model)
    print(model)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
   
    def calculate_rmse(outputs, labels):
        mask = labels > 0  # Create a boolean mask for values greater than zero
        if torch.sum(mask) == 0:
            return torch.tensor(float('nan'))  # Return NaN if no element is greater than zero
        rmse_log = torch.sqrt(torch.mean((torch.log(labels[mask]) - torch.log(outputs[mask])) ** 2))
        return rmse_log
    
    #mean_labels = torch.mean(torch.cat([labels for _, labels in val_loader], 0))   
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for rgb_images, depth_images in train_loader:
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device).view(depth_images.size(0), -1)  # Flatten depth maps
    
            # Forward pass
            outputs = model(rgb_images)
            outputs_flat = outputs.view(outputs.size(0), -1)  # Reshapes to [256, 196]
            loss = criterion(outputs_flat, depth_images)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update the progress bar with the loss value
            train_bar.update(1)
            train_bar.set_postfix({'loss': loss.item()})
        # Save the model after each epoch or iteration with the loss value in the filename
        loss_value = loss.item()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
        
        # Validation loop
        model.eval()
        total_rmse = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).view(labels.size(0), -1)
                outputs = model(images)
                outputs_flat = outputs.view(outputs.size(0), -1)
                total_rmse += calculate_rmse(outputs_flat, labels)
                total_samples += labels.size(0)
            
            average_rmse = total_rmse / total_samples
            print(f"Validation RMSE: {average_rmse:.8f}")
        
            # Save the model after each epoch or iteration with the loss value in the filename
            filename = f"weights_30_40/epoch_{epoch+1}_train_loss_{loss_value:.4f}_val_RMSE_{average_rmse:.8f}.pth"
            torch.save(model.state_dict(), filename)
