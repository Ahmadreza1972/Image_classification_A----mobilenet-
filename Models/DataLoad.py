import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import random

class DataLoad:
    
    def __init__(self,train_path,test_path,valdata_ratio,batch_size,height_transform,width_transform,label_path):
        self._train_path=train_path
        self._test_path=test_path
        self._valdata_ratio=valdata_ratio
        self._batch_size=batch_size
        self._height_transform=height_transform
        self._width_transform=width_transform
        self._label_path=label_path

    # Load dataset from .pth file
    def load_data(self,path):
        raw_data = torch.load(path, weights_only=False)
        data = raw_data['data']
        if isinstance(raw_data['labels'], list):
            labels = raw_data['labels']
        else:
            labels = raw_data['labels'].tolist()
        indices = raw_data['indices']
        return data, labels , indices
    
    def remap_labels(self,labels, class_mapping):
        return [class_mapping[label] for label in labels]

    # Prepare DataLoader with transform
    def create_dataloader(self,images, labels, batch_size, shuffle=False):
        #images = torch.tensor(images).float()  # Convert images to float tensor if necessary
        with open(self._label_path, "r") as file:
            labels_str = [line.strip() for line in file]  # Convert each line to a float
        self._group_labels=labels
        unique_labels = sorted(set(labels))
        unique_labels_str=[labels_str[i] if i != 6 else "Unknown" for i in unique_labels]
        class_mapping = {label: i for i, label in enumerate(unique_labels)}
        remapped_labels = self.remap_labels(labels, class_mapping)
        labels = torch.tensor(remapped_labels).long()  # Convert labels to long tensor (for classification)
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader,unique_labels,unique_labels_str

    def image_label_maker(self,path,test):
        # Load data
        images, labels, _ = self.load_data(path)
        
        # Transformations for input images
        if test:
            transform = transforms.Compose([
            transforms.Resize((self._height_transform, self._width_transform)),  # Resize images
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])
        else:
            
            transform = transforms.Compose([
            transforms.Resize((self._height_transform, self._width_transform)),  # Resize images
            #transforms.RandomHorizontalFlip(p=0.5),  # Flip images with 50% probability
            #transforms.RandomRotation(degrees=15),   # Rotate images randomly within ±15 degrees
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly shift images by 10%
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
            #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Apply slight perspective distortion
            #transforms.RandomResizedCrop((self._height_transform, self._width_transform), scale=(0.8, 1.0)),  # Random cropping
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
            ])
        # Convert tensor to PIL Image for transformation
        print(images[0].shape)
        if images[0].shape[0]==3:
            images_pil = [transforms.ToPILImage()(img.permute(0,1,2)) for img in images]    
        else:
            images_pil = [transforms.ToPILImage()(img.permute(2,0,1)) for img in images]
        images = torch.stack([transform(img) for img in images_pil])  # Apply transformation to each image
        
        return transform,images,labels
    
    def DataLoad(self):

        transform,images,labels=self.image_label_maker(self._train_path,test=False)
        # Convert labels to tensor

        # Shuffle the indices once
        indices = list(range(len(images)))
        random.shuffle(indices)  # Shuffling only once

        # Apply shuffled order
        images = images[indices]
        labels = [labels[i] for i in indices]
        
        if self._test_path!=None:
            train_size = int((1-self._valdata_ratio) * len(images))
            val_size = len(images) - train_size
            train_dataset, val_dataset = random_split(images, [train_size, val_size])
        
            train_images = images[train_dataset.indices]
            val_images = images[val_dataset.indices]
        
            train_labels = [labels[i] for i in train_dataset.indices]
            val_labels = [labels[i] for i in val_dataset.indices]
        
            train_loader,unique_labels,unique_labels_str = self.create_dataloader(train_images, train_labels, batch_size=self._batch_size)
            val_loader,unique_labels,unique_labels_str = self.create_dataloader(val_images, val_labels, batch_size=self._batch_size)
 
            # Load and transform test data
            transform,test_images,test_labels=self.image_label_maker(self._test_path,test=True)

            test_loader,unique_labels,unique_labels_str = self.create_dataloader(test_images, test_labels, batch_size=self._batch_size)
        else:
            train_loader,unique_labels,unique_labels_str = self.create_dataloader(images, labels, batch_size=self._batch_size)
            val_loader=[]
            test_loader=[]
        return train_loader,val_loader,test_loader,unique_labels,unique_labels_str
        
        
            
        