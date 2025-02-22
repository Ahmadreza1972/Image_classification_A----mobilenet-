import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

class DataLoad:
    
    def __init__(self,train_path,test_path,valdata_ratio,batch_size,height_transform,width_transform):
        self._train_path=train_path
        self._test_path=test_path
        self._valdata_ratio=valdata_ratio
        self._batch_size=batch_size
        self._height_transform=height_transform
        self._width_transform=width_transform

    # Load dataset from .pth file
    def load_data(self,path):
        raw_data = torch.load(path, weights_only=False)
        data = raw_data['data']
        if isinstance(raw_data['labels'], list):
            labels = raw_data['labels']
        else:
            labels = raw_data['labels'].tolist()
        indices = [] #raw_data['indices']
        return data, labels , indices
    
    def remap_labels(self,labels, class_mapping):
        return [class_mapping[label] for label in labels]

    # Prepare DataLoader with transform
    def create_dataloader(self,images, labels, batch_size, shuffle=False):
        #images = torch.tensor(images).float()  # Convert images to float tensor if necessary
        unique_labels = sorted(set(labels))
        class_mapping = {label: i for i, label in enumerate(unique_labels)}
        remapped_labels = self.remap_labels(labels, class_mapping)
        labels = torch.tensor(remapped_labels).long()  # Convert labels to long tensor (for classification)
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    
    def DataLoad(self):
        # Load data
        images, labels, _ = self.load_data(self._train_path)
        
        # Transformations for input images
        transform = transforms.Compose([
        transforms.Resize((self._height_transform, self._width_transform)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
        ])
        # Convert tensor to PIL Image for transformation
        print(images[0].shape)
        if images[0].shape[0]==3:
            images_pil = [transforms.ToPILImage()(img.permute(0,1,2)) for img in images]    
        else:
            images_pil = [transforms.ToPILImage()(img.permute(2,0,1)) for img in images]
        images = torch.stack([transform(img) for img in images_pil])  # Apply transformation to each image
        

        if self._test_path!=None:
            train_size = int((1-self._valdata_ratio) * len(images))
            val_size = len(images) - train_size
            train_dataset, val_dataset = random_split(images, [train_size, val_size])
        
            train_images = images[train_dataset.indices]
            val_images = images[val_dataset.indices]
        
            train_labels = [labels[i] for i in train_dataset.indices]
            val_labels = [labels[i] for i in val_dataset.indices]
        
            train_loader = self.create_dataloader(train_images, train_labels, batch_size=self._batch_size)
            val_loader = self.create_dataloader(val_images, val_labels, batch_size=self._batch_size)
 
            # Load and transform test data
            test_images, test_labels, _ = self.load_data(self._test_path)
            
            if test_images[0].shape[0]==3:
                test_images_pil = [transforms.ToPILImage()(img.permute(0,1,2)) for img in test_images]    
            else:
                test_images_pil = [transforms.ToPILImage()(img.permute(2,0,1)) for img in test_images]

            test_images = torch.stack([transform(img) for img in test_images_pil])
            test_loader = self.create_dataloader(test_images, test_labels, batch_size=self._batch_size)
        else:
            train_loader = self.create_dataloader(images, labels, batch_size=self._batch_size)
            val_loader=[]
            test_loader=[]
        return train_loader,val_loader,test_loader
        
        
            
        