from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform= None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [ p for p in self.files if p.endswith(('.jpg', '.png','jpeg'))]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform: 
            image = self.transform(image)
        
        return image
    

def get_transforms(size, crop, final_size): 
    
    transform_list = []
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    else:
        transform_list.append(transforms.Resize(size))
    
    transform_list.append(transforms.Resize(final_size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_features, style_features):
    size = content_features.size()
    style_mean, style_std = calc_mean_std(style_features)
    content_mean, content_std = calc_mean_std(content_features)
    normalized_features = (content_features - content_mean.expand(size)) / content_std.expand(size)
    return normalized_features * style_std + style_mean
    
    

def calc_mean_std(features):
    size = features.size()
    assert len(size) == 4
    batch_size, channels = size[:2]
    feat_mean = features.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = features.view(batch_size, channels, -1).var(dim=2, unbiased = False).view(batch_size, channels, 1, 1) + 1e-5
    feat_std = feat_var.sqrt()
    return feat_mean, feat_std  
