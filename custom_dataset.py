from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def __init__(self, 
                 root, 
                 data_type, 
                 transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]), 
                target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label> / <train/test> / <item>_<view>.png
        for label in self.classes:
            label_path = os.path.join(root, label, data_type)
            if os.path.isdir(label_path):
                items_views = {}
                for file in os.listdir(label_path):
                    if file.endswith(".png") and not file.startswith('.'):
                        item_view = file.rsplit('_', 1)  # Split filename on the last underscore
                        if len(item_view) == 2:  # Ensure the filename is correctly formatted
                            item, view = item_view
                            view_path = os.path.join(label_path, file)
                            if item not in items_views:
                                items_views[item] = []
                            items_views[item].append(view_path)
                
                # Append the collected views and labels into the lists
                for item, views in items_views.items():
                    self.x.append(views)
                    self.y.append(self.class_to_idx[label])

    def __getitem__(self, index):
        original_views = self.x[index]
        views = []
        errors = []

        for view_path in original_views:
            try:
                im = Image.open(view_path).convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                views.append(im)
            except (IOError, UnidentifiedImageError) as e:
                errors.append((view_path, str(e)))

        if errors:
            print(f"Failed to load some images for index {index}: {errors}")

        return views, self.y[index]

    def __len__(self):
        return len(self.x)
