from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        # root / <label> / <train/test> / <item>_<view>.png
        for label in os.listdir(root):  # Label
            label_path = os.path.join(root, label, data_type)
            if os.path.isdir(label_path):  # Checking if the path is valid
                items_views = {}
                for file in os.listdir(label_path):
                    if file.endswith(".png"):  # Ensure it's an image file
                        item_view = file.split('_')  # Split filename into item and view
                        item = '_'.join(item_view[:-1])  # Item name (handle cases where item name might contain '_')
                        view = item_view[-1]  # View (e.g., "1.png")
                        
                        if item not in items_views:
                            items_views[item] = []
                        items_views[item].append(os.path.join(label_path, file))
                
                # Now append the collected views and labels into the lists
                for item, views in items_views.items():
                    self.x.append(views)
                    self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
