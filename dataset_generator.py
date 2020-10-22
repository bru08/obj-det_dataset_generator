# %%
import os
import argparse
import re
import pandas as pd
from pathlib import Path
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from skimage import io
import matplotlib as mpl 
import matplotlib.pyplot as plt

# %%
class OdDataset:

    def __init__(self, img_size=512, obj_size=28, max_obj_per_img=5, obj_type="mnist", export_format="yolo"):
        self.img_size = img_size
        self.obj_size = obj_size
        self.max_obj = max_obj_per_img
        self.export_format = export_format

        if obj_type == "mnist":
            dataset = torchvision.datasets.MNIST(
                root="./data", download=True, train=True
                )
            self.data = dataset.data
            self.labels = dataset.train_labels
        else:
            raise ValueError("Dataset type not implemented")

    def generate_dataset(self, out_dir="./dataset", sets="train-val", n=100, prop_train=0.8):
        n_train = int(n * prop_train)
        out_dir = Path(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir/"valid", exist_ok=True)
        os.makedirs(out_dir/"train", exist_ok=True)

        for i in range(n_train):
            coords, img, labels = self.generate_img()
            self.save_img_coord(out_dir/"train", f"img_{i}", coords, img, labels)
        
        for i in range(n - n_train):
            coords, img, labels = self.generate_img()
            self.save_img_coord(out_dir/"valid", f"img_{i}", coords, img, labels)

    def generate_img(self):
        img = np.zeros((self.img_size, self.img_size))
        n_trgt = np.random.randint(1, self.max_obj) if self.max_obj > 1 else self.max_obj
        trgt_id = np.random.choice(len(self.data), size=n_trgt, replace=False)
        coords = []
        labels = []
        for idx in trgt_id:
            xr, yr = np.random.randint(0,self.img_size-self.obj_size, 2)
            cls_id = self.labels[idx].item()
            coord = [xr, yr, xr + self.obj_size, yr + self.obj_size]
            coords.append(coord)
            labels.append(cls_id)
            trgt = transform.resize(self.data[idx], (self.obj_size, self.obj_size))
            img[coord[1]: coord[3], coord[0]: coord[2]] = trgt

        return np.array(coords), img, labels
    
    def save_img_coord(self, dest, name, coords, img, labels):
        io.imsave(dest / f"{name}.png", (img*255).astype(np.uint8))
        with open(dest / f"{name}.txt", "w+") as f:
            for i, coord in enumerate(coords):

                if self.export_format == "yolo":
                    coord[2] = coord[2] - coord[0]
                    coord[3] = coord[3] - coord[1]
                    coord = (x / self.img_size for x in coord)

                f.write(f"{labels[i]},")
                f.write(",".join([str(x) for x in coord]))
                f.write("\n")


def plot_targets(name, coord_norm=False, modec="xyxy"):
    img = io.imread(name)
    with open(re.sub(".png", ".txt", name), "r") as f:
        annots = np.array([ [float(y) for y in x.split(",")] for x in f.readlines()] )
    plot_box(img, annots[:, 1:], annots[:, 0], modec=modec, coord_norm=coord_norm)
    plt.show()

def plot_box(img, coords, names=None, modec="xyxy", coord_norm=False):
    fig, ax = plt.subplots()
    ax.imshow(img)
    if coord_norm:
        coords = [x*img.shape[0] for x in coords]
    if modec == "xyxy":
        for i, annot in enumerate(coords):
            if names.shape[0]:
                ax.text(annot[0]+3, annot[1]-6, str(names[i]), backgroundcolor="green", fontsize=6)
            w, h = (-annot[0] + annot[2]), (-annot[1] + annot[3])
            patch = mpl.patches.Rectangle((annot[0], annot[1]), w, h, facecolor="none", linewidth=2, edgecolor="green")
            ax.add_patch(patch)
    elif modec == "xywh":
        for i, annot in enumerate(coords):
            print(annot)
            #x, y = round(annot[0] - annot[2]/2), round(annot[1] - annot[3]/2)
            x, y = annot[0], annot[1]
            if names.shape[0]:
                ax.text(x+3, y-6, str(names[i]), backgroundcolor="green", fontsize=6)
            patch = mpl.patches.Rectangle((x, y), annot[2], annot[3], facecolor="none", linewidth=2, edgecolor="green")
            ax.add_patch(patch)





# %%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--image-size', type=int, default=256)
    parser.add_argument('-o', '--object-size', type=int, default=256//5)
    parser.add_argument('-m', '--max-objects', type=int, default=4)
    parser.add_argument("-n", "--dataset-size", type=int, default=100)
    parser.add_argument("-f", "--annotations-format", type=str, default="")
    args = parser.parse_args()

    im_size = args.image_size
    obj_size = args.object_size
    max_obj_per_img = args.max_objects
    ds_size = args.dataset_size
    annot_format = args.annotations_format

    generator = OdDataset(im_size, obj_size, max_obj_per_img, export_format=annot_format)
    generator.generate_dataset(n=ds_size)

# %%

    plot_targets("./dataset/train/img_3.png", coord_norm=True, modec="xywh")



# %%
