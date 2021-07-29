import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
from model import DRModel
device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
model = DRModel(device)
checkpt = torch.load('/content/models/best_model.pth')
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
model.load_state_dict(checkpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False

model.eval()


class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, dim, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      
        img_name = os.path.join('/content/data/new_data/resized_aptos_2019/resized_test_19', self.data.loc[idx, 'id_code'] + '.jpg')
        image = Image.open(img_name)
        image = image.resize((self.dim, self.dim), resample=Image.BILINEAR)
        image = self.transform(image)
        return {'image': image}


test_dataset = RetinopathyDatasetTest('/content/data/new_data/resized_aptos_2019/labels/testLabels19.csv', 256, transform)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
test_preds = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

#coef = [0.5, 1.5, 2.5, 3.5]
#0.7043314282343803
#1.3705476070376668
#2.1951830069323504
#3.1268313658701907
#3.2926724085868417

coef = [0.7043314282343803, 1.3705476070376668, 2.1951830069323504, 3.1268313658701907]
for i, pred in enumerate(test_preds):
    if pred < coef[0]:
        test_preds[i] = 0
    elif coef[0] <= pred < coef[1]:
        test_preds[i] = 1
    elif coef[1] <= pred < coef[2]:
        test_preds[i] = 2
    elif coef[2] <= pred < coef[3]:
        test_preds[i] = 3
    else:
        test_preds[i] = 4

#test_preds.shape = (1,len(test_preds))
sample = pd.read_csv("/content/data/new_data/resized_aptos_2019/labels/testLabels19.csv")
#sample.diagnosis = pd.Series(test_preds[0])
sample.diagnosis = test_preds.astype(int)
sample.to_csv("/content/submission.csv", index=False)
