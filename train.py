import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from model.Module_Fusion import FCNFusion
import torch.utils.data
import cv2
from PIL import Image
import os
import glob
from tqdm import tqdm

from loss import ssim_lwy
from torch.utils.data import DataLoader
from loss_vif import L_Grad

class GetDataset_type2(torch.utils.data.Dataset):
    def __init__(self, split, IR_path=None, VIS_path=None):
        super(GetDataset_type2, self).__init__()
        self.split = split
        if split == 'train':
            data_dir_IR = IR_path
            data_dir_VIS = VIS_path
            self.filepath_IR, self.filenames_IR = prepare_data_path(data_dir_IR)
            self.filepath_VIS, self.filenames_VIS = prepare_data_path(data_dir_VIS)
            self.length = min(len(self.filenames_VIS), len(self.filenames_IR))

    def __getitem__(self, index):

        if self.split=='train':
            IR_path = self.filepath_IR[index]
            image_inf_IR = cv2.imread(IR_path, 0)
            image_IR = np.asarray(Image.fromarray(image_inf_IR), dtype=np.float32) / 255.0
            VIS_path = self.filepath_VIS[index]
            image_inf_VIS = cv2.imread(VIS_path, 0)
            image_VIS = np.asarray(Image.fromarray(image_inf_VIS), dtype=np.float32) / 255.
            image_IR = torch.tensor(image_IR).unsqueeze(dim=0)
            image_VIS = torch.tensor(image_VIS).unsqueeze(dim=0)
            return (
                image_IR,
                image_VIS,
            )
    def __len__(self):

        return self.length

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.png"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

my_date= GetDataset_type2(split="train", IR_path="/home/ccwydlq10/code/IVF/ICME_Code_2024_4_25/data/LLVIP/stage_two_sge_IR",
                          VIS_path="/home/ccwydlq10/code/IVF/ICME_Code_2024_4_25/data/LLVIP/stage_two_sge_VI")
train_loader = DataLoader(my_date, shuffle=True, batch_size=6, drop_last=True)
test_module = FCNFusion(64).cuda()
test_module = torch.nn.parallel.DataParallel(test_module, device_ids=[0])

#loss function
loss_l1=nn.L1Loss()
#optimizer
learing_rate=0.00001
optimizer=torch.optim.Adam(test_module.parameters(),lr=learing_rate)
total_train_step=0
total_epoch = 50
interval = 1
test_module.train()
for epoch in range(total_epoch):
    print("------第{}轮训练--------".format(epoch))
    for i, (ir, vi) in tqdm(enumerate(train_loader), total=len(train_loader)):

        # ir, vi = ir.cuda(), vi.cuda()
        ir, vi = ir.cuda(), vi.cuda()
        out_image=test_module((ir),(vi))
        loss_1=  loss_l1(out_image,ir)
        loss_2 =loss_l1(out_image, vi)
        loss_SSIMir_1 =0.2* ssim_lwy(vi, out_image)

        loss_SSIMir_2 =0.2*  ssim_lwy(ir, out_image)
        grad = L_Grad()
        loss_5 = grad(ir, vi, out_image)
        loss= loss_1+loss_2+loss_SSIMir_1+loss_SSIMir_2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1

    # print("Number of training sessions：{},loss{}".format(total_train_step,loss))
    print("Number of training sessions：{},loss1---{},----loss2---{}".format(total_train_step, loss,loss))
    #Save parameters
    model_folder = '/home/ccwydlq10/code/IVF/FCNFusion/chekpoints'
    os.makedirs(model_folder, exist_ok=True)
    # Inspection conditions
    if (epoch + 1) % interval == 0:
        model_out_path = os.path.join(model_folder, f'two_stagef8usss_{epoch:04d}.pth')
        try:
            torch.save(test_module.state_dict(), model_out_path)
            print(f"save {model_out_path}")
        except OSError as e:
            print(f"Error while saving file: {e}")

