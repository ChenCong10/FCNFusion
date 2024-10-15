import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
from PIL import Image
import time
from tqdm import tqdm
from fusion_md3 import fusion_md2

# from fusion_md_xr2 import fusion_md2
import glob
import torch.nn.functional as F

def rgb2ycbcr(img_rgb):
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    # Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255.0
    # Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255.0
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    return Y, Cb, Cr

def ycbcr2rgb(Y, Cb, Cr):
    # R = Y + 1.402 * (Cr - 128 / 255.0)
    # G = Y - 0.34414 * (Cb - 128 / 255.0) - 0.71414 * (Cr - 128 / 255.0)
    # B = Y + 1.772 * (Cb - 128 / 255.0)
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)
    return np.concatenate([R, G, B], axis=-1)

# ir_path = '/home/w_y/datasets/haze/sots/haze'           # 图像路径
# ir_path = r'E:\open_date\MSRS\train_IR'

# ir_path = r'E:\open_date\MSRS\train_IR'
# ir_path = r'E:\open_date\MSRS\sgem_vis_y'
# ir_path = '/home/w_y/datasets/haze/indoor/haze'
# ir_path = '/home/w_y/datasets/haze/Dense_Haze/haze'
# ir_path ="/public/home/lijiangwei/date/MSRS/sgem_vis_y"

ir_path =r"F:\work\test\fanhua\1"
vis_path =r"F:\work\test\fanhua\2"

# vis_path =r"E:\open_date\M3FD\vi"
# ir_path =r"E:\open_date\M3FD\ir"

# ir_path ='./F:\work/test\LLVIP\IR/'
# vis_path ='./F:\work/test\LLVIP\VI/'

# ir_path =r"E:\IR_CI_date\AVIID_ljw\test\ir_samename"
# vis_path =r"E:\IR_CI_date\AVIID_ljw\test\vis_samename"
#
# ir_path =r"E:\IR_CI_date\TNO_gai\new_name_png\train_IR"
# vis_path =r"E:\IR_CI_date\TNO_gai\new_name_png\train_VIS"


# ir_path =r"E:\IR_VIS_date_result\M3FD\ir"
# vis_path =r"E:\IR_VIS_date_result\M3FD\vi"

# ir_path =r"E:\open_date\road\SMALL_cropinfrared"
# vis_path =r"E:\open_date\road\VIS_Y"

# ir_path =r"E:\open_date\MSRS\sgem_IR"
# vis_path =r"E:\open_date\MSRS\sgem_vis_y"



# saving_path = '/public/home/lijiangwei/test_image'                            # 融合图像保存路径
# model_path = "/public/home/lijiangwei/parameter2/fus_0001.pth"


# saving_path = r'E:\open_date\MSRS\ss'                            # 融合图像保存路径


saving_path = r'F:\work\test\fanhua\3'
model_path = './cc/two_stagef8usss_0049.pth'
# saving_path = '/public/home/lijiangwei/date/MSRS/TEST'                            # 融合图像保存路径
# model_path = "/public/home/lijiangwei/date/MSRS/f2us_0000.pth"

# saving_path = r'E:\IR_VIS_date_result\M3FD\my_m3fd'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# model = down_module(64).to(device)
model = fusion_md2(64).to(device)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.load_state_dict(
#     {k.replace('module.', ''): v for k, v in torch.load(model_path, map_location='cpu').items()})
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()



with torch.no_grad():
    ir_path1 = glob.glob(ir_path + '/*')
    vis_path2 = glob.glob(vis_path + '/*')
    time_list = []
    for path1, path2, in zip(tqdm(ir_path1), vis_path2):

        save_path = os.path.join(saving_path, os.path.relpath(path1, ir_path))


        resize_bool = 0


        img_1 = cv2.imread(path1, 0)
        img_2 = cv2.imread(path2, 0)

        print(path2)

        h, w = img_1.shape

        if h%8 !=0  or w%8 !=0:
            h1 = h // 8 * 8
            w1 = w // 8 * 8

            img_1 = cv2.resize(img_1, (w1, h1))
            img_2 = cv2.resize(img_2, (w1, h1))
            resize_bool=1



        input_1 = torch.from_numpy(img_1).float().div(255.).unsqueeze(0).unsqueeze(0)

        input_1 = input_1.to(device)
        input_1 = input_1



        input_2 = torch.from_numpy(img_2).float().div(255.).unsqueeze(0).unsqueeze(0)

        input_2 = input_2.to(device)
        input_2 = input_2


        # Pad the input if not_multiple_of 8
        start = time.time()
        out = model(input_1,input_2)
        end = time.time()
        out = out.cpu().numpy()
        out = out[0, 0, :, :]
        out = out * 255.0

        if resize_bool==1:
            out = cv2.resize(out, (w,h))



        vis_iamge=cv2.imread(path2)

        y,cb,cr=rgb2ycbcr(vis_iamge)

        f_image=ycbcr2rgb(out,cb,cr)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, f_image)
        time_list.append(end - start)
    print('平均耗时：',np.mean(time_list[1:]), 's')
