import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm




images_list1 = glob.glob(r'F:\chengxu\LW\Infrared and Visible Image Fusion\ICME_Code_2024_4_25\ICME_Code_2024_4_25\LLVIP\IR/*')           # 图1路径
images_list2 = glob.glob(r'F:\chengxu\LW\Infrared and Visible Image Fusion\ICME_Code_2024_4_25\ICME_Code_2024_4_25\LLVIP\VI/*')             # 图2路径
# images_list3 = glob.glob(r'E:\IR_CI_date\ICME_DATE_4\stage_one_ir_private/*')           # 图1路径
# images_list4 = glob.glob(r'E:\IR_CI_date\ICME_DATE_4\stage_one_vis_private/*')             # 图2路径
out_path1 = r'F:\chengxu\LW\Infrared and Visible Image Fusion\ICME_Code_2024_4_25\ICME_Code_2024_4_25\LLVIP\stage_two_sge_IR'                                                              # 图1切割后保存路径
out_path2 = r'F:\chengxu\LW\Infrared and Visible Image Fusion\ICME_Code_2024_4_25\ICME_Code_2024_4_25\LLVIP\stage_two_sge_VI'



i = 1
for path1, path2 in zip(tqdm(images_list1), images_list2):
    ##创建新文件夹，如果该文件已经存在，则报错
    # os.makedirs('./hhh')
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # img3 = cv2.imread(path3)
    # img4 = cv2.imread(path4)
    ##因为opencv读取的图片是BGR格式
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    ##裁剪块的大小
    patchsize = 256
    ##裁剪的数量
    num_patches = 4

    h, w, _ = img1.shape
    if h == 128 and w == 128:
        img1 = cv2.resize(img1, (500, 500))
        img2 = img1
        h, w = 500, 500


    for j in range(num_patches):

        rr = np.random.randint(0, h - patchsize)
        cc = np.random.randint(0, w - patchsize)

        patchs1 = img1[rr:rr + patchsize, cc:cc + patchsize, :]
        patchs2 = img2[rr:rr + patchsize, cc:cc + patchsize, :]
        # patchs3 = img3[rr:rr + patchsize, cc:cc + patchsize, :]
        # patchs4 = img4[rr:rr + patchsize, cc:cc + patchsize, :]
        cv2.imwrite(os.path.join(out_path1, f'{i}.png'), patchs1)
        cv2.imwrite(os.path.join(out_path2, f'{i}.png'), patchs2)
        # cv2.imwrite(os.path.join(out_path3, f'{i}.png'), patchs3)
        # cv2.imwrite(os.path.join(out_path4, f'{i}.png'), patchs4)

        i = i+1
