# @Author :ZhenyuZhang
# @CrateTime   :2021/10/02 12:12
# @File   :20210917PredictEvalbySpecificPathPKL.py
# @Email  :201718018670196

from __future__ import absolute_import, division, print_function

import random
import cv2
import torch
import numpy as np
import os
import pathlib
import six
import soundfile as sf
from scipy.io import wavfile
import scipy
from PIL import Image
import  argparse
from datetime import datetime
import glob
from  CalculIntegralBinaryAcc20210907 import calcualateIntegralAcc




def parent(path):
    path = pathlib.Path(path)
    return str(path.parent)

def exist(path):
    return os.path.exists(str(path))

def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]

def convert_seg_gray_to_color(input, n_classes, output_path=None, colors=class_colors):
    """
    Convert the segmented image on gray to color.

    :param input: it is available to get two type(ndarray, string), string type is a file path.
    :param n_classes: number of the classes.
    :param output_path: output path. if it is None, this function return result array(ndarray)
    :param colors: refer to 'class_colors' format. Default: random assigned color.
    :return: if out_path is None, return result array(ndarray)
    """
    if isinstance(input, six.string_types):
        seg = cv2.imread(input, flags=cv2.IMREAD_GRAYSCALE)
    elif type(input) is np.ndarray:
        assert len(input.shape) == 2, "Input should be h,w "
        seg = input

    height = seg.shape[0]
    width = seg.shape[1]

    seg_img = np.zeros((height, width, 3))

    for c in range(n_classes):
        seg_arr = seg[:, :] == c
        seg_img[:, :, 0] += ((seg_arr) * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr) * colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr) * colors[c][2]).astype('uint8')

    if output_path:
        cv2.imwrite(output_path, seg_img)
    else:
        return seg_img

def predict(model, input_path, output_path, colors=class_colors):
    """
    This function can save a predicted result on the color from the trained model.

    :param model: a network model.
    :param input_path: the input file path.
    :param output_path: the output file path.
    :param colors: refer to 'class_colors' format. Default: random assigned color.
    :return: model result.
    """
    model.eval()

    # #read image file
    # img = cv2.imread(input_path, flags=cv2.IMREAD_COLOR)
    # ori_height = img.shape[0]
    # ori_width = img.shape[1]


    # # read audio file
    # # img = sf.read(input_path)
    # # image_0 = img[0]
    # sf,img = wavfile.read(input_path)
    # image_0 = img
    #
    # ori_height = 1
    # ori_width = image_0.shape[0]

    img = np.array(Image.open(input_path))
    ori_height = img.shape[0]
    ori_width = img.shape[1]

    # if np.max(seg) > 0:
    #     seg = seg / np.max(seg)

    model_width = model.img_width
    model_height = model.img_height

    # if model_width != ori_width or model_height != ori_height:
    #     img = cv2.resize(img, (model_width, model_height), interpolation=cv2.INTER_NEAREST)

    data = img.reshape((1, ori_height, ori_width))
    # data = img.transpose((2, 0, 1))
    data = data[None, :, :, :]
    data = torch.from_numpy(data).float()

    if next(model.parameters()).is_cuda:
        if not torch.cuda.is_available():
            raise ValueError("A model was trained via .cuda(), but this system can not support cuda.")
        data = data.cuda()

    score = model(data)

    lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
    # lbl_pred = lbl_pred.transpose((1, 2, 0))
    lbl_pred = lbl_pred.reshape(ori_height, ori_width)
    # n_classes = np.max(lbl_pred)
    # lbl_pred = lbl_pred.reshape(model_width)

    # seg_img = convert_seg_gray_to_color(lbl_pred, n_classes, colors=colors)
    # (filepath, tempfilename) = os.path.split(input_path)
    # (filename, extension) = os.path.splitext(tempfilename)
    # outputfilename=output_path+'/'+tempfilename
    # c = lbl_pred.astype(np.float64)
    # scipy.io.wavfile.write(output_path, 16000, c)

    image_label_array = lbl_pred * 255  # 变换为0-255的灰度值
    image_label_array=np.uint8(image_label_array)
    im_label = Image.fromarray(image_label_array)
    im_label = im_label.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im_label.save(output_path)


    # if model_width != ori_width or model_height != ori_height:
    #     seg_img = cv2.resize(seg_img, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)
    #
    # if not exist(parent(output_path)):
    #     mkdir(parent(output_path))
    #
    # cv2.imwrite(output_path, seg_img)
    # a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
    unique, counts = np.unique(lbl_pred, return_counts=True)
    dict(zip(unique, counts))

    return dict(zip(unique, counts))





if __name__ == '__main__':
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model Modified By Zhenyu Zhang')
    parser.add_argument('--test_feature_map_path', type=str, default= r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Dev_mfccfeatureimage_WholeNorm', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--test_feature_truemaks_path', type=str, default= r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Dev_mfcc_groundtruthmask_WholeNorm', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'

    parser.add_argument('--PKLmodel_path', type=str, default="/home1/zzy/Forensics/TraditionalEditorDetection/Codes/SemanticSegmentation_2DFeaMap/models/model_EnglishTIMIT_mfcc_fcn16_vgg16_200_64_1000000/epochDev_132.pth", help='eval mode') #fcn16_vgg16 fcn8_mobilenet_v2    fcn16_resnet34  fcn8_mobilenet_v2 #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--maxmum_count_load', type=int, default=1000000, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--gpu_id_begin0',  type=str, default='2', help='the GpuCrd number')  #StepLR50_09
    parser.add_argument('--PrintInterval', type=int, default=1000, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'

    args = parser.parse_args()

    IDgpuCard=args.gpu_id_begin0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] =  IDgpuCard   #export CUDA_VISIBLE_DEVICES=2,必须放在出现torch之前，否则将不会起效果


    print(args.PKLmodel_path)
    assert os.path.exists(args.PKLmodel_path), 'There is no checkpoint model in the model_path+ model_tage'
    model = torch.load(args.PKLmodel_path)  # 加载模型

    path=os.path.normpath(args.PKLmodel_path)
    pathdetaillist=path.split(os.sep)
    Outputpath=args.test_feature_truemaks_path+'/'+pathdetaillist[-2]+pathdetaillist[-1]
    if os.path.exists(Outputpath) is False:
        os.makedirs(Outputpath)


    files = glob.glob(os.path.join(args.test_feature_map_path+ '/*png'))
    for i, file in enumerate(files):
        filepath, tmpfilename = os.path.split(file)
        if i<args.maxmum_count_load:
            score=predict(model, args.test_feature_map_path+'/'+tmpfilename,Outputpath+'/'+tmpfilename)
            if i%args.PrintInterval==0:
                print('processing file:', i,args.test_feature_map_path+'/'+tmpfilename,score)
        else:
            break

    # ########根据标签文件夹下的所有文件，计算正确率。
    # rightcount2,wrongcount2,acc2,rightcount1,wrongcount1,acc1=calcualateIntegralAcc(outputfilepath)
    # with open(wavlabelfilepath+'/'+tempfilename+'.txt', 'w') as f:
    #     f.write('AllPartFileCount,rightcount2:{},wrongcount2:{},acc2:{}\n'.format(rightcount2, wrongcount2, acc2))
    #     f.write('OnlyLastHalfPartFileCount,rightcount1:{},wrongcount1:{},acc1:{}\n'.format(rightcount1, wrongcount1, acc1))
    #
