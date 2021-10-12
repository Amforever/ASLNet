# @Author :ZhenyuZhang
# @CrateTime   :2021/9/5 21:49
# @File   :20210905maintest.py 
# @Email  :201718018670196

from torchvision import transforms
import torch


from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger
import  argparse
from datetime import datetime
from  CalculIntegralBinaryAcc20210907 import calcualateIntegralAcc
import glob

# train_images = r'./segmentation/test/dataset/cityspaces/images/train'  # ./segmentation/test/dataset/cityspaces   dataset/cityspaces/images/train
# test_images = r'./segmentation/test/dataset/cityspaces/images/test'
# train_labled = r'./segmentation/test/dataset/cityspaces/labeled/train'
# test_labeled = r'./segmentation/test/dataset/cityspaces/labeled/test'




if __name__ == '__main__':

    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model Modified By Zhenyu Zhang')
    parser.add_argument('--train_images', type=str, default=r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Train_mfccfeatureimage_WholeNorm' , help='eval mode')                 #默认训练时为False  True
    parser.add_argument('--test_images', type=str, default=r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Dev_mfccfeatureimage_WholeNorm', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--train_labled', type=str, default=r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Train_mfcc_groundtruthmask_WholeNorm', help='eval mode')                 #默认训练时为False  True
    parser.add_argument('--test_labeled', type=str, default= r'/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Dev_mfcc_groundtruthmask_WholeNorm', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--test_images_Finals', type=str, default=r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Eval_spectLPCfeatureimage', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--test_Predictlabeled_Finals', type=str, default=r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Eval_spectLPC_groundtruthmask', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--Acousticfeatures', type=str, default='mfcc', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--FileKindEngChi', type=str, default='EnglishTIMIT', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'


    parser.add_argument('--model_name', type=str, default="fcn16_vgg16", help='eval mode') #fcn16_vgg16 fcn8_mobilenet_v2    fcn16_resnet34  fcn8_mobilenet_v2 #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--batch_size', type=int, default=2, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--num_epochs', type=int, default=2, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--gpu_id_begin0',  type=str, default='2', help='the GpuCrd number')  #StepLR50_09
    parser.add_argument('--AudioLength', type=str, default='2s', help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--maxmum_count_load', type=int, default=320, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'

    args = parser.parse_args()
    train_images = args.train_images
    test_images = args.test_images
    train_labled = args.train_labled
    test_labeled = args.test_labeled
    model_name = args.model_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    IDgpuCard=args.gpu_id_begin0
    maxmum_count_load=args.maxmum_count_load
    timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S')

    model_tag = 'model_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.FileKindEngChi, args.Acousticfeatures, args.model_name, args.num_epochs, args.batch_size, args.AudioLength,args.maxmum_count_load,timestamp)
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] =  IDgpuCard   #export CUDA_VISIBLE_DEVICES=2,必须放在出现torch之前，否则将不会起效果


    # train_images = r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Train'  # ./segmentation/test/dataset/cityspaces   dataset/cityspaces/images/train
    # test_images = r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Dev'
    # train_labled = r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Train_samplelabel'
    # test_labeled = r'/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/ClipsALL2sSeed15_Dev_samplelabel'
    # model_name = "fcn8_resnet34"  #fcn8_vgg16    fcn8_resnet34   fcn8_mobilenet_v2
    # batch_size = 8
    # num_epochs = 10
    # maxmum_count_load=320


    device = 'cuda'
    n_classes = 2
    image_axis_minimum_size = 200
    pretrained = False  #默认是True
    fixed_feature = False
    logger = Logger(model_name=model_name, data_name='example')

    ### Loader
    compose = transforms.Compose([
        Rescale(image_axis_minimum_size), #放大图片，到image_axis_minimum_size， w
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes,maxmum_count_load, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images, test_labeled, n_classes,maxmum_count_load, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)
    print(model)

    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    trainer = Trainer(model, optimizer, logger, num_epochs,model_name,batch_size,maxmum_count_load,args.Acousticfeatures,model_save_path,args.FileKindEngChi, train_loader, test_loader,scheduler)
    trainer.train()


    #### Writing the predict result.
    # predict(model, r'./segmentation/test/dataset/cityspaces/input.png',
    #          r'./segmentation/test/dataset/cityspaces/output.png')


    # wavfilepath='/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/smallFilesPredictsFromEval'
    # wavlabelfilepath='/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/smallFilesPredictsFromEval_samplelabel'

    wavfilepath=args.test_images_Finals
    wavlabelfilepath=args.test_Predictlabeled_Finals
    tempfilename=model_name+'_BS'+str(batch_size)+'_NumE'+str(num_epochs)+'_Max'+str(maxmum_count_load)+'_Tim'+timestamp
    outputfilepath=wavlabelfilepath+'/'+ tempfilename
    if os.path.exists(outputfilepath) is False:
        os.makedirs(outputfilepath)
    # files=os.listdir(wavfilepath)
    files = glob.glob(os.path.join(wavfilepath+ '/*png'))
    for i, file in enumerate(files):
        filepath, tmpfilename = os.path.split(file)
        if i<maxmum_count_load:
            score=predict(model, wavfilepath+'/'+tmpfilename,outputfilepath+'/'+tmpfilename)
            if i%100==0:
                print('processing file:', i,wavfilepath+'/'+tmpfilename,score)
        else:
            break

    ########根据标签文件夹下的所有文件，计算正确率。
    rightcount2,wrongcount2,acc2,rightcount1,wrongcount1,acc1=calcualateIntegralAcc(outputfilepath)
    with open(wavlabelfilepath+'/'+tempfilename+'.txt', 'w') as f:
        f.write('AllPartFileCount,rightcount2:{},wrongcount2:{},acc2:{}\n'.format(rightcount2, wrongcount2, acc2))
        f.write('OnlyLastHalfPartFileCount,rightcount1:{},wrongcount1:{},acc1:{}\n'.format(rightcount1, wrongcount1, acc1))


