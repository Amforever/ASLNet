# @Author :ZhenyuZhang
# @CrateTime   :2021/9/17 16:04
# @File   :CalculInterAcc20210929.py
# @Email  :201718018670196
''' 2021年10月02日，调通真实计算的方法，不需再进行个修改'''



import  os
import  soundfile as sf
from scipy.io import wavfile
import numpy as np
from PIL import Image
import glob
import  argparse
# wavfilepath = '/home1/zzy/Forensics/TraditionalEditorDetection/EnglishTIMIT/smallFilesPredictsFromEval'


def calcualateIntegralAcc(pnglabelfilepath,pngpredlabelfilepath,Threshold,Intervalprint):
    Reallabel=pnglabelfilepath
    predictlabel=pngpredlabelfilepath
    # Realfileslist = os.listdir(pnglabelfilepath)  #files只是文件，但是如果有文件夹或别的格式的文件，也会打印出来。

    Realfileslist = glob.glob(os.path.join(pnglabelfilepath + '/*png')) #files有整体文件路径
    Realfileslist=Realfileslist[:]
    ClipsFilePath=[]

    ClipsRealLabelList=np.zeros(Realfileslist.__len__())
    ClipsProblabelList=np.zeros(Realfileslist.__len__())
    for i, filereal in enumerate(Realfileslist):
        (filepath, tempfilename) = os.path.split(filereal)
        filepred=os.path.join(pngpredlabelfilepath,tempfilename)
        matrixreal = np.array(Image.open(filereal))
        matrixpred = np.array(Image.open(filepred))
        ori_height = matrixreal.shape[0]
        ori_width = matrixreal.shape[1]
        ElementCorrectCount1=0                              #整个矩阵中 所有元素判断正确的个数
        ElementCorrectPercolCount1=np.zeros(int(ori_width/2))    #矩阵中 每列中判断对的元素的个数
        ColCorrect1=0                                       #矩阵中 列判断对的个数

        ElementCorrectCount2=0                              #整个矩阵中 所有元素判断正确的个数
        ElementCorrectPercolCount2=np.zeros(int(ori_width/2))      #矩阵中 每列中判断对的元素的个数
        ColCorrect2=0                                       #矩阵中 列判断对的个数

        ElementCorrectCount3=0

        for clocount in range (int(ori_width/3)):
            for rowcount in range(ori_height):
                # print(clocount,rowcount)
                if matrixpred[rowcount,clocount]==0:
                    ElementCorrectCount1=ElementCorrectCount1+1
        rho1=ElementCorrectCount1/(int(ori_width/3)*ori_height)


        for clocount in range (int(ori_width/3),int(ori_width*2/3)):
            for rowcount in range(ori_height):
                # print(clocount,rowcount)
                if matrixpred[rowcount,clocount]==0:
                    ElementCorrectCount2=ElementCorrectCount2+1
        rho2=ElementCorrectCount2/((int(ori_width/3))*ori_height)


        for clocount in range (int(ori_width*2/3),ori_width):
            for rowcount in range(ori_height):
                # print(clocount,rowcount)
                if matrixpred[rowcount,clocount]==0:
                    ElementCorrectCount3=ElementCorrectCount3+1
        rho3=ElementCorrectCount3/((int(ori_width/3))*ori_height)


        if (1-rho1)>Threshold or (1-rho2)>Threshold or (1-rho3)>Threshold:  #rho1
            audioclippretlab=0        #0代表假样本，阳性，
        else:
            audioclippretlab=1        #1代表真样本，阴性，

        # cont01=len(set(matrixreal))
        cont01 = (np.unique(matrixreal)).size
        if cont01==2:
            audiocliprallab=0        #0代表假样本，阳性，
        elif cont01==1:
            audiocliprallab=1           #1代表真样本，阴性，
        else:
            print('Audio maxtix include 0, 255 and others')
            exit()

        ClipsProblabelList[i] = audioclippretlab
        ClipsRealLabelList[i]=audiocliprallab

        # DecAccElementProb=ElementCorrectCount/(ori_height*ori_width)
        # DecAccColProb=ColCorrect/ori_width
        # # print('DecAccElementProb:',DecAccElementProb,'DecAccColProb',DecAccColProb)
        ClipsFilePath.append(filepred)
        # ClipsElementProb[i]=DecAccElementProb
        # ClipsColProb[i]=DecAccColProb
        if i % Intervalprint==0:
            print('processing file:',i,'FileName:',filepred,'DecAccElementProb:', audioclippretlab, 'DecAccColProb', audiocliprallab)
    # print('ClipsElementProb:', np.mean(ClipsElementProb), 'ClipsColProb', np.mean(ClipsColProb))
    return ClipsFilePath,ClipsProblabelList,ClipsRealLabelList


if __name__ == '__main__':

    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model Modified By Zhenyu Zhang')
    parser.add_argument('--pnglabelfilepath', type=str, default='/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Eval_mfcc_groundtruthmask_WholeNorm')                 #默认训练时为False  True
    parser.add_argument('--pngpredlabelfilepath', type=str, default='/home1/zzy/Forensics/TraditionalEditorDetection/FinalData/EnglishTIMIT/ClipsALL2s_Eval_mfcc_groundtruthmask_WholeNorm/fcn16_vgg16_BS64_NumE2_Max1000000_Tim20210928_124218') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--Threshold', type=float, default=0.99, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'
    parser.add_argument('--PrintInterval', type=int, default=1000, help='eval mode') #default=r'D:\AudioForensics\ASVcode\asvspoof2019masterNesl\models\model_logical_MfccFilterHPFInter_resnext50_2000_8_1e-06\epoch_0.pth', help='Model checkpoint')        #,default=None)                 #默认训练时为None  r'D:\AudioForensics\ASVcode\CodeFomGithub\asvspoof2019masterNesl\models\model_logical_spect_20_16_0.0001\epoch_19.pth'

    args = parser.parse_args()

    pnglabelfilepath = args.pnglabelfilepath
    pngpredlabelfilepath=  args.pngpredlabelfilepath
    Threshold=args.Threshold
    resulttxtpath=''
    ClipsFilePath,ClipsProblabelList,ClipsRealLabelList=calcualateIntegralAcc(pnglabelfilepath,pngpredlabelfilepath,Threshold,args.PrintInterval)

    TurePoistive_OrigOrig=0
    TureNegative_SplicedSpliced=0
    FalsePoistive_SplicedOrig=0
    FalseNegative_OrigSpliced=0

    PredictTprTnrAccFile = os.path.basename(pngpredlabelfilepath)+'_Thr'+str(Threshold)+'.txt'


    with open(PredictTprTnrAccFile, 'w') as f:
        for i in range(ClipsRealLabelList.__len__()):
            if ClipsRealLabelList[i]==0 and ClipsProblabelList[i]==0:
                TureNegative_SplicedSpliced=TureNegative_SplicedSpliced+1
            if ClipsRealLabelList[i]==1 and ClipsProblabelList[i]==1:
                TurePoistive_OrigOrig=TurePoistive_OrigOrig+1
            if ClipsRealLabelList[i] == 0 and ClipsProblabelList[i] == 1:
                FalsePoistive_SplicedOrig = FalsePoistive_SplicedOrig+1
            if ClipsRealLabelList[i] == 1 and ClipsProblabelList[i] == 0:
                FalseNegative_OrigSpliced=FalseNegative_OrigSpliced+1
            f.write('Filename:{},ClipsProblabelList:{},ClipsRealLabelList:{}\n'.format(ClipsFilePath[i], ClipsProblabelList[i], ClipsRealLabelList[i]))

        f.write('Threshold:{},Filename:{},TurePoistive_OrigOrig:{},TureNegative_SplicedSpliced:{},FalsePoistive_SplicedOrig:{},FalseNegative_OrigSpliced:{}\n'.format(Threshold,'ALL', TurePoistive_OrigOrig, TureNegative_SplicedSpliced,FalsePoistive_SplicedOrig,FalseNegative_OrigSpliced))

        TPR=TurePoistive_OrigOrig/(TurePoistive_OrigOrig+FalseNegative_OrigSpliced)
        TNR=TureNegative_SplicedSpliced/(TureNegative_SplicedSpliced+FalsePoistive_SplicedOrig)
        ACC=(TurePoistive_OrigOrig+TureNegative_SplicedSpliced)/(TurePoistive_OrigOrig+FalseNegative_OrigSpliced+TureNegative_SplicedSpliced+FalsePoistive_SplicedOrig)
        f.write('Threshold:{},Filename:{},TPR:{},TNR:{},ACC:{}\n'.format(Threshold,'ALL',TPR, TNR,ACC))
