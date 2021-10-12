"""
The trainer class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

from util.validation import *
from util.logger import *
from datetime import datetime
import time

try:
    from tqdm import tqdm
    from tqdm import trange
except ImportError:
    print("tqdm and trange not found, disabling progress bars")

    def tqdm(iter):
        return iter

    def trange(iter):
        return iter

TQDM_COLS = 80
start = time.time()

def cross_entropy2d(input, target):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # input: (n*h*w, c)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0.0
    target = target[mask]

    func_loss = torch.nn.CrossEntropyLoss()
    loss = func_loss(input, target)

    return loss


class Trainer(object):

    def __init__(self, model, optimizer, logger, num_epochs,model_name,batch_size,maxmum_count_load,Acousticfeatures,model_save_path,FileKindEngChi, train_loader,
                 test_loader=None,   scheduler=None,
                 epoch=0,
                 log_batch_stride=100,
                 check_point_epoch_stride=60
                 ):
        """
        :param model: A network model to train.
        :param optimizer: A optimizer.
        :param logger: The logger for writing results to Tensorboard.
        :param num_epochs: iteration count.
        :param train_loader: pytorch's DataLoader
        :param test_loader: pytorch's DataLoader
        :param epoch: the start epoch number.
        :param log_batch_stride: it determines the step to write log in the batch loop.
        :param check_point_epoch_stride: it determines the step to save a model in the epoch loop.
        :param scheduler: optimizer scheduler for adjusting learning rate.
        """
        self.cuda = torch.cuda.is_available()
        self.model = model
        self.optim = optimizer
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_nameself = model_name
        self.batch_sizeself = batch_size
        self.maxmum_count_load = maxmum_count_load
        self.Acousticfeatures = Acousticfeatures
        self.model_save_pathself =model_save_path
        self.FileKindEngChiself =FileKindEngChi
        self.num_epoches = num_epochs
        self.check_point_step = check_point_epoch_stride
        self.log_batch_stride = log_batch_stride
        self.scheduler = scheduler

        self.epoch = epoch

    def train(self):

        if not next(self.model.parameters()).is_cuda and self.cuda:
            raise ValueError("A model should be set via .cuda() before constructing optimizer.")

        acc_best_TrainAllEpoch = 0
        acc_best_TrainInAllEpoch_CroEpoch = 0
        acc_best_TestInAllEpoch = 0
        acc_best_TestInAllEpoch_CroEpoch = 0
        for epoch in trange(self.epoch, self.num_epoches,
                          position=0,
                          desc='Train', ncols=TQDM_COLS):
            self.epoch = epoch


            # train
            acc_all_TrainInOneEpoch,acc_all_TestInOneEpoch=self._train_epoch()

            print(acc_all_TrainInOneEpoch,acc_all_TestInOneEpoch)
            acc_ave_TrainInOneEpoch = np.average(np.array(acc_all_TrainInOneEpoch))
            acc_ave_TestInOneEpoch = np.average(np.array(acc_all_TestInOneEpoch))
            if acc_ave_TrainInOneEpoch > acc_best_TrainAllEpoch:
                acc_best_TrainAllEpoch = acc_ave_TrainInOneEpoch
                acc_best_TrainInAllEpoch_CroEpoch=epoch
                torch.save(self.model, os.path.join(self.model_save_pathself, 'epochTrain_{}.pth'.format(
                    epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。

            if acc_ave_TestInOneEpoch > acc_best_TestInAllEpoch:
                acc_best_TestInAllEpoch = acc_ave_TestInOneEpoch
                acc_best_TestInAllEpoch_CroEpoch=epoch
                torch.save(self.model, os.path.join(self.model_save_pathself, 'epochDev_{}.pth'.format(
                    epoch)))  # 保存模型直接保存模型的所有内容，调用时也方便调用GFKD项目集成时；调用的模块。

            print('\n*********************************************************************','\nThe begining time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)),
                  ' The current running time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                  ' ModelName:', self.model_nameself,' Batchsize:',self.batch_sizeself,' MaximumAudioLoad:', self.maxmum_count_load,' Acousticfeatures:',self.Acousticfeatures, ' FileKindEngChiself:',self.FileKindEngChiself,  ' CurrentEpoch:', self.epoch)
            print(
                "acc_ave_TrainInOneEpoch %.10f, acc_best_TrainAllEpoch:%.10f, acc_best_TrainInAllEpoch_CroEpoch:%d, acc_ave_TestInOneEpoch:%.10f, acc_best_TestInAllEpoch:%.10f, acc_best_TestInAllEpoch_CroEpoch: %d" % (
                    acc_ave_TrainInOneEpoch, acc_best_TrainAllEpoch, acc_best_TrainInAllEpoch_CroEpoch, acc_ave_TestInOneEpoch, acc_best_TestInAllEpoch,acc_best_TestInAllEpoch_CroEpoch))


            # step forward to reduce the learning rate in the optimizer.
            if self.scheduler:
                self.scheduler.step()

            # model checkpoints
            if epoch%self.check_point_step == 0:
                self.logger.save_model_and_optimizer(self.model,
                                                     self.optim,
                                                     'epoch_{}'.format(epoch))




    def evaluate(self):
        num_batches = len(self.test_loader)
        self.model.eval()

        with torch.no_grad():
            for n_batch, (sample_batched) in tqdm(enumerate(self.test_loader),
                                total=num_batches,
                                leave=False,
                                desc="Valid epoch={}".format(self.epoch),
                                ncols=TQDM_COLS):
                self._eval_batch(sample_batched, n_batch, num_batches)

    def _train_epoch(self):
        acc_best_TrainInOneEpoch=0
        acc_all_TrainInOneEpoch=[]
        acc_best_TrainInOneEpoch_CroBatch=0
        acc_best_TestInOneEpoch=0
        acc_all_TestInOneEpoch=[]
        acc_best_TestInOneEpoch_CroBatch=0
        NumBatchFrom0=0


        num_batches = len(self.train_loader)

        if self.test_loader:
            dataloader_iterator = iter(self.test_loader)

        for n_batch, (sample_batched) in tqdm(enumerate(self.train_loader),
                                              total=num_batches,
                                              leave=False,
                                              desc="Train epoch={}".format(self.epoch),
                                              ncols=TQDM_COLS):
            self.model.train()
            data = sample_batched['image']
            target = sample_batched['annotation']

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            self.optim.zero_grad()

            torch.cuda.empty_cache()

            score = self.model(data)
            loss = cross_entropy2d(score, target)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()
            self.optim.step()

            # print('\nfirst:\n',n_batch)

            #如果不是log的整数倍那就不写如log日志，直接跳出本次循环。下面的都不再执行。
            if n_batch%self.log_batch_stride != 0:
                continue
            # print('Second:', n_batch)
            self.logger.store_checkpoint_var('img_width', data.shape[3])
            self.logger.store_checkpoint_var('img_height', data.shape[2])

            self.model.img_width = data.shape[3]
            self.model.img_height = data.shape[2]

            #write logs to Tensorboard.
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            # lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :]
            # lbl_pred=lbl_pred.reshape((:, 32000))
            # lbl_pred=lbl_pred.reshape(-1,32000)#降为二维，

            lbl_true = target.data.cpu().numpy()
            # lbl_true=lbl_true.reshape(-1,1,32000)
            acc, acc_cls, mean_iou, fwavacc = \
                label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])
            if acc>acc_best_TrainInOneEpoch:
                acc_best_TrainInOneEpoch=acc
                acc_best_TrainInOneEpoch_CroBatch=n_batch



            self.logger.log_train(loss, 'loss', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc, 'acc', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
            self.logger.log_train(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
            self.logger.log_train(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)
            print('\nThe begining time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)),
                  ' The current running time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                  ' ModelName:', self.model_nameself,' Batchsize:',self.batch_sizeself,' MaximumAudioLoad:', self.maxmum_count_load,' CurrentEpoch:', self.epoch)
            print(
                "Train loss %.10f, Train acc:%.10f, Train acc_cls:%.10f, Train mean_iou:%.10f, Train fwavacc:%.10f, n_batch: %d, num_batches: %d, acc_best_TrainInOneEpoch:%.10f, acc_best_TrainInOneEpoch_CroBatch:%d, LearningRate:%.10f" % (
                loss, acc, acc_cls, mean_iou, fwavacc,n_batch,num_batches, acc_best_TrainInOneEpoch, acc_best_TrainInOneEpoch_CroBatch,self.optim.param_groups[0]['lr']))

            #write result images when starting epoch.
            if n_batch == 0:
                log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
                log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
                self.logger.log_images_train(log_img, self.epoch, n_batch, num_batches,
                                             nrows=data.shape[0])

            #if the trainer has the test loader, it evaluates the model using the test data.
            if self.test_loader:
                self.model.eval()
                with torch.no_grad():
                    try:
                        sample_batched = next(dataloader_iterator)
                    except StopIteration:
                        dataloader_iterator = iter(self.test_loader)
                        sample_batched = next(dataloader_iterator)

                    loss_test,acc_test, acc_cls_test, mean_iou_test, fwavacc_test=self._eval_batch(sample_batched, n_batch, num_batches)
                    if acc_test > acc_best_TestInOneEpoch:
                        acc_best_TestInOneEpoch = acc_test
                        acc_best_TestInOneEpoch_CroBatch = n_batch
                    print('\nThe begining time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)),
                          ' The current running time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                    print(
                        "Test loss %.10f, Test acc:%.10f, Test acc_cls:%.10f, Test mean_iou:%.10f, Test fwavacc:%.10f, n_batch: %d, num_batches: %d, acc_best_TestInOneEpoch:%.10f, acc_best_TestInOneEpoch_CroBatch:%d " % (
                        loss_test, acc_test, acc_cls_test, mean_iou_test, fwavacc_test, n_batch, num_batches, acc_best_TestInOneEpoch,
                        acc_best_TestInOneEpoch_CroBatch))

            acc_all_TrainInOneEpoch.append(acc)
            acc_all_TestInOneEpoch.append(acc_test)

        return acc_all_TrainInOneEpoch,acc_all_TestInOneEpoch









    def _eval_batch(self, sample_batched, n_batch, num_batches):
        data = sample_batched['image']
        target = sample_batched['annotation']

        if self.cuda:
            data, target = data.cuda(), target.cuda()
        torch.cuda.empty_cache()

        score = self.model(data)

        loss = cross_entropy2d(score, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()
        # lbl_true = lbl_true.reshape(-1, 1, 32000)
        acc, acc_cls, mean_iou, fwavacc = \
            label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])

        self.logger.log_test(loss, 'loss', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc, 'acc', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
        self.logger.log_test(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
        self.logger.log_test(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)

        if n_batch == 0:
            log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
            log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
            self.logger.log_images_test(log_img, self.epoch, n_batch, num_batches,
                                        nrows=data.shape[0])

        return  loss,acc, acc_cls, mean_iou, fwavacc

    def _write_img(self, score, target, input_img, n_batch, num_batches):
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()

        log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        log_img = self.logger.concatenate_images([log_img, input_img.cpu().numpy()[:, :, :, :]])
        self.logger.log_images(log_img, self.epoch, n_batch, num_batches, nrows=log_img.shape[0])