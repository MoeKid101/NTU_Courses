import torch
import numpy as np
from torch.nn import Module

CIFAR_ORIG_PATH = 'cifar-10-batches-py'

def genData(cifar_path:str, fileNames:list, outName:str):
    '''
    The function to generate a file with content (data, label) from the original
    downloaded CIFAR-10 dataset. Here (data, label) are both np.ndarrays with
    shape <data>:(-1, 3, 32, 32) and <label>:(-1).
    '''
    def unpack(filename:str)->dict:
        import pickle
        with open(str.join('/', [cifar_path, filename]), 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
        return dict
    data_lst, label_lst = list(), list()
    for fileName in fileNames:
        dataDict = unpack(fileName)
        data_lst.append(dataDict[b'data'].reshape((-1, 3, 32, 32)))
        label_lst.append(dataDict[b'labels'])
    dataArr, labelArr = (np.float32(np.concatenate(data_lst, axis=0)) / 256,
                         np.concatenate(label_lst, axis=0))
    torch.save((dataArr, labelArr), outName)

def show(img, path):
    '''
    The function which outputs a certain image. Remember that it accepts only integer
    type arrays.
    '''
    import matplotlib.pyplot as plt
    from PIL import Image
    img = img.reshape((3, 32, 32))
    x = img
    x = np.zeros((32, 32, 3), dtype=int)
    for i in range(3):
        x[:,:,i] = img[i,:,:]
    plt.imshow(x)
    plt.savefig(path)
    plt.clf()

def augment(std_data:np.ndarray, std_label:np.ndarray, numCopy:int=4)->tuple:
    '''
    The function to implement data augmentation. Accepts <std_data>, <std_label>
    with shape (-1, 3, 32, 32) and (-1,) respectively as input.
    Adopted data augmentation: horizontal flipping and random erasing.
    '''
    def genEB_2(minlen:int, maxlen:int, num:int):
        from numpy import random
        hSize, vSize = (random.randint(minlen, maxlen+1, size=num),
                        random.randint(minlen, maxlen+1, size=num))
        hMMarg, vMMarg = maxlen - hSize, maxlen - vSize
        hMargCoef, vMargCoef = random.rand(num), random.rand(num)
        hMarg, vMarg = ((hMMarg*hMargCoef).astype(np.int32),
                        (vMMarg*vMargCoef).astype(np.int32))
        ud, lr = random.randint(2, size=num), random.randint(2, size=num)
        ub, db = ud*(31-hSize-hMarg)+(1-ud)*hMarg, ud*(31-hMarg)+(1-ud)*(hMarg+hSize)
        lb, rb = lr*(31-vSize-vMarg)+(1-lr)*vMarg, lr*(31-vMarg)+(1-lr)*(vMarg+vSize)
        return ub, db, lb, rb
    copyShape = (numCopy, 1, 1, 1, 1)
    eraMin, eraMax = 4, 18
    std_data, std_label = std_data.reshape((1, -1, 3, 32, 32)), std_label.reshape((1, -1))
    DataCopy, LabelCopy = np.tile(std_data, copyShape), np.tile(std_label, copyShape)
    for i in range(numCopy):
        if i >= numCopy//2: DataCopy[i] = np.flip(DataCopy[i], axis=3)
        # random erasing
        ub, db, lb, rb = genEB_2(eraMin, eraMax, DataCopy.shape[1])
        for era_index in range(DataCopy.shape[1]):
            DataCopy[i, era_index, :, ub[era_index]:db[era_index],
                     lb[era_index]:rb[era_index]] = 0
    dataRet, labelRet = DataCopy.reshape((-1, 3, 32, 32)), LabelCopy.reshape((-1))
    return dataRet, labelRet

def train(model:Module, data_file:str, loss_func, optimizer,
          max_epoch:int, temp_folder:str, from_epoch:int=0, shuffle:bool=True,
          augmentation:bool=True, batch_size:int=128, maxBatchInFile:int=128):
    '''
    A standard training routine for a pytorch model to do image classification on 
    CIFAR-10 dataset.
    '''
    import time, os
    from numpy import random
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    model.train()
    # define variables to record losses (which might help you analyze your codes).
    epoch_losses = list()
    LossSavePath = f'{temp_folder}/loss'
    fin_losses = (torch.load(LossSavePath) if (os.path.exists(LossSavePath) and
                  from_epoch > 0) else list())
    # if your training is interrupted by something else, you can directly train from
    # the <from_epoch>-th epoch to save time.
    ModelLoadPath = f'{temp_folder}/model_e{from_epoch}'
    if from_epoch > 0: model.load_state_dict(torch.load(ModelLoadPath))
    # load data file
    dataArr, labelArr = torch.load(data_file)
    for epoch in range(from_epoch, max_epoch+1):
        # record start time of pre-processing. Pre-processing includes augmentation,
        # shuffling and splitting data into subfiles in this case. File splitting is
        # intended to restrict GPU memory usage.
        preProc_start = time.time()
        tmpData, tmpLabel = (augment(dataArr, labelArr) if augmentation
                             else (dataArr, labelArr))
        if shuffle:
            shuffle_idx = np.arange(tmpData.shape[0])
            random.shuffle(shuffle_idx)
            tmpData, tmpLabel = tmpData[shuffle_idx], tmpLabel[shuffle_idx]
        batch_num = tmpData.shape[0] // batch_size
        total_num = batch_num * batch_size
        tmpData, tmpLabel = tmpData[:total_num], tmpLabel[:total_num]
        file_size = maxBatchInFile * batch_size
        file_num = (tmpData.shape[0] // file_size) + 1
        for file_idx in range(file_num):
            fileData, fileLabel = ((tmpData[(file_idx*file_size):((file_idx+1)*file_size)],
                                   tmpLabel[(file_idx*file_size):((file_idx+1)*file_size)])
                                   if file_idx < file_num - 1 else
                                   (tmpData[(file_idx*file_size):], tmpLabel[(file_idx*file_size):]))
            fileDataTSR, fileLabelTSR = (torch.tensor(fileData).reshape((-1, batch_size, 3, 32, 32)).to('cuda'),
                                         torch.tensor(fileLabel).reshape((-1, batch_size)).type(torch.LongTensor).to('cuda'))
            torch.save((fileDataTSR, fileLabelTSR), f'{temp_folder}/data_{file_idx}')
        preProc_end = time.time()
        # record end time of pre-processing and starts training.
        # The following part of code is a standard pytorch back propagation routine.
        # You can find tutorials on official websites whichever deep-learning framework
        # you choose.
        train_start = time.time()
        for file_idx in range(file_num):
            dataTSR, labelTSR = torch.load(f'{temp_folder}/data_{file_idx}')
            for batch_idx in range(dataTSR.shape[0]):
                prediction = model(dataTSR[batch_idx])
                loss = loss_func(prediction, labelTSR[batch_idx])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())
        train_end = time.time()
        # print necessary training info including time cost and average loss.
        epoch_loss_avg = torch.mean(torch.tensor(epoch_losses)).item()
        epoch_losses.clear()
        fin_losses.append(epoch_loss_avg)
        print(f'Epoch {epoch}. Prepare data {round(preProc_end - preProc_start,2)}s. ' + 
              f'Training {round(train_end-train_start, 2)}s with average loss ' + 
              f'{np.round(epoch_loss_avg, 5)}.')
        # save models and test model performances.
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'{temp_folder}/model_e{epoch}')
            torch.save(fin_losses, LossSavePath)
            err = testGenData(model, 'data/test', '')
            print(f'test error rate {err}.')
    pass

def testGenData(model:Module, dataFile:str, conf_mat_plot:str,
                batch_size:int=128):
    '''
    Function to perform a general testing process on pytorch. Returning a float value
    indicating the test error rate.
    '''
    dataArr, labelArr = torch.load(dataFile)
    batch_num = dataArr.shape[0] // batch_size
    dataArr, labelArr = dataArr[:batch_num*batch_size], labelArr[:batch_num*batch_size]
    dataTSR, labelTSR = (torch.tensor(dataArr).reshape((-1, batch_size, 3, 32, 32)).to('cuda'),
                         torch.tensor(labelArr).reshape((-1, batch_size,)).to('cuda'))
    model.eval()
    pred_lst, wrong_sum, total_sum = list(), 0, 0
    for batch in range(dataTSR.shape[0]):
        model_output = model(dataTSR[batch])
        prediction = torch.argmax(model_output, dim=1)
        pred_lst.append(prediction)
        diff = (prediction != labelTSR[batch]).sum()
        wrong_sum += diff.item()
        total_sum += batch_size
    return wrong_sum / total_sum