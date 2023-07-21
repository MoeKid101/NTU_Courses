from cifar_utils import *
from models import *

if __name__ == '__main__':
    
    ''' Generate data file for CIFAR '''
    trainFileList = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    testFileList = ['test_batch',]
    trainPath, testPath = 'data/train', 'data/test'
    genData(CIFAR_ORIG_PATH, trainFileList, trainPath)
    genData(CIFAR_ORIG_PATH, testFileList, testPath)
    ''' Train CNN model for CIFAR '''
    model = VGGModel().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train(model, trainPath, nn.CrossEntropyLoss(), optimizer, 40, 'T3')
    pass