import tensorflow.keras as kr
import torch
from torch import optim
from torch import nn
from cnews_loader import read_category, read_vocab,process_file
from model import TextRNN
import numpy as np
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#设置数据目录
vocab_file = 'cnews.vocab.txt'
train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
val_file = 'cnews.val.txt'
# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
#print(categories)
# 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab('cnews.vocab.txt')
#print(words)
#print(word_to_id)
#print(word_to_id)
#获取字数
vocab_size = len(words)

# 数据加载及分批
# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file('cnews.train.txt', word_to_id, cat_to_id, 600)
#print('x_train=', x_train)
x_val, y_val = process_file('cnews.val.txt', word_to_id, cat_to_id, 600)


#设置GPU
cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train),torch.Tensor(y_train)
x_val, y_val = torch.LongTensor(x_val),torch.Tensor(y_val)

train_dataset = Data.TensorDataset(x_train,y_train)
train_loader = Data.DataLoader(dataset = train_dataset,batch_size=128, shuffle=True)
val_dataset= Data.TensorDataset(x_val, y_val)
val_loader = Data.DataLoader(dataset=val_dataset,batch_size=128)

def train():
    model = TextRNN().to(device)
    #定义损失函数
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
#保存最好模型，先给一个定义为0
    best_val_acc = 0
    for epoch in range(10):
       # print('epoch=',epoch)
        #分批训练
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch.to(device)
            y = y_batch.to(device)
            out = model(x)
            loss = Loss(out, y)
            #print(out)
            #print('loss=',loss)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = np.mean((torch.argmax(out, 1)==torch.argmax(y,1)).cpu().numpy())

            print('accuracy:',accuracy)
#对模型进行验证
        if  (epoch+1)% 5 == 0:
            for step, (x_batch,y_batch) in enumerate(val_loader):
                x = x_batch.to(device)
                y = y_batch.to(device)
                out = model(x)
                #计算准确率
                accuracy = np.mean((torch.argmax(out, 1)==torch.argmax(y,1)).cpu().numpy())
                if accuracy > best_val_acc:
                    torch.save(model,'model.pkl')
                    best_val_acc = accuracy
                    print('model.pkl saved')
                    print('accuracy:',accuracy)
                    #print('best_accuracy:',best_val_acc)

if __name__ == '__main__':
    train()

"""ccuracy: 0.34375
accuracy: 0.375
accuracy: 0.265625
accuracy: 0.4375
accuracy: 0.3359375
accuracy: 0.3984375
accuracy: 0.4140625
accuracy: 0.3359375
accuracy: 0.3359375
accuracy: 0.390625
accuracy: 0.3828125
accuracy: 0.34375
accuracy: 0.3671875
accuracy: 0.328125
accuracy: 0.265625
accuracy: 0.4453125
accuracy: 0.4140625
accuracy: 0.3125
accuracy: 0.296875
accuracy: 0.3125
accuracy: 0.3515625
accuracy: 0.328125
accuracy: 0.4296875
accuracy: 0.3828125
accuracy: 0.328125
accuracy: 0.34375
accuracy: 0.390625
accuracy: 0.4453125
accuracy: 0.3203125
accuracy: 0.390625
accuracy: 0.34375
accuracy: 0.3359375
accuracy: 0.328125
accuracy: 0.3046875
accuracy: 0.3828125
accuracy: 0.3671875
accuracy: 0.3203125
accuracy: 0.375
accuracy: 0.3515625
accuracy: 0.3046875
accuracy: 0.359375
accuracy: 0.34375"""