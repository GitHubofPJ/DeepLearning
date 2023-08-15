import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

criteria = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检测是否有可用的GPU，否则使用cpu

train_batch_size = 64
test_batch_size = 100
epochs = 50
lr = 2e-3
log_interval = 10

data_transform = transforms.Compose([
    transforms.Resize(80),
    transforms.CenterCrop(80),#这两行代码得到不怎么会拉伸变形又包含大部分图像信息的正方形图片
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # RGB通道的均值与标准差
])

train_dataset = datasets.ImageFolder(root=r'D:\application\pycharm\UCAS_DL\cat_dog\all_data\train',transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataset = datasets.ImageFolder(root=r'D:\application\pycharm\UCAS_DL\cat_dog\all_data\test',transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# 该类继承自nn.Module类，重载了__init__方法和forward方法
class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.func = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),#32 * w * w
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3, 1, 1),# 32 * w * w
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # 32 * (w/2) * (w/2)

            nn.Conv2d(32, 64, 3, 1, 1), # 64 * (w/2) * (w/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1), # 128 * (w/2) * (w/2)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2), # 128 * (w/4) * (w/4)

            nn.Conv2d(128, 256, 3, 1, 1),  # 256 * (w/4) * (w/4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),  # 256 * (w/8) * (w/8)
        )

        self.linear = nn.Sequential(
            nn.Linear(256 * 10 * 10, 512), # 256 * 10 * 10
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.func(x)
        #print(x.shape)
        x = x.view(batchsz, 256 * 10 * 10)
        logits = self.linear(x)

        return logits


my_resnet = MyResNet().to(device)
optimizer = optim.Adam(my_resnet.parameters(), lr=lr) #优化器
#加载网络模型和优化器模型
#my_resnet.load_state_dict(torch.load("my_resnet9138.pkl"))
#optimizer.load_state_dict(torch.load("optimizer9138.pkl"))

train_losses=[]
train_counter=[]
test_accuracy = []
test_losses=[]
test_counter=[i*len(train_loader.dataset) for i in range(epochs+1)]
#训练函数
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) #迁移到gpu上进行训练
        optimizer.zero_grad()
        output = my_resnet(data)
        loss = criteria(output, target)
        # target_ = torch.nn.functional.one_hot(target, num_classes=2).float()
        # loss = nn.BCEWithLogitsLoss()(output, target_)
        #print(f'train_loss: {loss}')
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:    #每隔log_interval * batch_size个数据输出一次
            print(f'Train Epoch: {epoch}', end = '\t')
            print(f'{(batch_idx+1) * len(data)}/{len(train_loader.dataset)}',end='\t') #训练轮次， 本轮次已训练iteration数量/本轮次总训练iteration数量
            print(f'{100. * (batch_idx+1) / len(train_loader):.2f}%',end='\t') #当前训练进度百分比
            print(f'Loss: {loss.item():.6f}')  #当前训练损失

            train_losses.append(loss.item())#记录每log_interval个iteration的损失
            train_counter.append((batch_idx * train_batch_size) + ((epoch - 1) * len(train_loader.dataset)))#记录当前所有轮次中训练的总图片数

    torch.save(my_resnet.state_dict(), "my_resnet.pkl")  # 只保存模型权重参数(不保存模型结构)
    torch.save(optimizer.state_dict(), 'optimizer.pkl')  # 保存优化器参数

class_name=['cat', 'dog']
#测试函数
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) #迁移到gpu上
            # x = x.view(x.size(0), 28 * 28)
            output = model(data)
            test_loss += criteria(output, target) * len(data)#nn.CrossEntropyLoss()返回的是均值
            # target_ = torch.nn.functional.one_hot(target, num_classes=2).float()
            # test_loss = nn.BCEWithLogitsLoss()(output, target_) * len(data)
            pred = output.argmax(dim=1)
            total += len(target)
            correct += pred.eq(target).sum().float().item()
    test_loss /= len(test_loader.dataset)  # 计算平均损失
    test_losses.append(test_loss.item())  # 添加到损失列表
    test_accuracy.append(correct / len(test_loader.dataset))
    print(f'测试集精度:{(correct / total * 100.):.2f}%')

def validate(model, device):
    vali_dataset = datasets.ImageFolder(root=r'D:\application\pycharm\UCAS_DL\cat_dog\validation',transform=data_transform)
    vali_loader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
    model.eval()
    correct,total = 0, 0
    vali_loss = 0
    with torch.no_grad():
        for idx,(data, target) in enumerate(vali_loader):
            data, target = data.to(device), target.to(device) #迁移到gpu上
            output = model(data)
            print(vali_dataset.imgs[idx])  # 查看该图片，及对应的标签
            vali_loss += criteria(output, target)
            pred = output.argmax(dim=1)
            show = transforms.ToPILImage()
            #为了方便显示，进行逆变换
            invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                           transforms.Normalize(mean=[-0.485, -0.456, -0.406],std=[1., 1., 1.]), ])
            img=vali_dataset[idx][0]
            img = invTrans(img)
            img =show(img)
            plt.imshow(img)
            plt.title(f'pred: {class_name[pred]}')
            plt.show()

            for i in pred:
                print(class_name[i])

            total += len(target)
            correct += pred.eq(target).sum().float().item()
            # print('total:',total,'correct:',correct)

test(my_resnet, device, test_loader)
for epoch in range(1,epochs+1):
    train(my_resnet, train_loader, optimizer, epoch)
    test(my_resnet, device, test_loader)

# 绘制训练曲线图
fig, ax1 = plt.subplots()
ax1.plot(train_counter, train_losses, color='blue')
#ax1.scatter(test_counter, test_losses, color='red', zorder=4)
ax1.set_xlabel('cumulative training batch sampling')
ax1.set_ylabel('Loss')
ax2 = ax1.twinx()
ax2.plot(test_counter, test_accuracy, color='green')
ax2.set_ylabel('Test Accuracy')
fig.legend(['Train Loss',  'Test Accuracy'], loc='upper right')
plt.show()

validate(my_resnet, device)
