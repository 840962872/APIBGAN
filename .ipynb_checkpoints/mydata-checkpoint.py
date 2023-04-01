import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn import preprocessing
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
writer = SummaryWriter()
# 读取文件
data = pd.read_csv('yanfa_test.csv', encoding='GB2312')
# 分成数据部分和结果部分
'''train_data = data[[ 'IP', 'url', 'port', 'vlan', 'switchIP', 'day', 'hour', 'minute',
                   'account_IP', 'account_url', 'account_switchIP', 'account_url_IP',
                   'url_IP_switchIP', 'account_IP__count', 'account_url__count', 'account_switchIP__count']]'''
train_data = data[['day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret']]

test_test =  pd.read_csv('yanfa_final.csv', encoding='GB2312')
test_test = test_test[['day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret']]

# 加入数据预处理后的归一化  使用MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
#print(train_data)
train_data = DataFrame(train_data, columns=[ 'day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret'])

test_test = min_max_scaler.transform(test_test)
test_test = DataFrame(test_test, columns=[ 'day','hour','minute',
                'account_switchIP', 
                    'account_IP__count', 'account_url__count','account_switchIP__count',
                        'account_url_IP__count','url_IP_switchIP__count','ret'])
test_test.to_csv("yanfa_test_norm.csv",encoding="GB2312")

#norm_data.to_csv("renshi_final_norm.csv",encoding='UTF-8')




#print(train_data)
train_result = data[['ret']]
# Reshape
#train_data = torch.Tensor(np.array(train_data)).reshape(528690,4,4)
#train_data = torch.Tensor(np.array(train_data)).reshape(4381,4,4)
#train_data = torch.Tensor(np.array(train_data)).reshape(1582,3,4)
train_data = torch.Tensor(np.array(train_data)).reshape(600,2,5)
#train_result = torch.Tensor(np.array(train_result)).reshape(528690)
#train_result = torch.Tensor(np.array(train_result)).reshape(4381)
train_result = torch.Tensor(np.array(train_result)).reshape(600)
#print(train_data.size())




class MyDataset(Dataset):

    def __init__(self):
        self.x_data = train_data
        self.y_data = train_result
        #self.len = train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
dealDataset = MyDataset()
dataloader = DataLoader(dataset=dealDataset,
                          batch_size=32,
                          shuffle=True)
'''for i in train_loader:
    #print(i)
    img, label = i
    print(img)
    print(label)'''


num_epoch = 2000
z_dimension = 4


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(10, 256),                     # 输入特征数为16，输出为10
            nn.LeakyReLU(0.2),                      # 进行非线性映射
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x


# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(4, 256),                      # 用线性变换将噪声映射到10个特征
            nn.ReLU(True),                          # relu激活
            nn.Linear(256, 256),                    # 线性变换
            nn.ReLU(True),                          # relu激活
            nn.Linear(256, 10),                     # 线性变换
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.gen(x)
        return x






# 创建对象
D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion_d = nn.BCELoss()  # 是单目标二分类交叉熵函数
criterion_g = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.003)



# batch从128改为了32，因为有点欠拟合
# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    total_g_Loss = 0
    total_d_Loss = 0
    for i, (img, _) in enumerate(dataloader):
        #print(i)
        num_img = img.size(0)  # img。size(128*4*4)  num_img=128
        #print(img.size(),end="一\n")
        # view()函数作用是将一个多行的Tensor,拼接成一行
        # 第一个参数是要拼接的tensor,第二个参数是-1
        # =============================训练判别器==================
        img = img.view(num_img, -1)  # 展平  为128*16
        #print(img.shape,end="二\n")
        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
        real_label = Variable(torch.ones(num_img,1)).cuda()  # 定义真实的图片label为1
        #print(real_label)
        #real_label = real_label.reshape((128,1))
        #print(real_label)
        fake_label = Variable(torch.zeros(num_img,1)).cuda()  # 定义假的图片的label为0

        # ########判别器训练train#####################
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # 计算真实图片的损失

        #print("这是输入判别器的数据：{}".format(real_img))
        real_out = D(real_img)  # 将真实图片放入判别器中
        #print("这是判别器输出的数据：{}".format(real_out))


        #print(len(real_out))
        #real_out =real_out.reshape((128))
        #print(real_out.size())
        #print(real_label)
        d_loss_real = criterion_d(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, z_dimension)).cuda()  # 随机生成一些噪声
        fake_img = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离


        fake_out = D(fake_img)  # 判别器判断假的图片，
        d_loss_fake = criterion_d(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        # ###############################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的图片的损失
        for mutiG in range(3):
            z1 = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
            fake_img1 = G(z1)  # 随机噪声输入到生成器中，得到一副假的图片
            z2 = Variable(torch.randn(num_img, z_dimension)).cuda()  # 得到随机噪声
            fake_img2 = G(z2)  # 随机噪声输入到生成器中，得到一副假的图片
            # print("GGGGG生成的数据！！！！；{}".format(fake_img))

            # ms loss
            ms_value = torch.mean(torch.abs(fake_img1 - fake_img2)) / torch.mean(torch.abs(z1 - z2))
            ms_loss = 1 / (ms_value + 0.00001)

            output1 = D(fake_img1)  # 经过判别器得到的结果
            output2 = D(fake_img2)
            g_loss1 = criterion_g(output1, real_label)  # 得到的假的图片与真实的图片的label的loss
            g_loss2 = criterion_g(output2, real_label)
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss = g_loss1 + g_loss2 + ms_loss
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        total_g_Loss += g_loss
        total_d_Loss += d_loss

        # tensorboard
    writer.add_scalars('run_14h', {'D_loss': total_d_Loss,
                               'G_loss': total_g_Loss}, epoch+1)
    #if (epoch + 1) % 100 == 0:
    print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
          'D real: {:.6f},D fake: {:.6f}'.format(
        epoch+1, num_epoch, d_loss.data.item(), g_loss.data.item(),
        real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
    ))
    print("这一轮的Generator的Loss为：{}".format(total_g_Loss))
    print("这一轮的Discriminator的Loss为：{}".format(total_d_Loss))
    torch.save(G, 'model//generator_{}th.pth'.format(epoch+1))
writer.close()

# 保存模型
#torch.save(G.state_dict(), './generator.pth')
#torch.save(D.state_dict(), './discriminator.pth')