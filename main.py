import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import math
import random
from matplotlib import pyplot as plt

#定义正态分布函数，生成数据点
sample_point=1000
def f(x):
    return math.exp(-x**2/2)/math.sqrt(2*math.pi)
x=np.zeros(sample_point)
y=np.zeros(sample_point)
for i in range(sample_point):
    x[i]=random.uniform(-2.0,2.0)
    y[i]=f(x[i])
#转为Tensor，否则报错
x=paddle.to_tensor(x,dtype='float32')
y=paddle.to_tensor(y,dtype='float32')

#建立多层感知机
class Net(paddle.nn.Layer):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=Linear(n_feature,n_hidden)
        self.predict=Linear(n_hidden,n_output)
    def forward(self,x):
        x=self.hidden(x)
        x=F.relu(x)
        x=self.predict(x)
        return x
net=Net(2,32,2)#网络参数

#训练模型
epochs=200
batch_cnt=400
x_traingraph=[]
y_traingraph=[]#存放最后一轮epoch训练结果的数组
optimizer=paddle.optimizer.SGD(learning_rate=0.1,parameters=net.parameters())
for i in range(epochs):
    for j in range(batch_cnt):
        x_train=x[j*2:j*2+2]#与网络参数适配
        y_train=y[j*2:j*2+2]
        prediction=net(x_train)
        if i==(epochs-1):
            x_traingraph.append(x_train)
            y_traingraph.append(prediction)
        loss=F.square_error_cost(prediction,y_train)#两个多维Tensor分项平方之差
        loss_avg=paddle.mean(loss)
        if i%10==0:
            print("epoch:{},loss:{}".format(i,loss_avg.numpy()))
        loss_avg.backward()
        optimizer.step()
        optimizer.clear_grad()
paddle.save(net.state_dict(),'MLP.pdparams')#保存模型参数
plt.clf()
#画最后一轮训练结果图
x_traingraph=np.array(x_traingraph)
y_traingraph=np.array(y_traingraph)
plt.figure(1)
plt.plot(x_traingraph,y_traingraph,'r.')
#原始图像
x_gauss=x[0:800]
y_gauss=y[0:800]
x_gauss=np.array(x_gauss)
y_gauss=np.array(y_gauss)
plt.plot(x_gauss,y_gauss,'g.')
plt.show()

#测试模型
x_testgraph=[]
y_testgraph=[]
params_file_path='MLP.pdparams'
param_dict=paddle.load(params_file_path)#加载模型参数
net.load_dict(param_dict)
net.eval()#灌入数据
for j in range(100):
    x_test=x[800+j*2:800+j*2+2]
    y_test=y[800+j*2:800+j*2+2]
    test=net(x_test)
    x_testgraph.append(x_test)
    y_testgraph.append(test)
loss=F.square_error_cost(test,y_test)
loss_avg=paddle.mean(loss)
print("loss:{}".format(loss_avg.numpy()))
#画最后一轮训练结果图
x_testgraph=np.array(x_testgraph)
y_testgraph=np.array(y_testgraph)
plt.figure(2)
plt.plot(x_testgraph,y_testgraph,'r.')
#原始图像
x_gauss=x[800:1000]
y_gauss=y[800:1000]
x_gauss=np.array(x_gauss)
y_gauss=np.array(y_gauss)
plt.plot(x_gauss,y_gauss,'g.')
plt.show()

#绘制拟合后函数
#原函数
plt.figure(3)
x_gauss=x_gauss[np.argsort(x_gauss)]
for i in range(np.size(x_gauss)):
    y_gauss[i]=f(x_gauss[i])
plt.plot(x_gauss,y_gauss,ls='-',color='green')
#拟合后函数
arrIndex=np.argsort(x_testgraph[:,0],axis=0)
x_testgraph=x_testgraph[:,0][arrIndex]
y_testgraph=y_testgraph[:,0][arrIndex]
plt.plot(x_testgraph,y_testgraph,ls='-',color='red')
plt.legend(["Primitive Function","Fitted Function"])
plt.show()