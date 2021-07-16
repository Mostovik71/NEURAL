import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

'''
def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)
'''

def create_nn(batch_size=100, learning_rate=0.05, epochs=10,
              log_interval=10):#batch_size - размер пачки данных на подачу(например по 100 строк за итерацию)
                               #learning rate - скорость обучения(подбираем, чтобы не перескочить глобальный минимум и ошибку)
                               #epochs - количество эпох (одна эпоха - все итерации), по сути чем больше тем лучше



    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 200)#Синапс между слоями(входной(28*28 узлов, первый скрытый(200 узлов))
            self.fc2 = nn.Linear(200, 200)#Синапс между слоями(первый скрытый(200 узлов), второй скрытый(200 узлов))
            self.fc3 = nn.Linear(200, 10)#Синапс между слоями(второй скрытый(200 узлов), выходной(10 узлов))
        def forward(self, x):
            x = F.relu(self.fc1(x))#Функция активации первого слоя - ReLU(Проходит соединение входного слоя с первым скрытым и активируется через функию ReLU)
            x = F.relu(self.fc2(x))#Функция активации второго слоя - ReLU(Активированное через ReLU число проходит соединение первого скрытого слоя со вторым скрытым и опять проходит через ReLU)
            x = self.fc3(x)#Выход после 3 слоя(с выходных узлов)(Активированное через ReLU число проходит соединение второго скрытого слоя с выходным слоем)
            return F.log_softmax(x)#Функция активации выходного слоя - softmax(Вышедшее с выходного слоя число проходит через функию активации softmax)
    #На выходном слое используется softmax, чтобы гарантировать, что сумма компонент вектора равна 1(softmax=(exp(i)/sum(exp(1 to n))
    #В нашей задаче(10 выходных нейронов, цифры от 0 до 9, 10-мерный вектор на выходе, где наибольшая вероятность - такая и цифра(например на 7 выходе наибольшая вероятность, значит это цифра 7)
    net = Net()#Создается сеть (объект класса net)
    print(net)

    # Оптимизируем путем стохастического градиентного спуска, задача найти глобальный минимум функции, т.е. минимизировать функию потерь
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # Создаем функцию потерь - функция потерь отрицательного логарифмического правдоподобия(линейная регрессия)
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))

    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        pass
        #simple_gradient()
    elif run_opt == 2:
        create_nn()