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

def create_nn(batch_size=100, learning_rate=1, epochs=1,
              log_interval=10):#batch_size - размер пачки данных на подачу(например по 100 строк за итерацию)
                               #learning rate - скорость обучения(подбираем, чтобы не перескочить глобальный минимум и ошибку)
                               #epochs - количество эпох (одна эпоха - все итерации), по сути чем больше тем лучше



    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([#transforms.Compose - позволяет делать преобразования в датасете
                           transforms.ToTensor(),#.ToTensor - преобразует датасет в PyTorch тензор
                           transforms.Normalize((0.1307,), (0.3081,))#Нормализует данные(в диапазон от 0 до 1), 0.1307 и 0.3081 - среднее и стандартное отклонение датасета, значения нужно установить для каждого канала
                       ])),
        batch_size=batch_size, shuffle=True)#shuffle - перемешивать ли датасет
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

    # Оптимизируем путем стохастического(вес сразу обновляется когда найден delta(w)) градиентного спуска, задача найти глобальный минимум функции, т.е. минимизировать функию потерь
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # Создаем функцию потерь - функция потерь отрицательного логарифмического правдоподобия(линейная регрессия)
    criterion = nn.NLLLoss()



    #Данные скачиваются батчами, и поступают в цикл
    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):#Для номера батча, данных и таргетного признака
            data, target = Variable(data), Variable(target)#Делаем данные переменным PyTorch
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)#Грубо говоря, данные теперь будут размера (batch_size,784) - данные для тренировки, 1 - канал изображения
            optimizer.zero_grad()#Все происходит по методу обратного распространения(начальное значение параметров - 0)
            net_out = net(data)#На выходе имеем логарифмический softmax для партии данных, который возвращает метод forward
            loss = criterion(net_out, target)#Считается потеря, разница между таргетом и полученным значением
            loss.backward()#Метод обратного распространения ошибки(потери), чтобы обновить веса, тк все стохастически, то веса обновляются сразу
            optimizer.step()#Градиентный спуск по весам, которые были обновлены
            if batch_idx % log_interval == 0:#Выводим результат, когда проходит определенное количество итераций
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(#Вывод в формате:
                    epoch, batch_idx * len(data), len(train_loader.dataset), #номер эпохи, номер батча * размер батча,длина датасета
                           100. * batch_idx / len(train_loader), loss.data))#сколько обучилось в процентах, значение функции потерь на каждом шаге

    # Запускаем цикл на тестовом датасете, чтобы проверить обученную сеть
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)#Передаем в сеть тестовые данные

        test_loss += criterion(net_out, target).data#Суммируем потери со всех партий
        pred = net_out.data.max(1)[1]  # Получаем индекс максимального значения

        correct += pred.eq(target.data).sum()#Сравнивает значения в двух тензорах и при совпадении возвращает единицу. В противном случае, функция возвращает 0

    test_loss /= len(test_loader.dataset)#Средняя потеря
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(#Вывод в формате:
        test_loss, correct, len(test_loader.dataset),#Средняя потеря, количество правильных ответов/весь датасет,
        100. * correct / len(test_loader.dataset)))#количество правильных ответов в процентах



create_nn()