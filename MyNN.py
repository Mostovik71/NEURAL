import math
def sigm(x):
    return 1/(1+math.exp(-x))
def f(x):
    (1-x)*x
outi=1
E=float(input("E: "))
alpha=float(input("alpha: "))
#float(input("l1: "))
l1=1
l2=0
w1=0.45
w2=0.78
w3=-0.12
w4=0.13
w5=1.5
w6=-2.3
dw10=0
dw20=0
dw30=0
dw40=0
dw50=0
dw60=0
err=1
while err>0.01:
    h1 = l1 * w1 + l2 * w3
    h2 = l1 * w2 + l2 * w4

    o1 = sigm(h1) * w5 + sigm(h2) * w6
    o = sigm(o1)  # Выход output нейрона
    err = (outi - o) #** 2
    delto = (outi - o) * (1 - o) * o  # дельта для выходного нейрона
    deltah1 = (1 - sigm(h1)) * sigm(h1) * (w5) * delto
    deltah2 = (1 - sigm(h2)) * sigm(h2) * (w6) * delto

    gradh1o = delto * sigm(h1)
    gradh2o = delto * sigm(h2)
    gradl1h1 = deltah1 * l1
    gradl1h2 = deltah1 * l2
    gradl2h1 = deltah2 * l1
    gradl2h2 = deltah2 * l2
    dw5=E*gradh1o+alpha*dw50
    dw6=E*gradh1o+alpha*dw60
    dw3=E*gradl2h1+alpha*dw30
    dw4 = E * gradl2h2 + alpha * dw40
    dw1= E * gradl1h1 + alpha * dw10
    dw2=E * gradl1h2 + alpha * dw20
    w1+=dw1
    w2+=dw2
    w3+=dw3
    w4+=dw4
    w5+=dw5

    w6+=dw6



print(o)
weights=[w1,w2,w3,w4,w5,w6]
print(weights)