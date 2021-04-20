import math
import numpy as np
import matplotlib.pyplot as plt
from viz import viz_function # 実軸・虚軸に平行な直線を用いて可視化
from viz import viz_function2 # 原点を中心とする同心円と原点から放射状に伸びる直線を用いて可視化
from viz import viz_function3 # 原点から放射状に伸びる直線のみを用いて可視化


# 二乗
def sq(z):
    w = z*z
    return w

# 共役
def conj(z):
    w = z.conjugate()
    return w


# 適当な一次関数: w=f(z)=3z+2j
def f1(z):
    w = 3*z + 2j # j: 虚数単位
    return w

# 適当な二次関数: w=f(z)=z^2+2z-j
def f2(z):
    w = z*z + 2*z - 1j # pythonでは「j」は「1j」と書かなければならない
    return w

# z の多項式でない関数: z=x+yj に対し w=f(z)=(x+y)+(2x-y)j
def f3(z):
    x = z.real # zの実部
    y = z.imag # zの虚部
    u = x + y # wの実部
    v = 2*x - y # wの虚部
    w = u + v*1j # u+vj
    return w

# 共役を含む関数
def f4(z):
    c = z.conjugate() # 共役複素数
    w = 2*c + 1
    return w


# ぴったり z==0 は入力されないものとする
def f5(z):
    w = np.where(z == 0, 0, (z + z.conjugate()) / abs(z))
    return w


# 関数の可視化
if __name__ == '__main__':
    viz_function(sq)
    #viz_function(conj)
    #viz_function2(sq)
    #viz_function2(conj)
    #viz_function3(f5)
