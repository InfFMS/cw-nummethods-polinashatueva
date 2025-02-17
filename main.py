# Построить графики уравнения Ван-дер-Ваальса при температурах -140, -130, -120, -110, -
# 100 градусов Цельсия. Выделить красным цветом кривую, начиная с которой начинаются
# изменения в поведении функции (см. пояснение).
# Пояснение: Критическая температура и её влияние на график давления
# Критическая температура — это температура, выше которой исчезает различие между
# жидкостью и газом. При T=Tcrit изотерма Ван-дер-Ваальса имеет особенность:
#  График имеет единственную точку перегиба.
#  Исчезает запрещенная зона — область с локальными экстремумами.
#  Фазовый переход первого рода (конденсация) перестаёт быть возможным.
# Физически это означает, что при критической температуре жидкость и газ становятся
# неразличимыми по плотности и физическим свойствам, образуя сверхкритическое
# состояние.
# Напоминание: перевод из градусов Цельсия в Кельвины: T(K) = T(˚C) + 273.15
# Диапазон изменения объема: [b+10-5
# , 10-3
# ] (м3
# / моль). Настоятельно рекомендуется
# отрисовывать значения функции как минимум по 1000 точек!
def s(a,b,f,eps):
    while b-a >= 2*eps:
        c = (a+b)/2
        if f(a)*f(c)<=0:
            b = c
        else:
            a = c
    return c

import matplotlib.pyplot as plt
import numpy as np
a = 0.1382
b = 3.19*10**(-5)
R = 8.314
V = np.linspace( 3.19*10**(-5)+10**(-5), 10**(-3), 1000)
P1  = (R*(-140+273.15)/(V-b)) - a/(V**2)
P2  = (R*(-130+273.15)/(V-b)) - a/(V**2)
P3  = (R*(-120+273.15)/(V-b)) - a/(V**2)
P4  = (R*(-110+273.15)/(V-b)) - a/(V**2)
P5  = (R*(-100+273.15)/(V-b)) - a/(V**2)

def f1(V):
    P = (R*(-130+273.15)/(V-b)) - a/(V**2)
    return P

# tc = [-140, -130, -120, -110, -100]
# tk= [temp + 273.15 for temp in tc]
# critical_temperature_reached = False
# for T_celsius, T_kelvin in zip(tc, tk):
#     P = f1(V, T_kelvin)
# # Создание подграфиков
# if tc >= -120 and not critical_temperature_reached:  # Определить нужное значение температуры.
#     color = 'red'
#     critical_temperature_reached = True
# else:
#     color = None  # Matplotlib выбирает цвет автоматически

fig, axs = plt.subplots()
plt.plot(V, P1, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()
q = 0.01



fig1, axs1 = plt.subplots()
plt.plot(V, P2, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()

fig2, axs2 = plt.subplots()
plt.plot(V, P3, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()

fig3, axs3 = plt.subplots()
plt.plot(V, P4, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()

fig4, axs4 = plt.subplots()
plt.plot(V, P5, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()






#
# Рассмотрим изотерму при температуре, равной -130 ˚C. Все дальнейшие задание
# проводятся именно с этой изотермой!
#  Найти локальный максимум и минимум функции P(V,T).
#  Эти точки определяют границы запрещенной зоны, где изотерма имеет
# нестабильный участок.
# Напоминание: Запрещенная зона — это область, где производная давления по объему
# положительна, то есть это означает, что при расширении вещества растет и его давление,
# что невозможно для реального вещества.

def find_local_extrema(func, V, T):
    P = func(V, T)
    dP = np.diff(P)
    local_maxima_indices = np.where((dP[:-1] > 0) & (dP[1:] < 0))[0] + 1
    local_minima_indices = np.where((dP[:-1] < 0) & (dP[1:] > 0))[0] + 1
    local_maxima = V[local_maxima_indices], P[local_maxima_indices]
    local_minima = V[local_minima_indices], P[local_minima_indices]
    return local_maxima, local_minima


# tc = [-140, -130, -120, -110, -100]
# tk= [temp + 273.15 for temp in tc]
# critical_temperature_reached = False
# for T_celsius, T_kelvin in zip(tc, tk):
#     P = f1(V, T_kelvin)
# maxima, minima = find_local_extrema(vanderwal, V, float(temp_klvn))
# ax.scatter(maxima[0], maxima[1], color='red', label='Локальные максимумы', zorder = 5)
# ax.scatter(minima[0], minima[1], color='blue', label='Локальные минимумы', zorder=5)
#
# for i in range(0,len(V)-1):
#     if P[i]>P[i+1] and P[i]>P[i-1]:
#         max.append([i])
# for i in range(0,len(V)-1):
#     if P[i]<P[i+1] and P[i]<P[i-1]:
#         min.append([i])
# print(max,min)

import matplotlib.pyplot as plt
import numpy as np

V = np.linspace( 3.19*10**(-5)+10**(-5), 10**(-3), 1000)

P2  = (R*(-130+273.15)/(V-b)) - a/(V**2)

plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()
fig5, axs5 = plt.subplots()
plt.plot(P2, V, label="график P,V", color="blue")
plt.grid(True)
# Настройка расстояний
plt.tight_layout()
plt.show()

def f(P,V):
    V = np.linspace(3.19 * 10 ** (-5) + 10 ** (-5), 10 ** (-3), 1000)
    P = (R * (-130 + 273.15) / (V - b)) - a / (V ** 2)
    return(P)
def extrems(V, P):
    """
    Находит локальные максимумы и минимумы.

    Args:
        V: Массив молярных объемов.
        P: Массив давлений.

    Returns:
        tuple: (массив индексов максимумов, массив индексов минимумов)
    """
    maxima_indices = []
    minima_indices = []
    for i in range(1, len(P) - 1):
        if P[i] > P[i-1] and P[i] > P[i+1]:
            maxima_indices.append(i)
        elif P[i] < P[i-1] and P[i] < P[i+1]:
            minima_indices.append(i)
    return maxima_indices, minima_indices








# 4
# Найти длину кривой в запрещенной зоне. Настоятельно рекомендуется разбивать
# интервал подсчета длины кривой как минимум на 1000 точек!
# Замечание: Длина кривой не имеет физического смысла, но полезна для проверки
# навыков интегрирования.
#
a1 = ((1.1*(10**(-4)))-(6.39*(10**(-5))))/1000
def f1(V):
    P = (R*(-130+273.15)/(V-b)) - a/(V**2)
    return P
m = 0
for i in range(999):
    V = a1*i + (6.39*(10**(-5)))
    b1 = (((f1(V)-f1(V+a1))**2)+a1**2)**(0.5)
    m = m+b1
print(m)





# Рассчитать корни уравнения:
# ( , ) 0 P V T P 
# нас
# с точностью до 10-7
# .
# Здесь
# 3664186.998 Pнас
# 
# Па — давление насыщенного пара кислорода при -130°C.
# Ожидаемые три корня:
#  Самый левый (жидкость) назовем его Vl,
#  Центральный (нефизическая область),
#  Самый правый (газ) назовем его V

Pnas = 3664186.998
eps = 10**(-7)
def s(a,b,f7,eps):
    while b-a >= 2*eps:
        c = (a+b)/2
        if f7(a)*f7(c)<=0:
            b = c
        else:
            a = c
    return c
def f1(V):
    P = (R*(-130+273.15)/(V-b)) - a/(V**2)
    return P

def f7(V):
    P = (R * (-130 + 273.15) / (V - b)) - a / (V ** 2) - 3664186.998
    return P
# def f3()
d1 = (6.39*(10**(-5)))
d2 = (1.1*(10**(-4)))
print((6.39*(10**(-5))),(1.1*(10**(-4))),f7,eps)








import matplotlib.pyplot as plt

import numpy as np



x1 = np.linspace(0, np.pi, 100)

y1 = np.sin(2*x1) + 1

y2 = -0.2*x1**2 +0.5

fig, axs = plt.subplots()

plt.plot(x1, y1, label="y1 = sin(2x) +1", color="blue")

plt.plot(x1, y2, label="y2 = -0.2*x**2 +0.5", color="green")

plt.grid(True)

plt.show()

a = 3.14/99

def f(P):
    y = np.sin(2*x) + 1
     return y

def f1(P):
    y = -0.2*x**2 +0.5
    return y

s1 = 0
w=0
for g in range(999):
    s1 = a * (f(g*a)+f((g+1)*a ))/2
    w = w +s1
print(w)
s2 = 0
q =0
for p in range(999):
    s2 = a * (f1(p*a) + f1((p+1)* a)) / 2
    q= q+s2
print(q)

