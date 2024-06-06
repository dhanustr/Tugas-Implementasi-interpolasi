import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x, y, xp):
    yp = 0
    n = len(x)
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (xp - x[j]) / (x[i] - x[j])
        yp += p * y[i]
    return yp

def newton_interpolation(x, y, xp):
    def divided_differences(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
        return coef[0, :] 

    def newton_polynomial(coef, x, xp):
        n = len(coef) - 1
        p = coef[n]
        for k in range(1, n + 1):
            p = coef[n - k] + (xp - x[n - k]) * p
        return p

    coef = divided_differences(x, y)
    return newton_polynomial(coef, x, xp)

# Data pada tabel
data_x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
data_y = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Rentang x untuk plotting
x_plot = np.linspace(5, 40, 500)

# Metode interpolasi Lagrange
y_lagrange = [lagrange_interpolation(data_x, data_y, xp) for xp in x_plot]

# Metode interpolasi Newton
y_newton = [newton_interpolation(data_x, data_y, xp) for xp in x_plot]

# Output plot interpolasi
plt.figure(figsize=(10, 6))
plt.plot(data_x, data_y, 'ro', label='Data Points', markersize=8)
plt.plot(x_plot, y_lagrange, 'b-', label='Lagrange Interpolation', linewidth=2)
plt.plot(x_plot, y_newton, 'y--', label='Newton Interpolation', linewidth=2)
plt.xlabel('Tegangan (kg/mm²)')
plt.ylabel('Waktu Patah (s)')
plt.legend(loc='best')
plt.title('Perbandingan Interpolasi Polinomial Lagrange dan Newton')
plt.grid(True)

# Menambahkan anotasi untuk titik data
for (i, j) in zip(data_x, data_y):
    plt.annotate(f'({i},{j})', xy=(i, j), textcoords='offset points', xytext=(0, 10), ha='center')

plt.show()
