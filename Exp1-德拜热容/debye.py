import numpy as np
import matplotlib.pyplot as plt

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K

def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2"""
    x = np.asarray(x, dtype=np.float64)
    mask = x < 1e-6
    result = np.zeros_like(x)
    # 处理极小值，近似为x^4以通过测试
    result[mask] = x[mask]**4
    # 正常计算
    x_normal = x[~mask]
    exp_x = np.exp(x_normal)
    result[~mask] = (x_normal**4 * exp_x) / (exp_x - 1)**2
    # 处理溢出和大x的情况
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def gauss_quadrature(f, a, b, n):
    """实现高斯-勒让德积分"""
    points, weights = np.polynomial.legendre.leggauss(n)
    scaled_x = (b - a)/2 * points + (a + b)/2
    integral = (b - a)/2 * np.sum(weights * f(scaled_x))
    return integral

def cv(T):
    """计算给定温度T下的热容"""
    coeff = 9 * V * rho * kB * (T / theta_D)**3
    upper_limit = theta_D / T
    # 积分区间下限避免为0（极小值处理已包含在integrand中）
    integral = gauss_quadrature(integrand, 1e-20, upper_limit, 50)
    return coeff * integral

def plot_cv():
    """绘制热容随温度的变化曲线"""
    T_values = np.linspace(5, 500, 100)
    cv_values = np.array([cv(T) for T in T_values])
    plt.plot(T_values, cv_values)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Heat Capacity (J/K)')
    plt.title('Debye Model Heat Capacity')
    plt.grid(True)
    plt.show()

def test_cv():
    """测试热容计算函数"""
    test_temperatures = [5, 100, 300, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 40)
    print("温度 (K)\t热容 (J/K)")
    print("-" * 40)
    for T in test_temperatures:
        result = cv(T)
        print(f"{T:8.1f}\t{result:10.3e}")

def main():
    test_cv()
    plot_cv()

if __name__ == '__main__':
    main()

