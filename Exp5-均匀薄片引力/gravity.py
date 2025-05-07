
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# 物理常数
G = 6.67430e-11  # 万有引力常数 (m^3 kg^-1 s^-2)

def calculate_sigma(length, mass):
    """
    计算薄片的面密度
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片总质量 (kg)
        
    返回:
        面密度 (kg/m²)
    """
    return mass / (length ** 2)

def integrand(x, y, z):
    """
    被积函数，计算引力积分核
    
    参数:
        x, y: 薄片上点的坐标 (m)
        z: 测试点高度 (m)
        
    返回:
        积分核函数值
    """
    return 1 / (x**2 + y**2 + z**2) ** 1.5

def gauss_legendre_integral(length, z, n_points=100):
    """
    使用高斯-勒让德求积法计算二重积分
    
    参数:
        length: 薄片边长 (m)
        z: 测试点高度 (m)
        n_points: 积分点数 (默认100)
        
    返回:
        积分结果值
    """
    # 获取高斯点和权重
    xi, wi = np.polynomial.legendre.leggauss(n_points)
    x = xi * (length / 2)  # 映射到[-L/2, L/2]
    w_x = wi * (length / 2)
    yj, wj = np.polynomial.legendre.leggauss(n_points)
    y = yj * (length / 2)
    w_y = wj * (length / 2)
    
    integral = 0.0
    # 双重循环计算积分
    for i in range(n_points):
        for j in range(n_points):
            integral += w_x[i] * w_y[j] * integrand(x[i], y[j], z)
    return integral

def calculate_force(length, mass, z, method='gauss'):
    """
    计算给定高度处的引力
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z: 测试点高度 (m)
        method: 积分方法 ('gauss'或'scipy')
        
    返回:
        引力值 (N)
    """
    sigma = calculate_sigma(length, mass)
    if method == 'gauss':
        integral = gauss_legendre_integral(length, z)
    else:
        # 使用SciPy的dblquad进行积分
        integral, _ = dblquad(lambda y, x: integrand(x, y, z), 
                             -length/2, length/2, 
                             lambda x: -length/2, lambda x: length/2)
    # 计算引力值（质点质量m=1kg）
    return G * sigma * z * integral

def plot_force_vs_height(length, mass, z_min=0.1, z_max=10, n_points=100):
    """
    绘制引力随高度变化的曲线
    
    参数:
        length: 薄片边长 (m)
        mass: 薄片质量 (kg)
        z_min: 最小高度 (m)
        z_max: 最大高度 (m)
        n_points: 采样点数
    """
    z_values = np.logspace(np.log10(z_min), np.log10(z_max), n_points)
    F_gauss = [calculate_force(length, mass, z) for z in z_values]
    F_scipy = []
    try:
        for z in z_values:
            F_scipy.append(calculate_force(length, mass, z, method='scipy'))
    except ImportError:
        pass
    
    # 理论极限值（z→0时）
    sigma = calculate_sigma(length, mass)
    F_limit = 2 * np.pi * G * sigma * 1  # m=1kg
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, F_gauss, label='Gauss-Legendre (n=100)')
    if F_scipy:
        plt.plot(z_values, F_scipy, '--', label='SciPy dblquad')
    plt.axhline(F_limit, color='r', linestyle=':', label='Theoretical Limit (z→0)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Height z (m)')
    plt.ylabel('Gravitational Force F_z (N)')
    plt.title('Gravitational Force vs. Height')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    length = 10    # 边长 (m)
    mass = 1e4     # 质量 (kg)
    
    plot_force_vs_height(length, mass)
    
    # 打印关键点引力值
    print("关键高度引力值：")
    for z in [0.1, 1, 5, 10]:
        F = calculate_force(length, mass, z)
        print(f"z = {z:.1f}m: F_z = {F:.3e} N")
