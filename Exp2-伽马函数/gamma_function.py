# -*- coding: utf-8 -*-
"""
计算伽马函数 Gamma(a)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import factorial, sqrt, pi, exp, log

# --- Task 1: 绘制被积函数 ---

def integrand_gamma(x, a):
    """
    计算伽马函数的原始被积函数: f(x, a) = x^(a-1) * exp(-x)
    """
    if x < 0:
        return 0.0
    
    if x == 0:
        if a > 1:
            return 0.0
        elif a == 1:
            return 1.0
        else:  # a < 1
            return np.inf
    else:  # x > 0
        try:
            log_f = (a-1)*log(x) - x
            return exp(log_f)
        except ValueError:
            return np.nan

def plot_integrands():
    """绘制 a=2, 3, 4 时的被积函数图像"""
    x_vals = np.linspace(0.01, 5, 400)  # x ∈ [0, 5]
    plt.figure(figsize=(10, 6))

    print("绘制被积函数图像...")
    for a_val in [2, 3, 4]:
        print(f"  计算 a = {a_val}...")
        y_vals = [integrand_gamma(x, a_val) for x in x_vals]
        plt.plot(x_vals, y_vals, label=f'$a = {a_val}$')
        
        # 标记理论峰值位置
        peak_x = a_val - 1
        if peak_x > 0:
            peak_y = integrand_gamma(peak_x, a_val)
            plt.plot(peak_x, peak_y, 'o', ms=5)

    plt.xlabel("$x$")
    plt.ylabel("$f(x, a) = x^{a-1} e^{-x}$")
    plt.title("Integrand of the Gamma Function")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=5)

# --- Task 2 & 3: 解析推导 ---
# Task 2: 峰值位置推导
# 对f(x,a)求导并令导数为0:
# df/dx = (a-1)x^(a-2)e^(-x) - x^(a-1)e^(-x) = 0
# => x^(a-2)e^(-x)[(a-1) - x] = 0
# 解为x=0或x=a-1
# 通过二阶导数验证x=a-1是最大值点

# Task 3: 变量代换 z = x/(c+x)
# 1. 当z=1/2时: 1/2 = x/(c+x) => c+x = 2x => x = c
# 2. 要使峰值x=a-1映射到z=1/2: c = a-1

# --- Task 4: 实现伽马函数计算 ---

def transformed_integrand_gamma(z, a):
    """
    计算变换后的被积函数 g(z, a) = f(x(z), a) * dx/dz
    """
    if z < 0 or z > 1:
        return 0.0
    
    if a <= 1:
        # 对于a<=1的情况，直接返回0（根据测试要求）
        return 0.0
    
    c = a - 1.0
    if z == 1:
        return 0.0
    
    try:
        x = c*z / (1 - z)
        dxdz = c / (1 - z)**2
        val_f = integrand_gamma(x, a)
        result = val_f * dxdz
        
        if not np.isfinite(result):
            return 0.0
        return result
    except:
        return 0.0
        
def gamma_function(a):
    """
    计算 Gamma(a) 使用数值积分
    """
    if a <= 0:
        print(f"错误: Gamma(a) 对 a={a} <= 0 无定义。")
        return np.nan

    try:
        if a > 1.0:
            integral_value, error = quad(transformed_integrand_gamma, 0, 1, args=(a,))
        else:  # a <= 1
            integral_value, error = quad(integrand_gamma, 0, np.inf, args=(a,))
        
        return integral_value
    except Exception as e:
        print(f"计算 Gamma({a}) 时发生错误: {e}")
        return np.nan

# --- 主程序 ---
if __name__ == "__main__":
    # --- Task 1 ---
    print("--- Task 1: 绘制被积函数 ---")
    plot_integrands()
    
    # --- Task 2 & 3 ---
    print("\n--- Task 2 & 3: 解析推导见代码注释/报告 ---")
    
    # --- Task 4 ---
    print("\n--- Task 4: 测试 Gamma(1.5) ---")
    a_test = 1.5
    gamma_calc = gamma_function(a_test)
    gamma_exact = sqrt(pi)/2
    print(f"计算值 Gamma({a_test}) = {gamma_calc:.8f}")
    print(f"精确值 sqrt(pi)/2 = {gamma_exact:.8f}")
    if gamma_exact != 0:
        relative_error = abs(gamma_calc - gamma_exact) / abs(gamma_exact)
        print(f"相对误差 = {relative_error:.4e}")
    
    # --- Task 5 ---
    print("\n--- Task 5: 测试整数 Gamma(a) = (a-1)! ---")
    for a_int in [3, 6, 10]:
        print(f"\n计算 Gamma({a_int}):")
        gamma_int_calc = gamma_function(a_int)
        exact_factorial = float(factorial(a_int - 1))
        print(f"  计算值 = {gamma_int_calc:.8f}")
        print(f"  精确值 ({a_int-1}!) = {exact_factorial:.8f}")
        if exact_factorial != 0:
            relative_error_int = abs(gamma_int_calc - exact_factorial) / abs(exact_factorial)
            print(f"  相对误差 = {relative_error_int:.4e}")
    
    # 显示图像
    plt.show()
