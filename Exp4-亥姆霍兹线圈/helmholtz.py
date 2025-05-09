import numpy as np
import matplotlib.pyplot as plt

# --- 物理和线圈参数 ---
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
I = 1.0  # 电流 (A)

def Helmholtz_coils(r_low, r_up, d):
    '''
    计算亥姆霍兹线圈的磁场。
    下方线圈位于z=-d/2，上方线圈位于z=+d/2。
    '''
    print(f"开始计算磁场: r_low={r_low}, r_up={r_up}, d={d}")

    # 1. 定义积分角度和空间网格
    phi = np.linspace(0, 2 * np.pi, 20)
    r = max(r_low, r_up)
    y = np.linspace(-2 * r, 2 * r, 25)
    z = np.linspace(-1.5 * d, 1.5 * d, 25)

    # 2. 创建三维网格
    Y, Z, Phi = np.meshgrid(y, z, phi)

    # 3. 计算到下方线圈的距离
    dist1_sq = (r_low * np.cos(Phi))**2 + (Y - r_low * np.sin(Phi))**2 + (Z - d/2)**2
    dist1 = np.sqrt(dist1_sq)
    dist1[dist1 < 1e-9] = 1e-9  # 避免除零

    # 4. 计算到上方线圈的距离
    dist2_sq = (r_up * np.cos(Phi))**2 + (Y - r_up * np.sin(Phi))**2 + (Z + d/2)**2
    dist2 = np.sqrt(dist2_sq)
    dist2[dist2 < 1e-9] = 1e-9

    # 5. 计算被积函数
    dBy_integrand = r_low * (Z - d/2) * np.sin(Phi) / dist1**3 + r_up * (Z + d/2) * np.sin(Phi) / dist2**3
    dBz_integrand = r_low * (r_low - Y * np.sin(Phi)) / dist1**3 + r_up * (r_up - Y * np.sin(Phi)) / dist2**3

    # 6. 对phi积分
    By_unscaled = np.trapezoid(dBy_integrand)
    Bz_unscaled = np.trapezoid(dBz_integrand)

    # 7. 应用比例因子
    scaling_factor = (MU0 * I) / (4 * np.pi)
    By = scaling_factor * By_unscaled
    Bz = scaling_factor * Bz_unscaled

    print("磁场计算完成.")
    return Y, Z, By, Bz

def plot_magnetic_field_streamplot(r_low, r_up, d):
    """绘制磁场流线图"""
    Y, Z, by, bz = Helmholtz_coils(.5,.5,0.8)

    bSY = np.arange(-0.45,0.50,0.05) #磁力线的起点的y坐标
    bSY, bSZ = np.meshgrid(bSY,0) #磁力线的起点坐标
    points = np.vstack([bSY, bSZ])
    h1 = plt.streamplot(Y[:,:,0],Z[:,:,0], by, bz, 
                    density=2,color='k',start_points=points.T)

    plt.xlabel('y / m')
    plt.ylabel('z / m')
    plt.title(f'Magnetic Field Lines of Helmholtz Coils (R={coil_radius}, d={coil_distance})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    coil_radius = 0.5  # 两个线圈的半径 (m)
    coil_distance = 0.8  # 两个线圈之间的距离 (m)
    plot_magnetic_field_streamplot(coil_radius, coil_radius, coil_distance)

