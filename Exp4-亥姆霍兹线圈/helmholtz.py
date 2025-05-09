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
    phi_angles = np.linspace(0, 2 * np.pi, 20)
    max_r = max(r_low, r_up)
    y_coords = np.linspace(-2 * max_r, 2 * max_r, 25)
    z_coords = np.linspace(-1.5 * d, 1.5 * d, 25)

    # 2. 创建三维网格
    Y, Z, Phi = np.meshgrid(y_coords, z_coords, phi_angles)

    # 3. 计算到下方线圈的距离
    dist1_sq = (r_low * np.cos(Phi))**2 + (Y - r_low * np.sin(Phi))**2 + (Z + d/2)**2
    dist1 = np.sqrt(dist1_sq)
    dist1[dist1 < 1e-9] = 1e-9  # 避免除零

    # 4. 计算到上方线圈的距离
    dist2_sq = (r_up * np.cos(Phi))**2 + (Y - r_up * np.sin(Phi))**2 + (Z - d/2)**2
    dist2 = np.sqrt(dist2_sq)
    dist2[dist2 < 1e-9] = 1e-9

    # 5. 计算被积函数
    dBy_integrand = (r_low * (Z + d/2) * np.sin(Phi)) / dist1**3 + (r_up * (Z - d/2) * np.sin(Phi)) / dist2**3
    dBz_integrand = (r_low * (r_low - Y * np.sin(Phi)) / dist1**3 + (r_up * (r_up - Y * np.sin(Phi)))) / dist2**3

    # 6. 对phi积分
    By_unscaled = np.trapezoid(dBy_integrand, x=phi_angles, axis=-1)
    Bz_unscaled = np.trapezoid(dBz_integrand, x=phi_angles, axis=-1)

    # 7. 应用比例因子
    scaling_factor = (MU0 * I) / (4 * np.pi)
    By = scaling_factor * By_unscaled
    Bz = scaling_factor * Bz_unscaled

    print("磁场计算完成.")
    return Y, Z, By, Bz

def plot_magnetic_field_streamplot(r_coil_1, r_coil_2, d_coils):
    """绘制磁场流线图"""
    print(f"开始绘图准备: r_coil_1={r_coil_1}, r_coil_2={r_coil_2}, d_coils={d_coils}")
    Y_full, Z_full, By_field, Bz_field = Helmholtz_coils(r_coil_1, r_coil_2, d_coils)
    Y_plot = Y_full[:, :, 0]
    Z_plot = Z_full[:, :, 0]

    plt.figure(figsize=(8, 7))

    # 设置流线起始点
    max_r = max(r_coil_1, r_coil_2)
    y_start = np.linspace(-0.8*max_r, 0.8*max_r, 15)
    sy, sz = np.meshgrid(y_start, [0])
    start_points = np.vstack([sy.ravel(), sz.ravel()]).T

    # 绘制流线图
    plt.streamplot(Y_plot, Z_plot, By_field, Bz_field,
                   density=1.5, color='k', start_points=start_points,
                   linewidth=1, arrowstyle='->', arrowsize=1)

    # 绘制线圈截面
    plt.plot([-r_coil_1, -r_coil_1], [-d_coils/2-0.02, -d_coils/2+0.02], 'b-', lw=3)
    plt.plot([r_coil_1, r_coil_1], [-d_coils/2-0.02, -d_coils/2+0.02], 'b-', lw=3)
    plt.text(0, -d_coils/2-0.1*max_r, f'Coil 1 (R={r_coil_1})', color='b', ha='center')

    plt.plot([-r_coil_2, -r_coil_2], [d_coils/2-0.02, d_coils/2+0.02], 'r-', lw=3)
    plt.plot([r_coil_2, r_coil_2], [d_coils/2-0.02, d_coils/2+0.02], 'r-', lw=3)
    plt.text(0, d_coils/2+0.1*max_r, f'Coil 2 (R={r_coil_2})', color='r', ha='center')

    # 设置图形属性
    plt.xlabel('y / m')
    plt.ylabel('z / m')
    plt.title(f'Magnetic Field Lines (R1={r_coil_1}, R2={r_coil_2}, d={d_coils})')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    radius_1 = 0.5
    radius_2 = 0.5
    distance_between_coils = 0.5
    plot_magnetic_field_streamplot(radius_1, radius_2, distance_between_coils)
