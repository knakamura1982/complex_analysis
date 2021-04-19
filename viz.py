import math
import numpy as np
import matplotlib.pyplot as plt


# 実軸に平行な線（プログラム上は点列）を作る
#   - y: 虚部の値
#   - x_min: 実部の最小値
#   - x_max: 実部の最大値
#   - n: 点の個数
def make_horizontal_line(y, x_min=-2, x_max=2, n=200):
    pts_x = np.arange(x_min, x_max, (x_max-x_min)/n, dtype=np.float32)
    pts_y = y * np.ones(len(pts_x), dtype=np.float32)
    pts = pts_x + pts_y*1j
    return pts

# 虚軸に平行な線（プログラム上は点列）を作る
#   - x: 実部の値
#   - y_min: 虚部の最小値
#   - y_max: 虚部の最大値
#   - n: 点の個数
def make_vertical_line(x, y_min=-2, y_max=2, n=200):
    pts_y = np.arange(y_min, y_max, (y_max-y_min)/n, dtype=np.float32)
    pts_x = x * np.ones(len(pts_y), dtype=np.float32)
    pts = pts_x + pts_y*1j
    return pts

# 原点から放射状に伸びる線（プログラム上は点列）を作る
#   - theta: 角度実部の値
#   - r_max: 絶対値の最大値
#   - n: 点の個数
def make_radial_line(theta, r_max=2, n=100):
    x_max = r_max * math.cos(theta)
    y_max = r_max * math.sin(theta)
    x_step = x_max / n
    y_step = y_max / n
    if x_step != 0:
        pts_x = np.arange(x_step, x_max + x_step, x_step, dtype=np.float32)
    else:
        pts_x = np.zeros(n, dtype=np.float32)
    if y_step != 0:
        pts_y = np.arange(y_step, y_max + y_step, y_step, dtype=np.float32)
    else:
        pts_y = np.zeros(n, dtype=np.float32)
    pts = pts_x + pts_y*1j
    return pts

# 原点を中心とする円状の線（プログラム上は点列）を作る
#   - r: 半径
#   - n: 点の個数
def make_circle_line(r, n=200):
    theta = np.arange(0, 2*math.pi, 2*math.pi/n, dtype=np.float32)
    pts_x = r * np.cos(theta)
    pts_y = r * np.sin(theta)
    pts = pts_x + pts_y*1j
    return pts

# 矢印の羽の部分を作る
#   - p: 矢印の根元の座標
#   - q: 矢印の先端の座標
#   - w: 羽の長さ
def make_arrow_wing(p, q, w):
    if p == q:
        return np.asarray([q, q]), np.asarray([q, q])
    r = p - q
    r /= abs(r)
    a = math.pi / 6
    s = q + w * (math.cos(a) + math.sin(a) * 1j) * r
    t = q + w * (math.cos(a) - math.sin(a) * 1j) * r
    return np.asarray([q, s]), np.asarray([q, t])

# 複素関数 func を可視化する
#   - lines: 可視化に用いる曲線群
#   - colors: 曲線の色
#   - widths: 曲線の太さ
def visualize(func, lines, colors, widths):
    w_lines = []
    for k in range(len(lines)):
        z = lines[k]
        w = func(z)
        if k == 0:
            x_min = np.min(z.real)
            x_max = np.max(z.real)
            y_min = np.min(z.imag)
            y_max = np.max(z.imag)
            u_min = np.min(w.real)
            u_max = np.max(w.real)
            v_min = np.min(w.imag)
            v_max = np.max(w.imag)
        else:
            x_min = min(x_min, np.min(z.real))
            x_max = max(x_max, np.max(z.real))
            y_min = min(y_min, np.min(z.imag))
            y_max = max(y_max, np.max(z.imag))
            u_min = min(u_min, np.min(w.real))
            u_max = max(u_max, np.max(w.real))
            v_min = min(v_min, np.min(w.imag))
            v_max = max(v_max, np.max(w.imag))
        w_lines.append(w)
    dz = (x_max - x_min) - (y_max - y_min)
    dw = (u_max - u_min) - (v_max - v_min)
    if dz > 0:
        y_min -= dz / 2
        y_max += dz / 2
    else:
        x_min += dz / 2
        x_max -= dz / 2
    if dw > 0:
        v_min -= dw / 2
        v_max += dw / 2
    else:
        u_min += dw / 2
        u_max -= dw / 2
    plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('Domain (z-plane)')
    for k in range(len(lines)):
        pts_x = lines[k].real
        pts_y = lines[k].imag
        plt.plot(pts_x, pts_y, color=colors[k], linewidth=widths[k])
        s, t = make_arrow_wing(lines[k][-2], lines[k][-1], (x_max - x_min) / 25)
        plt.plot(s.real, s.imag, color=colors[k], linewidth=widths[k])
        plt.plot(t.real, t.imag, color=colors[k], linewidth=widths[k])
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.xlim(u_min, u_max)
    plt.ylim(v_min, v_max)
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title('Codomain (w-plane)')
    for k in range(len(w_lines)):
        pts_u = w_lines[k].real
        pts_v = w_lines[k].imag
        plt.plot(pts_u, pts_v, color=colors[k], linewidth=widths[k])
        s, t = make_arrow_wing(w_lines[k][-2], w_lines[k][-1], (u_max - u_min) / 25)
        plt.plot(s.real, s.imag, color=colors[k], linewidth=widths[k])
        plt.plot(t.real, t.imag, color=colors[k], linewidth=widths[k])
    plt.show()

# 複素関数 func を可視化する（直交座標型）
# 上記の関数 visualize を適当なパラメータの下で実行するラッパー
def viz_function(func):
    m = -4
    lines = []
    colors = []
    widths = []
    for k in np.arange(-1.6, 2, 0.4):
        lines.append(make_horizontal_line(y=k))
        if m < 0:
            colors.append('#{0:02x}{1:02x}{2:02x}'.format(255, -48*m, 0))
        else:
            colors.append('#{0:02x}{1:02x}{2:02x}'.format(255, 0, 48*m))
        lines.append(make_vertical_line(x=k))
        colors.append('#{0:02x}{1:02x}{2:02x}'.format(0, 128-31*m, 128+31*m))
        widths.append(3-0.2*(m+4))
        widths.append(3-0.2*(m+4))
        m += 1
    visualize(func, lines, colors, widths)

# 複素関数 func を可視化する（極座標型）
# 上記の関数 visualize を適当なパラメータの下で実行するラッパー
def viz_function2(func):
    m = -4
    lines = []
    colors = []
    widths = []
    for k in np.arange(0.2, 2, 0.2):
        lines.append(make_circle_line(r=k))
        if m < 0:
            colors.append('#{0:02x}{1:02x}{2:02x}'.format(255, -48*m, 0))
        else:
            colors.append('#{0:02x}{1:02x}{2:02x}'.format(255, 0, 48*m))
        lines.append(make_radial_line(theta=(m+4)*2*math.pi/9))
        colors.append('#{0:02x}{1:02x}{2:02x}'.format(0, 128-31*m, 128+31*m))
        widths.append(3-0.2*(m+4))
        widths.append(3-0.2*(m+4))
        m += 1
    visualize(func, lines, colors, widths)

# 複素関数 func を可視化する（放射型）
# 上記の関数 visualize を適当なパラメータの下で実行するラッパー
def viz_function3(func):
    m = -4
    lines = []
    colors = []
    widths = []
    for k in np.arange(0.2, 2, 0.2):
        lines.append(make_radial_line(theta=(m+4)*2*math.pi/9))
        colors.append('#{0:02x}{1:02x}{2:02x}'.format(0, 128-31*m, 128+31*m))
        widths.append(3-0.2*(m+4))
        m += 1
    visualize(func, lines, colors, widths)
