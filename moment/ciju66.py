# =========================================================
#   磁矩计算器（含 3D 可旋转示意图 + 独立公差 + μV·s·cm）
#   第 1 部分：导入库、数据表、数学基础函数
# =========================================================

import tkinter as tk
from tkinter import ttk, messagebox
import math

# matplotlib 用于 3D 绘图
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# =========================================================
#   磁钢牌号 → Br / Hcb(kA/m)
# =========================================================
MAGNET_GRADES = {
    "N35": (1.17, 920),
    "N38": (1.23, 955),
    "N40": (1.27, 979),
    "N42": (1.30, 1030),
    "N45": (1.35, 1070),
    "N48": (1.38, 1110),
    "N50": (1.40, 1150),
    "N52": (1.42, 1180),
}


# =========================================================
#   Pc 计算函数
# =========================================================
def pc_block(L, W, H):
    return (L / 1000) / (2 * math.sqrt((W / 1000) * (H / 1000)))


def pc_cylinder_axial(D, H):
    return (H / 1000) / (D / 1000)


def pc_ring_axial(Do, Di, H):
    return (H / 1000) / ((Do + Di) / 2000)


def pc_ring_radial(Do, Di, H):
    return ((Do - Di) / 1000) / (H / 1000)


# =========================================================
#   体积计算（mm → m）
# =========================================================
def volume_block(L, W, H):
    return (L / 1000) * (W / 1000) * (H / 1000)


def volume_cylinder(D, H):
    return math.pi * (D / 2000) ** 2 * (H / 1000)


def volume_ring(Do, Di, H):
    return math.pi * ((Do / 2000) ** 2 - (Di / 2000) ** 2) * (H / 1000)


# =========================================================
#   R2 倒角扣减体积（mm³ → m³）
# =========================================================
def chamfer_block(L, W, H, R):
    k = (1 - math.pi / 4)
    V = (
        4 * k * (R ** 2) * L +
        4 * k * (R ** 2) * W +
        4 * k * (R ** 2) * H
    )
    return V / 1e9


def chamfer_cylinder(D, H, R):
    k = (1 - math.pi / 4)
    perimeter = math.pi * D
    V = 2 * k * (R ** 2) * (perimeter / 2)
    return V / 1e9


def chamfer_ring(Do, Di, H, R):
    k = (1 - math.pi / 4)
    Vouter = 2 * k * (R ** 2) * (math.pi * Do / 2)
    Vinner = 2 * k * (R ** 2) * (math.pi * Di / 2)
    return (Vouter + Vinner) / 1e9
# =========================================================
#   第 2 部分：根据独立公差生成 3 组尺寸
# =========================================================

def apply_tolerance(nom, up, dn):
    """
    给定标称值 + 上公差 + 下公差
    返回 (最小值, 标称值, 最大值)
    """
    val_min = nom + dn
    val_nom = nom
    val_max = nom + up
    return val_min, val_nom, val_max


# =========================================================
#   计算给定一组具体尺寸的磁矩（统一输出 μV·s·cm）
# =========================================================

def compute_one_shape(shape, dims, Br, Hcb, t, R, Pc_func, vol_func, chamfer_func):
    """
    输入：
        shape: block / cylinder / ring
        dims: (L, W, H) 或 (D, H, 0) 或 (Do, Di, H)
        Br: 剩磁 (T)
        Hcb: A/m
        t: 镀层(mm)
        R: 倒角半径(mm)
        Pc_func: 对应 Pc 函数
        vol_func: 对应体积函数
        chamfer_func: 对应倒角体积函数

    输出：
        (有效体积 mm³, Pc, μr, 磁矩 μV·s·cm)
        若无效（体积<=0）返回 None
    """
    # 提取尺寸
    L = dims[0]
    W = dims[1]
    H = dims[2]

    # ========== 1. 扣减镀层（2t） ==========
    if shape == "block":
        Le = L - 2*t
        We = W - 2*t
        He = H - 2*t
        if Le <= 0 or We <= 0 or He <= 0:
            return None

        V_raw = vol_func(Le, We, He)
        V_ch = chamfer_func(Le, We, He, R)
        Pc = Pc_func(Le, We, He)

    elif shape == "cylinder":
        De = L - 2*t
        He = W - 2*t
        if De <= 0 or He <= 0:
            return None

        V_raw = vol_func(De, He)
        V_ch = chamfer_func(De, He, R)
        Pc = Pc_func(De, He)

    else:  # ring
        Doe = L - 2*t
        Die = W + 2*t
        He = H - 2*t
        if Doe <= 0 or Die <= 0 or He <= 0:
            return None
        if Die >= Doe:
            return None

        V_raw = vol_func(Doe, Die, He)
        V_ch = chamfer_func(Doe, Die, He, R)
        Pc = Pc_func(Doe, Die, He)

    # ========== 2. 有效体积 ==========
    V_final = V_raw - V_ch
    if V_final <= 0:
        return None

    # 单位换算：m³ → mm³
    V_final_mm3 = V_final * 1e9

    # ========== 3. 计算 μr ==========
    mu_r = Br / Hcb

    # ========== 4. 计算磁矩（SI：A·m²）==========
    m_SI = Br * V_final * Pc / (mu_r + Pc)

    # ========== 5. 转成 μV·s·cm ==========
    # 已确认转换关系：1 A·m² = 1e8 μV·s·cm
    m_uVs_cm = m_SI * 1e8

    return V_final_mm3, Pc, mu_r, m_uVs_cm
# =========================================================
#   第 3 部分：3D 图形绘制（可旋转）
# =========================================================

def draw_block(ax, L, W, H):
    """绘制方块（不显示倒角）"""
    X = [0, L, L, 0, 0, L, L, 0]
    Y = [0, 0, W, W, 0, 0, W, W]
    Z = [0, 0, 0, 0, H, H, H, H]

    edges = [
        (0,1), (1,2), (2,3), (3,0),   # bottom
        (4,5), (5,6), (6,7), (7,4),   # top
        (0,4), (1,5), (2,6), (3,7)    # sides
    ]

    for e in edges:
        ax.plot([X[e[0]], X[e[1]]],
                [Y[e[0]], Y[e[1]]],
                [Z[e[0]], Z[e[1]]], color='blue')


def draw_cylinder(ax, D, H):
    """绘制圆柱（方向 A：高度沿 Z）"""
    r = D / 2
    z = np.linspace(0, H, 20)
    theta = np.linspace(0, 2*np.pi, 40)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = r * np.cos(theta_grid)
    y_grid = r * np.sin(theta_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='cyan')
    ax.plot(r*np.cos(theta), r*np.sin(theta), 0, color='blue')
    ax.plot(r*np.cos(theta), r*np.sin(theta), H, color='blue')


def draw_ring(ax, Do, Di, H):
    """绘制圆环（方向 A：平面在 XY，厚度沿 Z）"""
    Ro = Do / 2
    Ri = Di / 2

    z = np.linspace(0, H, 3)
    theta = np.linspace(0, 2*np.pi, 60)

    for Z in [0, H]:
        ax.plot(Ro*np.cos(theta), Ro*np.sin(theta), Z, color='blue')
        ax.plot(Ri*np.cos(theta), Ri*np.sin(theta), Z, color='blue')

    Z_top = np.ones_like(theta) * H
    Z_bot = np.zeros_like(theta)

    # 外壁
    ax.plot(Ro*np.cos(theta), Ro*np.sin(theta), Z_top, color='cyan')
    ax.plot(Ro*np.cos(theta), Ro*np.sin(theta), Z_bot, color='cyan')

    # 内壁
    ax.plot(Ri*np.cos(theta), Ri*np.sin(theta), Z_top, color='cyan')
    ax.plot(Ri*np.cos(theta), Ri*np.sin(theta), Z_bot, color='cyan')


# =========================================================
#   3D 绘图刷新函数
# =========================================================
def update_3d_plot():
    """根据当前形状输入刷新 3D 图形"""
    shape = shape_var.get()

    # 清空 AX
    ax.clear()

    try:
        v1 = float(entry1.get())
        v2 = float(entry2.get())
        v3 = float(entry3.get()) if entry3.winfo_ismapped() else 0
    except:
        return

    # 根据形状绘制
    if shape == "方块":
        draw_block(ax, v1, v2, v3)
        ax.set_title("方块 3D 示意图")

    elif shape == "圆柱（轴向磁化）":
        draw_cylinder(ax, v1, v2)
        ax.set_title("圆柱 3D 示意图")

    elif shape == "圆环（轴向磁化）" or shape == "圆环（径向磁化）":
        draw_ring(ax, v1, v2, v3)
        ax.set_title("圆环 3D 示意图")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.set_box_aspect([1,1,0.7])  # 更美观比例

    canvas.draw()
# =========================================================
#   计算按钮回调 —— 计算最小/中值/最大磁矩（μV·s·cm）
# =========================================================
# =========================================================
#   计算按钮回调 —— 计算最小/中值/最大磁矩（μV·s·cm）
# =========================================================
def calculate():
    try:
        shape_text = shape_var.get()

        # 材料参数
        Br = float(entry_Br.get())
        Hcb = float(entry_Hcb.get()) * 1000  # kA/m → A/m

        # 镀层、倒角
        t_um = float(entry_coating.get())
        t = t_um / 1000  # μm → mm
        R = float(entry_R.get())

        # =========================================================
        #   公差获取函数（空输入 → 默认 0.04 / -0.04）
        # =========================================================
        def get_tol(up_box, dn_box):
            try:
                up = float(up_box.get()) if up_box.get().strip() != "" else 0.04
                dn = float(dn_box.get()) if dn_box.get().strip() != "" else -0.04
            except:
                up, dn = 0.04, -0.04
            return up, dn

        # 获取公差
        tol1_up, tol1_dn = get_tol(tol_up_1, tol_dn_1)
        tol2_up, tol2_dn = get_tol(tol_up_2, tol_dn_2)
        if entry3.winfo_ismapped():
            tol3_up, tol3_dn = get_tol(tol_up_3, tol_dn_3)
        else:
            tol3_up, tol3_dn = 0.04, -0.04

        # 获取标称尺寸
        v1 = float(entry1.get())
        v2 = float(entry2.get())
        v3 = float(entry3.get()) if entry3.winfo_ismapped() else 0

        dims_min = (v1 + tol1_dn, v2 + tol2_dn, v3 + tol3_dn)
        dims_nom = (v1,           v2,          v3)
        dims_max = (v1 + tol1_up, v2 + tol2_up, v3 + tol3_up)

        # =========================================================
        #   选择对应公式
        # =========================================================
        if shape_text == "方块":
            shape = "block"
            Pc_func = pc_block
            vol_func = volume_block
            chamfer_func = chamfer_block

        elif shape_text == "圆柱（轴向磁化）":
            shape = "cylinder"
            Pc_func = pc_cylinder_axial
            vol_func = volume_cylinder
            chamfer_func = chamfer_cylinder

        else:
            shape = "ring"
            vol_func = volume_ring
            chamfer_func = chamfer_ring

            if shape_text == "圆环（轴向磁化）":
                Pc_func = pc_ring_axial
            else:
                Pc_func = pc_ring_radial

        # =========================================================
        #   分别计算三组（min/nom/max）
        # =========================================================
        results = {}
        for name, dims in zip(
            ["最小值", "中值", "最大值"],
            [dims_min, dims_nom, dims_max]
        ):
            res = compute_one_shape(shape, dims, Br, Hcb, t, R,
                                    Pc_func, vol_func, chamfer_func)

            if res is None:
                results[name] = "无效尺寸（镀层或倒角过大）"
            else:
                Vmm3, Pc_val, mu_r_val, m_val = res
                results[name] = f"{m_val:.4f} μV·s·cm   （体积 {Vmm3:.2f} mm³）"

        # =========================================================
        #   在 GUI 下方显示结果（而不是弹窗）
        # =========================================================
        msg = (
            f"最小磁矩：{results['最小值']}\n"
            f"中值磁矩：{results['中值']}\n"
            f"最大磁矩：{results['最大值']}\n"
        )

        result_text.config(text=msg)

    except Exception as e:
        messagebox.showerror("错误", str(e))

from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from tkinter import filedialog

def save_pdf():
    try:
        # 选择中文字体（请确保路径存在）
        font_path = r"C:\Windows\Fonts\msyh.ttc"  # 微软雅黑
        pdfmetrics.registerFont(TTFont("CN", font_path))

        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存 PDF",
            defaultextension=".pdf",
            filetypes=[("PDF 文件", "*.pdf")],
            initialfile="磁矩计算结果.pdf"
        )
        if not file_path:
            return

        c = pdfcanvas.Canvas(file_path, pagesize=A4)
        width, height = A4

        # =====================================================
        #   1. 中文淡水印
        # =====================================================
        c.saveState()
        c.setFont("CN", 60)
        c.setFillColorRGB(0.90, 0.90, 0.90)
        c.translate(width/2, height/2)
        c.rotate(30)
        c.drawCentredString(0, 0, "Willmat")
        c.restoreState()

        # =====================================================
        #   2. 标题
        # =====================================================
        c.setFont("CN", 18)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(40, height - 60, "磁矩计算结果报告")

        # =====================================================
        #   3. 输出输入参数与结果
        # =====================================================
        c.setFont("CN", 10)
        y = height - 100

        def write(text, gap=16):
            nonlocal y
            c.drawString(40, y, text)
            y -= gap
            if y < 80:
                c.showPage()
                c.setFont("CN", 10)
                y = height - 80

        # 输入参数
        write("【输入参数】", 20)
        write(f"磁钢形状：{shape_var.get()}")
        write(f"尺寸 1：{entry1.get()}   (上公差 {tol_up_1.get()} / 下公差 {tol_dn_1.get()})")
        write(f"尺寸 2：{entry2.get()}   (上公差 {tol_up_2.get()} / 下公差 {tol_dn_2.get()})")
        if entry3.winfo_ismapped():
            write(f"尺寸 3：{entry3.get()}   (上公差 {tol_up_3.get()} / 下公差 {tol_dn_3.get()})")

        write(f"镀层厚度：{entry_coating.get()} μm")
        write(f"倒角 R：{entry_R.get()} mm")
        write(f"磁钢牌号：{grade_var.get()}")
        write(f"剩磁 Br：{entry_Br.get()} T")
        write(f"矫顽力 Hcb：{entry_Hcb.get()} kA/m")

        # 输出磁矩结果
        write("", 20)
        write("【磁矩计算结果（μV·s·cm）】", 20)
        for line in result_text.cget("text").split("\n"):
            write(line)

        c.showPage()
        c.save()

        messagebox.showinfo("成功", f"PDF 已保存：\n{file_path}")

    except Exception as e:
        messagebox.showerror("PDF 错误", str(e))


        # ============================
        # 4. 写入磁矩计算结果（来自 result_text）
        # ============================
        write("")
        write("【磁矩计算结果（μV·s·cm）】")
        for line in result_text.cget("text").split("\n"):
            write(line)

        c.showPage()
        c.save()

        messagebox.showinfo("PDF 已保存", f"已生成：{pdf_file}")

    except Exception as e:
        messagebox.showerror("PDF 错误", str(e))

# =========================================================
#   第 4 部分：GUI 主界面（布局 A）
# =========================================================

root = tk.Tk()
root.title("磁矩计算器（3D可旋转独立公差μV·s·cm）")

# ---------------------------------------------------------
#   左：3D 图区域
# ---------------------------------------------------------
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111, projection='3d')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, rowspan=20, padx=10, pady=10)

# 创建一个 frame 用来放 toolbar
toolbar_frame = tk.Frame(root)
toolbar_frame.grid(row=21, column=0, pady=5)

# 把 toolbar 放到 frame，而不是 root
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()


# ---------------------------------------------------------
#   右：输入区域
# ---------------------------------------------------------
shape_var = tk.StringVar()

ttk.Label(root, text="磁钢形状：").grid(row=0, column=1, sticky='w')
shape_combo = ttk.Combobox(
    root,
    textvariable=shape_var,
    values=["方块", "圆柱（轴向磁化）", "圆环（轴向磁化）", "圆环（径向磁化）"],
    state="readonly",
    width=18
)
shape_combo.grid(row=0, column=2, sticky='w')
shape_combo.current(0)


# ---------------------------------------------------------
#   尺寸 + 公差输入表格
# ---------------------------------------------------------
ttk.Label(root, text="尺寸(标称/上公差/下公差) (mm)").grid(
    row=1, column=1, columnspan=3, sticky='w'
)

# ------- 通用创建函数 -------
def make_dim_row(label, row):
    ttk.Label(root, text=label).grid(row=row, column=1, sticky='w')
    e_nom = ttk.Entry(root, width=10)
    e_up = ttk.Entry(root, width=8)
    e_dn = ttk.Entry(root, width=8)
    e_nom.grid(row=row, column=2)
    e_up.grid(row=row, column=3)
    e_dn.grid(row=row, column=4)
    return e_nom, e_up, e_dn


entry1, tol_up_1, tol_dn_1 = make_dim_row("尺寸 1：", 2)
entry2, tol_up_2, tol_dn_2 = make_dim_row("尺寸 2：", 3)
entry3, tol_up_3, tol_dn_3 = make_dim_row("尺寸 3：", 4)


# ---------------------------------------------------------
#   镀层 & 倒角
# ---------------------------------------------------------
ttk.Label(root, text="镀层厚度 t (μm)：").grid(row=5, column=1, sticky='w')
entry_coating = ttk.Entry(root, width=10)
entry_coating.grid(row=5, column=2, sticky='w')

ttk.Label(root, text="倒角 R (mm)：").grid(row=6, column=1, sticky='w')
entry_R = ttk.Entry(root, width=10)
entry_R.grid(row=6, column=2, sticky='w')


# ---------------------------------------------------------
#   牌号选择
# ---------------------------------------------------------
ttk.Label(root, text="磁钢牌号：").grid(row=7, column=1, sticky='w')
grade_var = tk.StringVar()
grade_combo = ttk.Combobox(
    root,
    textvariable=grade_var,
    values=list(MAGNET_GRADES.keys()),
    state="readonly",
    width=18
)
grade_combo.grid(row=7, column=2, sticky='w')


def on_grade_selected(event=None):
    g = grade_var.get()
    if g in MAGNET_GRADES:
        Br, Hcb = MAGNET_GRADES[g]
        entry_Br.delete(0, tk.END)
        entry_Br.insert(0, Br)
        entry_Hcb.delete(0, tk.END)
        entry_Hcb.insert(0, Hcb)


grade_combo.bind("<<ComboboxSelected>>", on_grade_selected)


# ---------------------------------------------------------
#   Br / Hcb
# ---------------------------------------------------------
ttk.Label(root, text="剩磁 Br (T)：").grid(row=8, column=1, sticky='w')
entry_Br = ttk.Entry(root, width=10)
entry_Br.grid(row=8, column=2, sticky='w')

ttk.Label(root, text="矫顽力 Hcb (kA/m)：").grid(row=9, column=1, sticky='w')
entry_Hcb = ttk.Entry(root, width=10)
entry_Hcb.grid(row=9, column=2, sticky='w')


# ---------------------------------------------------------
#   形状变化时刷新输入框和 3D 图
# ---------------------------------------------------------
def update_shape_inputs(event=None):
    shape = shape_var.get()

    # 圆柱：尺寸 1 = D, 尺寸 2 = H
    if shape == "圆柱（轴向磁化）":
        entry3.grid_remove()
        tol_up_3.grid_remove()
        tol_dn_3.grid_remove()
    else:
        entry3.grid()
        tol_up_3.grid()
        tol_dn_3.grid()

    update_3d_plot()


shape_combo.bind("<<ComboboxSelected>>", update_shape_inputs)


# ---------------------------------------------------------
#   任意输入变化时更新 3D 图
# ---------------------------------------------------------
def bind_update_3d(entry):
    entry.bind("<KeyRelease>", lambda e: update_3d_plot())


for wid in [
    entry1, entry2, entry3,
    tol_up_1, tol_dn_1,
    tol_up_2, tol_dn_2,
    tol_up_3, tol_dn_3
]:
    bind_update_3d(wid)

entry_coating.bind("<KeyRelease>", lambda e: update_3d_plot())
entry_R.bind("<KeyRelease>", lambda e: update_3d_plot())


# ---------------------------------------------------------
#   计算按钮
# ---------------------------------------------------------
calc_btn = ttk.Button(root, text="计算磁矩（μV·s·cm）", command=calculate)
calc_btn.grid(row=15, column=1, columnspan=3, pady=10)
save_pdf_btn = ttk.Button(root, text="保存为 PDF", command=save_pdf)
save_pdf_btn.grid(row=17, column=1, columnspan=3, pady=5)
# 显示结果的文本框（多行 Label）
result_text = tk.Label(root, text="", justify="left", anchor="w", bg="white", fg="black", width=45, height=10, relief="sunken")
result_text.grid(row=16, column=1, columnspan=3, padx=5, pady=10, sticky='w')

# 初始化 3D 图
update_3d_plot()
# =========================================================
#   第 5 部分：程序主循环
# =========================================================

# 启动 Tkinter 事件循环
root.mainloop()
