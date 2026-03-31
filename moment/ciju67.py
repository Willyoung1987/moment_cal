##############################################
# Streamlit 版磁矩计算器（含 3D、PDF、公差）
##############################################

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
from io import BytesIO

st.set_page_config(page_title="磁矩计算器（WEB版）", layout="wide")


##############################################
# 牌号表
##############################################
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


##############################################
# Pc 函数
##############################################
def pc_block(L, W, H):
    return (L/1000) / (2 * math.sqrt((W/1000)*(H/1000)))

def pc_cylinder_axial(D, H):
    return (H/1000) / (D/1000)

def pc_ring_axial(Do, Di, H):
    return (H/1000) / ((Do + Di) / 2000)

def pc_ring_radial(Do, Di, H):
    return ((Do - Di)/1000) / (H/1000)


##############################################
# 体积函数
##############################################
def volume_block(L, W, H):
    return (L/1000)*(W/1000)*(H/1000)

def volume_cylinder(D, H):
    return math.pi*(D/2000)**2*(H/1000)

def volume_ring(Do, Di, H):
    return math.pi*((Do/2000)**2 - (Di/2000)**2)*(H/1000)


##############################################
# 倒角扣减 R2
##############################################
def chamfer_block(L, W, H, R):
    k = (1 - math.pi/4)
    V = 4*k*(R**2)*L + 4*k*(R**2)*W + 4*k*(R**2)*H
    return V / 1e9

def chamfer_cylinder(D, H, R):
    k = (1 - math.pi/4)
    perimeter = math.pi * D
    V = 2*k*(R**2)*(perimeter/2)
    return V / 1e9

def chamfer_ring(Do, Di, H, R):
    k = (1 - math.pi/4)
    Vouter = 2*k*(R**2)*(math.pi*Do/2)
    Vinner = 2*k*(R**2)*(math.pi*Di/2)
    return (Vouter + Vinner)/1e9


##############################################
# 计算一次磁矩
##############################################
def compute_one(shape, dims, Br, Hcb, t, R, Pc_func, Vol_func, Chamfer_func):

    L = dims[0]
    W = dims[1]
    H = dims[2]

    # 扣镀层
    if shape=="block":
        Le = L - 2*t
        We = W - 2*t
        He = H - 2*t
        if Le<=0 or We<=0 or He<=0: return None
        Vr = Vol_func(Le,We,He)
        Vc = Chamfer_func(Le,We,He,R)
        Pc = Pc_func(Le, We, He)

    elif shape=="cylinder":
        De = L - 2*t
        He = W - 2*t
        if De<=0 or He<=0: return None
        Vr = Vol_func(De, He)
        Vc = Chamfer_func(De, He, R)
        Pc = Pc_func(De, He)

    else:
        Doe = L - 2*t
        Die = W + 2*t
        He = H - 2*t
        if Doe<=0 or Die<=0 or He<=0 or Die>=Doe: return None
        Vr = Vol_func(Doe, Die, He)
        Vc = Chamfer_func(Doe, Die, He, R)
        Pc = Pc_func(Doe, Die, He)

    V_final = Vr - Vc
    if V_final <= 0: return None

    V_mm3 = V_final * 1e9

    mu_r = Br / Hcb
    m_SI = Br * V_final * Pc / (mu_r + Pc)

    # 转 μV·s·cm
    m_final = m_SI * 1e8
    return V_mm3, Pc, mu_r, m_final



##############################################
# 3D plotly 绘图函数
##############################################
def plot_block(L, W, H):
    fig = go.Figure()

    # 画一个立方体
    x=[0,L,L,0,0,L,L,0]
    y=[0,0,W,W,0,0,W,W]
    z=[0,0,0,0,H,H,H,H]
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[x[e[0]],x[e[1]]],
            y=[y[e[0]],y[e[1]]],
            z=[z[e[0]],z[e[1]]],
            mode="lines",
            line=dict(color="blue")
        ))
    fig.update_layout(scene_aspectmode="data")
    return fig

def plot_cylinder(D, H):
    fig = go.Figure()
    r = D/2
    theta = np.linspace(0,2*np.pi,50)
    x = r*np.cos(theta)
    y = r*np.sin(theta)

    fig.add_trace(go.Scatter3d(x=x,y=y,z=[0]*50,mode="lines",line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(x=x,y=y,z=[H]*50,mode="lines",line=dict(color='blue')))

    fig.update_layout(scene_aspectmode="data")
    return fig

def plot_ring(Do, Di, H):
    fig = go.Figure()
    Ro = Do/2
    Ri = Di/2
    theta=np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter3d(x=Ro*np.cos(theta),y=Ro*np.sin(theta),z=[0]*60,mode="lines"))
    fig.add_trace(go.Scatter3d(x=Ro*np.cos(theta),y=Ro*np.sin(theta),z=[H]*60,mode="lines"))
    fig.add_trace(go.Scatter3d(x=Ri*np.cos(theta),y=Ri*np.sin(theta),z=[0]*60,mode="lines"))
    fig.add_trace(go.Scatter3d(x=Ri*np.cos(theta),y=Ri*np.sin(theta),z=[H]*60,mode="lines"))
    fig.update_layout(scene_aspectmode="data")
    return fig


##############################################
# Streamlit UI
##############################################
st.title("🔩 磁矩计算器（Streamlit 网页版）")
st.caption("含 3D 示意、公差、镀层、倒角、PDF 导出，所有计算单位自动统一为 μV·s·cm")

left, right = st.columns([1.2,1])

##############################################
# 右侧输入区
##############################################
with right:

    shape = st.selectbox("磁钢形状", ["方块","圆柱（轴向磁化）","圆环（轴向磁化）","圆环（径向磁化）"])

    st.subheader("📐 尺寸（mm）及公差")
    size1 = st.number_input("尺寸 1 标称(mm)",min_value=0.0, value=10.0)
    up1 = st.number_input("尺寸1 上公差(mm)", value=0.04)
    dn1 = st.number_input("尺寸1 下公差(mm)", value=-0.04)

    size2 = st.number_input("尺寸 2 标称(mm)",min_value=0.0, value=5.0)
    up2 = st.number_input("尺寸2 上公差(mm)", value=0.04)
    dn2 = st.number_input("尺寸2 下公差(mm)", value=-0.04)

    if "圆柱" in shape:
        size3, up3, dn3 = 0, 0, 0
    else:
        size3 = st.number_input("尺寸 3 标称(mm)",min_value=0.0, value=3.0)
        up3 = st.number_input("尺寸3 上公差(mm)", value=0.04)
        dn3 = st.number_input("尺寸3 下公差(mm)", value=-0.04)

    st.subheader("🧲 材料")
    grade = st.selectbox("磁钢牌号", list(MAGNET_GRADES.keys()))
    Br_default, Hcb_default = MAGNET_GRADES[grade]

    Br = st.number_input("剩磁 Br (T)", value=Br_default)
    Hcb = st.number_input("矫顽力 Hcb (kA/m)", value=Hcb_default)

    st.subheader("🛠 镀层 & 倒角")
    t_um = st.number_input("镀层厚度 t (μm)", value=10.0)
    R = st.number_input("倒角半径 R (mm)", value=0.0)

    if st.button("计算磁矩"):
        dims_min = (size1+dn1, size2+dn2, size3+dn3)
        dims_nom = (size1, size2, size3)
        dims_max = (size1+up1, size2+up2, size3+up3)

        # 选择函数
        if shape=="方块":
            shp="block"
            Pc_f=pc_block
            Vol_f=volume_block
            Cham_f=chamfer_block
        elif shape=="圆柱（轴向磁化）":
            shp="cylinder"
            Pc_f=pc_cylinder_axial
            Vol_f=volume_cylinder
            Cham_f=chamfer_cylinder
        else:
            shp="ring"
            Pc_f=pc_ring_axial if shape=="圆环（轴向磁化）" else pc_ring_radial
            Vol_f=volume_ring
            Cham_f=chamfer_ring

        results={}
        for name,d in zip(["最小值","中值","最大值"],[dims_min,dims_nom,dims_max]):
            res=compute_one(shp,d,Br,Hcb,t_um/1000,R,Pc_f,Vol_f,Cham_f)
            if res:
                results[name]=f"{res[3]:.4f} μV·s·cm （体积 {res[0]:.2f} mm³）"
            else:
                results[name]="无效尺寸"

        st.subheader("📊 计算结果")
        for k,v in results.items():
            st.write(f"**{k}：** {v}")

        st.session_state["last_results"] = results
        st.session_state["input_params"] = {
            "shape":shape,
            "dims":(size1,size2,size3),
            "tol":(up1,dn1,up2,dn2,up3,dn3),
            "Br":Br,
            "Hcb":Hcb,
            "t_um":t_um,
            "R":R,
            "grade":grade
        }


##############################################
# 左侧 3D 示意图
##############################################
with left:
    st.subheader("🧊 3D 示意图")

    if shape=="方块":
        fig=plot_block(size1,size2,size3)
    elif shape=="圆柱（轴向磁化）":
        fig=plot_cylinder(size1,size2)
    else:
        fig=plot_ring(size1,size2,size3)

    st.plotly_chart(fig, use_container_width=True)


##############################################
# PDF 导出
##############################################
st.subheader("📄 导出 PDF")

def build_pdf():
    buffer=BytesIO()
    pdf=canvas.Canvas(buffer,pagesize=A4)
    width,height=A4

    pdfmetrics.registerFont(TTFont("CN","C:/Windows/Fonts/msyh.ttc"))

    # 水印
    pdf.saveState()
    pdf.setFont("CN",60)
    pdf.setFillColorRGB(0.9,0.9,0.9)
    pdf.translate(width/2,height/2)
    pdf.rotate(30)
    pdf.drawCentredString(0,0,"威尔迈（嘉兴）")
    pdf.restoreState()

    pdf.setFont("CN",16)
    pdf.drawString(40,height-60,"磁矩计算报告")

    y=height-100
    pdf.setFont("CN",10)

    def write(text):
        nonlocal y
        pdf.drawString(40,y,text)
        y-=16

    params=st.session_state.get("input_params",None)
    results=st.session_state.get("last_results",None)

    if not params or not results:
        write("无计算结果。请先点击计算。")
    else:
        write("【输入参数】")
        write(f"磁钢形状：{params['shape']}")
        s1,s2,s3=params["dims"]
        u1,d1,u2,d2,u3,d3=params["tol"]

        write(f"尺寸1：{s1} (+{u1}/ {d1})")
        write(f"尺寸2：{s2} (+{u2}/ {d2})")
        if "圆柱" not in params["shape"]:
            write(f"尺寸3：{s3} (+{u3}/ {d3})")

        write(f"镀层：{params['t_um']} μm")
        write(f"倒角：{params['R']} mm")
        write(f"牌号：{params['grade']}")
        write(f"Br：{params['Br']}")
        write(f"Hcb：{params['Hcb']} kA/m")

        write("")
        write("【磁矩结果 μV·s·cm】")
        for k,v in results.items():
            write(f"{k}: {v}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


if st.button("📥 导出 PDF"):
    pdf_file=build_pdf()
    st.download_button(
        label="点击下载 PDF",
        data=pdf_file,
        file_name="磁矩计算结果.pdf",
        mime="application/pdf"
    )
