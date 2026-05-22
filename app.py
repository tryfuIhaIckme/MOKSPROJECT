import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap
from backend_ml import PredictorAPI
import numpy as np

st.set_page_config(
    page_title="Symmetry-BioPredictor",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_api():
    return PredictorAPI()

api = load_api()

st.sidebar.header("⚙️ Характеристики каркаса (Host MOF)")

with st.sidebar:
    st.subheader("Текстурные свойства")
    pld = st.slider("PLD (Å)", 2.0, 15.0, 4.3, help="Pore Limiting Diameter - минимальный диаметр окна поры.")
    lcd = st.slider("LCD (Å)", 2.0, 25.0, 11.4, help="Largest Cavity Diameter - максимальный диаметр полости.")
    porosity = st.slider("Porosity", 0.1, 0.9, 0.66, help="Геометрическая пористость каркаса.")
    asa = st.number_input("ASA (m2/g)", 0.0, 6000.0, 1334.0, help="Accessible Surface Area - доступная удельная поверхность.")
    pore_vol = st.number_input("Pore Volume (cm3/g)", 0.1, 3.0, 0.7012, help="Объем пор.")
    vf = st.slider("vf (Void Fraction)", 0.1, 0.9, 0.6122, help="Доля свободного объема.")
    sa_acc = st.number_input("sa_acc_m2g", 0.0, 5000.0, 1271.46, help="Удельная поверхность (расчетная).")

    st.subheader("Симметрийные свойства")
    pore_pg = st.selectbox(
        "Pore Point Group", 
        ['m-3m', 'mmm', '622', '4/mmm', '2/m', 'mm2', '4mm', '6mm', '3m', '-3m', 'm-3', '222', '1', '-1', '2', 'm', '4', '-4', '4/m', '422', '-42m', '3', '-3', '32', '6', '-6', '6/m', '-6m2', '23', '432', '-43m'],
        index=0,
        help="Точечная группа симметрии порового канала."
    )
    crystal_system = st.selectbox(
        "Crystal System",
        ['cubic', 'orthorhombic', 'hexagonal', 'tetragonal', 'monoclinic', 'triclinic', 'trigonal'],
        index=0,
        help="Кристаллическая система материала."
    )
    space_group = st.text_input("Space Group", "Fm-3m", help="Пространственная группа симметрии.")
    pore_symmetry_order = st.number_input("Pore Symmetry Order", 1, 192, 192, help="Порядок точечной группы (число элементов симметрии).")

st.title("🧬 In-Silico Bio-Designer: Геометрический Резонанс")
st.markdown("""
Система прогнозирования проницаемости газов через MOF-каркасы на основе **геометрической симметрии**. 
Здесь мы исследуем, как "соответствие" симметрий молекулы-гостя и поры-хозяина влияет на транспортные свойства.
""")

# Guest Molecule Selection
st.subheader("🎯 Селекция молекулы-гостя (Guest Molecule)")
available_gases = list(api.GAS_DATABASE.keys())
gas_name = st.selectbox("Выберите газ:", available_gases, index=available_gases.index("Carbon dioxide") if "Carbon dioxide" in available_gases else 0)

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Прогноз диффузии", 
    "🔬 Научный анализ (SHAP)", 
    "🏆 Bio-Reticular Leaderboard",
    "🧪 3D Lab"
])

# Prepare MOF params
mof_params = {
    "PLD": pld, "LCD": lcd, "Porosity": porosity, "ASA (m2/g)": asa,
    "Pore Volume (cm3/g)": pore_vol, "vf": vf, "sa_acc_m2g": sa_acc,
    "Pore_Symmetry_Order": pore_symmetry_order, "Pore_PointGroup": pore_pg,
    "Crystal_System": crystal_system, "Space_Group": space_group,
    "MOF_id": "User_Input", # Placeholder for feature calculation
    "MOF_Formula": "Zn" # Dummy for bio-active check
}

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 24}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-8, -2], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2c3e50"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-8, -6], 'color': '#ff4b4b'},
                {'range': [-6, -4], 'color': '#ffa15a'},
                {'range': [-4, -2], 'color': '#00d580'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#2c3e50", 'family': "Arial"},
        height = 400
    )
    return fig

with tab1:
    try:
        log_d = api.predict(mof_params, gas_name)
        
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.plotly_chart(create_gauge_chart(log_d, f"Predicted log(D) for {gas_name}"), use_container_width=True)
        
        with st.expander("🛠️ Посмотреть рассчитанные индексы"):
            processed_df = api._preprocess_input(mof_params, gas_name)
            st.write("Сгенерированные признаки взаимодействия:")
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Symmetry Order Ratio", f"{processed_df['Symmetry_Order_Ratio'].values[0]:.2f}")
            metrics_cols[1].metric("Shape Fit Factor", f"{processed_df['Shape_Fit_Factor'].values[0]:.4f}")
            metrics_cols[2].metric("Size Exclusion Δ", f"{processed_df['size_exclusion_delta'].values[0]:.2f} Å")
            metrics_cols[3].metric("Bio-active", "Yes" if processed_df['is_bioactive'].values[0] else "No")
            
            st.dataframe(processed_df)
    except Exception as e:
        st.error(f"❌ Ошибка при расчете: {e}")

with tab2:
    st.subheader("🔬 Локальная интерпретация (SHAP Waterfall)")
    try:
        with st.spinner("Генерация SHAP объяснения..."):
            shap_values = api.explain_prediction(mof_params, gas_name)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        st.info("""
        **Как читать этот график:**
        *   Красные стрелки увеличивают прогноз D, синие - снижают.
        *   Признаки симметрии и bio-reticular параметры (is_bioactive, linker_type) теперь включены в анализ!
        """)
    except Exception as e:
        st.error(f"❌ Ошибка отрисовки SHAP: {e}")

with tab3:
    st.subheader("🏆 Bio-Reticular Leaderboard")
    st.markdown("""
    Ранжирование материалов на основе их **био-совместимости** и прогнозируемой диффузии. 
    Используйте фильтры для поиска материалов с аминокислотными линкерами и безопасными металлами.
    """)
    
    @st.cache_data
    def get_leaderboard_data():
        df = pd.read_csv("FINAL_ML_READY_DATASET.csv")
        metadata = pd.read_csv("mof_metadata.csv")
        df = df.merge(metadata, on="MOF_id", how="left")
        
        # Add bio features
        from backend_ml import get_bio_features
        df["is_bioactive"], df["linker_type"], df["c_multiplicity"] = zip(*df.apply(
            lambda x: get_bio_features(x["MOF_id"], x["MOF_Formula"], x["Total_C"]), axis=1
        ))
        return df

    lb_df = get_leaderboard_data()
    
    col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
    with col_l1:
        bio_only = st.checkbox("🍀 Bio-active Metals (Fe, Mn, Zn, Cu, Ni, Mo)", value=False)
    with col_l2:
        amino_only = st.checkbox("🧬 Amino Acid Linkers Only", value=True)
    with col_l3:
        mult_only = st.checkbox("🔢 Carbon Multiplicity Rule", value=False)
    
    filtered_df = lb_df[lb_df["Gas"] == gas_name]
    if bio_only:
        filtered_df = filtered_df[filtered_df["is_bioactive"] == 1]
    if amino_only:
        filtered_df = filtered_df[filtered_df["linker_type"] != "Other"]
    if mult_only:
        filtered_df = filtered_df[filtered_df["c_multiplicity"] == 1]
        
    st.markdown(f"**Топ-10 био-совместимых MOF для газа {gas_name}:**")
    display_cols = ["MOF_id", "MOF_Formula", "PLD", "LCD", "linker_type", "is_bioactive", "c_multiplicity", "log(D) 1bar(cm/s)"]
    res_df = filtered_df[display_cols].sort_values("log(D) 1bar(cm/s)", ascending=False).head(10)
    
    st.dataframe(res_df.style.highlight_max(axis=0, subset=["log(D) 1bar(cm/s)"], color='#d4edda'))
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать полный Subset (CSV)", csv, f"bio_leaderboard_{gas_name}.csv", "text/csv")

with tab4:
    st.subheader("🧪 3D Lab: Structural & Gas Visualization")
    st.markdown("Интерактивный анализ взаимного расположения поры и молекулы-гостя.")
    
    cif_file = st.file_uploader("Загрузите .cif файл MOF-каркаса", type=["cif"])
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.info("### 🏰 MOF Structure")
        if cif_file:
            content = cif_file.read().decode("utf-8")
            st.success(f"File {cif_file.name} successfully loaded.")
            # Simplified placeholder for 3Dmol since it requires external JS
            st.code(content[:300] + "\n...", language="text")
        else:
            st.warning("Please upload a .cif file to visualize the 3D structure.")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/IRMOF-1_structure.png/300px-IRMOF-1_structure.png", caption="Sample MOF Structure (IRMOF-1)")
            
    with col_v2:
        st.info(f"### 🎈 Gas Molecule: {gas_name}")
        st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{gas_name}/PNG", width=250)
        
        gas_props = api.GAS_DATABASE[gas_name]
        st.json({
            "Symmetry Group": gas_props["Symmetry"],
            "Point Group Order": gas_props["Gas_Symmetry_Order"],
            "Kinetic Diameter": f"{gas_props['KineticDiameter_A']} Å",
            "Polarizability": gas_props["Polarizability_1e-25_cm3"]
        })

# Footer
st.markdown("---")
st.markdown("Developed for MOF Gas Permeation Research | 2026")
