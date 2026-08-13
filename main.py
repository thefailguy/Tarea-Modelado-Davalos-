
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.optimize import minimize

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Gestión y Optimización de Columna de Destilación",
    page_icon="🧪",
    layout="wide"
)

# --- Carga de Recursos (Modelo y Datos) ---
@st.cache_resource
def load_assets():
    """Carga el modelo entrenado y los datos históricos de proceso."""
    try:
        model = joblib.load('modelo_xgboost_final.joblib')
        df = pd.read_csv('transformed_data.csv')
        return model, df
    except Exception as e:
        st.error(f"Error al cargar los archivos: {e}")
        return None, None

model, df = load_assets()

if model is not None and df is not None:
    X_ref = df.drop('mpy', axis=1)

    # --- Barra Lateral: Parámetros Operacionales de Entrada ---
    st.sidebar.header("⚙️ Parámetros Operacionales")
    st.sidebar.markdown("Ajusta las condiciones de operación para predecir la corrosión:")

    # Rangos derivados del dataset histórico
    p_min, p_max = float(df['presion_cabeza_psi'].min()), float(df['presion_cabeza_psi'].max())
    f_min, f_max = float(df['agua_BAPD'].min()), float(df['agua_BAPD'].max())
    t_min, t_max = float(df['cloruros_ppm'].min()), float(df['cloruros_ppm'].max())

    p_mean = float(df['presion_cabeza_psi'].median())
    f_mean = float(df['agua_BAPD'].mean())
    t_mean = float(df['cloruros_ppm'].mean())

    pressure = st.sidebar.slider("Presión Cabeza (presion_cabeza_psi)", min_value=p_min, max_value=p_max, value=p_mean, step=0.1)
    flow = st.sidebar.slider("Agua BAPD (agua_BAPD)", min_value=f_min, max_value=f_max, value=f_mean, step=1.0)
    temp = st.sidebar.slider("Cloruros PPM (cloruros_ppm)", min_value=t_min, max_value=t_max, value=t_mean, step=0.5)

    # DataFrame con las entradas actuales
    df_current = pd.DataFrame({
        'presion_cabeza_psi': [pressure],
        'agua_BAPD': [flow],
        'cloruros_ppm': [temp]
    })

    # Predicción actual
    current_pred = model.predict(df_current)[0]

    # --- Header Principal ---
    st.title("🧪 Sistema Inteligente de Corrosión: Predicción, Interpretabilidad y Prescripción")
    st.markdown("Esta plataforma web integra Machine Learning avanzado (RandomForest), Interpretabilidad Explicable (SHAP) y Optimización de Setpoints para la corrosión.")

    # --- Pestañas de la Aplicación ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predicción en Tiempo Real",
        "🧠 Interpretabilidad SHAP",
        "🗺️ Sensibilidad Operacional",
        "🎯 Setpoint Operacional Óptimo"
    ])

    # === TAB 1: PREDICCIÓN ===
    with tab1:
        st.subheader("📈 Resultado de Predicción de Corrosión")
        col1, col2, col3 = st.columns(3)
        col1.metric("Presión Seleccionada", f"{pressure:.2f} psi")
        col2.metric("Agua Seleccionada", f"{flow:.1f} m³/h")
        col3.metric("Cloruros Seleccionados", f"{temp:.1f} ppm")

        st.markdown("---")
        st.success(f"### **Corrosión Predicha (MPY): `{current_pred:.2f} MPY`**")
        st.info("La corrosión predicha refleja la tasa de corrosión estimada en MPY bajo el setpoint actual.")

    # === TAB 2: SHAP INTERPRETABILIDAD ===
    with tab2:
        st.subheader("🧠 Interpretabilidad Local con SHAP")
        st.markdown("Explicación detallada de cómo cada variable operacional está influyendo en la predicción actual de corrosión respecto al valor promedio de la planta.")

        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer(df_current)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        shap.plots.waterfall(shap_values_single[0], show=False)
        st.pyplot(fig)

    # === TAB 3: SENSIBILIDAD OPERACIONAL ===
    with tab3:
        st.subheader("🗺️ Mapa de Sensibilidad Operacional (Cloruros PPM vs Agua BAPD)")
        st.markdown("Explora cómo cambia la corrosión predicha al variar los Cloruros PPM y el Agua BAPD manteniendo la presión de cabeza actual.")

        cloruros_range = np.linspace(t_min, t_max, 40)
        agua_range = np.linspace(f_min, f_max, 40)
        cloruros_grid, agua_grid = np.meshgrid(cloruros_range, agua_range)

        grid_df = pd.DataFrame({
            'presion_cabeza_psi': pressure,
            'agua_BAPD': agua_grid.ravel(),
            'cloruros_ppm': cloruros_grid.ravel()
        })

        mpy_grid = model.predict(grid_df).reshape(cloruros_grid.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        contour = ax.contourf(cloruros_grid, agua_grid, mpy_grid, levels=20, cmap='viridis_r') # Invert colormap for corrosion (higher is worse)
        plt.colorbar(contour, ax=ax, label='Corrosión Predicha (MPY)')
        ax.scatter([temp], [flow], color='red', s=120, marker='X', label='Punto Actual Seleccionado')
        ax.set_title('Superficie de Sensibilidad Operacional', fontweight='bold')
        ax.set_xlabel('Cloruros PPM (cloruros_ppm)')
        ax.set_ylabel('Agua BAPD (agua_BAPD)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

    # === TAB 4: SETPOINT OPERACIONAL ÓPTIMO ===
    with tab4:
        st.subheader("🎯 Optimización Prescriptiva de Setpoint")
        st.markdown("Búsqueda matemática del punto óptimo de operación $(presion\_cabeza\_psi^*, agua\_BAPD^*, cloruros\_ppm^*)$ que **minimiza la corrosión** dentro de límites seguros de planta (percentiles 5% a 95%).")

        bounds = [
            (df['presion_cabeza_psi'].quantile(0.05), df['presion_cabeza_psi'].quantile(0.95)),
            (df['agua_BAPD'].quantile(0.05), df['agua_BAPD'].quantile(0.95)),
            (df['cloruros_ppm'].quantile(0.05), df['cloruros_ppm'].quantile(0.95))
        ]

        def obj_func(x):
            in_df = pd.DataFrame([x], columns=['presion_cabeza_psi', 'agua_BAPD', 'cloruros_ppm'])
            return model.predict(in_df)[0] # Minimize MPY directly

        x0 = [pressure, flow, temp]
        res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds)

        opt_p, opt_f, opt_t = res.x
        opt_mpy = res.fun # The minimized MPY

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📍 Operación Actual")
            st.write(f"- Presión Cabeza: `{pressure:.2f} psi`")
            st.write(f"- Agua BAPD: `{flow:.1f} m³/h`")
            st.write(f"- Cloruros PPM: `{temp:.1f} ppm`")
            st.write(f"- **Corrosión Actual:** `{current_pred:.2f} MPY`")

        with col_b:
            st.markdown("#### 🎯 Setpoint Óptimo Prescripto")
            st.write(f"- Presión Cabeza Óptima: `{opt_p:.2f} psi`")
            st.write(f"- Agua BAPD Óptima: `{opt_f:.1f} m³/h`")
            st.write(f"- Cloruros PPM Óptimos: `{opt_t:.1f} ppm`")
            st.write(f"- **Corrosión Mínima Posible:** `{opt_mpy:.2f} MPY`")

        reduction = current_pred - opt_mpy
        st.success(f"💡 **Reducción Potencial de Corrosión:** `-{reduction:.2f} MPY` de reducción alcanzable ajustando al Setpoint Óptimo.")
else:
    st.error("No se pudo iniciar la aplicación. Verifica la existencia de 'modelo_xgboost_final.joblib' y 'transformed_data.csv'.")
