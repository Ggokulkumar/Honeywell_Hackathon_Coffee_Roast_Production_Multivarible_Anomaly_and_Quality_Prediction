import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import requests
import json

st.set_page_config(
    page_title="Roastmaster's AI Assistant",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Main header style */
    .main-header {
        font-size: 2.8rem;
        color: #D2B48C; /* Tan color for contrast */
        text-align: center;
        font-weight: bold;
        padding-top: 1rem;
    }
    /* Metric card styling for dark theme */
    .metric-card {
        background-color: #161A21;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #D2B48C; /* Default border color */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-card-warning {
        border-left-color: #ffc107 !important; /* Yellow for warning */
    }
    .metric-card-danger {
        border-left-color: #dc3545 !important; /* Red for danger */
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161A21;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #161A21;
    }
    /* Status text styling */
    .status-good { color: #28a745; font-weight: bold; font-size: 1.2rem; }
    .status-warning { color: #ffc107; font-weight: bold; font-size: 1.2rem; }
    .status-danger { color: #dc3545; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all necessary model files with error handling."""
    models = {}
    model_files = {
        'quality_model': 'enhanced_coffee_quality_model.pkl',
        'anomaly_model': 'enhanced_coffee_anomaly_model.pkl',
        'scaler': 'enhanced_coffee_scaler.pkl',
        'preprocessors': 'enhanced_coffee_preprocessors.pkl'
    }
    
    missing_files = [f for f in model_files.values() if not os.path.exists(f)]
    if missing_files:
        st.error(f"Fatal Error: Missing model files: {', '.join(missing_files)}. Please run the training script first.")
        return None

    try:
        for name, path in model_files.items():
            models[name] = joblib.load(path)
        st.success("✅ AI Models Loaded Successfully")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


class RoastApp:
    def __init__(self, models):
        self.models = models
        if models:
            self.quality_model = models['quality_model']
            self.anomaly_model = models['anomaly_model']
            self.scaler = models['scaler']
            self.preprocessors = models['preprocessors']
            self.feature_columns = self.preprocessors['feature_columns']
            self.feature_selector = self.preprocessors['feature_selector']

    def _engineer_features(self, df):
        """Replicates the full feature engineering from the training script."""
        temp_cols = ['drying_temp_avg_c', 'maillard_temp_avg_c', 'development_temp_avg_c', 'final_temp_c']
        available_temp_cols = [col for col in temp_cols if col in df.columns]
        if len(available_temp_cols) >= 2:
            df['temp_range'] = df[available_temp_cols].max(axis=1) - df[available_temp_cols].min(axis=1)
        else:
            df['temp_range'] = 0
        
        if 'total_energy_units' in df.columns and 'batch_size_kg' in df.columns:
            df['energy_efficiency'] = df['total_energy_units'] / (df['batch_size_kg'] + 1e-6)
        else:
            df['energy_efficiency'] = 0
        
        if 'initial_moisture_pct' in df.columns and 'roast_duration_min' in df.columns:
            estimated_final_moisture = df['initial_moisture_pct'] * (1 - df.get('weight_loss_pct', 15) / 100)
            df['moisture_loss_rate'] = (df['initial_moisture_pct'] - estimated_final_moisture) / (df['roast_duration_min'] + 1e-6)
        else:
            df['moisture_loss_rate'] = 0
        
        if 'first_crack_time_min' in df.columns and 'roast_duration_min' in df.columns:
            df['crack_timing_ratio'] = df['first_crack_time_min'] / (df['roast_duration_min'] + 1e-6)
        else:
            df['crack_timing_ratio'] = 0
        
        if 'heat_rate_c_per_min' in df.columns and 'roast_duration_min' in df.columns:
            df['roast_intensity'] = df['heat_rate_c_per_min'] * df['roast_duration_min']
        else:
            df['roast_intensity'] = 0
            
        if 'ambient_temp_c' in df.columns and 'humidity_pct' in df.columns:
            df['environmental_factor'] = df['ambient_temp_c'] * (1 + df['humidity_pct'] / 100)
        else:
            df['environmental_factor'] = 0
            
        stability_cols = ['gas_level_pct', 'airflow_rate_pct', 'drum_speed_rpm']
        available_stability_cols = [col for col in stability_cols if col in df.columns]
        if available_stability_cols:
            df['process_stability'] = df[available_stability_cols].std(axis=1)
        else:
            df['process_stability'] = 0
            
        if 'batch_size_kg' in df.columns and 'final_weight_kg' in df.columns:
            df['weight_change_ratio'] = df['final_weight_kg'] / (df['batch_size_kg'] + 1e-6)
        else:
            df['weight_change_ratio'] = 1.0
            
        if 'first_crack_time_min' in df.columns and 'second_crack_time_min' in df.columns and df['second_crack_time_min'].notna().all():
            df['crack_gap'] = df['second_crack_time_min'] - df['first_crack_time_min']
        else:
            df['crack_gap'] = 0
            
        quality_cols = ['aroma_score', 'body_score', 'acidity_score']
        available_quality_cols = [col for col in quality_cols if col in df.columns]
        if available_quality_cols:
            df['quality_composite'] = df[available_quality_cols].mean(axis=1)
        else:
            df['quality_composite'] = 0
            
        if all(col in df.columns for col in ['drying_temp_avg_c', 'maillard_temp_avg_c', 'development_temp_avg_c']):
            temp_diff1 = df['maillard_temp_avg_c'] - df['drying_temp_avg_c']
            temp_diff2 = df['development_temp_avg_c'] - df['maillard_temp_avg_c']
            df['temp_progression_rate'] = (temp_diff1 + temp_diff2) / 2
        else:
            df['temp_progression_rate'] = 0
            
        return df

    def predict(self, input_data):
        """Makes predictions on new user input."""
        df = pd.DataFrame([input_data])

        # Apply label encoding
        for col, le in self.preprocessors['label_encoders'].items():
            df[col + '_encoded'] = le.transform(df[col])

        # Engineer features
        df = self._engineer_features(df)

        # Ensure all columns are present and in the correct order
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Scale the data
        X_scaled = self.scaler.transform(df)

        # Select features for anomaly model
        X_selected = self.feature_selector.transform(X_scaled)

        # Make predictions
        quality_pred = self.quality_model.predict(X_scaled)[0]
        anomaly_prob = self.anomaly_model.predict_proba(X_selected)[:, 1][0]

        return quality_pred, anomaly_prob

    @st.cache_data
    def get_gemini_analysis(_self, quality_score, anomaly_prob, params):
        """Generates human-readable analysis using the Gemini API."""
        API_KEY = "" # Your Gemini API key would go here, but it's handled by Streamlit secrets
        if not API_KEY:
             API_KEY = st.secrets.get("GEMINI_API_KEY")
        if not API_KEY:
            return "Gemini API key not found. Please add it to your Streamlit secrets."

        prompt = f"""
        Act as an expert coffee roastmaster. You are analyzing a simulated coffee roast batch.
        Based on the following data, provide a concise, human-readable analysis and one actionable recommendation.

        Data:
        - Predicted Final Quality Score: {quality_score:.2f} / 10
        - Anomaly Risk Probability: {anomaly_prob*100:.1f}%
        - Target Roast Level: {params['target_roast_level']}
        - Key Roaster Settings: Gas Level at {params['gas_level_pct']}%, Airflow at {params['airflow_rate_pct']}%

        Your analysis should be 2-3 sentences.
        Your recommendation should be a single, clear bullet point.
        """
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"Error contacting Gemini API: {e}"

    def render(self):
        st.markdown('<h1 class="main-header">☕ Roastmaster\'s AI Assistant</h1>', unsafe_allow_html=True)
        
        st.sidebar.title("🎛️ Simulate New Roast Batch")
        params = self.get_sidebar_params()

        tab1, tab2, tab3 = st.tabs(["📈 Live Dashboard", "💡 Process Insights", "ℹ️ About"])

        with tab1:
            self.render_live_dashboard(params)
        
        with tab2:
            self.render_insights_tab()

        with tab3:
            self.render_about_tab()

    def render_live_dashboard(self, params):
        col1, col2 = st.columns([3, 2])

        with col1:
            self.display_temperature_chart(params)

        with col2:
            quality_score, anomaly_prob = self.display_predictions(params)
            self.display_gemini_analysis(quality_score, anomaly_prob, params)
            self.display_process_summary(params)

    def get_sidebar_params(self):
        with st.sidebar.expander("Bean & Batch Settings", expanded=True):
            bean_type = st.selectbox("Bean Type", ['Arabica', 'Robusta', 'Blend'])
            target_roast_level = st.selectbox("Target Roast Level", ['Light', 'Medium', 'Dark'])
            batch_size_kg = st.slider("Batch Size (kg)", 50, 150, 100)
            initial_moisture_pct = st.slider("Initial Moisture (%)", 8.0, 16.0, 11.5, 0.1)

        with st.sidebar.expander("Roaster Machine Settings", expanded=True):
            preheat_temp_c = st.slider("Preheat Temperature (°C)", 180, 230, 205)
            gas_level_pct = st.slider("Gas Level (%)", 40, 100, 75)
            airflow_rate_pct = st.slider("Airflow Rate (%)", 60, 100, 80)
            drum_speed_rpm = st.slider("Drum Speed (RPM)", 50, 80, 65)
        
        roast_duration_min = st.sidebar.slider("Target Duration (min)", 10, 20, 14)
        
        final_temp_c = preheat_temp_c + (roast_duration_min * 10) * (gas_level_pct / 75)
        
        return {
            'bean_type': bean_type, 'target_roast_level': target_roast_level,
            'batch_size_kg': batch_size_kg, 'initial_moisture_pct': initial_moisture_pct,
            'preheat_temp_c': preheat_temp_c, 'gas_level_pct': gas_level_pct,
            'airflow_rate_pct': airflow_rate_pct, 'drum_speed_rpm': drum_speed_rpm,
            'roast_duration_min': roast_duration_min,
            'final_temp_c': final_temp_c,
            'drying_temp_avg_c': preheat_temp_c + (final_temp_c - preheat_temp_c) * 0.2,
            'maillard_temp_avg_c': preheat_temp_c + (final_temp_c - preheat_temp_c) * 0.5,
            'development_temp_avg_c': preheat_temp_c + (final_temp_c - preheat_temp_c) * 0.8,
            'first_crack_time_min': roast_duration_min * 0.6,
            'heat_rate_c_per_min': (final_temp_c - preheat_temp_c) / roast_duration_min if roast_duration_min > 0 else 0,
            'total_energy_units': gas_level_pct * roast_duration_min * batch_size_kg / 100,
            'weight_loss_pct': 15 + (final_temp_c - 220) * 0.1,
            'defects_count': 1, 'bean_density_g_cm3': 1.3, 'bean_size_mm': 6.5, 'second_crack_time_min': 12,
            'ambient_temp_c': 25, 'humidity_pct': 60, 'final_weight_kg': 85,
            'color_score_agtron': 55, 'aroma_score': 8, 'body_score': 8, 'acidity_score': 8,
        }

    def display_temperature_chart(self, params):
        st.subheader("🌡️ Real-Time Temperature Profile")
        duration = params['roast_duration_min']
        time_points = np.linspace(0, duration, 50)
        ideal_temp = 150 + (70 * (time_points / duration)**0.9)
        gas_effect = (params['gas_level_pct'] - 75) * 0.2
        live_temp = ideal_temp + gas_effect + np.random.normal(0, 1.5, 50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_points, y=ideal_temp, mode='lines', name='Ideal Profile', line=dict(color='cyan', dash='dash')))
        fig.add_trace(go.Scatter(x=time_points, y=live_temp, mode='lines', name='Live Roast', line=dict(color='#FF8C00', width=3)))
        
        fig.add_vrect(x0=0, x1=duration*0.4, fillcolor="#FDB813", opacity=0.1, line_width=0, annotation_text="Drying")
        fig.add_vrect(x0=duration*0.4, x1=duration*0.75, fillcolor="#F68D2E", opacity=0.1, line_width=0, annotation_text="Maillard")
        fig.add_vrect(x0=duration*0.75, x1=duration, fillcolor="#D95319", opacity=0.1, line_width=0, annotation_text="Development")

        fig.update_layout(title="Roast Temperature vs. Time", xaxis_title="Time (minutes)", yaxis_title="Temperature (°C)", height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    def display_predictions(self, params):
        st.subheader("📊 AI Predictions")
        quality_score, anomaly_prob = self.predict(params)
        
        status = "✅ ON TRACK"
        status_class = "status-good"
        card_class = "metric-card"
        progress_color = "#28a745" # Green

        if anomaly_prob > 0.7:
            status = "🔥 CRITICAL ANOMALY"
            status_class = "status-danger"
            card_class = "metric-card metric-card-danger"
            progress_color = "#dc3545" # Red
        elif anomaly_prob > 0.4:
            status = "⚠️ WARNING"
            status_class = "status-warning"
            card_class = "metric-card metric-card-warning"
            progress_color = "#ffc107" # Yellow

        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.markdown(f"**System Status:** <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        c1.metric("Predicted Quality Score", f"{quality_score:.2f} / 10", delta=f"{quality_score-8.5:.2f} vs Target", delta_color="normal")
        c2.metric("Anomaly Risk", f"{anomaly_prob*100:.1f}%")
        
        st.progress(anomaly_prob)
        st.markdown(f"""
        <style>
            .stProgress > div > div > div > div {{
                background-color: {progress_color};
            }}
        </style>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        return quality_score, anomaly_prob

    def display_gemini_analysis(self, quality_score, anomaly_prob, params):
        st.subheader("AI Roastmaster's Analysis")
        with st.spinner("Generating AI analysis..."):
            analysis = self.get_gemini_analysis(quality_score, anomaly_prob, params)
            st.info(analysis)

    def display_process_summary(self, params):
        with st.expander("View Full Batch Configuration"):
            st.json({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()})
            
    def render_insights_tab(self):
        st.header("💡 Process Insights & Anomaly Drivers")
        st.markdown("The AI model has identified the key factors that most often contribute to a process anomaly. Monitoring these parameters closely is crucial for maintaining high quality.")
        
        importances = self.anomaly_model.feature_importances_
        selected_indices = self.feature_selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]
        
        imp_df = pd.DataFrame({'feature': selected_features, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        
        fig = go.Figure(go.Bar(x=imp_df['importance'], y=imp_df['feature'], orientation='h', marker_color='#D2B48C'))
        fig.update_layout(title="Top 10 Most Important Features for Anomaly Detection", yaxis={'categoryorder':'total ascending'}, height=500, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    def render_about_tab(self):
        st.header("ℹ️ About the Project")
        st.markdown("""
        This dashboard is a deliverable for the **Honeywell Hackathon**, designed to demonstrate a predictive maintenance and quality assurance system for the Food & Beverage industry, specifically focusing on coffee roasting.
        
        **Key Deliverables Addressed:**
        - **1. F&B Process Identification:** We have modeled the key stages of industrial coffee roasting: Drying, Maillard reaction, and Development.
        - **4. Final Product Quality Definition:** Quality is quantified through a composite score (`overall_quality_score`) based on aroma, body, acidity, and color, predicted by our model.
        - **5. Multivariable Prediction Model:** Two models were developed:
            - A **LightGBM Regressor** to predict the final quality score.
            - A **tuned LightGBM Classifier** to predict the probability of a process anomaly.
        - **6. Dashboard Visualization:** This interactive dashboard visualizes the real-time process (temperature profile) and the predicted quality and anomaly risk from the AI models.
        
        **Technology Stack:**
        - **Backend & ML:** Python, Pandas, Scikit-learn, LightGBM, Imbalanced-learn
        - **Frontend:** Streamlit, Plotly
        - **Generative AI:** Google Gemini Pro for human-readable analysis.
        """)

if __name__ == "__main__":
    models = load_models()
    if models:
        app = RoastApp(models)
        app.render()
