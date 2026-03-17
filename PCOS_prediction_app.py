import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="PCOS Diagnostic Hub", page_icon="🌺", layout="wide")

# ==========================================
# 2. LOAD AI ENGINE
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('deployed_models/pcos_rf_model.pkl')
    scaler = joblib.load('deployed_models/pcos_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
    expected_features = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading models. Did you save the .pkl files? Details: {e}")
    st.stop()

# ==========================================
# 3. HEADER & INTRODUCTION
# ==========================================
st.title("🌺 PCOS Awareness & Risk Assessment Hub")
st.markdown("Welcome. This tool uses data science to analyze your symptoms against thousands of clinical records. **Please note:** This is a screening tool, not a medical diagnosis.")
st.divider()

# ==========================================
# 4. MAIN LAYOUT: TWO COLUMNS
# ==========================================
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📝 Step 1: Enter Your Details")
    
    with st.expander("🩺 **Ultrasound & Endocrine Data**", expanded=True):
        sub1, sub2 = st.columns(2)
        with sub1:
            follicle_r = st.slider("Right Ovary Follicles [Normal: 4 - 11]", 0, 30, 5)
            follicle_l = st.slider("Left Ovary Follicles [Normal: 4 - 11]", 0, 30, 5)
        with sub2:
            amh = st.number_input("AMH Level (ng/mL) [Normal: 1.5 - 4.0]", 0.0, 20.0, 2.5)
            cycle_length = st.slider("Cycle Length (Days) [Normal: 21 - 35]", 20, 50, 28)

    with st.expander("⚖️ **Body Metrics & Lifestyle**", expanded=True):
        sub3, sub4 = st.columns(2)
        with sub3:
            age = st.number_input("Age (Years)", 12, 60, 25)
            weight = st.number_input("Weight (Kg)", 30.0, 150.0, 65.0)
            bmi = st.number_input("BMI [Normal: 18.5 - 24.9]", 10.0, 50.0, 22.0)
        with sub4:
            weight_gain = st.selectbox("Unusual Weight Gain?", ["No", "Yes"])
            skin_darkening = st.selectbox("Skin Darkening (Neck/Armpits)?", ["No", "Yes"])
            fast_food = st.selectbox("High Fast Food Intake?", ["No", "Yes"])

yes_no_map = {"Yes": 1, "No": 0}

with col2:
    st.subheader("📊 Step 2: Your Assessment")
    
    if st.button("Calculate My Risk Profile", type="primary", use_container_width=True):
        with st.spinner("Analyzing data patterns..."):
            
            # Map Inputs (The Zero-Trap happens here, but we will adjust the threshold below to compensate)
            input_data = {feature: 0 for feature in expected_features} 
            input_data["Follicle No. (R)"] = follicle_r
            input_data["Follicle No. (L)"] = follicle_l
            input_data["Cycle length(days)"] = cycle_length
            input_data["AMH(ng/mL)"] = amh
            input_data["Weight gain(Y/N)"] = yes_no_map[weight_gain]
            input_data["Skin darkening (Y/N)"] = yes_no_map[skin_darkening]
            input_data["Fast food (Y/N)"] = yes_no_map[fast_food]
            input_data["Weight (Kg)"] = weight
            input_data["BMI"] = bmi
            input_data["Age (yrs)"] = age
            
            # Predict
            input_df = pd.DataFrame([input_data])[expected_features]
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1] * 100
            
            # Draw an interactive Plotly Gauge Chart with NEW MEDICAL THRESHOLDS
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability,
                title = {'text': "PCOS Risk Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [0, 25], 'color': "#a8e6cf"},   # Green (Strict Normal)
                        {'range': [25, 40], 'color': "#ffd3b6"},  # Orange (Warning)
                        {'range': [40, 100], 'color': "#ff8b94"}  # Red (Clinical High Risk)
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': probability
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)

            # --- UPDATED MEDICAL THRESHOLD (40%) ---
            if prediction == 1 or probability >= 40:
                st.error("🚨 **Elevated Risk Detected**")
                st.write("Your symptoms share a strong statistical overlap with PCOS profiles. Note: Clinical screening models are calibrated to flag risks above a 40% probability threshold.")
                
                st.markdown("### 👩‍⚕️ Recommended Next Steps")
                
                st.markdown("""
                <div style="background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b; margin-bottom: 12px;">
                    <h4 style="margin:0; color: #ff8b8b;">🩺 1. Consult a Professional</h4>
                    <p style="margin:5px 0 0 0; font-size: 15px; color: #e0e0e0;">Please schedule an appointment with a gynecologist or endocrinologist. Show them these specific numbers.</p>
                </div>
                
                <div style="background-color: rgba(255, 165, 0, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #ffa500; margin-bottom: 12px;">
                    <h4 style="margin:0; color: #ffd180;">🥗 2. Dietary Adjustments</h4>
                    <p style="margin:5px 0 0 0; font-size: 15px; color: #e0e0e0;">Focus on whole foods and a low-glycemic index diet to manage insulin levels.</p>
                </div>
                
                <div style="background-color: rgba(0, 160, 240, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00a0f0; margin-bottom: 12px;">
                    <h4 style="margin:0; color: #80d4ff;">🏃‍♀️ 3. Active Lifestyle</h4>
                    <p style="margin:5px 0 0 0; font-size: 15px; color: #e0e0e0;">Regular, moderate exercise (like brisk walking or yoga) can significantly improve metabolic health.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.success("✅ **Low Risk Detected**")
                st.balloons() 
                st.write("Your symptoms fall within normal baseline parameters.")
                
                st.markdown("### 🌿 Wellness Advice")
                
                st.markdown("""
                <div style="background-color: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 12px;">
                    <h4 style="margin:0; color: #88d49e;">🥑 1. Maintain Routine</h4>
                    <p style="margin:5px 0 0 0; font-size: 15px; color: #e0e0e0;">Continue your healthy habits, balanced diet, and active lifestyle.</p>
                </div>
                
                <div style="background-color: rgba(0, 160, 240, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #00a0f0; margin-bottom: 12px;">
                    <h4 style="margin:0; color: #80d4ff;">🏥 2. Routine Check-ups</h4>
                    <p style="margin:5px 0 0 0; font-size: 15px; color: #e0e0e0;">Ensure you attend your annual physicals and gynecological exams.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill out the details on the left and click the button to generate your personalized interactive report and wellness advice.")