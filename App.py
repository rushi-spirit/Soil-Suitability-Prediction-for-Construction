import streamlit as st
import numpy as np
import pickle

# Load the model and encoders
model = pickle.load(open('soil_model.pkl', 'rb'))
scaler = pickle.load(open('soil_scaler.pkl', 'rb'))
label_encoder = pickle.load(open('soil_encoder.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Soil Suitability App", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
         .title {
            text-align: center;
            font-size: 40px;
            color: #FF6347;
        }
       .subtitle {
            font-size:20px;
            text-align:center;
            color:#666666;
        }
        .result-box {
            background-color:#f1f1f1;
            padding:20px;
            border-radius:10px;
            text-align:center;
            margin-top:20px;
        }
        .suitable {
            color:green;
            font-size:24px;
            font-weight:bold;
        }
        .not-suitable {
            color:red;
            font-size:24px;
            font-weight:bold;
        }
        .emoji {
            font-size:50px;
        }
        .stApp {
            background-image: url('https://i.pinimg.com/736x/f4/80/f1/f480f1c58a5132eb514809c1b8d34d28.jpg');
            background-size: cover;
            background-position: center;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="title">🏗️ Soil Suitability Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter soil details to check construction suitability</div>', unsafe_allow_html=True)
st.markdown("---")

# User input
ph = st.slider("Select Soil pH", 0.0, 14.0, 7.0, 0.1)
moisture = st.slider("Select Moisture %", 0.0, 100.0, 30.0, 0.1)
soil_type = st.selectbox("Choose Soil Type", ["Sandy", "Silty", "Peaty", "Loamy"])

if st.button("Predict"):
    try:
        # Encoding the soil type using the pre-trained label encoder
        soil_encoded = label_encoder.transform([soil_type])[0]

        # Prepare input data for prediction
        input_data = np.array([[ph, moisture, soil_encoded]])
        input_scaled = scaler.transform(input_data)  # Scale the input data

        # Make prediction using the trained model
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            # Suitable message
            st.markdown("""
                <div class="result-box">
                    <div class="emoji">🎉</div>
                    <div class="suitable">The soil is Suitable for Construction!</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Not suitable message
            st.markdown("""
                <div class="result-box">
                    <div class="emoji">😞</div>
                    <div class="not-suitable">The soil is Not Suitable for Construction.</div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error occurred: {e}")


with st.expander("📘 Soil Suitability Info"):
    st.markdown("""
        <style>
        .info-table th, .info-table td {
            padding: 10px;
            border: 1px solid #ccc;
            text-align: center;
        }
        .info-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }
        </style>

        ### 🧪 **Soil Types and Their Construction Suitability**

        <table class='info-table'>
            <tr>
                <th><b>Soil Type</b></th>
                <th><b>Properties</b></th>
                <th><b>Ideal pH Range</b></th>
                <th><b>Ideal Moisture %</b></th>
                <th><b>Suitability</b></th>
            </tr>
            <tr>
                <td>Sandy</td>
                <td>Loose, drains quickly</td>
                <td>6.0 – 7.5</td>
                <td>10 – 20%</td>
                <td>✅ Good (with compaction)</td>
            </tr>
            <tr>
                <td>Silty</td>
                <td>Smooth, retains water</td>
                <td>6.0 – 7.5</td>
                <td>20 – 40%</td>
                <td>⚠️ Moderate (needs drainage)</td>
            </tr>
            <tr>
                <td>Peaty</td>
                <td>High organic matter, soft</td>
                <td>4.0 – 5.5</td>
                <td>40 – 60%</td>
                <td>❌ Poor (too soft for building)</td>
            </tr>
            <tr>
                <td>Loamy</td>
                <td>Balanced texture, fertile</td>
                <td>6.0 – 7.0</td>
                <td>15 – 30%</td>
                <td>✅ Excellent</td>
            </tr>
        </table>

        ---

        ### 📊 **Ideal pH & Moisture for Construction Suitability**
        - **pH Range:**  
          - Ideal: **6.0 – 7.5**
          - Too acidic (< 5.5) or too alkaline (> 8) weakens soil structure.

        - **Moisture %:**  
          - Optimal: **10 – 30%**
          - Very wet soils (> 40%) may be unstable.

        ---

        

        ### 🏗️ **Construction Tips**
        - Use **pH meters** and **moisture sensors** before building.
        - **Sandy and Loamy soils** are usually the safest for construction.
    """, unsafe_allow_html=True)