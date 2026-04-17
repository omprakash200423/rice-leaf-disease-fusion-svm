import streamlit as st
import numpy as np
import joblib
from PIL import Image

# 🌿 Title
st.markdown("""
<h1 style='text-align: center; color: #22c55e;'>
 Rice Leaf Disease Detection
</h1>
""", unsafe_allow_html=True)

# Load model
svm = joblib.load("svm_model.pkl")

CLASSES = ["Brown Spot", "Healthy", "Leaf Blast", "Leaf Scald", "Sheath Blight"]

# 📤 Upload
uploaded_file = st.file_uploader("📤 Upload rice leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Analyze Leaf"):

        with st.spinner("Analyzing... ⏳"):

            # 🔥 FIX: match feature size (simulate training input size)
            img = image.resize((224, 224))
            img = np.array(img).flatten().reshape(1, -1)

            # IMPORTANT: remove scaler if mismatch happens
            try:
                scaler = joblib.load("scaler.pkl")
                img = scaler.transform(img)
            except:
                pass

            pred = svm.predict(img)[0]
            predicted_class = CLASSES[pred]

        # 🌿 Result
        st.markdown(f"### 🌿 Disease Detected: **{predicted_class}**")

        st.markdown("---")

        # 🧪 Recommendations
        recommendations = {
            "Brown Spot": {
                "DO": "Spray Mancozeb, maintain proper nutrition",
                "DONT": "Avoid dry soil conditions",
                "Action": "Spray within 5 days"
            },
            "Leaf Blast": {
                "DO": "Use Tricyclazole fungicide",
                "DONT": "Avoid excess nitrogen fertilizers",
                "Action": "Immediate spraying required"
            },
            "Leaf Scald": {
                "DO": "Apply Carbendazim",
                "DONT": "Do not reuse infected seeds",
                "Action": "Spray within 3 days"
            },
            "Sheath Blight": {
                "DO": "Apply Validamycin",
                "DONT": "Avoid dense planting",
                "Action": "Immediate action needed"
            }
        }

        if predicted_class in recommendations:
            rec = recommendations[predicted_class]

            st.markdown("## 🧪 Recommendations")

            st.markdown(f"""
            <div style="
                background-color:#1e293b;
                padding:15px;
                border-radius:10px;
                border-left:5px solid #22c55e;
            ">
            <b>✅ DO:</b> {rec['DO']} <br><br>
            <b>❌ DON'T:</b> {rec['DONT']} <br><br>
            <b>⏱ Action:</b> {rec['Action']}
            </div>
            """, unsafe_allow_html=True)