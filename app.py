import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# ----------------- TITLE -----------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2e7d32;'>🌱 Crop Disease Detection System</h1>
    <p style='text-align:center; font-size:18px;'>
    Upload a leaf image to detect disease, severity, and treatment recommendations
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------- API CONFIG -----------------
API_URL = "http://127.0.0.1:8000/predict"

# ----------------- LAYOUT -----------------
left_col, right_col = st.columns([1, 2])

# ----------------- LEFT COLUMN (UPLOAD) -----------------
with left_col:
    st.subheader("📷 Upload Leaf Image")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    analyze = False

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        analyze = st.button("🔍 Analyze Leaf")

# ----------------- RIGHT COLUMN (RESULTS) -----------------
with right_col:
    if analyze and uploaded_file is not None:

        with st.spinner("Sending image to AI server..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(API_URL, files=files, timeout=30)

            except Exception as e:
                st.error(f"API connection error: {e}")
                st.stop()

        # ----------------- HANDLE RESPONSE -----------------
        if response.status_code != 200:
            st.error("❌ API error occurred")
            st.write(response.text)
            st.stop()

        result = response.json()

        # ----------------- EXTRACT DATA -----------------
        disease = result.get("disease", "N/A")
        confidence = result.get("confidence", 0)
        severity_percent = result.get("severity_percent", 0)
        severity_level = result.get("severity_level", "N/A")
        remedies = result.get("remedies", {})
        mask_b64 = result.get("mask", None)

        # ----------------- DISPLAY RESULTS -----------------
        st.success("✅ Analysis Complete")

        st.subheader("🧠 Analysis Summary")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🦠 Disease", disease)
        with m2:
            st.metric("📊 Confidence", f"{confidence}%")
        with m3:
            st.metric("⚠️ Severity", severity_level)

        st.progress(min(int(severity_percent), 100))
        st.caption(f"Estimated infected area: {severity_percent}%")

        # ----------------- MASK IMAGE -----------------
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask_image = Image.open(BytesIO(mask_bytes))

            st.subheader("🖼️ Infected Area Visualization")
            st.image(mask_image, width = 400)

        # ----------------- REMEDIES -----------------
        st.subheader("💊 Treatment Recommendations")

        if remedies:
            tab1, tab2 = st.tabs(["🌿 Organic", "🧪 Chemical"])

            with tab1:
                for r in remedies.get("organic", []):
                    st.write("•", r)

            with tab2:
                for r in remedies.get("chemical", []):
                    st.write("•", r)
        else:
            st.warning("No remedy information available for this disease.")
