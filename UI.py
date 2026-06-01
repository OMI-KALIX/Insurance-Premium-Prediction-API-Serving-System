import streamlit as st
import requests
import base64

# PAGE CONFIG

st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="centered"
)

# API URL

API_URL = "https://insurance-premium-prediction-api-serving.onrender.com/predict"

# BACKGROUND + CUSTOM CSS

def add_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
    f"""
    <style>

    /* Hide Streamlit Header */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Hide Menu */
    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    /* Background */
    .stApp {{
        background-image:
            linear-gradient(
                rgba(0,0,0,0.45),
                rgba(0,0,0,0.45)
            ),
            url("data:image/png;base64,{encoded}");

        background-size: 100% auto;
        background-position: top center;
        background-repeat: no-repeat;
        background-attachment: scroll;
    }}

    /* Remove Streamlit Top Padding */
    .block-container {{
        padding-top: 0rem !important;
    }}

    /* Main Form Container */
    .main .block-container {{
        margin-top: 220px;

        background: rgba(0,0,0,0.30);
        backdrop-filter: blur(12px);

        padding: 2rem;
        border-radius: 20px;

        border: 1px solid rgba(255,255,255,0.15);
    }}

    /* Title */
    h1 {{
        color: white !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;

        text-shadow:
            2px 2px 10px rgba(0,0,0,0.8);
    }}

    /* Paragraph */
    p {{
        color: white !important;
        text-align: center;
    }}

    /* Labels */
    label {{
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }}

    /* Number Input */
    .stNumberInput input {{
        background: rgba(15,15,25,0.90) !important;
        color: white !important;
        border-radius: 12px !important;
    }}

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] {{
        background: rgba(15,15,25,0.90) !important;
        color: white !important;
        border-radius: 12px !important;
    }}

    /* Button */
    .stButton button {{
        width: 100%;
        height: 55px;

        border: none;
        border-radius: 12px;

        font-size: 18px;
        font-weight: bold;

        color: white;

        background: linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );
    }}

    .stButton button:hover {{
        transform: scale(1.02);
        transition: 0.3s;
    }}

    /* Warning */
    div[data-baseweb="notification"] {{
        border-radius: 12px !important;
    }}

    /* Success */
    .stSuccess {{
        border-radius: 12px !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: #2563eb;
        border-radius: 10px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)
add_bg("background_image.png")

# HEADER

st.markdown(
    """
    <h1>🏥 Insurance Premium Prediction</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center;
              font-size:18px;
              color:white;
              margin-bottom:20px;'>

    Predict insurance premium categories using Machine Learning and Risk Analysis.

    </p>
    """,
    unsafe_allow_html=True
)

st.warning(
    "This application uses an AI model to predict insurance premium categories. "
    "Please provide accurate information for the most reliable results."
)

# INPUTS


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income_lpa = st.number_input(
    "Income (LPA)",
    min_value=0.0,
    value=5.0
)

occupation = st.selectbox(
    "Occupation",
    [
        'Factory Worker', 'Businessman', 'Sales Manager', 'Banker',
        'Marketing Manager', 'Insurance Agent', 'HR Manager',
        'Pharmacist', 'Teacher', 'Software Engineer', 'Consultant',
        'Driver', 'Shop Owner', 'Nurse', 'Accountant',
        'Government Employee', 'Architect', 'Engineer',
        'Real Estate Agent', 'Civil Servant', 'Plumber',
        'Retail Manager', 'Chef', 'Electrician',
        'Carpenter', 'Doctor', 'Lab Technician',
        'Data Analyst', 'Lawyer', 'Content Writer'
    ]
)

smoker = st.selectbox(
    "Smoker",
    [0, 1]
)

city = st.selectbox(
    "City",
    [
        'Agra', 'Allahabad', 'Srinagar', 'Meerut', 'Varanasi',
        'Hyderabad', 'Ahmedabad', 'Chennai', 'Amritsar',
        'Vadodara', 'Lucknow', 'Bhopal', 'Mumbai',
        'Ludhiana', 'Rajkot', 'Surat', 'Ghaziabad',
        'Ranchi', 'Pune', 'Kanpur', 'Nagpur',
        'Faridabad', 'Bangalore', 'Jaipur', 'Kolkata',
        'Nashik', 'Patna', 'Indore', 'Delhi',
        'Visakhapatnam'
    ]
)

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0
)

height = st.number_input(
    "Height (m)",
    min_value=1.0,
    max_value=2.5,
    value=1.75
)

# PREDICTION

if st.button("Predict Insurance Premium Category"):

    with st.spinner("Analyzing customer profile..."):

        try:

            response = requests.post(
                API_URL,
                json={
                    "age": age,
                    "weight": weight,
                    "height": height,
                    "income_lpa": income_lpa,
                    "smoker": smoker,
                    "city": city,
                    "occupation": occupation
                }
            )

            if response.status_code == 200:

                prediction = response.json().get(
                    "predicted_insurance_premium_category"
                )

            else:
                st.error(
                    "Unable to get prediction from API."
                )

            # PREDICTION RESULT
            # =========================

            if prediction == "Low":
                st.success("✅ Low Premium Category")
                st.image("images/low.png", use_container_width=True)

            elif prediction == "Medium":
                st.warning("⚠️ Medium Premium Category")
                st.image("images/medium.png", use_container_width=True)

            elif prediction == "High":
                st.error("🚨 High Premium Category")
                st.image("images/high.png", use_container_width=True)

            # =========================
            # GENERATE RECOMMENDATIONS
            # =========================

            bmi = weight / (height ** 2)

            tips = []

            # Smoking
            if smoker == 1:
                tips.append("🚭 Consider quitting smoking to significantly reduce health risks and insurance costs.")
            else:
                tips.append("✅ Great! Being a non-smoker lowers your overall health risk.")

            # BMI Analysis
            if bmi < 18.5:
                tips.append("🥗 Consider a balanced diet and nutrient-rich meals to reach a healthy BMI.")
            elif bmi <= 24.9:
                tips.append("💪 Your BMI is within the healthy range. Keep maintaining your lifestyle.")
            elif bmi <= 29.9:
                tips.append("🏃 Regular exercise and healthy eating can help improve your BMI.")
            else:
                tips.append("⚠️ Obesity increases health risks. Consider consulting a healthcare professional.")

            # Age Analysis
            if age < 30:
                tips.append("🌱 Building healthy habits now can significantly reduce future health risks.")
            elif age < 50:
                tips.append("🩺 Schedule annual preventive health checkups.")
            else:
                tips.append("🩺 Regular health screenings and preventive care are highly recommended.")

            # Income Analysis
            if income_lpa < 10:
                tips.append("💰 Build an emergency healthcare fund for unexpected medical expenses.")
            elif income_lpa < 25:
                tips.append("📈 Consider increasing your insurance coverage as your income grows.")
            else:
                tips.append("🛡️ Comprehensive insurance plans may provide better long-term protection.")

            # Weight Analysis
            if weight > 90:
                tips.append("⚖️ Maintaining a healthy weight can lower future health risks.")

            # Occupation Analysis
            sedentary_jobs = [
                "Software Engineer",
                "Data Analyst",
                "Accountant",
                "Content Writer",
                "Banker"
            ]

            if occupation in sedentary_jobs:
                tips.append("🪑 Take regular breaks, stretch often, and stay physically active.")

            # Premium Category Recommendations
            if prediction == "Low":
                tips.extend([
                    "🎉 Your profile indicates lower insurance risk.",
                    "💵 Consider investing premium savings into long-term financial goals.",
                    "🏃 Continue maintaining a healthy lifestyle.",
                    "🩺 Keep up with regular health checkups."
                ])

            elif prediction == "Medium":
                tips.extend([
                    "⚖️ Small lifestyle improvements may help reduce future premiums.",
                    "🏃 Aim for at least 150 minutes of exercise per week.",
                    "🥗 Focus on balanced nutrition and weight management.",
                    "📋 Review your insurance coverage annually."
                ])

            elif prediction == "High":
                tips.extend([
                    "🚨 Focus on preventive healthcare and routine medical checkups.",
                    "❤️ Prioritize fitness, nutrition, and risk-factor management.",
                    "🛡️ Consider comprehensive health insurance coverage.",
                    "📉 Lifestyle improvements may help reduce future insurance costs.",
                    "👨‍⚕️ Consult healthcare professionals for personalized guidance."
                ])

            # =========================
            # DISPLAY RECOMMENDATIONS
            # =========================

            st.markdown("---")
            st.subheader("💡 Personalized Recommendations")

            for i, tip in enumerate(tips, start=1):
                st.info(f"{i}. {tip}")    

        except Exception as e:
            st.error(f"Connection Error: {e}")

st.markdown(
    "<div style='text-align:center; color:#d1d5db; font-size:14px; margin-top:30px;'>© 2025 OMI-KALIX | Insurance Premium Prediction API & Serving System</div>",
    unsafe_allow_html=True
)
