import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Heart Disease App", layout="wide")


# ----------------------------
# BACKGROUND FUNCTION
# ----------------------------
def add_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.6);
            z-index: -1;
        }}

        label {{
            color: yellow !important;
            font-weight: bold !important;
        }}

        h1, h2, h3 {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# SESSION STATE
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# USER DATABASE FILE
# ----------------------------
import json
import os

USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

users = load_users()

# ----------------------------
# LOGIN PAGE
# ----------------------------
if not st.session_state.logged_in:

    add_bg("https://images.unsplash.com/photo-1505751172876-fa1923c5c528")

    st.markdown("<h1 style='text-align:center;'>🔐 Heart Disease Prediction</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        option = st.radio("Select Option", ["Login", "Create Account"])

        # ---------------- LOGIN ----------------
        if option == "Login":

            st.markdown("<h3>Login</h3>", unsafe_allow_html=True)

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                # Admin Login
                if username == "admin" and password == "1234":
                    st.session_state.logged_in = True
                    st.success("Admin Login Successful ✅")
                    st.rerun()

                # User Login
                elif username in users and users[username] == password:
                    st.session_state.logged_in = True
                    st.success("User Login Successful ✅")
                    st.rerun()

                else:
                    st.error("Invalid Username or Password ❌")

        # ---------------- REGISTER ----------------
        if option == "Create Account":

            st.markdown("<h3>Create Account</h3>", unsafe_allow_html=True)

            new_user = st.text_input("Create Username")
            new_pass = st.text_input("Create Password", type="password")

            if st.button("Register"):
                if new_user in users or new_user == "admin":
                    st.warning("Username already exists ⚠️")
                elif new_user == "" or new_pass == "":
                    st.warning("Please enter valid details")
                else:
                    users[new_user] = new_pass
                    save_users(users)
                    st.success("Account Created Successfully ✅")
                    st.info("Now go to Login and sign in.")

    st.stop()


# ----------------------------
# MAIN PAGE
# ----------------------------
add_bg("https://images.unsplash.com/photo-1576091160550-2173dba999ef")

st.sidebar.title("Navigation")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

option = st.sidebar.radio("Go to", [
    "Dashboard",
    "Prediction",
    "🏥 Recommended Hospitals",
    "🧘 Yoga for Heart"
])

st.title("❤️ Heart Disease Prediction App")


# ----------------------------
# LOAD DATASET
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

try:
    df = load_data()
except:
    st.error("heart.csv file not found in folder.")
    st.stop()


# ----------------------------
# DASHBOARD
# ----------------------------
if option == "Dashboard":

    st.header("📊 Data Dashboard")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Missing Values")
    st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.countplot(x=df['target'], ax=ax)
    st.pyplot(fig)


# ----------------------------
# MODEL TRAINING
# ----------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))





# ----------------------------
# PREDICTION PAGE
# ----------------------------
if option == "Prediction":

    st.header("🔍 Heart Disease Prediction")
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")

    input_data = []

    for column in X.columns:
        value = st.number_input(f"Enter {column}", value=0.0)
        input_data.append(value)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
            st.info("👉 Please check Recommended Hospitals and Yoga page.")

        else:
            st.success("✅ No Heart Disease")


# ----------------------------
# HOSPITAL PAGE
# ----------------------------
if option == "🏥 Recommended Hospitals":

    st.header("🏥 Best Heart Care Hospitals in Chennai")

    hospitals = [
        {"name": "Apollo Hospitals", "speciality": "Advanced Cardiology", "contact": "+91-44-2829 3333"},
        {"name": "Fortis Malar Hospital", "speciality": "Cardiac & Emergency Care", "contact": "+91-44-4289 2222"},
        {"name": "MIOT International", "speciality": "Heart & Vascular Institute", "contact": "+91-44-4200 2288"},
        {"name": "Royal Care", "speciality": "Heart & Emergency Care", "contact": "+91-44-2489-5481"},
        {"name": "BM Birla Heart Hospital", "speciality": "Interventional Cardiology", "contact": "+91-22-6698-6666"}
    ]

    for hospital in hospitals:
        st.subheader("🏥 " + hospital["name"])
        st.write("❤️ Speciality:", hospital["speciality"])
        st.write("📞 Contact:", hospital["contact"])
        st.markdown("---")


# ----------------------------
# YOGA PAGE
# ----------------------------
if option == "🧘 Yoga for Heart":

    st.header("🧘 Yoga for Heart Health")

    yoga_list = [
        {
            "name": "Bhujangasana (Cobra Pose)",
            "image": "data:image/jpeg;base64,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",
            "steps": ["Lie on stomach", "Place palms under shoulders", "Lift chest", "Hold 20 sec"]
        },
        {
            "name": "Tadasana (Mountain Pose)",
            "image": "data:image/jpeg;base64,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",
            "steps": ["Stand straight", "Raise arms", "Stretch body", "Hold 20 sec"]
        },
        {
            "name": "Setu Bandhasana (Bridge Pose)",
            "image": "data:image/jpeg;base64,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",
            "steps": ["Lie on back", "Bend knees", "Lift hips", "Hold 15 sec"]
        },
        {
            "name": "Vrikshasana (Tree Pose)",
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4phUYxhF7GmYmB1HpH-5Jrmhxz2ZHjjIefg&s",
            "steps": ["Stand straight", "Place foot on thigh", "Join palms", "Hold 20 sec"]
        },
        {
            "name": "Shavasana (Relaxation Pose)",
            "image": "data:image/jpeg;base64,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",
            "steps": ["Lie flat", "Relax body", "Close eyes", "Breathe normally 5 min"]
        }
    ]

    for yoga in yoga_list:
        st.subheader(yoga["name"])
        st.image(yoga["image"], width=400)
        for step in yoga["steps"]:
            st.write("•", step)
        st.markdown("---")

        # ----------------------------
# HEART HEALTH GUIDE PAGE
# ----------------------------
elif option == "Heart Health Guide":

    st.title("💚 Complete Heart Health Guide")

    st.image("https://images.unsplash.com/photo-1505751172876-fa1923c5c528", 
             use_container_width=True)

    # Daily Habits
    st.header("🕒 Daily Healthy Habits")
    st.write("Follow these habits every day to protect your heart:")
    st.markdown("""
    - 🚶 Walk at least 30 minutes daily
    - 😴 Sleep 7–8 hours
    - 🚭 Avoid smoking & alcohol
    - ⚖ Maintain healthy weight
    - 🩺 Check blood pressure regularly
    """)

    # Food
    st.header("🥗 Food Recommendations")
    st.write("Eat healthy foods for better heart function:")
    st.markdown("""
    ✅ Recommended:
    - Vegetables & Fruits
    - Whole grains
    - Nuts & Seeds
    - Fish rich in Omega-3

    ❌ Avoid:
    - Junk food
    - Fried food
    - Sugary drinks
    """)

    # Exercise
    st.header("🏃 Exercise Suggestions")
    st.markdown("""
    - Brisk walking
    - Yoga & breathing exercises
    - Cycling
    - Swimming
    - Light jogging
    """)

    # Water
    st.header("💧 Water Intake Tips")
    st.markdown("""
    - Drink 2–3 liters daily
    - Start your day with warm water
    - Avoid sugary beverages
    """)

    # Stress
    st.header("🧘 Stress Management")
    st.markdown("""
    - Practice meditation
    - Deep breathing exercises
    - Spend time with family
    - Listen to calming music
    """)

    st.success("💡 Healthy lifestyle reduces heart disease risk significantly.")
# ----------------------------
# HEALTH TIPS PAGE
# ----------------------------
if option == "📈 Health Tips":

    st.header("📈 Heart Disease Prevention & Health Tips")

    st.subheader("🥗 1. Healthy Diet")
    st.write("""
    • Eat more fruits and vegetables  
    • Reduce salt intake  
    • Avoid oily and junk food  
    • Choose whole grains  
    • Drink plenty of water  
    """)

    st.subheader("🏃 2. Regular Exercise")
    st.write("""
    • Walk 30 minutes daily  
    • Do light jogging  
    • Practice yoga  
    • Maintain healthy weight  
    • Avoid sitting long hours  
    """)

    st.subheader("🚭 3. Avoid Smoking & Alcohol")
    st.write("""
    • Stop smoking  
    • Limit alcohol consumption  
    • Avoid passive smoking  
    """)

    st.subheader("🧘 4. Stress Management")
    st.write("""
    • Practice meditation  
    • Sleep 7–8 hours daily  
    • Take short breaks  
    • Stay positive  
    """)

    st.subheader("🩺 5. Regular Checkups")
    st.write("""
    • Monitor blood pressure  
    • Check cholesterol levels  
    • Regular sugar test  
    • Consult doctor yearly  
    """)

    st.success("💚 Prevention is better than cure!")