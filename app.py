import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Student Academic Performance System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Academic Performance Prediction & Analysis System")
st.caption("Python • Data Analysis • Machine Learning")


# ==============================
# LOAD CSV
# ==============================

@st.cache_data
def load_data():
    return pd.read_csv("student_data_100.csv")


df = load_data()


# ==============================
# MACHINE LEARNING MODEL
# ==============================

features = [
    "Attendance",
    "Study_Hours",
    "Assignment_Score",
    "Internal_Marks",
    "Previous_Sem_Marks"
]

model = LinearRegression()

model.fit(
    df[features],
    df["Final_Marks"]
)


# ==============================
# FUNCTIONS
# ==============================

def get_level(mark):

    if mark >= 85:
        return "Excellent"

    elif mark >= 75:
        return "Good"

    elif mark >= 60:
        return "Average"

    else:
        return "Needs Improvement"


def get_recommendation(student):

    advice = []

    if student["Attendance"] < 75:
        advice.append("Improve attendance")

    if student["Study_Hours"] < 3:
        advice.append("Increase study hours")

    if student["Assignment_Score"] < 70:
        advice.append("Focus on assignments")

    if student["Internal_Marks"] < 65:
        advice.append("Improve internal marks")

    if len(advice) == 0:
        return "Maintain the current study routine."

    return " and ".join(advice) + "."


# ==============================
# SIDEBAR
# ==============================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Performance Prediction",
        "Students Data",
        "Visualizations",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "B.Tech CSE (Data Science)\n\n"
    "Python Mini Project\n\n"
    "Dataset: 100 Students"
)


# ==============================
# DASHBOARD
# ==============================

if page == "Dashboard":

    st.header("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        len(df)
    )

    c2.metric(
        "Average Final Marks",
        f"{df['Final_Marks'].mean():.1f}"
    )

    c3.metric(
        "Average Attendance",
        f"{df['Attendance'].mean():.1f}%"
    )

    top_student = df.loc[
        df["Final_Marks"].idxmax(),
        "Student_Name"
    ]

    c4.metric(
        "Top Student",
        top_student
    )

    st.subheader("Performance Overview")

    col1, col2 = st.columns(2)

    with col1:

        top15 = df.sort_values(
            "Final_Marks",
            ascending=False
        ).head(15)

        fig = px.bar(
            top15,
            x="Student_Name",
            y="Final_Marks",
            title="Top 15 Students by Final Marks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            df,
            x="Attendance",
            y="Final_Marks",
            size="Study_Hours",
            hover_name="Student_Name",
            title="Attendance vs Final Marks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("📌 Quick Insights")

    st.write(
        f"• Average study time: "
        f"**{df['Study_Hours'].mean():.1f} hours/day**"
    )

    st.write(
        f"• Highest final marks: "
        f"**{df['Final_Marks'].max()}/100**"
    )

    st.write(
        f"• Lowest final marks: "
        f"**{df['Final_Marks'].min()}/100**"
    )

    st.write(
        "• Attendance, study hours, assignment marks "
        "and internal marks are useful indicators."
    )

    st.write(
        "• Linear Regression is used to predict expected final marks."
    )


# ==============================
# PERFORMANCE PREDICTION
# ==============================

elif page == "Performance Prediction":

    st.header("🔮 Performance Prediction")

    roll = st.selectbox(
        "Select Roll No.",
        df["Student_ID"].tolist()
    )

    student = df[
        df["Student_ID"] == roll
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:

        st.text_input(
            "Student Name",
            str(student["Student_Name"]),
            disabled=True
        )

        st.text_input(
            "Attendance (%)",
            str(student["Attendance"]),
            disabled=True
        )

        st.text_input(
            "Study Hours / Day",
            str(student["Study_Hours"]),
            disabled=True
        )

    with col2:

        st.text_input(
            "Assignment Score",
            str(student["Assignment_Score"]),
            disabled=True
        )

        st.text_input(
            "Internal Marks",
            str(student["Internal_Marks"]),
            disabled=True
        )

        st.text_input(
            "Previous Semester Marks",
            str(student["Previous_Sem_Marks"]),
            disabled=True
        )

    if st.button(
        "Predict Performance",
        type="primary"
    ):

        input_data = pd.DataFrame([{
            feature: student[feature]
            for feature in features
        }])

        prediction = float(
            model.predict(input_data)[0]
        )

        prediction = max(
            0,
            min(100, prediction)
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Predicted Final Marks",
            f"{prediction:.1f}/100"
        )

        r2.metric(
            "Performance Level",
            get_level(prediction)
        )

        r3.metric(
            "Actual Final Marks",
            f"{int(student['Final_Marks'])}/100"
        )

        st.success(
            "Recommendation: "
            + get_recommendation(student)
        )


# ==============================
# STUDENTS DATA
# ==============================

elif page == "Students Data":

    st.header("👨‍🎓 Students Data")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🏆 Top 10 Students")

    top10 = df.sort_values(
        "Final_Marks",
        ascending=False
    ).head(10)

    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )


# ==============================
# VISUALIZATIONS
# ==============================

elif page == "Visualizations":

    st.header("📈 Visualizations")

    chart = st.selectbox(
        "Choose Visualization",
        [
            "Attendance vs Final Marks",
            "Study Hours vs Final Marks",
            "Assignment Score vs Final Marks",
            "Internal Marks vs Final Marks",
            "Previous Semester Marks vs Final Marks",
            "Final Marks Distribution"
        ]
    )

    if chart == "Attendance vs Final Marks":

        fig = px.scatter(
            df,
            x="Attendance",
            y="Final_Marks",
            hover_name="Student_Name",
            title=chart
        )

    elif chart == "Study Hours vs Final Marks":

        fig = px.scatter(
            df,
            x="Study_Hours",
            y="Final_Marks",
            hover_name="Student_Name",
            title=chart
        )

    elif chart == "Assignment Score vs Final Marks":

        fig = px.scatter(
            df,
            x="Assignment_Score",
            y="Final_Marks",
            hover_name="Student_Name",
            title=chart
        )

    elif chart == "Internal Marks vs Final Marks":

        fig = px.scatter(
            df,
            x="Internal_Marks",
            y="Final_Marks",
            hover_name="Student_Name",
            title=chart
        )

    elif chart == "Previous Semester Marks vs Final Marks":

        fig = px.scatter(
            df,
            x="Previous_Sem_Marks",
            y="Final_Marks",
            hover_name="Student_Name",
            title=chart
        )

    else:

        fig = px.histogram(
            df,
            x="Final_Marks",
            nbins=10,
            title=chart
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==============================
# ABOUT PROJECT
# ==============================

else:

    st.header("ℹ️ About Project")

    st.markdown("""
### 🎯 Objective

Analyze student academic data and predict expected final performance.

### 🛠 Technologies

- Python
- Pandas
- Plotly
- Scikit-learn
- Streamlit
- CSV Dataset

### ⭐ Main Features

1. Student Data Management
2. Performance Analysis
3. Interactive Dashboard
4. Data Visualization
5. Final Marks Prediction
6. Personalized Recommendation

### 🤖 Machine Learning

Linear Regression is used for prediction.

### 📊 Input Features

- Attendance
- Study Hours
- Assignment Score
- Internal Marks
- Previous Semester Marks

### 📁 Dataset

The project uses 100 synthetic/demo student records.
""")
