import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load model and label encoder
model = joblib.load("iris_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Flower falling animation (short duration)
flower_animation = """
<style>
@keyframes fall {
  0% {transform: translateY(-10%); opacity:1;}
  100% {transform: translateY(75vh); opacity:0;}
}

.flower {
  position: fixed;
  top: -20px;
  font-size: 35px;
  animation: fall 5s linear 1;
}

.flower:nth-child(1){left:10%;}
.flower:nth-child(2){left:20%;}
.flower:nth-child(3){left:30%;}
.flower:nth-child(4){left:40%;}
.flower:nth-child(5){left:50%;}
.flower:nth-child(6){left:60%;}
.flower:nth-child(7){left:70%;}
.flower:nth-child(8){left:80%;}
.flower:nth-child(9){left:90%;}
</style>

<div class="flower">🌸</div>
<div class="flower">🌺</div>
<div class="flower">🌼</div>
<div class="flower">🌷</div>
<div class="flower">🌹</div>
<div class="flower">🌸</div>
<div class="flower">🌺</div>
<div class="flower">🌼</div>
<div class="flower">🌷</div>
"""

# Title
st.title("🌸 Iris Flower Classification App")

st.write("This application predicts the **species of an Iris flower** based on flower measurements.")

# Top flower images
st.subheader("🌼 Types of Iris Flowers")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
        caption="Setosa"
    )

with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        caption="Versicolor"
    )

with col3:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        caption="Virginica"
    )

st.divider()

# Sidebar inputs
st.sidebar.header("🌸 Enter Flower Measurements")

input_method = st.sidebar.radio(
    "Choose Input Method",
    ["Slider Input", "Keyboard Input"]
)

# Input fields
if input_method == "Keyboard Input":

    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

else:

    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.0)
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 1.0)

st.divider()

# Prediction
if st.button("🌸 Predict Species"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)

    species = label_encoder.inverse_transform(prediction)[0]

    # Show result
    st.success(f"🌼 Predicted Species: **{species}**")

    # Confetti effect
    st.snow()

    # Show flower image based on prediction
    if species.lower() == "setosa":

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
            caption="Iris Setosa",
            width=350
        )

    elif species.lower() == "versicolor":

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            caption="Iris Versicolor",
            width=350
        )

    elif species.lower() == "virginica":

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
            caption="Iris Virginica",
            width=350
        )

    # Show floating flowers briefly
    st.markdown(flower_animation, unsafe_allow_html=True)

# Footer
st.markdown("---")

st.markdown(
"""
<div style='text-align: center; color: gray;'>

🌸 Developed for Machine Learning Project  
Iris Flower Classification using Scikit-Learn  

© 2026

</div>
""",
unsafe_allow_html=True
)