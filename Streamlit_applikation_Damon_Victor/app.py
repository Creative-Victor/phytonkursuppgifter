import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image


# Ladda den sparade modellen
model = keras.models.load_model("mnist_model.keras")

# Skapa appens titel
st.title("🔢 Välkommen till Victors & Damons guessbot, nu ska du rita en siffra och låta vår guessbot gissa rätta siffran! 🔢")
st.image ("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZnA1eDd6ejlyY21nYXk4cGJjdGZjajBqZWl5YWFreG5ocmVrNzUzMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/OXvOMPcbm4sb7oBfKf/giphy.gif")
st.write("🖌️ Ladda upp en bild på en siffra (0-9) så gissar vår guessbot vilken det är!")

# Ladda upp en bild från användaren
uploaded_file = st.file_uploader("Ladda upp en bild", type=["png", "gif", "jpg", "jpeg"])

# Knapp
if st.button("🔄 Ta bort denna bild"):
    uploaded_file = None


if uploaded_file is not None:
    # Öppna bilden och visa den
    image = Image.open(uploaded_file).convert("L")  # Gör bilden svartvit
    st.image(image, caption="Din uppladdade siffra", use_column_width=True)

    # Förbehandla bilden så den matchar modellen
    image = image.resize((28, 28))  # Ändra storlek till 28x28 pixlar
    image = np.array(image)  # Gör om till en array
    image = image / 255.0  # Normalisera (gör alla värden mellan 0 och 1)
    image = image.reshape(1, 28, 28, 1)  # Gör den redo för modellen

    # AI gissar siffran
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    # Visa resultatet
    st.write(f"🤖 AI tror att siffran är: **{predicted_digit}**")
    
    
