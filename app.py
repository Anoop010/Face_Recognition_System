import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from utils import add_person, remove_person, train_model, recognize_face

DATASET_DIR = "dataset"
MODEL_PATH = "model/face_model.yml"

st.title("Face Recognition System")

# Sidebar menu
option = st.sidebar.selectbox("Choose an option", ["Add Person", "Remove Person", "Train Model", "Recognize Face"])

if option == "Add Person":
    st.header("Add New Person")
    name = st.text_input("Enter person's name:")

    if st.button("Capture Image"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            img_path = add_person(name, frame)
            st.image(frame, channels="BGR", caption=f"Image saved: {img_path}")
        else:
            st.error("Failed to capture image.")

elif option == "Remove Person":
    st.header("Remove Person from Dataset")
    if os.path.exists(DATASET_DIR):
        persons = os.listdir(DATASET_DIR)
        person_to_remove = st.selectbox("Select a person to remove:", persons)

        if st.button("Remove"):
            remove_person(person_to_remove)
            st.success(f"Removed {person_to_remove} from dataset.")
    else:
        st.error("Dataset folder does not exist!")

elif option == "Train Model":
    st.header("Train Face Recognition Model")
    if st.button("Start Training"):
        try:
            train_model()
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Recognize Face":
    st.header("Recognize Face")
    mode = st.radio("Select mode", ["Image", "Video", "Live Camera"])

    if mode == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            name, confidence, face_coords = recognize_face(frame)
            st.image(frame, channels="BGR", caption=f"Recognized: {name} (Confidence: {confidence:.2f})")

    elif mode == "Video":
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
        if video_file:
            st.video(video_file)

    elif mode == "Live Camera":
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            name, confidence, face_coords = recognize_face(frame)
            stframe.image(frame, channels="BGR", caption=f"Recognized: {name} (Confidence: {confidence:.2f})")

        cap.release()
