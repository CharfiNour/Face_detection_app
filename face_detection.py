import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Lumiere\Desktop\Data projects\face detection\haarcascade_frontalface_default.xml'
)

def app():
    st.title("Face Detection App")
    st.write("Press the button below to start detecting faces from your webcam")

    if "run" not in st.session_state:
        st.session_state.run = False

    col1, col2 = st.columns(2)
    if col1.button("Detect Faces"):
        st.session_state.run = True
    if col2.button("Stop"):
        st.session_state.run = False

    FRAME_WINDOW = st.image([])

    if st.session_state.run:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

if __name__ == "__main__":
    app()    