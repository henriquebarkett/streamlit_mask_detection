import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import torch
from torchvision.transforms import ToTensor

# Carregando o modelo
classificador = torch.load('mask_detection_model.pth')

#Detecção da face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Erro ao iniciar detecção de face.")

categorias = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'} # Dicionário com as categorias e seus índices correspondentes

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class FaceMask(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convertendo a imagem para escala de cinza
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)       

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                # Pré-processar a ROI e converter para tensor
                roi = roi_gray.astype('float') / 255.0
                roi = ToTensor()(roi).unsqueeze(0)

                # Realizar a inferência utilizando o modelo PyTorch
                with torch.no_grad():
                    prediction = classificador(roi)
                    maxindex = int(torch.argmax(prediction))
                    finalout = categorias[maxindex]
                    output = str(finalout)

            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    activiteis = ["Home", "Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Desenvolvido por Amon Menezes Negreiros   
            Email : menezes.am031@gmal.com 
            [LinkedIn] (https://www.linkedin.com/in/amonmenezesnegreiros/)""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Mask detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                The application has two functionalities.

                1. Real time face mask detection using web cam feed.

                2. Real time face mask recognization.

                """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect if you using mask or no")
        webrtc_streamer(key="example", video_transformer_factory=FaceMask)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face mask detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                                    <div style="background-color:#98AFC7;padding:10px">
                                    <h4 style="color:white;text-align:center;">This Application is developed by Amon Menezes Negreiros using Streamlit Framework, Opencv, PyTorch library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or want to comment just write a mail at menezes.am031@gmail.com. </h4>
                                    <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                    </div>
                                    <br></br>
                                    <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()