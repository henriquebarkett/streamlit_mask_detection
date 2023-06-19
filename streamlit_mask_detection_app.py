import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import torch
from torchvision import transforms
from cnn_classifier import Classifier


categorias = {0: 'incorrect_mask', 1: 'with_mask', 2: 'without_mask'}  # Dicionário com as categorias e seus índices correspondentes

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def home():
    st.title('Boas vindas ao Face Mask Detection')
    st.subheader("Uma aplicação para detecção de máscara facial em tempo real feita com OpenCV, PyTorch e Streamlit.")
    st.write(
        '''
            No menu a esquerda você irá encontrar as principais funcionalidades dessa aplicação.
            1. Detecção do uso de máscara facial em tempo real usando webcam.
            2. Classificação do uso de mascara facial através de imagens enviadas pelo usuário.
        '''
    )


def image_classification():
    pre_trained_weights = torch.load('mask_detection_model_stdict.pt')  # Carregar pesos do modelo
    classificador = Classifier()
    classificador.load_state_dict(pre_trained_weights)
    classificador.eval()

    st.header("Classificação por Imagem")
    st.write("Faça o upload de uma foto e receba a classificação de máscara facial.")

    # Cria um campo de upload de arquivo
    uploaded_file = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem enviada', use_column_width=True)
        image = image.resize((224, 224), resample=Image.BILINEAR)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)

        with torch.no_grad():
            prediction = classificador(image_tensor)
            maxindex = int(torch.argmax(prediction))
            finalout = categorias[maxindex]
            output = str(finalout)

        # Exibe o resultado da classificação
        st.subheader("Resultado da classificação")
        st.write(f"A imagem é classificada como: {output}")


def about():
    st.header("Sobre esse projeto")
    st.write(
        '''
            Essa é uma aplicação utiliza um modelo CNN treinado com PyTorch e uma interface Web feita com OpenCV e Streamlit.

            Desenvolvido por:
            - Amon Menezes Negreiros
            - Henrique Barkett
            - Pedro Carvalho Almeida
            - Wander Araújo Buraslan
        '''
    )


pages = {
    'Home': home,
    'Classificação por Imagem': image_classification,
    'Sobre': about,
}

page = st.sidebar.selectbox('Escolha uma página', pages.keys())

if page:
    pages[page]()
