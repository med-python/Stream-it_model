import streamlit as st
import os
from werkzeug.utils import secure_filename
import pydicom
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import paramiko
import pandas as pd

def download_model(remote_path, local_path, key_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-18-218-205-80.us-east-2.compute.amazonaws.com', username='ubuntu', key_filename=key_path)

    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()

# Загрузка модели
remote_path = '/home/ubuntu/service/models/resnet50_modelmammae.pth'
local_path = 'resnet50_modelmammae.pth'  # Локальный путь, куда будет сохранена загруженная модель

key_path = "vkr.pem"  # Используйте относительный путь к файлу ключа

download_model(remote_path, local_path, key_path)

# Функция классификации DICOM файла
def classify_dicom(filepath):
    # Функция нормализации и визуализации DICOM
    def normalize_visualize_dicom_1(dcm_file):
        dicom_file = pydicom.dcmread(dcm_file)
        dicom_array = dicom_file.pixel_array.astype(float)
        normalized_dicom_array = ((np.maximum(dicom_array, 0))/dicom_array.max()) * 255.0
        uint8_image = np.uint8(normalized_dicom_array)
        return uint8_image

    # Классификация DICOM файла
    image_1 = normalize_visualize_dicom_1(filepath)
    if image_1 is not None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.fromarray(image_1)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        resnet50.eval()
        with torch.no_grad():
            output = resnet50(img_tensor.to(device))
            predicted_class = torch.round(output).item()

        if predicted_class == 1:
            result_text = "Данное изображение соответствует 1 (или 2) категории по шкале BI-RADS_MRT."
        else:
            result_text = "Данное изображение соответствует 3 (или 4) категории по шкале BI-RADS_MRT. Требуется консультация специалиста."
        
        return img, result_text
    
# Загрузка предварительно обученной модели ResNet50
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

def get_dicom_info(dicom_file_path):
    dicom_info = {}
    dicom = pydicom.dcmread(dicom_file_path)
    dicom_info['PatientName'] = dicom.PatientName
    dicom_info['PatientID'] = dicom.PatientID
    dicom_info['StudyDescription'] = dicom.StudyDescription
    dicom_info['Modality'] = dicom.Modality
    dicom_info['Rows'] = dicom.Rows
    dicom_info['Columns'] = dicom.Columns
    dicom_info['PixelSpacing'] = dicom.PixelSpacing
    dicom_info['PatientAge'] = dicom.PatientAge
    dicom_info['Manufacturer'] = dicom.Manufacturer
    dicom_info['InstitutionName'] = dicom.InstitutionName
    return dicom_info

def main():
    st.title('Классификация маммографических изображений')
    uploaded_file = st.sidebar.file_uploader("Загрузите файл DICOM", type=['dcm'])

    if uploaded_file is not None:
        # Сохраняем файл DICOM локально
        uploaded_filepath = os.path.join('uploads', secure_filename(uploaded_file.name))
        with open(uploaded_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Отображаем информацию о файле DICOM
        dicom_info = get_dicom_info(uploaded_filepath)
        st.write("### Информация о файле DICOM:")
        # Преобразуем словарь в список кортежей
        dicom_info_tuples = [(key, value) for key, value in dicom_info.items()]
        df = pd.DataFrame(dicom_info_tuples, columns=['Параметр', 'Значение'])
        st.table(df)

        # Отображаем спиннер перед классификацией изображения
        with st.spinner('Классификация изображения...'):
            # Классифицируем изображение и получаем изображение и результат классификации
            image, classification_result = classify_dicom(uploaded_filepath)
            st.write("### Заключение  классификатора:")
            # Форматируем вывод в цветной блок
            if "здоровое" in classification_result.lower() or "1 (или 2)" in classification_result:
                st.markdown('<div style="background-color: #008000; padding: 10px; border-radius: 5px;">'
                            '<p style="color: white;">{}</p></div>'.format(classification_result), unsafe_allow_html=True)
            elif "3 (или 4)" in classification_result:
                st.markdown('<div style="background-color: #FF33FF; padding: 10px; border-radius: 5px;">'
                            '<p style="color: black;">{}</p></div>'.format(classification_result), unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color: #FF5733; padding: 10px; border-radius: 5px;">'
                            '<p style="color: white;">{}</p></div>'.format(classification_result), unsafe_allow_html=True)

            # Отображаем изображение
            st.image(image, caption='Обработанное изображение', use_column_width=True)

if __name__ == '__main__':
    main()

       
