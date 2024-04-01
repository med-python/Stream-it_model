# Stream-it_model

<a name="readme-top"></a>
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[!['Black'](https://img.shields.io/badge/code_style-black-black?style=for-the-badge)](https://github.com/psf/black)

<!-- Библиотеки проекта -->

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

# Classification of breast cancer Project

This project contains a Flask web application for predicting outcomes using a model trained on DICOM files.

## Description

The project consists of the following components:

- [app.py](https://raw.githubusercontent.com/med-python/Classification_of_breast_cancer/main/app.py?token=GHSAT0AAAAAACOJ74RZC7Z7UIQ6IPTADEYWZPLFD4Q): Flask application for serving the web interface.
- [script_model.py](https://raw.githubusercontent.com/med-python/Classification_of_breast_cancer/main/script_model.py?token=GHSAT0AAAAAACOJ74RYUCU3UUUXPQZ4O6WOZPLFFSQ): Script containing the machine learning model. Note that the model files are not included in this repository.
- [script_dcm_png_model.py](https://raw.githubusercontent.com/med-python/Classification_of_breast_cancer/main/script_dcm_png_model.py?token=GHSAT0AAAAAACOJ74RYRKRU7F2HMAZ5NFCUZPLFGPQ): Script that takes a DICOM file as input, loads the trained model, and generates predictions.
- [uploads](): Folder for uploading DICOM files to be tested. The `script_dcm_png_model.py` script will access files from this folder.

## Usage

To run the Flask application:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
## Run the Flask application:
   
Access the web application in your browser at http://localhost:5000.

Note: Make sure you have the trained model files available on your computer to use the script_model.py and script_dcm_png_model.py scripts.

##    Authors

* **Anna Permiakova** [med-python](https://github.com/med-python),

* **Ekaterina Gaponenko**  [egaponenko](https://github.com/egaponenko)

## License

This project is licensed under the MIT License - see the [LICENSE.tx](LICENSE.tx) file for details

