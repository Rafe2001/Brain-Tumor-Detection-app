# Brain Tumor Detection
![image](https://github.com/Rafe2001/Brain-Tumor-Detection-app/assets/108533597/8137a599-d003-4fce-a108-8f2b1e1c4711)
![image](https://github.com/Rafe2001/Brain-Tumor-Detection-app/assets/108533597/22bd535d-14a2-464b-a082-7bc2bf916f5c)
![image](https://github.com/Rafe2001/Brain-Tumor-Detection-app/assets/108533597/42e7ef40-2289-46f1-9604-0671795c71f0)


## Table of Contents
- [About](#about)
- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About
The "Brain Tumor Detection" project is an advanced deep learning application designed to detect brain tumors in medical images, such as MRI scans. It plays a crucial role in assisting medical professionals with early and accurate brain tumor diagnosis, potentially saving lives and improving patient outcomes.

### Algorithm Overview
This project employs a state-of-the-art convolutional neural network (CNN) architecture to perform brain tumor classification. The CNN is trained on a diverse dataset of brain MRI scans, enabling it to distinguish between tumor and non-tumor images with remarkable accuracy. With a validation accuracy of over 91% and a training accuracy exceeding 96%, the model demonstrates its robustness and effectiveness.

## Features
- **Accurate Tumor Detection**: The deep learning model accurately detects brain tumors in MRI scans.
- **User-Friendly Web Interface**: A user-friendly web interface simplifies the process of uploading and analyzing medical images.
- **Real-Time Predictions**: The application provides real-time predictions using a pre-trained deep learning model.
- **Data Augmentation**: During model training, data augmentation techniques are employed to enhance accuracy further.
- **Healthcare Integration** (Optional): Integration with other healthcare systems for seamless data sharing can be implemented.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```shell
   git clone https://github.com/Rafe2001/Brain-Tumor-Detection-app.git
   cd Brain-Tumor-Detection-app
   ```

2. Install the required dependencies from the `requirements.txt` file:
   ```shell
   pip install -r requirements.txt
   ```

## Usage
Using the Brain Tumor Detection application is straightforward:

1. Run the Flask web application:
   ```shell
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Upload a brain MRI scan image using the provided form.

4. Click the "Upload and Predict" button to see the prediction result.

## Contributing
We welcome contributions to enhance the capabilities of the Brain Tumor Detection project. If you'd like to contribute, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Implement your changes and commit them.

4. Push your changes to your fork.

5. Submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code as per the terms of the license.

## Acknowledgments
We would like to express our gratitude to the contributors of open-source libraries and frameworks used in this project. Their work has been instrumental in the development of this application.
