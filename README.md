# Federated_Learning_MNIST
My GitHub profile README
🔐 Federated Learning with CNN on MNIST

Privacy-preserving distributed image classification using Federated Averaging (FedAvg) across multiple clients — built with PyTorch.


📌 Overview
This project implements a Federated Learning system where multiple clients collaboratively train a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset — without sharing their raw data with a central server.
The server aggregates model weights using the FedAvg algorithm, preserving data privacy across all clients.

🏗️ Architecture
Central Server
     │
     ├── Sends global model weights to clients
     │
     ├── Client 1 → trains on local data → sends weights back
     ├── Client 2 → trains on local data → sends weights back
     ├── Client 3 → trains on local data → sends weights back
     ├── Client 4 → trains on local data → sends weights back
     └── Client 5 → trains on local data → sends weights back
     │
     └── Aggregates weights using FedAvg → updates global model

⚙️ Configuration
ParameterValueDatasetMNISTNumber of Clients5Communication Rounds10Aggregation MethodFedAvgModelCNN (Conv → ReLU → Pool → FC)FrameworkPyTorchPlatformGoogle Colab

📁 Project Structure
federated-learning-mnist/
│
├── README.md
├── requirements.txt
├── fl_main.py          # Main training loop (rounds + aggregation)
├── client.py           # Client-side local training
├── server.py           # FedAvg aggregation logic
├── model.py            # CNN architecture
├── utils.py            # Data loading and splitting utilities
└── results/
    ├── accuracy_plot.png
    └── loss_plot.png

🧠 Model Architecture
Input (28x28 grayscale)
  → Conv2D(1, 32, kernel=3) + ReLU
  → MaxPool2D(2)
  → Conv2D(32, 64, kernel=3) + ReLU
  → MaxPool2D(2)
  → Flatten
  → Linear(1600, 128) + ReLU
  → Linear(128, 10)
  → Output (10 classes)

📊 Results
RoundTrain AccuracyValidation Accuracy1~%~%5~%~%10~%~%

📌 (Update this table with your actual results)

<!-- Add your accuracy/loss plot here -->
<!-- ![Accuracy Plot](results/accuracy_plot.png) -->

🚀 How to Run
1. Clone the repository
bashgit clone https://github.com/your-username/federated-learning-mnist.git
cd federated-learning-mnist
2. Install dependencies
bashpip install -r requirements.txt
3. Run federated training
bashpython fl_main.py

📦 Requirements
torch
torchvision
numpy
matplotlib

📚 Key Concepts

Federated Learning — a distributed ML approach where training happens on local client data; only model weights (not raw data) are shared with the server
FedAvg — aggregation algorithm that averages client weights proportionally to their dataset sizes
CNN — Convolutional Neural Network used for image feature extraction and classification


👩‍💻 Author
Hafiza Kainat Siraj
MS Computer Science — MNSUAM, Multan, Pakistan
LinkedIn · GitHub

📄 License
This project is open source under the MIT License.
