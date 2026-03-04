# Alzheimer’s MRI Classification Using CNN

**Goal:** Detect early-stage Alzheimer’s disease from MRI scans using convolutional neural networks, with interpretability and knowledge retrieval.

**Status:** In Progress — CNN implemented, Grad-CAM working, RAG framework integrated.

### Pipeline
1. Preprocess MRI images (normalize, resize, augment)
2. CNN model (ResNet18 backbone)
3. Grad-CAM interpretability
4. RAG knowledge retrieval for clinical context
5. Evaluate with cross-validation, accuracy, precision/recall, F1

### Dependencies
Python, PyTorch, OpenAI API, NumPy, OpenCV, Torchvision
