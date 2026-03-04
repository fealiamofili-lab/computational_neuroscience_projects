# Brain–Computer Interface: EEG-Based Virtual Arm

**Goal:** Decode motor imagery EEG signals to control a virtual arm in real-time.

**Status:** In Progress

### Pipeline
1. Preprocess EEG signals (filtering, artifact removal)
2. Extract spectral/time-frequency features
3. Train CNN or RNN classifier
4. Map predictions to virtual arm simulation
5. Evaluate with cross-validation, confusion matrices, feature importance
6. Framework designed for potential closed-loop feedback

### Dependencies
Python, NumPy, PyTorch, Matplotlib, PyBullet
