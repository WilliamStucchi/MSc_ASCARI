# MSc Thesis: Vehicle Dynamics Modeling with Neural Networks (ASCARI)

This repository contains the research, simulation scripts, and neural network architectures developed for my **Master of Science thesis**.  
The project explores the application of **Deep Learning** to model complex vehicle dynamics and estimate non-measurable states, 
with a specific focus on **longitudinal and lateral vehicle behavior**.

---

## 🚀 Project Overview

Traditional vehicle dynamics modeling often relies on simplified physical models (e.g., Linear / Non-linear Bicycle Models).  
This project investigates how **Neural Networks** can:

- Capture non-linearities and tire–road interactions more accurately
- Provide real-time state estimation where physical sensors are absent

### Key Research Areas

- **State Estimation**  
  Estimation of logitudinal velocity (`vx`), lateral velocity (`vy`), respective accelerations and sideslip angle using Neural Networks.

- **Model Comparison**  
  Benchmarking Neural Network performance against the classical Bicycle Model.

- **Incremental Learning**  
  Techniques allowing models to adapt to new driving conditions or vehicle configurations over time.

- **Handling Analysis**  
  Analysis of steering behaviors (Step, Ramp, Sine steering) and GG-plots for performance evaluation.

---

## 🛠 Tech Stack

### Languages
- Python (≈96%)
- MATLAB (core simulation data)

### Libraries
- **TensorFlow / Keras** – Neural Network design and training
- **NumPy & Pandas** – Data manipulation and outlier detection
- **Matplotlib & Seaborn** – GG-plots and dynamic results visualization
- **SciPy** – Signal processing and control analysis

---

## 🚦 Key Functionalities

### 1. Neural Network vs. Physical Models

Neural Networks are benchmarked against a classical and an optimized **Bicycle Model** to evaluate predictive accuracy and generalization across different driving maneuvers and operating conditions.

### 2. State Estimation

Neural Networks are used as **virtual sensors** to estimate lateral vehicle behavior, a quantity that is difficult and costly to measure directly in production vehicles.

### 3. Handling & Performance Visualization

The project includes several analysis and visualization tools:

- **GG-Plots** – Longitudinal vs. lateral acceleration limits
- **Steering Analysis** – Vehicle response to:
  - Step steering
  - Ramp steering
  - Sinusoidal steering inputs

---

## 📊 Results Summary

- Successful implementation of a **Neural Network–based observer** for lateral state estimation, improving estimation accuracy by 80%
- Improved accuracy in modeling **non-linear tire behavior** compared to linear physical models
- Exploration of **incremental learning** techniques to mitigate catastrophic forgetting under varying road friction conditions

---

## 👥 Contributors

- **William Stucchi**
- Master’s Thesis conducted within the **ASCARI research framework**
- **Institution:** Politecnico di Milano
