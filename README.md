# Comparative Analysis of Different GAN Techniques for Face Generation

A comparative study and implementation of three Generative Adversarial Network (GAN) architectures — **DCGAN, ACGAN, and BEGAN** — for generating synthetic human face images.

This project explores the differences between these architectures in terms of **image quality, training stability, convergence, and controllability of generated outputs**.

---

## 📌 Overview

Generative Adversarial Networks (GANs) are deep learning models capable of learning complex data distributions and generating new data samples that resemble real-world examples.

In this project, I implemented and analyzed three different GAN architectures:

- **DCGAN (Deep Convolutional GAN)**
- **ACGAN (Auxiliary Classifier GAN)**
- **BEGAN (Boundary Equilibrium GAN)**

The models were trained for face generation using the **UTKFace dataset** and implemented using **PyTorch**.

The main objective of this project is to understand how different GAN architectures affect the quality and stability of generated images.

---

## 🧠 GAN Architectures

### 1. DCGAN — Deep Convolutional GAN

DCGAN uses convolutional neural networks to improve the generation and discrimination of images.

The architecture consists of:

- A **Generator** that generates synthetic face images from random noise.
- A **Discriminator** that distinguishes between real and generated images.
- Convolutional and transposed convolutional layers for image processing.

DCGAN serves as the baseline model for comparison in this project.

**Implementation:**

```text
train_dcgan.py
train_acgan.py
train_began.py

Comparative-Analysis-of-GAN-Techniques/
│
├── train_dcgan.py          # DCGAN implementation
├── train_acgan.py          # ACGAN implementation
├── train_began.py          # BEGAN implementation
│
├── samples_dcgan/           # Generated DCGAN samples
├── samples_acgan/           # Generated ACGAN samples
├── samples_began/           # Generated BEGAN samples
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


### One important recommendation 🔥

I would **not** put the comparison table with claims like:

> DCGAN = Moderate stability  
> ACGAN = Good stability  
> BEGAN = High stability

unless you actually measured these experimentally.

Since your project is titled **"Comparative Analysis"**, it will look much stronger if your README has a section like:

```markdown
## 📊 Experimental Results

| Model | Epochs | Training Time | Image Quality | Stability | FID Score |
|---|---:|---:|---|---|---:|
| DCGAN | 5 | TBD | TBD | TBD | TBD |
| ACGAN | 5 | TBD | TBD | TBD | TBD |
| BEGAN | 5 | TBD | TBD | TBD | TBD |
