# A-heterogeneous-vision-system-using-MPSoC
This repository contains the source code and experiments for the dissertation project **"A Heterogeneous Computer Vision System Using MPSoC"**,  
which focuses on deploying a DenseNet-based chest X-ray classification model onto the **Xilinx Kria KV260** platform using **Vitis AI**.

---

##  Project Structure

### `Train_NIH/`
Contains the training pipeline for the **NIH ChestX-ray14** dataset.  
Includes:
- Data preprocessing scripts  
- Model definition (DenseNet variants)  
- Training scripts for generating model checkpoints (`.pth`)

### `Quantification_PTQ/`
Implements the **Post-Training Quantization (PTQ)** workflow using Vitis AI.  
Includes multiple quantization versions for experimentation and testing:
- **Version 1 & 2:** Preliminary quantization attempts with accuracy issues  
- **Version 3:** Final quantized model used for deployment on the MPSoC platform

---

##  Usage

1. **Train the model** using the scripts in `Train_NIH/`.  
2. **Apply quantization** using the scripts in `Quantification_PTQ/`.  
3. Use **Version 3** of the quantized model for deployment.

---

## Notes

- The code is tailored for the **NIH ChestX-ray14** dataset.  
- Deployment was tested on **Xilinx Kria KV260** with **Vitis AI 2.5**.  
- Additional setup (Docker, Vitis AI tools) may be required to reproduce the results.

---

© 2025 Yalun Lee. All rights reserved.
