# Pneumonia Detection using ResNet-50

This project uses transfer learning with ResNet-50 to classify chest X-ray images as either pneumonia or normal. The dataset used is PneumoniaMNIST, a subset of the MedMNIST collection.

---

## 📁 Files

* `pneumonia_detection_resnet50.ipynb` – Jupyter notebook with complete code and training pipeline.
* `requirements.txt` – List of Python packages required to run the notebook.
* `best_resnet50.pth` – (Optional) Saved model weights after best validation AUC.

---

## ⚙️ How to Run This Project

### 1. Create a Python Virtual Environment

Make sure you have Python 3.12 installed.

```bash
python3.12 -m venv venv
```

Activate the environment:

* On **Windows**:

```bash
.\venv\Scripts\activate
```

* On **Mac/Linux**:

```bash
source venv/bin/activate
```

### 2. Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you're using a **NVIDIA GPU** and want to enable CUDA, make sure the CUDA version matches what's specified in the `requirements.txt`. For example, the provided file installs Torch with CUDA 12.6. If your system has a different CUDA version, you can visit:

️ [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

...to get the correct install command for your setup.

---

## 📌 Hyperparameters Note

The model was trained with the following settings:

* **Backbone**: Pretrained ResNet-50
* **Batch size**: 32
* **Learning rate**: `1e-4` for frozen backbone, `1e-5` for fine-tuning
* **Loss function**: CrossEntropyLoss (with optional label smoothing)
* **Optimizer**: Adam
* **Dropout**: 0.5 to reduce overfitting
* **Augmentations**: Rotation, horizontal flip, resized crop
* **Scheduler**: ReduceLROnPlateau to reduce learning rate on validation plateau
* **Early stopping**: Training stops after 5 epochs of no improvement in AUC

These values were chosen based on common transfer learning setups and adjusted slightly after monitoring validation AUC.

---

## 📊 Evaluation

The model is evaluated using:

* **AUC-ROC** – Measures model’s ability to separate classes regardless of threshold.
  ![auc-roc](https://github.com/user-attachments/assets/adf100b0-f7aa-42db-981a-b961607a03ab)
* **Confusion matrix** – Shows TP, FP, TN, FN.
  ![confusion-matrix](https://github.com/user-attachments/assets/207530db-503f-4a95-b567-ded47b8a7d2f)
* **Classification report** – Includes accuracy, precision, recall, F1.
  ![classification-report](https://github.com/user-attachments/assets/bfca57cf-fb2b-4074-9039-4f435b5880eb)

  

The notebook will save the best-performing model automatically and load it for final evaluation.


---

## ✅ Tips

* If you get an error about image channels, make sure grayscale images are converted to RGB using `transforms.Grayscale(num_output_channels=3)`.
* If your GPU is not used, check if `torch.cuda.is_available()` returns `True`.

---

## 📬 Questions?

Feel free to open an issue or email if you run into any problems getting the notebook to run.
