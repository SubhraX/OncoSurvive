# OncoSurvive

This is a Streamlit web application that predicts the survival duration of cancer patients using a machine learning model trained on clinical features.

The app uses a Keras-based neural network to classify survival durations into the following categories:

- `<1yr`
- `1-3yrs`
- `3-5yrs`
- `>5yrs`

---

## Getting Started

Follow these steps to run the application in a **virtual environment** (`.venv`).

### 1. Clone the Repository

```bash
git clone https://github.com/SubhraX/OncoSurvive.git
cd cancer-survival-predictor
```

### 2. Set Up a Virtual Environment

#### On **Windows**:

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### On **macOS/Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` does not exist, run:
>
> ```bash
> pip install streamlit pandas numpy scikit-learn tensorflow joblib
> pip freeze > requirements.txt
> ```

---

### 4. Place Required Files

Ensure the following files are in the project root directory:

* `app.py` — the Streamlit app (your script)
* `cancer_dataset.csv` — dataset used for preprocessing structure
* `cancer_survival_model.h5` — pre-trained Keras model

---

### 5. Run the Application

```bash
streamlit run app.py
```

This will open the app in your default web browser at `http://localhost:8501`.

---

## Project Structure

```
cancer-survival-predictor/
│
├── .venv/                     # Python virtual environment
├── app.py                     # Streamlit app script
├── cancer_dataset.csv         # Dataset file
├── cancer_survival_model.h5   # Trained model
├── requirements.txt           # Dependency list
└── README.md                  # Project README
```

---

## Features

* Clean UI for entering patient data
* Handles both numeric and categorical features
* Applies preprocessing (scaling, one-hot encoding)
* Displays survival class and confidence probabilities
* Utilizes caching for performance

---

## Example Inputs

* **Age**: Integer (e.g., 45)
* **Tumor Size**: Float (e.g., 3.2 cm)
* **Stage**: One of `Stage I`, `Stage II`, `Stage III`, `Stage IV`
* **Other categorical fields**: Selected from dropdowns

---

## Deactivating the Virtual Environment

After you're done:

```bash
deactivate
```

---

## Contact

Created by Ayushmaan and ShubhraX

---

