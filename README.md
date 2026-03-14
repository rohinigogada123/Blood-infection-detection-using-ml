# 🧬 Blood Infection Risk Detection System (Sepsis Detection using Machine Learning)

A **Machine Learning based diagnostic support system** designed to detect the **risk of blood infection (sepsis)** using patient vital signs and laboratory values.

This system helps estimate infection probability and provides clinical recommendations through an **interactive web dashboard built with Streamlit**.

⚠️ **This project is intended for educational and research purposes only and is not a substitute for professional medical diagnosis.**

---

# 📌 Project Overview

Sepsis is a life-threatening condition caused by the body's extreme response to infection. Early detection is crucial for preventing organ failure and improving survival rates.

This project implements a **machine learning pipeline** that:

* Processes patient clinical data
* Applies feature engineering
* Uses trained ML models
* Predicts infection risk
* Displays results in an interactive medical dashboard

The system allows:

✔ Manual patient data entry
✔ Batch prediction using CSV files
✔ Sample data testing
✔ Clinical risk assessment

---

# 🧠 Machine Learning Models Used

The system uses two supervised learning models:

### Random Forest Classifier

* Ensemble learning method
* Handles non-linear patterns
* Provides feature importance
* Good for medical prediction problems

### Logistic Regression

* Linear classification model
* Provides probability scores
* Easy to interpret

The **best performing model is automatically selected** during training.

---

# 📊 Features Used for Prediction

The model analyzes **8 clinical indicators**:

| Feature          | Unit   | Normal Range | Description              |
| ---------------- | ------ | ------------ | ------------------------ |
| WBC Count        | 10³/μL | 4.5 – 11     | White blood cell count   |
| Temperature      | °C     | 36.5 – 37.5  | Body temperature         |
| Heart Rate       | bpm    | 60 – 100     | Heart beats per minute   |
| Respiratory Rate | /min   | 12 – 20      | Breathing rate           |
| Lactate          | mmol/L | 0.5 – 2      | Tissue oxygen indicator  |
| Glucose          | mg/dL  | 70 – 100     | Blood sugar level        |
| Platelet Count   | 10³/μL | 150 – 400    | Blood clotting indicator |
| Bilirubin        | mg/dL  | 0.1 – 1.2    | Liver function indicator |

---

# 🏗 Feature Engineering

Two additional features are created during preprocessing:

```
wbc_temp_interaction = wbc_count * temperature
lactate_glucose_ratio = lactate / glucose
```

These features help the model capture **clinical relationships between variables**.

---

# 📁 Project Structure

```
blood-infection-risk-detection/
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── data/
│   └── sepsis_dataset.csv      # Medical dataset
│
├── models/
│   ├── infection_model.pkl     # Trained ML model
│   └── scaler.pkl              # Feature scaler
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
│
├── notebooks/
│   └── data_analysis.ipynb     # Exploratory analysis
│
├── main.py                     # Model training pipeline
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone or Download the Project

```
git clone <repository-url>
cd blood-infection-risk-detection
```

or download and extract the project ZIP file.

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment:

**Windows**

```
venv\Scripts\activate
```

**Mac / Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 🚀 Running the Project

### Step 1: Train the Model

Run:

```
python main.py
```

This will:

* Load dataset
* Preprocess data
* Train ML models
* Evaluate performance
* Save trained model

Generated files:

```
models/infection_model.pkl
models/scaler.pkl
```

---

### Step 2: Launch the Web Application

Run:

```
streamlit run app/app.py
```

Streamlit will start a local server.

Open the application in your browser:

```
http://localhost:8501
```

---

# 🖥 Web Application Features

### 🔍 Detection Page

Allows prediction using:

**1️⃣ Manual Input**

Enter patient medical values and click:

```
Detect Infection Risk
```

The system displays:

* Risk level
* Risk probability
* Clinical recommendations

---

**2️⃣ CSV Upload**

Upload patient data in this format:

```
wbc_count,temperature,heart_rate,respiratory_rate,lactate,glucose,platelet_count,bilirubin
7.5,37.2,75,16,1.5,100,250,0.8
15.2,39.5,120,28,4.5,250,80,3.2
```

The system performs **batch prediction**.

---

**3️⃣ Sample Data**

Run predictions on predefined sample patients.

---

# 📈 Model Output

The system returns:

### Risk Classification

```
LOW RISK
HIGH RISK
```

### Risk Score

```
Probability of infection (%)
```

Example:

```
Risk Score: 72.4%
```

---

# 📊 Model Performance Metrics

The model evaluation includes:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Sensitivity
* Specificity

---

# 🛠 Input Validation

The application restricts unrealistic values using Streamlit.

Example constraints:

| Feature          | Min | Max |
| ---------------- | --- | --- |
| WBC              | 0   | 50  |
| Temperature      | 30  | 42  |
| Heart Rate       | 40  | 200 |
| Respiratory Rate | 10  | 50  |
| Lactate          | 0   | 20  |
| Glucose          | 40  | 400 |
| Platelets        | 0   | 500 |
| Bilirubin        | 0   | 20  |

---

# ⚠️ Important Disclaimer

This system is developed **only for educational and research purposes**.

It should **NOT be used for real clinical decision making**.

Always consult qualified healthcare professionals for medical diagnosis and treatment.

---

# 🧰 Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn

---

# 👨‍💻 Author

Blood Infection Risk Detection System
Machine Learning Project

---

# 📚 References

* SEPSIS-3 Clinical Guidelines
* Machine Learning in Healthcare Research
* Scikit-learn Documentation

---

# ⭐ Future Improvements

Possible enhancements:

* Real-time patient monitoring
* Deep learning models
* Risk visualization dashboards
* Hospital database integration
* API deployment

---
