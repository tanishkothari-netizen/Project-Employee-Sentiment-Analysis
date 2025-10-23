# **📧 Employee Engagement & Sentiment Analyzer**

**Tagline:** Deriving actionable insights on employee morale and retention risk from internal communications data.

## **✨ Description**

This project analyzes an unlabeled dataset of employee messages to assess sentiment, measure engagement, and predict retention risk. It is a comprehensive data science solution that moves from raw text to predictive modeling, providing leadership with actionable metrics.

The primary goal is to transform unstructured text into quantifiable engagement metrics for early identification of morale issues and potential "flight risks."

### **Key Features**

1. **Sentiment Labeling (NLP):** Automatically labels thousands of messages as Positive, Negative, or Neutral.  
2. **Engagement Scoring:** Computes a monthly sentiment score for every employee.  
3. **Flight Risk Identification:** Flags employees demonstrating patterns of frequent negative communication (4+ negative emails in a rolling 30-day window).  
4. **Predictive Modeling:** Develops a Linear Regression model to quantify how engagement factors (like message frequency and length) influence monthly sentiment scores.  
5. **Automated Reporting:** Generates visualizations and structured ranking tables for reporting key findings.

### **Technologies Used**

| Category | Tool / Library | Purpose |
| :---- | :---- | :---- |
| **Language** | Python (3.8+) | Core scripting and data manipulation. |
| **Data Handling** | Pandas, NumPy | Data cleaning, feature engineering, aggregation, and time-series analysis. |
| **NLP** | NLTK (VADER) | Efficient, rule-based lexicon for sentiment scoring and labeling. |
| **Modeling** | Scikit-learn | Linear Regression model implementation and evaluation. |
| **Visualization** | Matplotlib, Seaborn | Generating professional charts for EDA and model performance. |

## **⚙️ Installation/Setup**

These instructions guide you through setting up the required environment for running the analysis.

### **Prerequisites**

* Python 3.8 or higher.  
* Familiarity with Jupyter Notebooks or Google Colab.  
* A stable internet connection (for data download and initial NLTK setup).

### **Step-by-step Setup**

1. **Clone the Repository** (If applicable):  
   git clone \[Your\_Repository\_Link\]  
   cd \[repository-name\]

2. **Create a Virtual Environment** (Recommended):  
   python \-m venv venv  
   source venv/bin/activate  \# On Linux/macOS  
   venv\\Scripts\\activate      \# On Windows

3. **Install Dependencies** (Using pip): Since a requirements.txt is not provided, install the necessary libraries directly:  
   pip install pandas numpy matplotlib seaborn scikit-learn nltk

4. **Data Setup:** The notebook primarily accesses the data via a Google Sheets URL. For local running, ensure the test.csv file is present in the project's root directory alongside the notebook.

## **🏃 Usage**

The project is designed to be executed entirely within the Jupyter Notebook environment.

### **Running the Analysis**

1. **Launch Jupyter/Colab:**  
   jupyter notebook  \# If running locally  
   \# OR open the .ipynb file directly in Google Colab.

2. **Open the Notebook:** Open Tanish\_kothari\_AI-project-submission.ipynb.  
3. **Execute All Cells:** Run the entire script sequentially.  
   * In Jupyter/Colab: Navigate to Run \-\> Run All Cells.  
   * The notebook handles all data loading, cleaning, NLTK downloads, analysis, and visualization generation automatically.

### **Outputs**

Upon completion, the script will print final metrics and generate the following artifacts:

* **Rankings:** Tables showing Top 3 Positive and Top 3 Negative employees per month.  
* **Flight Risk List:** A list of flagged employee IDs.  
* **Visualizations:** Image files saved in the newly created visualization/ directory.

## **📐 Methodology**

The analysis follows a six-step process, combining NLP, time-series aggregation, and statistical modeling.

### **1\. Data Preprocessing and Cleaning**

* Raw email data is loaded, and employee IDs are extracted from email addresses.  
* Dates are standardized, and messages are cleaned to remove email artifacts (=01, \=20) to prevent VADER misinterpretation.

### **2\. Sentiment Labeling (NLTK VADER)**

* **Model Selection:** The VADER lexicon is used for sentiment scoring due to its effectiveness in analyzing social media and informal communications.  
* **Labeling:** A **VADER compound score** is calculated for each message. Labels are assigned based on a standard threshold:  
  * Positive: Score $\\ge 0.05$  
  * Negative: Score $\\le \-0.05$  
  * Neutral: Score between \-0.05 and 0.05

### **3\. Employee Engagement Scoring**

* **Metric:** A numeric score is assigned to each message (+1 for Positive, \-1 for Negative, 0 for Neutral).  
* **Aggregation:** Scores are aggregated using Pandas.groupby() to determine a monthly\_sentiment\_score for each employee.

### **4\. Flight Risk Identification (Rolling 30 Days)**

* **Criteria:** An employee is flagged if they send **4 or more negative messages** within any continuous **30-day window**.  
* **Implementation:** This is implemented using Pandas' time-based rolling('30D', closed='left') function to count preceding negative messages accurately.

### **5\. Feature Engineering for Predictive Model**

* **Target Variable (**$Y$**):** monthly\_sentiment\_score  
* **Features (**$X$**):**  
  * monthly\_message\_count: Total messages sent by the employee in the month (Engagement/Activity).  
  * avg\_message\_length: Average word count of messages in the month (Effort/Context).

### **6\. Predictive Modeling (Linear Regression)**

* **Model:** A simple **Linear Regression** model (via Scikit-learn) is trained to understand the relationship between the engineered features ($X$) and the monthly sentiment score ($Y$).  
* **Training:** Data is split 70/30 into training and testing sets (random\_state=42).  
* **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.

## **📈 Results**

The analysis provided strong evidence linking high employee communication volume to positive sentiment, offering a clear metric for engagement.

### **Key Performance Metrics (Linear Regression)**

| Metric | Value | Interpretation |
| :---- | :---- | :---- |
| **R-squared (**$R^2$**) Score** | **0.5890** | The features explain approximately 59% of the variance in the monthly sentiment score, indicating a moderately strong fit and predictability. |
| **Root Mean Squared Error (RMSE)** | **1.0541** | The average prediction error is around $\\pm 1.05$ points on the cumulative monthly score scale. |

### **Model Interpretation (Coefficients)**

| Feature | Coefficient | Significance |
| :---- | :---- | :---- |
| **Monthly Message Count** | **\+0.6742** | **Strongest Predictor.** Each additional message sent per month strongly increases the predicted monthly sentiment score. |
| **Avg Message Length** | **\+0.0096** | **Weak Predictor.** Message length has a minimal, positive effect. |

### **Flight Risk Outcome**

* **Flagged Employees:** No employees met the strict criteria of 4 or more negative mails in a rolling 30-day period in the analyzed dataset.

## **📁 Directory Structure**

The project structure ensures all components are organized and reproducible.

.  
├── Tanish\_kothari\_AI-project-submission.ipynb  \# Main Executable Notebook (Code, Output, Commentary)  
├── README.md                                   \# Project Overview and Documentation (This file)  
├── test.csv                                    \# Raw Dataset (Included for local execution)  
├── environment\_example.txt                     \# Structure file (No API key needed for VADER)  
└── visualization/                              \# Directory for all generated plots  
    ├── 2\_1\_sentiment\_distribution.png          \# Overall sentiment distribution  
    ├── 2\_2\_sentiment\_trend\_monthly.png         \# Time series of sentiment counts  
    └── 6\_1\_prediction\_vs\_actual.png            \# Actual vs. Predicted score plot

## **✅ Conclusion**

This project successfully leveraged NLP and classic machine learning to quantify abstract concepts like employee morale and engagement. The strong positive correlation identified between message count and sentiment score suggests that **fostering an active communication culture is directly linked to higher morale**. The implemented flight risk detection system provides a valuable tool for early HR intervention.