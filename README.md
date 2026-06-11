# Women's E-Commerce Clothing Reviews Dashboard

This repository contains an interactive Streamlit dashboard designed for data mining and information retrieval tasks using the Women's E-Commerce Clothing Reviews dataset.

## Course Information
* **Student:** Muhammad Luqman bin Aziz
* **Course:** Data Mining and Information Retrieval
* **Lecturer:** Dr. Nurashikin Saaludin
* **Semester:** March 2026

## Project Structure
* `app.py`: The main Streamlit dashboard application file containing the visual styling and execution logic.
* `Womens Clothing E-Commerce Reviews.csv`: The clothing reviews dataset used as the data source.
* `requirements.txt`: The file containing list of Python dependencies required to run the dashboard.

## Dashboard Modules
The dashboard is structured into eight modules corresponding to the syllabus:
1. **Home:** Overview of the application with key metrics (total reviews, average rating, recommendation rate).
2. **Dataset Overview:** Data structure, missing values table, correlation matrix, and distribution histograms.
3. **Preprocessing & Feature Engineering:** Duplicate cleaning, missing text handling, original vs. encoded category mappings, and scaling.
4. **Classification Analysis:** Prediction of recommendation status using Decision Tree, Naive Bayes, Logistic Regression, KNN, and Random Forest.
5. **Clustering Analysis:** Customer segmentation using K-Means and K-Medoids, evaluated via the Elbow Method and Silhouette Score, with automatic text interpretations.
6. **Information Retrieval:** Search engine based on TF-IDF term-document weights and Cosine Similarity, showing the built Inverted Index (Term -> Document IDs).
7. **Text Mining & Sentiment:** Word Cloud, TF-IDF top keyword scoring, and review text examples categorized by sentiment labels.
8. **Conclusion & Recommendation:** Summary of key analytical findings and actionable business insights.

## How to Run Locally

### 1. Set Up Environment
Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```

## How to Deploy to Streamlit Community Cloud

1. Create a free account at [Streamlit Share](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Select this repository, choose `app.py` as the entry file, and click **Deploy**.
