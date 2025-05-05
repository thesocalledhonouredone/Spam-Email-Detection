# Email Spam Detection with Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Apaulgithub/oibsip_taskno4/blob/main/Email_Spam_Detection_with_Machine_Learning.ipynb)

Oasis Infobyte Internship Project - Task 4

![Spam Image](https://www.easyspace.com/blog/wp-content/uploads/2019/03/spam-1.png)
*Image Courtesy: [https://www.easyspace.com/blog/wp-content/uploads/2019/03/spam-1.png](https://www.easyspace.com/blog/wp-content/uploads/2019/03/spam-1.png)*

---

## Problem Statement

Email spam, or junk mail, remains a persistent issue, flooding inboxes with unsolicited and often malicious content. These emails may contain cryptic messages, scams, or dangerous phishing attempts. This project, undertaken during a data science internship provided by Oasis Infobyte, aims to create an effective email spam detection system using Python and machine learning.

---

## Project Objectives & Highlights

1.  **Data Preprocessing:** Preprocessing a substantial email dataset (`spam.csv`), including data cleaning, handling missing values (dropping unnamed columns), and transforming text data.
2.  **Feature Engineering/Extraction:** Using techniques like `CountVectorizer` to extract meaningful features from email text data for model training.
3.  **Machine Learning Model Selection:** Implementing and evaluating a robust spam detection model. The primary model used is Multinomial Naive Bayes (`MultinomialNB`).
4.  **Model Evaluation:** Assessing model performance using metrics like accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.
5.  **Validation:** Using train-test split and evaluating on a dedicated test dataset to confirm the model's ability to generalize to new, unseen email data.
6.  **Spam Detection System:** Creating a simple function (`detect_spam`) to classify new email text as 'Spam' or 'Ham'.

---

## Dataset

* The dataset used is `spam.csv`, loaded directly from a GitHub repository.
* It contains email messages labeled as either 'ham' (legitimate) or 'spam'.
* Initial analysis showed the dataset consists of approximately 13.41% spam messages and 86.59% ham messages.

---

## Methodology

1.  **Import Libraries:** Necessary libraries like pandas, numpy, sklearn, matplotlib, seaborn, and wordcloud are imported.
2.  **Load Data:** The dataset is loaded from `https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv`.
3.  **Data Cleaning & Preprocessing:**
    * Columns 'v1' and 'v2' are renamed to 'Category' and 'Message'.
    * Columns 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4' are dropped due to many missing values.
    * A binary 'Spam' column is created (1 for spam, 0 for ham).
4.  **Exploratory Data Analysis (EDA):**
    * The distribution of spam vs. ham is visualized using a pie chart.
    * A word cloud is generated to identify common words in spam messages.
5.  **Train-Test Split:** The data is split into training (75%) and testing (25%) sets.
6.  **Model Pipeline:** A scikit-learn `Pipeline` is created combining `CountVectorizer` for text feature extraction and `MultinomialNB` for classification.
7.  **Training & Evaluation:** The pipeline is trained on the training data and evaluated on both training and test sets using a custom `evaluate_model` function, which calculates and visualizes performance metrics.

---

## Key Insights

* The dataset is imbalanced, with significantly more ham messages than spam.
* Common keywords frequently found in spam messages include 'free,' 'call,' 'text,' 'txt,' and 'now'.
* The Multinomial Naive Bayes model performed exceptionally well, achieving **98.49% recall** on the test set, indicating its effectiveness in identifying spam emails. The ROC AUC was 0.98 for the train set and 0.96 for the test set.

---

## How to Use

1.  **Dependencies:** Ensure you have Python installed along with the following libraries:
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`
    * `wordcloud`
2.  **Run the Notebook:** Execute the cells in the `Email_Spam_Detection_with_Machine_Learning.ipynb` notebook sequentially. This will load the data, train the model, and evaluate it.
3.  **Detect Spam:** Use the `detect_spam` function defined in the notebook to classify an email text:
    ```python
    # Example from notebook
    sample_email = 'Free Tickets for IPL'
    result = detect_spam(sample_email)
    print(result) # Output: This is a Spam Email!
    ```

---

## Conclusion

This project successfully demonstrates the application of machine learning, specifically the Multinomial Naive Bayes model combined with `CountVectorizer`, for effective email spam detection. The high recall achieved signifies the model's capability to minimize the impact of spam messages, contributing to enhanced email security and user experience.

---

## Author

* Katuru V Venkata Sairama Anirudh

## Reference

* Oasis Infobyte