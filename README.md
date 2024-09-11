# Fake_news_prediction

```markdown
# Fake News Detection

This project aims to build a machine learning model to classify news articles as reliable or unreliable based on the article's content. The dataset contains the title, author, and text of the articles, as well as a label indicating whether the article is considered unreliable (1) or reliable (0).

## Project Structure

- `train.csv`: The dataset used for training the model. It contains the following columns:
  - `id`: Unique identifier for each news article.
  - `title`: The title of the news article.
  - `author`: The author of the news article.
  - `text`: The text of the news article, which may be incomplete.
  - `label`: Label indicating whether the article is unreliable (1) or reliable (0).

- `fake_news_detection.ipynb`: The main Jupyter notebook where data preprocessing, feature extraction, model training, and evaluation are performed.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

You can install the required packages using:

```bash
pip install pandas numpy scikit-learn nltk
```

## Data Preprocessing

1. **Handling Missing Values**: Missing values in the `author`, `title`, and `text` columns are replaced with an empty string.
2. **Text Preprocessing**:
   - **Stopwords Removal**: Common English stopwords are removed using NLTK.
   - **Stemming**: Words are stemmed to their root form using `PorterStemmer`.
3. **Feature Extraction**:
   - A new feature `content` is created by concatenating the `author` and `title`.
   - Text data is converted into numerical features using `TfidfVectorizer`.

## Model Training

- **Train/Test Split**: The data is split into training and testing sets.
- **Model**: A Logistic Regression model is used to classify the news articles.
- **Evaluation**: The model's accuracy is evaluated on the test set.

## Results

The model achieves a reasonable level of accuracy, demonstrating its ability to distinguish between reliable and unreliable news articles based on textual features.

## How to Run

1. Clone the repository:

```bash
git clone <https://github.com/Ashwadhama2004/Fake_news_prediction/tree/main>
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter notebook and run the cells to preprocess the data, train the model, and evaluate its performance:

```bash
jupyter notebook fake_news_detection.ipynb
```

## Acknowledgments

- The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com).
- The NLTK library was used for text preprocessing.
- Scikit-learn was used for model training and evaluation.

-Link fo the data set:- https://www.kaggle.com/c/fake-news/data?select=train.csv
