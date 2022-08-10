import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


def load_data(database_filepath):
    """
    Loads the cleaned dataframe
    Input: filepath
    Returns: Messages and categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df.message.values
    y = df.iloc[:, 4:]
    return X, y


def tokenize(text):
    """
    Transforms input text to be suitable for training
    Input: Input text
    Returns: Transformed text
    """
    # Normalized
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenized
    words = word_tokenize(text)

    # Stop Word Removal
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatized
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    # Stem
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


def build_model():
    """
    Builds the pipeline object containing transformation and categorizing
    Importantly, the pipeline is included in a Gridsearch, first finding the
    an optimal set of parameters for the classification.
    Input: None
    Returns: GridSearchCV Object containing Pipeline
    """
    pipeline = Pipeline([("vect", CountVectorizer(tokenizer=tokenize)),
                         ("tfidf", TfidfTransformer()),
                         ("clf", MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'clf__estimator__n_estimators': [10, 20, 30],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


def evaluate_model(model, X_test, y_test):
    """
    Computes performance of the trained model
    Input: Trained model, test messages, test categories
    Returns: None
    """
    y_pred = model.predict(X_test)

    for inx, colm in enumerate(y_test.columns):
        print("Column: ", colm)
        print(classification_report(y_test[colm], y_pred[:, inx]))
        print("_" * 10)
        print("\n")

    correct = y_pred == y_test
    print("Total accuracy: ", correct.mean().mean())


def save_model(model, model_filepath):
    """
    Saves the model into a selected folder
    Input: Dataframe, filepath
    Returns: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
