import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# 1. Build a linear classifier for MNIST dataset.

def build_classifier():
    """
    Classifies  MNIST Digits dataset using
    Logistic Regression
    Arguments: None
    Result: Confusion Matrix
    """

    digits = load_digits()
    print("Image Data Shape", digits.data.shape)
    print("Label Data Shape", digits.target.shape)

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                        test_size=0.25, random_state=5)

    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train,)
    predictions = model.predict(x_test)

    score = model.score(x_test, y_test)
    print("Score = ", score)

    cm = metrics.confusion_matrix(y_test, predictions)
    metrics.ConfusionMatrixDisplay(cm).plot()
    plt.show()


if __name__ == "__main__":
    build_classifier()
