import matplotlib.pyplot as plt

"""
Implementation of Gradient Descent
Author: Amruth Karun M V
Date:   07 Sep 2021
"""


def grad_desc(data, itr, lr):
    """
    Performs gradient descent
    Arguments:
    data    -- input data
    itr     -- number of iteration
    lr      -- learning rate
    Returns: calculated slope and intercept
    """
    m = 0
    b = 0
    for _ in range(itr):
        for point in zip(data[0], data[1]):
            x = point[0]
            y_actual = point[1]
            y_prediction = m*x + b
            error = y_prediction - y_actual
            delta_m = -1 * (error * x) * lr
            delta_b = -1 * (error) * lr
            m = m + delta_m
            b = b + delta_b
    return m,b


def create_plot(data, m, b, xlab, ylab):
    """
    Creates plot for input data
    Arguments:
    data    -- input data
    m       -- final slope
    b       -- final intercept
    xlab    -- label for x axis
    ylab    -- label for y axis
    """
    plt.plot(data[0], data[1], 'o')
    plt.title("Linear Regression with Gradient Descent")
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # Plot regression lines
    regression_y = []
    for x in data[0]:
        y = m * x + b
        regression_y.append(y)
    print("Y Predicted: ", regression_y)
    plt.plot(data[0], regression_y)


def load_height_weight():
    """
    Gets height and weight data
    Arguments: None
    Returns: height and weight
    """
    height = [70, 72, 75, 78, 83, 72, 72, 64]
    weight = [180, 190, 200, 240, 218, 200, 172, 170]
    return [height, weight]


def load_hour_scores():
    """
    Gets hour and score data
    Arguments: None
    Returns: hour and score
    """
    hour = [2.3, 5.0, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5,
            3.4, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.7, 6.8, 7.4]
    score = [18, 45, 25, 72, 30, 20, 88, 60, 81, 25, 85, 62, 41, 44, 17,
              95, 30, 24, 67, 69, 30, 54, 33, 70, 85]
    return [hour, score]


if __name__ == "__main__":

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    data1 = load_height_weight()
    print("1. Height-Weight Data:")
    for lr in learning_rates:
        m, b = grad_desc(data1, 10, lr)
        create_plot(data1, m, b, "Height", "Weight")
        plt.savefig("height_weight_" + str(lr) + ".png")
        plt.show()

    data2 = load_hour_scores()
    print("\n2. Hour-Score Data:")
    for lr in learning_rates:
        m, b = grad_desc(data2, 10, lr)
        create_plot(data2, m, b, "Hour", "Score")
        plt.savefig("hour_score_" + str(lr) + ".png")
        plt.show()
