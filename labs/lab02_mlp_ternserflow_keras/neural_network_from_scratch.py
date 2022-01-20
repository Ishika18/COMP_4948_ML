import matplotlib.pyplot as plt
import numpy as np


def visualize_data(data, mystery_flower):
    # scatter plot
    for i in range(len(data)):
        point = data[i]
        color = 'r'
        if point[2] == 0:
            color = 'b'
        plt.scatter(point[0], point[1], c=color)

    plt.scatter(mystery_flower[0], mystery_flower[1], color='green')
    plt.title("Red, Blue and Mystery flowers")
    plt.show()


# compress linear eq result b/w 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def show_losses(losses):
    plt.plot(losses)
    plt.title("Loss")
    plt.show()


def get_optimal_weights_and_bias(data):
    w1 = np.random.rand()
    w2 = np.random.rand()
    b = np.random.rand()

    costs = []
    learning_rate = 0.2

    for i in range(1000):
        # get random element from dataset
        ri = np.random.randint(len(data))
        point = data[ri]

        # regression line for flowers
        z = w1 * point[0] + w2 * point[1] + b

        # compress prediction to values between 0 and 1
        pred = sigmoid(z)
        target = point[2]

        # loss function (residual square)
        cost = (pred - target) ** 2

        # derivative of cost function wrt prediction
        dcost_pred = 2 * (pred - target)

        # change in cost wrt target
        dcost_dz = dcost_pred * pred

        # change in cost wrt weight 1
        dcost_dw1 = dcost_dz * point[0]

        # change in cost wrt weight 2
        dcost_dw2 = dcost_dz * point[1]

        # update weights and bias
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_dz

        # cal cost every 10 iterations
        if i % 10 == 0:
            cost = 0
            point = data[ri]

            # cal cost for current prediction
            z = point[0] * w1 + point[1] * w2 + b
            pred = sigmoid(z)
            target = point[2]
            cost += np.square(pred - target)
            costs.append(cost)

    show_losses(costs)
    return w1, w2, b


def show_predictions(data, title, w1, w2, b):
    print("\n***Predictions Using " + title)
    # Show predictions for each data point.
    for i in range(len(data)):
        point = data[i]
        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoid(z)

        color = 'blue'
        if pred > 0.5:
            color = 'red'
        color = str(pred) + " " + color
        print(str(point) + " predicted color: " + color)


# Input data.
data = [
    [3, 1.5, 1],
    [2, 1, 0],
    [4, 1.5, 1],
    [3, 1, 0],
    [3.5, .5, 1],
    [2, .5, 0],
    [5.5, 1, 1],
    [1, 1, 0]]

# Unknown value.
mystery_flower = [4.5, 1]

w1, w2, b = get_optimal_weights_and_bias(data)
show_predictions(data, "Training Data", w1, w2, b)
show_predictions([mystery_flower], "Mystery Flower", w1, w2, b)
