def getRes(weights, heights, intercept):
    sum_ssr = 0
    BETA = 0.64
    for i in range(0, len(weights)):
        sum_ssr += -2 * (heights[i] - intercept - BETA * weights[i])

    print("Intercept: " + str(intercept) + " Res: " + str(round(sum_ssr, 2)))


if __name__ == '__main__':
    test_weights = [0.5, 2.3, 2.9]
    test_heights = [1.4, 1.9, 3.2]
    test_intercept = 0.95
    getRes(test_weights, test_heights, test_intercept)
