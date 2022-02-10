import time
import pandas as pd

# -------------------------------------------------------------------
# Logs method execution times to a data frame.
# -------------------------------------------------------------------
timeItList = []


def showTimeResults():
    timeItDf = pd.DataFrame()
    for i in range(0, len(timeItList)):
        timeItDf = timeItDf.append(timeItList[i], ignore_index=True)
    print("\n")
    print(timeItDf)
    return timeItDf


# -------------------------------------------------------------------
# Enables @timeit decorator.
# -------------------------------------------------------------------
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            diff = (te - ts) * 1000
            timeItList.append({'Method': method.__name__, 'Time (ms)': diff})
            print('%r  %2.2f ms' % (method.__name__, diff))
        return result

    return timed


@timeit  # Prefix any function with this code to automatically log execution time.
def doSomething():
    for i in range(0, 1000):
        print(i)

@timeit
def doSomethingElse():
    for _ in range(0, 100):
        print("Shagun")


for x in range(0, 3):
    doSomething()

doSomethingElse()

showTimeResults()
