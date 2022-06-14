import pandas as pd
from pandas import read_csv, Series, DataFrame
import numpy as np
from sklearn.metrics import accuracy_score

RAW_DATA_URL = r'https://covid.ourworldindata.org/data/owid-covid-data.csv'

# Prepare dataset
covid19data = read_csv(RAW_DATA_URL, usecols=['date', 'new_cases', 'location'], parse_dates=True)

# Select Location
covid19data = covid19data[covid19data['location'] == 'India']

# Select only date and counts
covid19data = Series(covid19data['new_cases'].values, index=pd.to_datetime(covid19data['date']))

# Weekly-fi
covid19data = covid19data.resample('W-Sun').sum()


def _adjust(idx: int, left: int, right: int):
    if idx < left:
        return left
    elif idx > right:
        return right
    else:
        return idx


# Window Size is assumed to be always equal to 5
def _window(center: int):
    left = _adjust(center - 2, 0, len(covid19data) - 1)
    right = _adjust(center + 2 + 1, 0, len(covid19data))
    return list(covid19data.iloc[left:right])


def _accumulate(n: int):
    seq = []
    i = 1
    while True:
        w = _window(n - (i * 52))
        if not w or i > 2:   # considering as latest as of 2 years of historical data
            break
        seq.extend(w)
        i = i + 1
    return seq, int(covid19data.iloc[n])


def perform_analysis(q):
    result = DataFrame()
    for q0 in q:
        result['P' + str(q0)] = []

    split = len(covid19data) - 1 - 52
    for x in range(split + 1, len(covid19data)):
        history, current = _accumulate(x)
        outbreak = current > np.percentile(history, q)
        result.loc[len(result.index)] = outbreak.astype('int')

    history = covid19data.iloc[:(split + 1)]
    current = covid19data.iloc[(split + 1):].values
    mean = np.mean(history)
    std = np.std(history)
    threshold = mean + 2 * std
    outbreak = current > threshold
    result['mean+2std'] = outbreak.astype('int')

    return result


# Debug
if __name__ == '__main__':
    K = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    result = perform_analysis(K)
    print(result)
    print()

    K_max = 0
    m = float('-inf')
    for i in K:
        acc = accuracy_score(result['mean+2std'], result['P' + str(i)])
        print("Accuracy of prediction by {}th percentile : {:.2f}%".format(i, acc*100))
        if acc > m:
            m = acc
            K_max = i

    print()
    print("The value of K for threshold calculation by Kth percentile method :", K_max)

    print("Predictions :", result['P' + str(K_max)].values)

