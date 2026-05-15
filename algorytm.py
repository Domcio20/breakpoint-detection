import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data.txt", sep="\t", header=None)

x = df[0]
y = df[1]


def least_squares(start, stop):

    x_data = x[start:stop]
    y_data = y[start:stop]

    n = len(x_data)
    if n < 2:
        return np.nan, np.nan

    sx = sum(x_data)
    sy = sum(y_data)
    sx2 = sum(a*a for a in x_data)
    sxy = sum(a*b for a, b in zip(x_data, y_data))

    denom = (n * sx2 - sx * sx)
    if denom == 0:
        return np.nan, np.nan

    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n

    return a, b


def build_slopes(dictionary, step=10, max_window=None):

    if max_window is None:
        max_window = len(x) // 2

    for w in range(50, max_window, step):

        start = len(x) - w
        end = len(x)

        a, _ = least_squares(start, end)

        dictionary[w] = a


def breakpoint_detection(slopes, windows):

    da = np.gradient(slopes)
    d2a = np.gradient(da)

    d2a_n = (d2a - np.mean(d2a)) / (np.std(d2a) + 1e-9)

    threshold = np.mean(np.abs(d2a_n)) + 1.5 * np.std(d2a_n)

    candidates = np.where(np.abs(d2a_n) > threshold)[0]

    if len(candidates) == 0:
        print("No clear breakpoint — fallback")
        jump_idx = len(windows) // 2
    else:
        jump_idx = candidates[0]

    jump_window = windows[jump_idx]

    break_idx = len(x) - jump_window + 10

    print("BREAK WINDOW:", jump_window)

    return break_idx


def pre_regression(break_idx):

    window_before = int(0.1 * len(x))
    window_before = np.clip(window_before, 30, 200)

    end1 = break_idx
    start1 = max(0, end1 - window_before)

    a1, b1 = least_squares(start1, end1)

    return a1, b1


def post_regression(break_idx):

    window_after = int(0.1 * len(x))
    window_after = np.clip(window_after, 30, 200)

    start2 = break_idx
    end2 = min(len(x), start2 + window_after)

    a2, b2 = least_squares(start2, end2)

    print("POST WINDOW:", window_after)

    return a2, b2


def lines_intersection(a1, b1, a2, b2):

    if not np.isnan(a1) and not np.isnan(a2) and abs(a1 - a2) > 1e-12:
        x_cross = (b2 - b1) / (a1 - a2)
    else:
        x_cross = None

    print("INTERSECTION:", x_cross)

    return x_cross


def closest_point(x_cross):

    if x_cross is not None:

        idx = np.argmin(np.abs(x - x_cross))

        x_meas = x.iloc[idx]
        y_meas = y.iloc[idx]

        print("CLOSEST POINT:", x_meas, y_meas)

        return x_meas, y_meas

    return None, None


def plot_results(x_cross, x_meas, y_meas, a1, b1, a2, b2):

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=5, alpha=0.4)

    x_line = np.linspace(x.min(), x.max(), 300)

    plt.plot(x_line, a1 * x_line + b1, color="blue", label="Before")
    plt.plot(x_line, a2 * x_line + b2, color="orange", label="After")

    if x_cross is not None:

        idx = np.argmin(np.abs(x - x_cross))

        plt.scatter(x.iloc[idx], y.iloc[idx],
                    color="red", s=120, label="Intersection")

        plt.scatter(x_meas, y_meas,
                    color="green", s=120, label="Closest")

    diff = y.max() - y.min()
    plt.ylim(y.min() - 0.1 * diff, y.max() + 0.1 * diff)

    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    slopes_dict = {}

    build_slopes(slopes_dict)

    windows = np.array(sorted(slopes_dict.keys()))
    slopes = np.array([slopes_dict[w] for w in windows])

    break_idx = breakpoint_detection(slopes, windows)

    a1, b1 = pre_regression(break_idx)
    a2, b2 = post_regression(break_idx)

    x_cross = lines_intersection(a1, b1, a2, b2)

    x_meas, y_meas = closest_point(x_cross)

    plot_results(x_cross, x_meas, y_meas, a1, b1, a2, b2)


main()