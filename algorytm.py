import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "orygn_pomiar_0418_normalized.txt"

df = pd.read_csv(filename, sep="\t", header=None)

x = df[0]
y = df[2]
z = df[4]

def least_squares(start, stop):
    x_data = x[start:stop]
    y_data = y[start:stop]
    n = len(x_data)

    if n < 2:
        return np.nan, np.nan
    
    sx = sum(x_data)
    sy = sum(y_data)
    sx2 = sum(a * a for a in x_data)
    sxy = sum(a * b for a, b in zip(x_data, y_data))

    denom = (n * sx2 - sx * sx)
    if denom == 0:
        return np.nan, np.nan
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n

    return a, b

def build_slopes(dictionary, step=10, max_window=None):
    n = len(x)
    if max_window is None:
        max_window = n//2

    start_range = int(n/10)
    for w in range(start_range, max_window, step):
        start = n - w
        end = n
        a, _ = least_squares(start, end)
        dictionary[w] = a

def breakpoint_detection(slopes, windows, step=10):
    da = np.gradient(slopes)
    d2a = np.gradient(da)

    d2a_n = (d2a - np.mean(d2a)) / (np.std(d2a) + 1e-9)
    threshold = np.mean(np.abs(d2a_n)) + 1.5 * np.std(d2a_n)
    candidates = np.where(np.abs(d2a_n) > threshold)[0]
    fallback = False

    if len(candidates) == 0:
        print("No clear breakpoint — fallback")
        jump_idx = len(windows) // 2
        fallback = True
    else:
        jump_idx = candidates[np.argmax(np.abs(d2a_n[candidates]))]

    #break_idx accroding to windows idx which are calculated from behinde
    jump_window = windows[jump_idx]
    #we transform our jump window to normal idx; +step because we want to use the last "good" window
    break_idx = len(x) - jump_window + step
    print("BREAK WINDOW:", jump_window)
    return break_idx, fallback, jump_window

def pre_regression(break_idx):
    window_before = int(0.1 * len(x))
    window_before = np.clip(window_before, 30, 200)
    start = break_idx - window_before
    print(start, break_idx)
    return least_squares(start, break_idx)


def post_regression(break_idx):
    window_after = int(0.1 * len(x))
    window_after = np.clip(window_after, 30, 200)
    end = break_idx + window_after
    return least_squares(break_idx, end)

def lines_intersection(a1, b1, a2, b2):
    if (not np.isnan(a1) and not np.isnan(a2)
        and abs(a1 - a2) > 1e-12):
        return (b2 - b1) / (a1 - a2)
    return None

def closest_point(x_cross):
    if x_cross is None:
        return None, None
    idx = np.argmin(np.abs(x - x_cross))
    return x.iloc[idx], y.iloc[idx]

def log_failed_case(filename, x_cross, handmade_x, fallback):
    x_meas, _ = closest_point(x_cross)
    should_log = False
    if fallback:
        should_log = True
    if x_cross is None or handmade_x is None:
        should_log = True
    elif abs(x_meas - handmade_x) > 0.5:
        should_log = True
    if not should_log:
        return
    
    log_file = "failed_cases.txt"
    already_logged = False
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if filename in line:
                    already_logged = True
                    break
    except FileNotFoundError:
        pass

    if already_logged:
        print("CASE ALREADY LOGGED")
        return
    
    diff = abs(x_meas - handmade_x)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{filename} | " f"alg={x_meas} | " f"handmade={handmade_x}  | "f"diff={diff} | "f"fallback={fallback}\n")
    print("CASE LOGGED")

def plot_results(x_cross, x_meas, y_meas, handmade_x, handmade_y, a1, b1, a2, b2, break_idx):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=5, alpha=0.4)

    x_line = np.linspace(x.min(), x.max(), 300)
    plt.plot(x_line, a1 * x_line + b1, label="Before")
    plt.plot(x_line, a2 * x_line + b2, label="After")

    if break_idx is not None:
        break_x = x.iloc[break_idx]
        plt.axvline(break_x, color="red", linestyle="--", label="Breakpoint")
    if handmade_x is not None:
        plt.scatter(handmade_x, handmade_y, color="purple", s=120, label="Handmade")
    if x_meas is not None:
        plt.scatter(x_meas, y_meas, color="green", s=120, label="Algorithm")
    
    #plt.axvline(x.iloc[int(0.9*len(x))], color="yellow", linestyle="--")
    diff = y.max() - y.min()
    plt.ylim(y.min() - 0.1 * diff, y.max() + 0.1 * diff)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    slopes_dict = {}
    step = int(len(x) / 50)

    build_slopes(slopes_dict, step)

    windows = np.array(sorted(slopes_dict.keys()))
    slopes = np.array([slopes_dict[w] for w in windows])
    print(slopes_dict)

    break_idx, fallback, _ = breakpoint_detection(slopes, windows, step)
    a1, b1 = pre_regression(break_idx)
    a2, b2 = post_regression(break_idx)
    x_cross = lines_intersection(a1, b1, a2, b2)
    x_meas, y_meas = closest_point(x_cross)
    handmade_x = z.iloc[0]
    handmade_y = y.iloc[np.argmin(np.abs(x - handmade_x))]
    log_failed_case(filename, x_cross, handmade_x, fallback)
    plot_results(x_cross, x_meas, y_meas, handmade_x, handmade_y, a1, b1, a2, b2, break_idx)

main()