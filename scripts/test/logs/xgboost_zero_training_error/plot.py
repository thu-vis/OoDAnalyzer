import numpy as np
import matplotlib.pyplot as plt

def process_data(filename):
    file_strings = open(filename, "r").read()
    lines = file_strings.split("\n")
    val_error = []
    train_error = []
    for line in lines:
        line = line.split("\t")
        v = float(line[1].split(":")[1])
        t = float(line[2].split(":")[1])
        val_error.append(v)
        train_error.append(t)
    return val_error, train_error

def plot(filename):
    val_error, train_error = process_data(filename)
    x = range(len(val_error))
    plt.plot(x, val_error, mec="r", mfc="w", label="test error")
    plt.plot(x, train_error, ms=10, label="training error")
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("error rate")
    plt.title("training and test error rate of sEMG dataset (XGBoost)")
    plt.show()

if __name__ == '__main__':
    plot(".\loss.txt")
