import matplotlib.pyplot as plt


def plot_signal(x, highlight=None):
    plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    if highlight:
        plt.axvspan(highlight[0], highlight[1], color='red', alpha=0.5)
    plt.plot(x)
    plt.show()
