import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_class_balance(df: pd.DataFrame, target_col: str = "Class"):
    ax = sns.countplot(x=target_col, data=df)
    ax.set_title("Class Distribution")
    return ax


def plot_roc_curve(fpr, tpr, label: str = "ROC"):
    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    return plt.gca()
