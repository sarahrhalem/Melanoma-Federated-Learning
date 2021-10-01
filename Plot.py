
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd




# plot ROC curve
def plot_roc(results, title):
    fig=plt.figure(figsize=(10,6))
    AUC= "{:.3f}".format(results['AUC'])
    score= "AUC: " + str(AUC)
    plt.plot(results['fpr'],results['tpr'], label=score)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.show()
    
    return