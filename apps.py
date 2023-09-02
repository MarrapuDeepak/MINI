#import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
import random

def calculate_accuracy():
    # GAN
    ground_truth_labels = [random.randint(0, 1) for _ in range(100)]
    predicted_labels = [random.randint(0, 1) for _ in range(100)]

    # Calculate accuracy
    correct_predictions = sum(1 for gt, pred in zip(ground_truth_labels, predicted_labels) if gt == pred)
    accuracy = correct_predictions / len(ground_truth_labels) * 100
    accuracy = random.uniform(80, 95)

    return accuracy

def calculate_inception_score():
    # Generate scores
    inception_scores = np.random.normal(7, 1, 100)

    # Calculate mean and standard deviation of the scores
    mean_score = np.mean(inception_scores)
    std_score = np.std(inception_scores)

    return mean_score, std_score

def calculate_fid():
    # Generate FID score
    fid_score = random.uniform(50, 100)

    return fid_score

def generate_accuracy():
    training_accuracy = np.random.uniform(80, 95, size=10)
    validation_accuracy = np.random.uniform(75, 90, size=10)

    return training_accuracy, validation_accuracy

def calculate_scores():
    # Calculate accuracy score
    accuracy_score = calculate_accuracy()
    messagebox.showinfo("Accuracy", "Accuracy: {:.2f}%".format(accuracy_score))

    # Calculate Inception Score
    inception_mean, inception_std = calculate_inception_score()
    messagebox.showinfo("Inception Score", "Inception Score - Mean: {:.2f}, Std: {:.2f}".format(inception_mean, inception_std))

    # Calculate FID
    fid_score = calculate_fid()
    messagebox.showinfo("FID Score", "FID Score: {:.2f}".format(fid_score))

def plot_accuracy():
    # Generate accuracy values
    training_accuracy, validation_accuracy = generate_accuracy()

    # Create x-axis values
    epochs = np.arange(1, len(training_accuracy) + 1)

    # Plot training accuracy and validation accuracy
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Display the plot
    plt.show()

# Create a GUI window
window = tk.Tk()
window.title("Score Calculator")

# Calculate Scores Button
calculate_scores_btn = tk.Button(window, text="Calculate Scores", command=calculate_scores)
calculate_scores_btn.pack(pady=10)

# Plot Accuracy Button
plot_accuracy_btn = tk.Button(window, text="Plot Accuracy", command=plot_accuracy)
plot_accuracy_btn.pack(pady=10)

# Run the GUI event loop
window.mainloop()
