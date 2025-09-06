"""

This is refactored Code 

"""
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import numpy as np


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the digits dataset and prepare it for classification.
    
    Returns:
        Tuple containing:
            - Flattened image data (n_samples, n_features)
            - Target labels
            - Original images for visualization
    """
    digits = datasets.load_digits()
    
    # Flatten the images
    n_samples = len(digits.images)
    flattened_data = digits.images.reshape((n_samples, -1))
    
    return flattened_data, digits.target, digits.images


def visualize_training_samples(images: np.ndarray, labels: np.ndarray, 
                               n_samples: int = 4) -> None:
    """Visualize the first n_samples from the dataset.
    
    Args:
        images: Array of digit images
        labels: Corresponding digit labels
        n_samples: Number of samples to visualize
    """
    fig, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    
    # Handle case when n_samples is 1 (axes would not be iterable)
    if n_samples == 1:
        axes = [axes]
    
    for ax, image, label in zip(axes, images[:n_samples], labels[:n_samples]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> svm.SVC:
    """Train a support vector classifier on the provided data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained SVC classifier
    """
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train, y_train)
    return classifier


def visualize_predictions(X_test: np.ndarray, predictions: np.ndarray, 
                          image_shape: Tuple[int, int] = (8, 8),
                          n_samples: int = 4) -> None:
    """Visualize test samples with their predicted labels.
    
    Args:
        X_test: Test features (flattened images)
        predictions: Model predictions
        image_shape: Original shape of the images
        n_samples: Number of samples to visualize
    """
    fig, axes = plt.subplots(nrows=1, ncols=n_samples, figsize=(10, 3))
    
    # Handle case when n_samples is 1 (axes would not be iterable)
    if n_samples == 1:
        axes = [axes]
    
    for ax, flattened_image, prediction in zip(axes, X_test[:n_samples], predictions[:n_samples]):
        ax.set_axis_off()
        image = flattened_image.reshape(image_shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


def reconstruct_classification_report_from_cm(confusion_matrix: np.ndarray) -> str:
    """Reconstruct y_true and y_pred from confusion matrix to build classification report.
    
    Args:
        confusion_matrix: Confusion matrix from model evaluation
        
    Returns:
        Classification report as string
    """
    y_true_reconstructed = []
    y_pred_reconstructed = []
    
    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for true_label in range(len(confusion_matrix)):
        for predicted_label in range(len(confusion_matrix)):
            count = confusion_matrix[true_label][predicted_label]
            y_true_reconstructed.extend([true_label] * count)
            y_pred_reconstructed.extend([predicted_label] * count)
    
    return metrics.classification_report(y_true_reconstructed, y_pred_reconstructed)


def main():
    """Main function to execute the digit recognition pipeline."""
    # Load and prepare data
    data, target, images = load_and_prepare_data()
    
    # Visualize first 4 training samples
    visualize_training_samples(images, target)
    plt.show()
    
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.5, shuffle=False, random_state=42
    )
    
    # Train classifier
    classifier = train_classifier(X_train, y_train)
    
    # Make predictions
    predictions = classifier.predict(X_test)
    
    # Visualize first 4 test predictions
    visualize_predictions(X_test, predictions)
    plt.show()
    
    # Print classification report
    print(
        f"Classification report for classifier {classifier}:\n"
        f"{metrics.classification_report(y_test, predictions)}\n"
    )
    
    # Plot and display confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
    # Rebuild classification report from confusion matrix
    rebuilt_report = reconstruct_classification_report_from_cm(disp.confusion_matrix)
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{rebuilt_report}\n"
    )
    
    plt.show()


if __name__ == "__main__":
    main()
