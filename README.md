# CNN Image Classification Portfolio Project

Welcome to my portfolio project on image classification using Convolutional Neural Networks (CNN). This project showcases my ability to build, train, and evaluate a CNN model for image classification tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Results and Insights](#results-and-insights)
- [Future Work](#future-work)
- [Contact](#contact)

## Project Overview
This project involves developing a CNN model to classify images into predefined categories. The goal is to demonstrate the process of building an effective CNN from scratch and to showcase the results obtained from the model.

## Objectives
- Implement a CNN for image classification.
- Preprocess and augment the dataset for improved performance.
- Train the CNN model and monitor its performance.
- Evaluate the model on a test set and visualize the results.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - TensorFlow (for model implementation)
  - NumPy (for numerical operations)
  - Matplotlib (for visualization)
  - Pandas (for data manipulation)
  - Scikit-learn (for evaluation metrics)


## Project Workflow

The project is organized in the following steps, each corresponding to a section in the Jupyter Notebook `CNNimageClassification.ipynb`:

1. **Import Libraries**: Import all necessary libraries and modules.
2. **Data Extraction**: Load the dataset and perform preprocessing steps such as normalization and data augmentation.
3. **CNN Architecture Definition**: Define the CNN architecture, including layers and activation functions. The architecture used in this project is based on LeNet-5.
4. **MaxPooling**: Implement max pooling to reduce the spatial dimensions of the feature maps.
5. **Same Padding**: Use same padding to ensure the output size matches the input size after convolutions.
6. **Increasing Number of Convolutional Layers**: Experiment with increasing the number of convolutional layers to capture more complex features.
7. **Kernel Size Reduction**: Reduce the kernel size to capture finer details.
8. **Regularization Techniques**: Apply regularization techniques to prevent overfitting.
    - **Batch Normalization**: Apply batch normalization after convolutional layers.
    - **Dropout**: Use dropout in fully connected and convolutional layers to improve generalization.
9. **Summary**: Summarize the model architecture and parameters.
10. **Data Augmentation**: Perform data augmentation to artificially increase the size of the training dataset and improve model robustness.

## Results and Insights

The project includes detailed results and insights:
- Accuracy and loss plots to track the model's performance during training.
- Confusion matrix and classification report for detailed performance analysis.
- Sample predictions to visualize the model's classification capabilities.

## Future Work

Potential improvements and future directions for this project include:
- Experimenting with different CNN architectures and hyperparameters.
- Incorporating more advanced data augmentation techniques.
- Applying transfer learning using pre-trained models.

## Contact

If you have any questions or feedback, feel free to reach out:
- [LinkedIn](https://www.linkedin.com/in/vlad-plyusnin-b65b501b2/)
- [GitHub](https://github.com/VladPlusIn)

Thank you for reviewing my portfolio project on CNN image classification!
