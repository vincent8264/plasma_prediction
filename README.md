# Plasma Measurement Prediction
Predicting plasma measurements, electron temperature (t<sub>e</sub>) and electron density (n<sub>e</sub>), from He line ratios using SVM (Support Vector Machine), XGBoost, and deep neural networks.

The dataset is from the paper https://doi.org/10.1063/5.0028000, which originally proposed the SVM method

## Code
lineratio.py runs the prediction of SVM and XGBoost models. lineratio_DNN runs the neural network prediction using Keras

## Results comparison
![comparison](/comparison.png)

## Conclusion
1. Based on the given data, it is proven that SVM can predict electron density and temperature when given the He line ratios.
2. DNN can reach higher accuracies than SVM and XGBoost but use more training time
3. SVM and XGBoost has better results at predicting electron temperatures, while DNN is better at predicting electron density. 
