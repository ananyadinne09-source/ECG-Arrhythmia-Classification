\# Explainable AI-Based ECG Arrhythmia Classification Using Deep Learning



A deep learning-based system for classifying 12-lead electrocardiogram (ECG) recordings into five rhythm classes, with Explainable AI (XAI) techniques used to visualize which temporal regions of the ECG input contributed most strongly to the model's predictions.



\## Overview



Electrocardiograms contain important information about cardiac electrical activity, but interpreting large volumes of ECG recordings manually can be time-consuming. This project develops a deep learning pipeline for automated ECG rhythm classification and complements the predictions with explainability methods.



The system uses an InceptionTime-style 1D convolutional neural network with Squeeze-and-Excitation (SE) blocks and residual connections.



The final model was evaluated on a held-out test set and achieved:



| Metric | Result |

|---|---:|

| Test Accuracy | \*\*97.33%\*\* |

| Weighted F1-score | \*\*97.32%\*\* |

| Macro F1-score | \*\*96.59%\*\* |

| Macro ROC-AUC | \*\*99.53%\*\* |

| Weighted ROC-AUC | \*\*99.67%\*\* |



\---



\## Classes



The final classification task contains five ECG rhythm classes:



1\. Atrial Fibrillation

2\. Sinus Bradycardia

3\. Sinus Rhythm

4\. Sinus Tachycardia

5\. Supraventricular Tachycardia



\---



\## Dataset



The project uses the \*\*Chapman-Shaoxing/Ningbo 12-lead ECG dataset\*\*.



The raw dataset contains 12-lead ECG recordings sampled at 500 Hz. Each processed recording is represented as a 10-second ECG segment with:



```text

12 leads × 5000 samples

