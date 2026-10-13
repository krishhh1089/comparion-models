# MLP from scratch vs library MLP: Student Course Grades

## What this project does
Compares a NumPy MLP implemented from scratch with `sklearn.neural_network.MLPClassifier` on the UCI Student Performance dataset.

- Dataset: UCI Student Performance (Math or Portuguese course)
- Target: binary classification — **Pass** if final grade `G3 >= 10`, otherwise **Fail**
- Inputs: student attributes and earlier grades (`G1`, `G2`); final grade `G3` is excluded from features to prevent target leakage
- Scratch network: one hidden layer, sigmoid activation, binary cross-entropy, backpropagation, mini-batch gradient descent
- Library network: scikit-learn MLP with logistic/sigmoid activation; uses Adam, a gradient-based optimizer. This is not identical to vanilla gradient descent, so the difference is documented.

## Setup
Python 3.9+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python compare_mlps.py
```
Default is Mathematics. To use Portuguese course grades:
```bash
python compare_mlps.py --subject portuguese
```
To change training epochs:
```bash
python compare_mlps.py --epochs 300
```

The script prints test accuracy and training time and displays/saves:
- `learning_curves.png` — epoch vs training loss and validation loss for both models
- `confusion_matrices.png` — confusion matrices for both models

## Confusion matrix
Rows are actual labels and columns are predicted labels, with order `[Fail, Pass]`:

```text
                 Predicted
               Fail    Pass
Actual Fail     TN      FP
Actual Pass     FN      TP
```

## Notes on evaluation
- Train/validation/test split: 64% / 16% / 20% of the complete dataset.
- Stratified splitting preserves approximate class proportions.
- Preprocessing is fitted only on the training split to avoid data leakage.
- The reported training time covers model fitting, not data download or preprocessing.
- The library's per-epoch training loss includes its L2 regularization term; its validation loss is plain binary cross-entropy. The curves are useful for trend comparison but are not perfectly identical objectives.
- Results depend on package versions and runtime. Run the script to obtain the actual accuracy, timing, and plots on your machine.

## Dataset citation
Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5TG7T
