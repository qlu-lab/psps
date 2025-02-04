## psps: Post-prediction summary statistics-based inference

<img src="fig.png" align="right" height="300" style="float:right; height:150px;">

Repository for [R](https://github.com/qlu-lab/psps?tab=readme-ov-file#install-r-package-psps) and [Python](https://github.com/qlu-lab/psps?tab=readme-ov-file#python-package) packages `psps` that implements [Task-Agnostic Machine Learning-Assisted Inference](https://arxiv.org/abs/2405.20039). 


`psps` is a simple and task-agnotsic protocol for valid and efficient machine learning (ML)-assited infernece. It can be easily adapted to a variety of statistical tasks.

# R Package
## Install R Package `psps`
```R
# install.packages("devtools")
devtools::install_github("qlu-lab/psps", subdir = "psps-r")
```

## TL;DR
```R
fit_psps <- psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
```
Suppose we are interested in estimating a K-dimensional parameter, then
* `est_lab_y`: a K-dimensional vector of Point estimates using `Y` in labeled data.
* `est_lab_yhat`: a K-dimensional vector of  Point estimates using `Yhat` in labeled data.
* `est_unlab_yhat`: a K-dimensional vector of  Point estimates using `Yhat` in unlabeled data.
* `Sigma`: a 3K x 3K Variance-covariance matrix for the above three estimators (Note: not the asymptotic variance).

## Example 
Here is an example of psps for logistic regression. For other tasks, simply replace logistic regression with other algorithms to produce summary statistics.
```R
# Load the package
library(psps)

# Load the Labeled and unlabelled data
lab <- read.csv("./psps/test_data/lab.csv")
unlab <- read.csv("./psps/test_data/unlab.csv")

```

### Logistic regression
```R
# Point Estimates Preparation

## Fit logistic regression models and extract the second coefficient (typically the slope)
est_lab_y <- coef(glm(Y ~ X, data = lab, family = binomial("logit")))[2]
est_lab_yhat <- coef(glm(Yhat ~ X, data = lab, family = binomial("logit")))[2]
est_unlab_yhat <- coef(glm(Yhat ~ X, data = unlab, family = binomial("logit")))[2]

# Variance-Covariance Matrix Preparation

## Bootstrap for Covariance Calculation between est_lab_y and est_lab_yhat
B <- 1000  # Number of bootstrap iterations
n <- nrow(lab)  # Total observations in labeled data

### Initialize matrices to store bootstrap estimates
est_lab_y_boot <- vector("numeric", B)
est_lab_yhat_boot <- vector("numeric", B)

### Perform bootstrap to estimate the variance-covariance of est_lab_y and est_lab_yhat
set.seed(123)  # Set seed for reproducibility
for (i in 1:B) {
  boot_indices <- sample(1:n, n, replace = TRUE)
  boot_lab <- lab[boot_indices, ]
  est_lab_y_boot[i] <- coef(glm(Y ~ X, data = boot_lab, family = binomial("logit")))[2]
  est_lab_yhat_boot[i] <- coef(glm(Yhat ~ X, data = boot_lab, family = binomial("logit")))[2]
}

### Compute the covariance matrix
Sigma <- matrix(0, nrow = 3, ncol = 3)
Sigma[1:2, 1:2] <- cov(cbind(est_lab_y_boot, est_lab_yhat_boot))  # Covariance of bootstrap estimates
Sigma[3, 3] <- summary(glm(Yhat ~ X, data = unlab, family = binomial("logit")))$coefficients[2, "Std. Error"]^2

## One-step Debiasing with the psps Method
fit_psps <- psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
print(fit_psps)

### Example output from psps function:
# Estimate  Std.Error  Lower.CI  Upper.CI  P.value
# 0.7537844 0.09366147 0.5702113 0.9373575 8.418062e-16
```

# Python package
## Install Python Package `psps`
```bash
pip install psps_py
```

## TL;DR
psps inputs summary statistics from three separate analyses and returns the ML-assisted estimator.
```R
fit_psps = psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
```
Suppose we are interested in estimating a K-dimensional parameter, then
* `est_lab_y`: a K-dimensional vector of Point estimates using `Y` in labeled data.
* `est_lab_yhat`: a K-dimensional vector of  Point estimates using `Yhat` in labeled data.
* `est_unlab_yhat`: a K-dimensional vector of  Point estimates using `Yhat` in unlabeled data.
* `Sigma`: a 3K x 3K Variance-covariance matrix for the above three estimators (Note: not the asymptotic variance).

## Example 
Here is an example of `psps` for logistic regression. For other tasks, simply replace logistic regression with other algorithms to produce summary statistics.
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
# Load the package
from psps_py import psps

# Load data
lab = pd.read_csv("./psps/test_data/lab.csv")
unlab = pd.read_csv("./psps/test_data/unlab.csv")
```


### Logistic regression
```python
# Fit logistic regression models and extract the second coefficient
est_lab_y = glm('Y ~ X', data=lab, family=sm.families.Binomial()).fit().params['X']
est_lab_yhat = glm('Yhat ~ X', data=lab, family=sm.families.Binomial()).fit().params['X']
est_unlab_yhat = glm('Yhat ~ X', data=unlab, family=sm.families.Binomial()).fit().params['X']

# Bootstrap for covariance calculation
B = 1000  # Number of bootstrap iterations
n = len(lab)  # Total observations in labeled data
np.random.seed(123)  # For reproducibility

est_lab_y_boot = np.zeros(B)
est_lab_yhat_boot = np.zeros(B)

for i in range(B):
    boot_indices = np.random.choice(lab.index, size=n, replace=True)
    boot_lab = lab.loc[boot_indices]
    est_lab_y_boot[i] = glm('Y ~ X', data=boot_lab, family=sm.families.Binomial()).fit().params['X']
    est_lab_yhat_boot[i] = glm('Yhat ~ X', data=boot_lab, family=sm.families.Binomial()).fit().params['X']

# Compute the covariance matrix
Sigma = np.zeros((3, 3))
Sigma[0:2, 0:2] = np.cov(np.vstack([est_lab_y_boot, est_lab_yhat_boot]))
Sigma[2, 2] = glm('Yhat ~ X', data=unlab, family=sm.families.Binomial()).fit().bse['X']**2

# Run psps
fit_psps = psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
print(fit_psps)

### Example output from psps function:
# Estimate  Std.Error  Lower.CI  Upper.CI  P.value
# 0.755438   0.090828  0.577417  0.933458  9.007197e-17
```

## Analysis script
We provide the script for analysis in the `psps` paper [here](https://github.com/jmiao24/psps_analysis).

## Contact 
Please submit an issue or contact Jiacheng (jiacheng.miao@wisc.edu) or Qiongshi (qlu@biostat.wisc.edu) for questions.

## Reference
[Task-Agnostic Machine Learning-Assisted Inference](https://arxiv.org/abs/2405.20039)

[Assumption-Lean and Data-Adaptive Post-Prediction Inference](https://arxiv.org/abs/2311.14220)

[Valid Inference for Machine Learning-Assisted GWAS](https://www.medrxiv.org/content/10.1101/2024.01.03.24300779v1)

## Familial links
* [POP-TOOLS](https://github.com/qlu-lab/POP-TOOLS) (**PO**st-**P**rediction **TOOLS**) is a toolkit for conducting valid and powerful machine learning (ML)-assisted genetic association studies. It currently implements `POP-GWAS`, where statistical and computational methods are optimized for GWAS applications.
