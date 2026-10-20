| family | model | detail | metric | n_calm | n_volatile | mean_calm | mean_volatile | diff | ci_lo | ci_hi | p_perm | p_placebo | placebo_lo | placebo_hi | q_bh | significant_q05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accuracy | xgb | all test days | correct | 6950 | 2885 | 0.502 | 0.538 | 0.036 | 0.018 | 0.057 | 5.0e-04 | 5.0e-04 | -0.023 | 0.021 | 0.028 | True |
| accuracy | lgbm | all test days | correct | 6950 | 2885 | 0.501 | 0.537 | 0.035 | 0.017 | 0.057 | 1.0e-03 | 0.003 | -0.020 | 0.023 | 0.028 | True |
| accuracy | mlp | all test days | correct | 6950 | 2885 | 0.501 | 0.522 | 0.021 | 0.001 | 0.040 | 0.029 | 0.049 | -0.021 | 0.021 | 0.150 | False |
| accuracy | xgb_shuffled | all test days | correct | 6950 | 2885 | 0.500 | 0.506 | 0.006 | -0.015 | 0.027 | 0.584 | 0.599 | -0.021 | 0.021 | 0.758 | False |
| agreement | lgbm | kernel~lime | 1-jaccard | 310 | 240 | 0.687 | 0.674 | -0.014 | -0.047 | 0.023 | 0.499 | 0.524 | -0.041 | 0.039 | 0.758 | False |
| agreement | lgbm | tree~kernel | 1-jaccard | 310 | 240 | 0.375 | 0.311 | -0.065 | -0.115 | -0.014 | 0.013 | 0.028 | -0.063 | 0.047 | 0.121 | False |
| agreement | lgbm | tree~lime | 1-jaccard | 310 | 240 | 0.683 | 0.658 | -0.025 | -0.063 | 0.010 | 0.244 | 0.267 | -0.050 | 0.040 | 0.531 | False |
| agreement | mlp | kernel~lime | 1-jaccard | 310 | 240 | 0.567 | 0.555 | -0.013 | -0.063 | 0.034 | 0.540 | 0.601 | -0.047 | 0.049 | 0.758 | False |
| agreement | xgb | kernel~lime | 1-jaccard | 310 | 240 | 0.661 | 0.683 | 0.022 | -0.012 | 0.060 | 0.214 | 0.329 | -0.045 | 0.045 | 0.500 | False |
| agreement | xgb | tree~kernel | 1-jaccard | 310 | 240 | 0.330 | 0.293 | -0.037 | -0.086 | 0.014 | 0.156 | 0.175 | -0.056 | 0.052 | 0.438 | False |
| agreement | xgb | tree~lime | 1-jaccard | 310 | 240 | 0.679 | 0.672 | -0.008 | -0.044 | 0.029 | 0.699 | 0.722 | -0.044 | 0.045 | 0.808 | False |
| agreement | xgb_shuffled | kernel~lime | 1-jaccard | 310 | 240 | 0.716 | 0.708 | -0.009 | -0.047 | 0.028 | 0.714 | 0.697 | -0.044 | 0.042 | 0.808 | False |
| agreement | xgb_shuffled | tree~kernel | 1-jaccard | 310 | 240 | 0.353 | 0.345 | -0.008 | -0.055 | 0.036 | 0.730 | 0.779 | -0.059 | 0.057 | 0.808 | False |
| agreement | xgb_shuffled | tree~lime | 1-jaccard | 310 | 240 | 0.699 | 0.668 | -0.031 | -0.075 | 0.011 | 0.275 | 0.174 | -0.043 | 0.047 | 0.570 | False |
| noise_floor | lgbm | kernel | 1-jaccard | 310 | 240 | 0.071 | 0.086 | 0.015 | -0.010 | 0.042 | 0.204 | 0.214 | -0.025 | 0.021 | 0.500 | False |
| noise_floor | lgbm | lime | 1-jaccard | 310 | 240 | 0.379 | 0.356 | -0.023 | -0.062 | 0.017 | 0.120 | 0.283 | -0.046 | 0.033 | 0.375 | False |
| noise_floor | mlp | kernel | 1-jaccard | 310 | 240 | 0.126 | 0.122 | -0.004 | -0.030 | 0.022 | 0.831 | 0.780 | -0.030 | 0.028 | 0.878 | False |
| noise_floor | mlp | lime | 1-jaccard | 310 | 240 | 0.345 | 0.292 | -0.052 | -0.082 | -0.020 | 0.021 | 0.009 | -0.043 | 0.035 | 0.150 | False |
| noise_floor | xgb | kernel | 1-jaccard | 310 | 240 | 0.063 | 0.060 | -0.003 | -0.022 | 0.016 | 0.729 | 0.788 | -0.022 | 0.024 | 0.808 | False |
| noise_floor | xgb | lime | 1-jaccard | 310 | 240 | 0.375 | 0.345 | -0.030 | -0.065 | 0.003 | 0.064 | 0.148 | -0.039 | 0.041 | 0.258 | False |
| noise_floor | xgb_shuffled | kernel | 1-jaccard | 310 | 240 | 0.052 | 0.058 | 0.006 | -0.012 | 0.024 | 0.583 | 0.619 | -0.021 | 0.024 | 0.758 | False |
| noise_floor | xgb_shuffled | lime | 1-jaccard | 310 | 240 | 0.386 | 0.375 | -0.011 | -0.044 | 0.025 | 0.753 | 0.596 | -0.043 | 0.036 | 0.811 | False |
| retrain | lgbm | bootstrap | 1-jaccard | 310 | 240 | 0.687 | 0.671 | -0.015 | -0.034 | 0.004 | 0.246 | 0.165 | -0.019 | 0.024 | 0.531 | False |
| retrain | lgbm | seed | 1-jaccard | 310 | 240 | 0.422 | 0.413 | -0.009 | -0.032 | 0.014 | 0.531 | 0.526 | -0.029 | 0.029 | 0.758 | False |
| retrain | mlp | bootstrap | 1-jaccard | 310 | 240 | 0.568 | 0.536 | -0.033 | -0.057 | -0.008 | 0.024 | 0.071 | -0.033 | 0.036 | 0.150 | False |
| retrain | mlp | seed | 1-jaccard | 310 | 240 | 0.698 | 0.672 | -0.026 | -0.044 | -0.007 | 0.002 | 0.014 | -0.020 | 0.021 | 0.028 | True |
| retrain | xgb | bootstrap | 1-jaccard | 310 | 240 | 0.689 | 0.686 | -0.003 | -0.021 | 0.016 | 0.736 | 0.815 | -0.021 | 0.021 | 0.808 | False |
| retrain | xgb | seed | 1-jaccard | 310 | 240 | 0.370 | 0.384 | 0.014 | -0.011 | 0.040 | 0.324 | 0.442 | -0.034 | 0.037 | 0.604 | False |
| retrain | xgb_shuffled | bootstrap | 1-jaccard | 310 | 240 | 0.710 | 0.721 | 0.011 | -0.004 | 0.027 | 0.314 | 0.302 | -0.020 | 0.020 | 0.604 | False |
| retrain | xgb_shuffled | seed | 1-jaccard | 310 | 240 | 0.405 | 0.410 | 0.005 | -0.028 | 0.038 | 0.711 | 0.753 | -0.032 | 0.035 | 0.808 | False |
| perturbation | lgbm | kernel eps=0.01 | top1_changed | 310 | 240 | 0.117 | 0.104 | -0.013 | -0.054 | 0.026 | 0.565 | 0.572 | -0.042 | 0.047 | 0.758 | False |
| perturbation | lgbm | kernel eps=0.05 | top1_changed | 310 | 240 | 0.300 | 0.254 | -0.046 | -0.104 | 0.013 | 0.208 | 0.195 | -0.053 | 0.077 | 0.500 | False |
| perturbation | lgbm | lime eps=0.01 | top1_changed | 310 | 240 | 0.295 | 0.228 | -0.067 | -0.134 | -9.3e-04 | 0.028 | 0.088 | -0.081 | 0.074 | 0.150 | False |
| perturbation | lgbm | lime eps=0.05 | top1_changed | 310 | 240 | 0.311 | 0.262 | -0.048 | -0.107 | 0.013 | 0.101 | 0.212 | -0.080 | 0.069 | 0.336 | False |
| perturbation | lgbm | tree eps=0.01 | top1_changed | 310 | 240 | 0.085 | 0.094 | 0.008 | -0.020 | 0.036 | 0.572 | 0.596 | -0.030 | 0.031 | 0.758 | False |
| perturbation | lgbm | tree eps=0.05 | top1_changed | 310 | 240 | 0.295 | 0.256 | -0.039 | -0.087 | 0.013 | 0.087 | 0.192 | -0.046 | 0.066 | 0.327 | False |
| perturbation | mlp | kernel eps=0.01 | top1_changed | 310 | 240 | 0.086 | 0.067 | -0.019 | -0.055 | 0.022 | 0.285 | 0.367 | -0.035 | 0.044 | 0.570 | False |
| perturbation | mlp | kernel eps=0.05 | top1_changed | 310 | 240 | 0.182 | 0.165 | -0.016 | -0.067 | 0.036 | 0.461 | 0.559 | -0.056 | 0.057 | 0.758 | False |
| perturbation | mlp | lime eps=0.01 | top1_changed | 310 | 240 | 0.290 | 0.208 | -0.082 | -0.135 | -0.033 | 0.006 | 0.017 | -0.062 | 0.069 | 0.067 | False |
| perturbation | mlp | lime eps=0.05 | top1_changed | 310 | 240 | 0.392 | 0.283 | -0.109 | -0.166 | -0.054 | 0.002 | 0.002 | -0.064 | 0.075 | 0.028 | True |
| perturbation | xgb | kernel eps=0.01 | top1_changed | 310 | 240 | 0.102 | 0.085 | -0.017 | -0.053 | 0.021 | 0.384 | 0.386 | -0.037 | 0.044 | 0.673 | False |
| perturbation | xgb | kernel eps=0.05 | top1_changed | 310 | 240 | 0.242 | 0.219 | -0.022 | -0.074 | 0.027 | 0.522 | 0.449 | -0.059 | 0.056 | 0.758 | False |
| perturbation | xgb | lime eps=0.01 | top1_changed | 310 | 240 | 0.258 | 0.233 | -0.025 | -0.079 | 0.032 | 0.363 | 0.512 | -0.065 | 0.078 | 0.655 | False |
| perturbation | xgb | lime eps=0.05 | top1_changed | 310 | 240 | 0.292 | 0.249 | -0.044 | -0.097 | 0.011 | 0.141 | 0.255 | -0.065 | 0.079 | 0.417 | False |
| perturbation | xgb | tree eps=0.01 | top1_changed | 310 | 240 | 0.108 | 0.079 | -0.029 | -0.059 | 0.001 | 0.049 | 0.098 | -0.033 | 0.035 | 0.229 | False |
| perturbation | xgb | tree eps=0.05 | top1_changed | 310 | 240 | 0.293 | 0.263 | -0.031 | -0.087 | 0.023 | 0.183 | 0.304 | -0.056 | 0.059 | 0.488 | False |
| perturbation | xgb_shuffled | kernel eps=0.01 | top1_changed | 310 | 240 | 0.130 | 0.122 | -0.008 | -0.056 | 0.041 | 0.707 | 0.728 | -0.044 | 0.042 | 0.808 | False |
| perturbation | xgb_shuffled | kernel eps=0.05 | top1_changed | 310 | 240 | 0.301 | 0.297 | -0.004 | -0.059 | 0.050 | 0.900 | 0.894 | -0.064 | 0.062 | 0.916 | False |
| perturbation | xgb_shuffled | lime eps=0.01 | top1_changed | 310 | 240 | 0.352 | 0.350 | -0.002 | -0.074 | 0.069 | 0.942 | 0.975 | -0.079 | 0.085 | 0.942 | False |
| perturbation | xgb_shuffled | lime eps=0.05 | top1_changed | 310 | 240 | 0.370 | 0.386 | 0.016 | -0.056 | 0.087 | 0.595 | 0.683 | -0.077 | 0.086 | 0.758 | False |
| perturbation | xgb_shuffled | tree eps=0.01 | top1_changed | 310 | 240 | 0.112 | 0.109 | -0.003 | -0.037 | 0.031 | 0.868 | 0.884 | -0.037 | 0.039 | 0.900 | False |
| perturbation | xgb_shuffled | tree eps=0.05 | top1_changed | 310 | 240 | 0.290 | 0.308 | 0.018 | -0.030 | 0.066 | 0.453 | 0.536 | -0.051 | 0.061 | 0.758 | False |
| headline | pooled | retrain seed | 1-jaccard | 930 | 720 | 0.497 | 0.490 | -0.007 | -0.022 | 0.009 | 0.492 | 0.507 | -0.021 | 0.021 | 0.758 | False |
| headline | pooled | retrain bootstrap | 1-jaccard | 930 | 720 | 0.648 | 0.631 | -0.017 | -0.033 | -9.0e-04 | 0.064 | 0.093 | -0.018 | 0.021 | 0.258 | False |
| headline | pooled | perturbation primary eps=0.05 | top1_changed | 930 | 720 | 0.257 | 0.228 | -0.029 | -0.060 | 0.003 | 0.029 | 0.133 | -0.034 | 0.040 | 0.150 | False |
| headline | pooled | agreement all pairs | 1-jaccard | 2170 | 1680 | 0.569 | 0.549 | -0.020 | -0.041 | 0.002 | 0.102 | 0.130 | -0.027 | 0.023 | 0.336 | False |
