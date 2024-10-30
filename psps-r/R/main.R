#---------------------------------------------------------------
# psps: PoSt-Prediction Sumstats-based inference
# Jiacheng Miao and Qiongshi Lu
# Available from https://arxiv.org/abs/2405.20039
#---------------------------------------------------------------

#' psps
#'
#' \code{psps} function Task-Agnostic Machine Learning-Assisted Inference.
#' @param est_lab_y: a K-dimensional Array of Point estimates using Y in labeled data.
#' @param est_lab_yhat: a K-dimensional Array of Point estimates using Yhat in labeled data.
#' @param est_unlab_yhat: a K-dimensional Array of Point estimates using Yhat in unlabeled data.
#' @param Sigma: a 3K x 3K Variance-covariance matrix for the above three estimators (Note: not the asymptotic variance).
#' @param alpha Specifies the confidence level as 1 - alpha for confidence intervals.
#' @return A summary table presenting point estimates, standard error, confidence intervals (1 - alpha), P-values for ML-assisted inference.
#' @export

psps <- function(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma, alpha = 0.05) {
  # Prepare inputs
  K <- length(est_lab_y)
  Sigma <- as.matrix(Sigma)
  v_est <- Sigma[1:K, 1:K]
  r <- Sigma[(K + 1):(2 * K), 1:K]
  v_eta_lab <- Sigma[(K + 1):(2 * K), (K + 1):(2 * K)]
  v_eta_unlab <- Sigma[(2 * K + 1):(3 * K), (2 * K + 1):(3 * K)]
  V <- v_eta_lab + v_eta_unlab
  omega_0 <- solve(V) %*% r

  # Get the results
  est <- est_lab_y + omega_0 %*% (est_unlab_yhat - est_lab_yhat)
  standard_errors <- sqrt(diag(v_est - t(omega_0) %*% r))
  lower_ci <- est - qnorm(1 - alpha / 2) * standard_errors
  upper_ci <- est + qnorm(1 - alpha / 2) * standard_errors
  p_values <- 2 * pnorm(abs(est / standard_errors), 0, 1, lower.tail = F)
  output_table <- data.frame(
    Estimate = est, Std.Error = standard_errors,
    Lower.CI = lower_ci, Upper.CI = upper_ci, P.value = p_values
  )
  colnames(output_table) <- c("Estimate", "Std.Error", "Lower.CI", "Upper.CI", "P.value")
  return(output_table)
}
