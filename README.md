# Bern02: Hierarchical Models and Hypothesis Testing for Hotel Towel Reuse

**Objective:** This exercise evaluates the effectiveness of descriptive social norms on hotel towel reuse using a collection of randomized case-control studies.

---

## 1. Model Choice and Justification

When analyzing data pooled from multiple independent studies, standard pooled or unpooled approaches fail to capture the underlying data structure:

*   **Complete pooling** ignores study-to-study variation (e.g., different locations, guest demographics, and message phrasing).
*   **No pooling** estimates each study separately, ignoring shared information and inflating uncertainty.

To address these limitations, we implement a **Hierarchical Binomial Logistic Model**. This approach accounts for both the binomial outcome structure and between-study heterogeneity through random effects.

### Mathematical Formulation

We model the number of successes $y_{ij}$ out of $n_{ij}$ total trials for group $i$ in study $j$:

$$y_{ij} \sim \text{Binomial}(n_{ij}, p_{ij})$$

$$\text{logit}(p_{ij}) = \ln\left(\frac{p_{ij}}{1 - p_{ij}}\right) = \beta_0 + \beta_{\text{social}} x_{ij} + \mu_{0j} + \mu_{1j} x_{ij}$$

Where the parameters and variables are defined as:
*   $y_{ij}$: Number of customers who reused towels.
*   $n_{ij}$: Total number of customers.
*   $x_{ij} \in \{0, 1\}$: Group indicator ($0 = \text{control}$, $1 = \text{social norm}$).
*   $\beta_0$: Population-level baseline log-odds of towel reuse in the control group.
*   $\beta_{\text{social}}$: Overall population-level effect of the social norm intervention on the log-odds scale.
*   $\mu_{0j} \sim \mathcal{N}(0, \sigma_0^2)$: Random intercept for study $j$, capturing varying baseline reuse rates across hotels.
*   $\mu_{1j} \sim \mathcal{N}(0, \sigma_1^2)$: Random slope for study $j$, capturing between-study heterogeneity in intervention efficacy.

---

## 2. Estimation Method and Implementation

The model is implemented in Python using **Bambi**.

### Model Definition and Fitting

```python
import bambi as bmb
import arviz as az
import numpy as np

# Define the hierarchical binomial logistic model
model_hierarchical = bmb.Model(
    "p(reuse, total) ~ group + (group | study)",
    data=combined_data,
    family="binomial"
)

# Fit the model using MCMC sampling
results = model_hierarchical.fit(
    draws=2000, 
    tune=2000, 
    target_accept=0.95,
    return_inferencedata=True,
    random_seed=42
)
```

To ensure reliable posterior sampling, we configured Markov Chain Monte Carlo (MCMC) with 2,000 tuning steps, 2,000 sampling draws, and an increased `target_accept` rate of `0.95` to prevent divergent transitions.

---

## 3. Results Analysis and Reliability

We summarize the posterior distribution using a **90%** Equal-Tail Interval (ETI) and inspect diagnostic metrics (`r_hat` and effective sample size `ess_bulk`) to verify convergence and reliability.

```python
summary_table = az.summary(results, ci_prob=0.90)
columns_to_key = ["mean", "eti90_lb", "eti90_ub", "r_hat", "ess_bulk"]

# Print summary filtered for key metrics
print(summary_table)
```

### Posterior Summary Table

| Parameter | Mean | 90% ETI Lower | 90% ETI Upper | $\hat{R}$ (r_hat) | Bulk ESS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Intercept ($\beta_0$) | 0.42 | -0.30 | 1.10 | 1.00 | 2294 |
| group[social] ($\beta_{\text{social}}$) | 0.18 | -0.05 | 0.37 | 1.00 | 3870 |
| study_sigma ($\sigma_0$) | 1.17 | 0.66 | 2.00 | 1.00 | 1902 |
| group\|study_sigma[social] ($\sigma_1$) | 0.19 | 0.01 | 0.55 | 1.00 | 2215 |

**Reliability Check:** All parameters exhibit an $\hat{R}$ value of `1.00` and robust bulk effective sample sizes ($\text{ESS} > 1900$), confirming that the MCMC chains successfully converged and mixed well.

---

## 4. Hypothesis Testing and Conclusion

We evaluate whether the social norm intervention increases towel reuse by testing the posterior distribution of the intervention effect.

*   **Null Hypothesis ($H_0$):** $\beta_{\text{social}} \le 0$ (No positive effect)
*   **Alternative Hypothesis ($H_1$):** $\beta_{\text{social}} > 0$ (Positive effect)

### Computation of Posterior Probability and Odds Ratios

```python
# Extract posterior draws for the social norm effect
posterior_draws = results.posterior["group"].values.flatten()

# Calculate probability of a positive effect
prob_positive = np.mean(posterior_draws > 0)

# Convert log-odds to Odds Ratios (OR)
odds_ratios = np.exp(posterior_draws)
mean_or = np.mean(odds_ratios)

print(f"Probability of positive effect: {prob_positive:.4f}")
print(f"Mean Odds Ratio: {mean_or:.4f}")
```

### Findings

*   **Posterior Probability:** $P(\beta_{\text{social}} > 0 \mid \text{data}) \approx 0.9215$ (92.15%).
*   **Expected Odds Ratio:** $\text{Mean OR} \approx 1.205$.

### Interpretation

Although the mean odds ratio indicates a positive trend ($\approx 20\%$ increase in odds of reuse), the posterior probability of a positive effect (92.15%) falls short of a traditional $95\%$ threshold. Combined with the $90\%$ ETI spanning $-0.05$ to $0.37$ (which includes zero), this indicates **weak or inconclusive evidence** for the intervention once between-study heterogeneity is properly controlled.
