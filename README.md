# Bern02-Hierarchical-models-and-testing

The purpose of this exercise is to use a collection of randomized case-control studies to determine the effectiveness of descriptive social norms on reuse of towels at hotels.

## Model Formulation

Since we are considering multiple different studies we need both an appropriate family distribution for the response variable and to consider heterogeneity between studies. We want to account for both the outcome structure and variations between individual studies. Thus we formulate a Hierarchical Binomial Logistic Model, where $p(\text{reuse}, \text{total})$ specifies proportion of successes out of total trials, as follows:

$y_{ij} \sim \text{Binomial}(n_{ij}, p_{ij})$

$\text{logit}(p_{ij}) = \ln{\frac{p_{ij}}{1-p_{ij}}} = \beta_0 + \beta_{\text{social}} * x_{ij} + \mu_{0j} + \mu_{1j} * x_{ij}$

where each variable is defined as

$y_{ij}$: number of people who reuse towels (reuse) in study j and group i

$n_{ij}$: total number of customers (total) in study j and group i

$x_{ij}$ in {0,1}: Group indicator (0=control, 1=social norm)

$\beta_0$ : Population-level baseline log-odds of towel reuse in the control group

$\beta_{\text{social}}$: Overall population-level effect of the social norm intervention on the log-odds scale

$\mu_{0j} \sim \mathcal{N}(0,\sigma^2_0)$: Random intercept for study j, capture varying baseline reuse rates across hotels

$\mu_{1j} \sim \mathcal{N}(0,\sigma^2_1)$: Random slope for study j, capture between-study heterogeneity in intervention efficacy


Using Bambi, we have the following code to describe the model
```
model_hierarchical = bmb.Model(
    "p(reuse, total) ~ group + (group | study)",
    data=combined_data,
    family="binomial"
)
```

We account for the heterogeneity between hotels (from things like location, wording of message, guest demographic, etc) by using random effects (ie $\mu_{0j}$ and $\mu_{1j}$) to allow for adaption between aggregate data. A fixed-effects model would assume a single underlying true effect, underestimating standard errors.

## Model Analysis
We want to consider the parameter values of the posterior means and the 90% probability interval. We must first fit the model to get the posterior values. For reproducibility, a random seed of 42 was chosen.

```
results = model_hierarchical.fit(
    draws=2000, 
    tune=2000, 
    target_accept=0.95,
    return_inferencedata=True,
    random_seed=42
)
```

We then want to summarize the fit and consider a confidence level of 90%. 

```
summary_table = az.summary(results, ci_prob=0.90)
columns_to_keep = ["mean", summary_table.columns[2], summary_table.columns[3], "r_hat", "ess_bulk"]
print(summary_table[columns_to_keep])
```
This results in the following table

| | mean | eti90_lb | eti90_ub | r_hat | ess_bulk |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Intercept                  |  0.42   |  -0.3     | 1.1  |1.00    | 2294|
|group[social]              | 0.178   |-0.049     |0.37  |1.00    | 3870|
| 1 study_sigma             |   1.17  |   0.66    |    2 | 1.00   |  1902|
| group|study_sigma[social] |  0.188  |  0.013    | 0.55 | 1.00   |  2215|
| 1, study[1]               |   -0.97  |   -1.7   | -0.23|  1.00   |  2348|
| 1,study[2]                |   -0.9   |  -1.6    |-0.16 | 1.00    | 2345|
|1,study[3]                 | -0.13    |-0.84     |0.61  |1.00     |2385|
|1,study[4]                 | -0.64    | -1.4    |0.097 | 1.00     |2362|
|1,study[5]                 |  1.15    |  0.3    |  2.1 | 1.00     |3299|
|1,study[6]                 |  1.02    | 0.31    |  1.8 | 1.00     |2423|
|1,study[7]                 |   0.9    | 0.12    |  1.8 | 1.00     |2577|
|group,study[social, 1]     | 0.078    |-0.12    | 0.41 | 1.00    | 3974|
|group,study[social, 2]     | 0.065    |-0.12    | 0.34 | 1.00    | 4045|
|group,study[social, 3]     | 0.017    | -0.2    | 0.27 | 1.00    | 4930|
|group,study[social, 4]     | 0.035    |-0.17    | 0.31 | 1.00    | 4271|
|group,study[social, 5]     | 0.035    |-0.28    | 0.42 | 1.00     |7001|
|group,study[social, 6]     |-0.071    |-0.38    | 0.15  |1.00     |5264|
|group,study[social, 7]     |-0.145    |-0.68    | 0.11  |1.00     |3928|


## Hypothesis Testing

We want to test whether the social norm intervention increases towel reuse.

Null Hypothesis ($H_0$): $\beta_{\text{social}} \leq 0$, ie the social norm has no positive effect on towel reuse

Alternate Hypothesis ($H_1$): $\beta_{\text{social}} > 0$, ie the social norm has a positive effect on towel reuse

This can also be expressed in log odds, $OR = \exp{\beta_{\text{social}}}$

$H_0: OR \leq 1$

$H_1: OR > 1$

