## Bern02-Hierarchical-models-and-testing

The purpose of this exercise is to use a collection of randomized case-control studies to determine the effectiveness of descriptive social norms on reuse of towels at hotels.

# Model Formulation

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

# Hypothesis Testing

We want to test whether the social norm intervention increases towel reuse.

Null Hypothesis ($H_0$): $\beta_{\text{social}} \leq 0$, ie the social norm has no positive effect on towel reuse

Alternate Hypothesis ($H_1$): $\beta_{\text{social}} > 0$, ie the social norm has a positive effect on towel reuse

This can also be expressed in log odds, $OR = \exp{\beta_{\text{social}}}$

$H_0: OR \leq 1$

$H_1: OR > 1$

