# Bern02-Hierarchical-models-and-testing

The model should use an appropriate family distribution for the response variable, and consider between study heterogeneity

$y_{ij}$: number of people who reuse towels (reuse) in study j and group i

$n_{ij}$: total number of customers (total) in study j and group i

$x_{ij}$ in {0,1}: Group indicator (0=control, 1=social norm)

$\beta_0$ : Population-level baseline log-odds of towel reuse in the control group

$\beta_{\text{social}}$: Overall population-level effect of the social norm intervention on the log-odds scale

$\mu_{0j} \sim \mathcal{N}(0,\sigma^2_0)$: Random intercept for study j, capture varying baseline reuse rates across hotels

$\mu_{1j} \sim \mathcal{N}(0,\sigma^2_1)$: Random slope for study j, capture between-study heterogeneity in intervention efficacy

we want to account for both the outcome structure and variations between individual studies thus we formulate a Hierarchical Binomial Logistic Model as follows

$y_{ij} \sim \text{Binomial}(n_{ij}, p_{ij})$

$\text{logit}(p_{ij}) = \ln{\frac{p_{ij}}{1-p_{ij}}} = \beta_0 + \beta_{\text{social}} * x_{ij} + \mu_{0j} + \mu_{1j} * x_{ij}$

Specify Hierarchical Binomial Model 

$p(\text{reuse}, \text{total})$ specifies proportion of successes out of total trials
