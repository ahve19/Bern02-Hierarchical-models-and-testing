# Bern02-Hierarchical-models-and-testing

The model should use an appropriate family distribution for the response variable, and consider between study heterogeneity

$y_ij$: number of people who reuse towels (reuse) in study j and group i
n_ij: total number of customers (total) in study j and group i
x_ij in {0,1}: Group indicator (0=control, 1=social norm)
β_0: Population-level baseline log-odds of towel reuse in the control group
β_social: Overall population-level effect of the social norm intervention on the log-odds scale
u _0j ~ N(0,σ^2_0): Random intercept for study j, capture varying baseline reuse rates across hotels
u _1j ~ N(0,σ^2_1): Random slope for study j, capture between-study heterogeneity in intervention efficacy

we want to account for both the outcome structure and variations between individual studies
thus we formulate a Hierarchical Binomial Logistic Model as follows
y_ij ~ Binomial(n_ij, p_ij)
logit(p_ij) = ln(p_ij/(1-p_ij)) = β_0 + β_social * x_ij + u_0j + u_1j * x_ij

Specify Hierarchical Binomial Model 
p(reuse, total) specifies proportion of successes out of total trials
