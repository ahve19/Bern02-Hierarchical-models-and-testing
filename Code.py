#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import bambi as bmb
import numpy as np
import pandas as pd
import arviz as az

# read in data
data_file = "towelData.csv"
data = pd.read_csv(data_file, sep=';', encoding='latin1')
count = data.iloc[:, -1] # get the last column with numbers or yes/no

# count has the number of yes and no for control and social norm groups
control_yes = count[::4].to_numpy() # every 4th starting from 0 - control group + yes
control_no = count[2::4].to_numpy() # every 4th starting from 2 - control group + no
control_total = np.array([y + n for y, n in zip(control_yes, control_no)])

social_yes = count[1::4].to_numpy() # every 4th starting from 1 - social norm group + yes
social_no = count[3::4].to_numpy() # every 4th starting from 3 - social norm group + no
social_total = np.array([y + n for y, n in zip(social_yes, social_no)])

study = np.arange(1,len(control_yes)+1) # 7 diferent studies

control_data = pd.DataFrame({"reuse": control_yes, "total": control_total, "group": "control", "study": study})
social_data = pd.DataFrame({ "reuse": social_yes, "total": social_total, "group": "social", "study": study})

combined_data = pd.concat([control_data, social_data], ignore_index=True)

# Convert data types - important for bambi that they are correctly set 
# can give errors otherwise
combined_data['reuse'] = combined_data['reuse'].astype(int)
combined_data['total'] = combined_data['total'].astype(int)
combined_data['group'] = combined_data['group'].astype('category')
combined_data['study'] = combined_data['study'].astype('category')

model_hierarchical = bmb.Model(
    "p(reuse, total) ~ group + (group | study)",
    data=combined_data,
    family="binomial"
)


# response variable consists of y successes out of n total trials
# this can be modeled by a binomial distribution

# we account for the heterogeneity between hotels (from things like location, wording of message, guest demographic, etc)
# by using random effects (ie u_0j and u_1j) to allow for adaption between aggregate data
# fixed-effects model would assume a single underlying true effect, underestimating standard errors




# 5. Formulate hypotheses for the test of the effectiveness of the intervention referring to one or several parameters within the model

# we want to test whether the social norm intervention increases towel reuse

# Null Hypothesis (H_0): β_social <= 0
# ie the social norm has no positive effect on towel reuse

# Alternate Hypothesis (H_1): β_social > 0
# ie the social norm has a positive effect on towel reuse

# this can also be expressed in log odds, OR = exp(β_social)
# H_0: OR <= 1
# H_1: OR > 1

# 1. Fit the model with a higher target_accept to eliminate divergences
# We also increase tuning steps to help the sampler adapt better
results = model_hierarchical.fit(
    draws=2000, 
    tune=2000, 
    target_accept=0.95, # Forces the sampler to take smaller, safer steps
    return_inferencedata=True,
    random_seed=42
)

# 2. Extract the summary using 'ci_prob' as requested by your ArviZ version
# We will use .iloc to select columns by position to avoid future name changes 
# (e.g., in case 'hdi_5%' is named 'ci_5%' or 'hdi_3%' in your version)
summary_table = az.summary(results, ci_prob=0.90)

# Filter for the relevant columns: mean, lower 90%, upper 90%, r_hat, ess_bulk
# Typically, these are the 1st, 3rd, 4th, and the last two columns.
columns_to_keep = ["mean", summary_table.columns[2], summary_table.columns[3], "r_hat", "ess_bulk"]
print(summary_table[columns_to_keep])



# 1. Extract the posterior draws for the intervention effect
# Note: Bambi might name this "group" or "group[social]" depending on the version
posterior_draws = results.posterior["group"].values.flatten()

# 2. Calculate the Posterior Probability: P(beta_social > 0 | data)
prob_positive = np.mean(posterior_draws > 0)

# 3. Convert the log-odds effect to an Odds Ratio (OR) for interpretability
odds_ratios = np.exp(posterior_draws)
mean_or = np.mean(odds_ratios)

print(f"Posterior probability of a positive effect: {prob_positive:.4f} (or {prob_positive*100:.2f}%)")
print(f"Expected Odds Ratio: {mean_or:.3f}")
