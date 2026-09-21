import bambi as bmb
import arviz as az
import numpy as np
import pandas as pd

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

summary_table = az.summary(results, ci_prob=0.90)
columns_to_key = ["mean", "eti90_lb", "eti90_ub", "r_hat", "ess_bulk"]

# Print summary filtered for key metrics
print(summary_table)

# Extract posterior draws for the social norm effect
posterior_draws = results.posterior["group"].values.flatten()

# Calculate probability of a positive effect
prob_positive = np.mean(posterior_draws > 0)

# Convert log-odds to Odds Ratios (OR)
odds_ratios = np.exp(posterior_draws)
mean_or = np.mean(odds_ratios)

print(f"Probability of positive effect: {prob_positive:.4f}")
print(f"Mean Odds Ratio: {mean_or:.4f}")
