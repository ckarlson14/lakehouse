import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load the CSV data
data = pd.read_csv('survey_responses.csv')

# Strip any leading/trailing spaces in column headers
data.columns = data.columns.str.strip()

# Verify the columns
logging.debug(f"Columns: {data.columns}")

# Process data
names = data['Name']
payment_levels = data[['Happiness Level 1', 'Happiness Level 2', 'Happiness Level 3', 'Happiness Level 4', 'Happiness Level 5', 'Happiness Level 6', 'Happiness Level 7', 'Happiness Level 8', 'Happiness Level 9', 'Happiness Level 10']].values

# Define satisfaction function with linear interpolation
def satisfaction(contributions, payment_levels):
    satisfactions = []
    for i, contribution in enumerate(contributions):
        satisfaction_level = 10
        for j, level in enumerate(payment_levels[i]):
            if contribution <= level:
                if j == 0:
                    satisfaction_level = 1
                else:
                    prev_level = payment_levels[i][j-1]
                    satisfaction_level = j + (contribution - prev_level) / (level - prev_level)
                break
        satisfactions.append(satisfaction_level)
        logging.debug(f"Person {i+1} - Contribution: {contribution}, Satisfaction: {satisfaction_level:.2f}")
    return np.array(satisfactions)

# Constants
total_cost = 200000
num_individuals = len(data)

# Initial contributions equally distributed
initial_contributions = np.full(num_individuals, total_cost / num_individuals)

# Objective function to minimize variance of satisfaction
def objective(contributions):
    satisfactions = satisfaction(contributions, payment_levels)
    mean_satisfaction = np.mean(satisfactions)
    variance = np.sum((satisfactions - mean_satisfaction) ** 2)
    return variance

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda contributions: np.sum(contributions) - total_cost}
]

# Bounds based on payment levels
bounds = [(0, payment_levels[i][-1]) for i in range(num_individuals)]

# Optimize using SLSQP method
result = minimize(objective, initial_contributions, method='SLSQP', constraints=constraints, bounds=bounds)

if not result.success:
    logging.error("Optimization failed to find a solution.")
else:
    # Optimal contributions
    optimal_contributions = result.x

    # Calculate satisfaction levels for optimal contributions
    satisfactions = satisfaction(optimal_contributions, payment_levels)

    # Output the results and calculate dispersion
    total_dispersion = 0
    mean_satisfaction = np.mean(satisfactions)

    logging.info(f"{'Name':<10} {'Contribution':<15} {'Satisfaction':<15} {'Dispersion':<10}")
    for i, person in enumerate(names):
        satisfaction_level = satisfactions[i]
        dispersion = satisfaction_level - mean_satisfaction
        total_dispersion += abs(dispersion)
        logging.info(f"{person:<10} ${optimal_contributions[i]:<15.2f} {satisfaction_level:<15.2f} {dispersion:<10.2f}")

    average_dispersion = total_dispersion / num_individuals
    logging.info(f"\nAverage Dispersion in Happiness Levels: {average_dispersion:.2f}")

    # Save results to a CSV file
    results = pd.DataFrame({
        'Name': names,
        'Optimal Contribution': optimal_contributions,
        'Satisfaction Level': satisfactions
    })
    results.to_csv('optimal_contributions.csv', index=False)
