import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import simpy
import plotly.express as px

# Define constants
TARGET_INFLATION = 7  # Midpoint of the 6-8% target band
R_MIN, R_MAX = 0, 30  # Bounds for the policy rate
RR_MIN, RR_MAX = 5, 30  # Bounds for the reserve ratio

# Load the data
st.title("Monetary Policy Optimization and Simulation")
data = pd.read_csv("rates.csv")

# Extract relevant columns from data
X = data[['base_rate', 'interbank_rate', 'reserve', 'exchange_rate', '10_year_minus_3_year']]
y_inflation = data['inflation']
y_lending_rate = data['lending_rate']
y_exchange_rate = data['exchange_rate']

# Plot inflation using Plotly
fig = px.line(data, x=data['Date'], y=['inflation', 'base_rate', 'reserve'], title='Inflation Over Time')
st.plotly_chart(fig)

# Get the latest lending rate
current_lending_rate = y_lending_rate.iloc[-1]

# Fit Random Forest models to estimate relationships using the whole dataset
st.write("Fitting Random Forest models to estimate relationships...")
rf_inflation = RandomForestRegressor().fit(X[['base_rate', 'interbank_rate', 'reserve', 'exchange_rate']], y_inflation)
rf_interest_rate = RandomForestRegressor().fit(X[['base_rate', 'interbank_rate', 'reserve']], y_lending_rate)
rf_exchange_rate = RandomForestRegressor().fit(X[['base_rate', 'reserve', '10_year_minus_3_year']], y_exchange_rate)

# Allow user to input weights for the objective function
st.write("Set Weights for Objective Function Components:")
W_INFLATION = st.sidebar.slider('Weight for Inflation', 0.0, 1.0, 1.0)
W_INTEREST_RATE = st.sidebar.slider('Weight for Interest Rate', 0.0, 1.0, 1.0)
W_EXCHANGE_RATE = st.sidebar.slider('Weight for Exchange Rate', 0.0, 1.0, 1.0)
W_BANK_STABILITY = st.sidebar.slider('Weight for Bank Stability', 0.0, 1.0, 1.0)

# Define function to calculate objective value
def calculate_objective(vars):
    policy_rate, reserve_ratio = vars
    
    # Get the mean values for interbank rate and bond spread
    interbank_rate_mean = X['interbank_rate'].mean()
    bond_spread_mean = X['10_year_minus_3_year'].mean()
    
    # Create DataFrames with appropriate column names for prediction
    df_rf_exchange = pd.DataFrame([[policy_rate, reserve_ratio, bond_spread_mean]], columns=['base_rate', 'reserve', '10_year_minus_3_year'])
    df_rf_inflation = pd.DataFrame([[policy_rate, interbank_rate_mean, reserve_ratio, 0]], columns=['base_rate', 'interbank_rate', 'reserve', 'exchange_rate'])
    df_rf_interest = pd.DataFrame([[policy_rate, interbank_rate_mean, reserve_ratio]], columns=['base_rate', 'interbank_rate', 'reserve'])

    # Predict the exchange rate using the trained model
    exchange_rate_t = rf_exchange_rate.predict(df_rf_exchange)[0]
    
    # Update df_rf_inflation with the predicted exchange rate
    df_rf_inflation['exchange_rate'] = exchange_rate_t
    
    # Predict the inflation using the trained model
    inflation_t = rf_inflation.predict(df_rf_inflation)[0]
    
    # Predict the interest rate using the trained model
    interest_rate_t = rf_interest_rate.predict(df_rf_interest)[0]

    # Bank stability (represented by deviation from current lending rate)
    bank_stability_t = (current_lending_rate - interest_rate_t) ** 2

    # Objective function components
    inflation_diff = inflation_t - TARGET_INFLATION
    obj = (W_INFLATION * inflation_diff**2 +
           W_INTEREST_RATE * interest_rate_t**2 +
           W_EXCHANGE_RATE * exchange_rate_t**2 +
           W_BANK_STABILITY * bank_stability_t)

    return obj

# Function to perform optimization and return results
def optimize_policy():
    best_mse = float('inf')
    best_initial_guess = None

    for _ in range(50):
        # Add random perturbations to initial guess
        initial_guess = [13.5 + np.random.uniform(0, 2), 26 + np.random.uniform(0, 1)]
        
        # Optimize using the current weights
        result = minimize(calculate_objective, initial_guess, method='SLSQP', bounds=[(R_MIN, R_MAX), (RR_MIN, RR_MAX)])

        # Calculate MSE
        mse = calculate_objective(result.x)

        # Update best initial guess if necessary
        if mse < best_mse:
            best_mse = mse
            best_initial_guess = result.x

    # Optimize using the best initial guess
    result = minimize(calculate_objective, best_initial_guess, method='SLSQP', bounds=[(R_MIN, R_MAX), (RR_MIN, RR_MAX)])
    
    return result.x

# SimPy simulation function
def simulate_policy(env, policy_rate, reserve_ratio):
    while True:
        # Perform optimization
        optimal_policy = optimize_policy()
        policy_rate, reserve_ratio = optimal_policy

        # Output current policy
        st.write(f"At time {env.now}: Optimal policy rate: {policy_rate}, Optimal reserve ratio: {reserve_ratio}")

        # Wait for the next decision period
        yield env.timeout(1)  # Simulate policy update every 1 time unit

# Starting policy values
initial_policy_rate = 13.5
initial_reserve_ratio = 26.0

# Run the simulation
def run_simulation():
    env = simpy.Environment()
    env.process(simulate_policy(env, initial_policy_rate, initial_reserve_ratio))
    env.run(until=4)  # Run the simulation for 10 time units

if st.button('Run Simulation'):
    run_simulation()
