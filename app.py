import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from interest2 import Month, StocksSavingsPlan, AnnuityLoan, Parallel, TAX_INFO_STOCKS


def get_spread(plans, V_keys=("V_end",)):
  plan_dfs = [plan.to_year_dataframe() for plan in plans]

  spread_df = {V_key: pd.concat([plan_df[V_key] for plan_df in plan_dfs], axis=1,
                                keys=[plan.description for plan in plans])
               for V_key in V_keys}
  return spread_df


def get_end_distribution(plans, V_keys=("V_end",)):
  return {V_key: pd.DataFrame({V_key: [plan[-1][V_key] for plan in plans]},
                      index=[plan.description for plan in plans])
          for V_key in V_keys}


section = st.sidebar.radio("Section", ["ETF", "Real Estate"])

if section == "ETF":
  st.title("ETF Savings Plan")
  """
Simulates an stocks savings plan respecting German tax laws for different average interest rates.
The Plan with p=0% is a comparison to an savings strategy without any interest. 
* **V_end** is the expected value of the total depot at each time
* **V_end** is the expected value of the total depot **after tax** at each time 

TODO: respect marriage state, plot age, make start selectable, use historic index data
"""

  col1, col2, col3 = st.beta_columns(3)

  V_0 = col1.number_input("Enter start capital V_0", 0.0, 1000_000_000.00, 100_000.00, step=5_000.00)
  rate = col2.number_input("Enter monthly rate", 0.0, 1000_000_000.00, 1_350.00, step=50.00)
  start = Month(2022, 1)
  runtime_years = col3.number_input("Desired runtime", 1, 100, 30)
  tax_variants = list(TAX_INFO_STOCKS.keys())
  tax_variant = col1.selectbox("Tax variant", tax_variants, index=tax_variants.index("married"))

  etf_interest_rates = st.multiselect("ETF interest rates", list(range(0, 21)), default=list(range(0, 10, 1)))

  plans = [StocksSavingsPlan("ETF_{:02d}%".format(p), V_0, rate, p_year=p, start=start, tax_info=tax_variant).year_steps(runtime_years)
           for p in etf_interest_rates]

  V_keys = ["V_end", "V_net", "interest_cum", "tax_cum", "tax_sell"]
  V_keys_selected = st.multiselect("Values to plot", V_keys, default=["V_end"])
  spread_df = get_spread(plans, V_keys=V_keys_selected)
  end_distribution_df = get_end_distribution(plans, V_keys_selected)

  for V_key in spread_df.keys():
    st.header(V_key)
    st.line_chart(spread_df[V_key])
    st.bar_chart(end_distribution_df[V_key])
    st.dataframe(spread_df[V_key].iloc[::-1])

elif section == "Real Estate":
  st.title("Real Estate Financing")

  col1, col2, col3 = st.beta_columns(3)

  price = col1.number_input("Price", 0.0, 1000_000_000.00, 450_000.00, step=10_000.00)
  total_capital = col2.number_input("Total capital", 0.0, 1000_000_000.00, 120_000.00, step=10_000.00)
  eigenkapital = col3.number_input("Total capital to use for buying", 0.0, 1000_000_000.00, 90_000.00, step=10_000.00)
  buy_costs_rate = col1.number_input("Buy costs rate", 0.0, 100.00, 6.5, step=0.5)

  total_monthly_budget = col2.number_input("Total monthly budget", 0.0, 1000_000_000.00, 2_000.00, step=100.00)
  loan_interest_rate = col3.number_input("Loan interest rate", 0.0, 100.00, 1.1, step=0.1)
  loan_runtime = col1.number_input("Loan run time", 5, 50, 20, step=5)
  loan_rate = col2.number_input("Loan monthly rate", 0.0, 100_000.00, 1350.00, step=100.00)

  etf_interest_rates = st.multiselect("ETF interest rates", list(range(0,21)), default=list(range(0,10,2)))

  buy_costs = price * buy_costs_rate/100
  total_price = price + buy_costs
  loan_value = total_price - eigenkapital
  capital_remaining = total_capital - eigenkapital

  f"""
**Total price: {total_price:_.2f}**
  
**Buy costs: {buy_costs:_.2f}**

**Loan value: {loan_value:_.2f}**

**Capital remaining: {capital_remaining:_.2f}**
  """

  start = Month(2021, 1)

  plans = [Parallel("Plan_{:02d}%".format(p),
                    start=start,
                    plans=[AnnuityLoan("SPK", -loan_value, loan_rate, p_year=loan_interest_rate, start=start),
                           StocksSavingsPlan("ETF", capital_remaining, total_monthly_budget-loan_rate, p_year=p, start=start)],
                   ).year_steps(loan_runtime)
           for p in etf_interest_rates]

  V_keys = ["V_end", "V_net", "rate_cum"]
  V_keys_selected = V_keys=st.multiselect("Values to plot", V_keys, default=V_keys)
  spread_df = get_spread(plans, V_keys_selected)

  for V_key in spread_df.keys():
    st.header(V_key)
    st.line_chart(spread_df[V_key])
    st.dataframe(spread_df[V_key].iloc[::-1])

  #st.header("Raw Plan Data")
  #st.dataframe(plan.to_dataframe().iloc[::-1])