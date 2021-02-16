import collections

from contexttimer import timer
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from interest2 import Month, StocksSavingsPlan, StocksSavingsPlanDataBased, AnnuityLoan, Parallel, TAX_INFO_STOCKS


def get_spread(plans, V_keys=("V_end",)):
  plan_dfs = [plan.to_year_dataframe() for plan in plans]

  spread_df = {V_key: pd.concat([plan_df[V_key] for plan_df in plan_dfs], axis=1,
                                keys=[plan.description for plan in plans]
                                ).sort_index()
                                 .interpolate(method="time", limit_direction="both", limit_area="inside")
               for V_key in V_keys}
  return spread_df


def get_end_distribution(plans, V_keys=("V_end",), V_keys_percentage=("interest_eff",)):
  value_without_interest = plans[0][-1]["rate_cum"] + plans[0][0]["V_start"]
  return {V_key: pd.DataFrame({V_key: [plan[-1][V_key] for plan in plans] + ([value_without_interest] if V_key not in V_keys_percentage else [])},
                              index=[plan.description for plan in plans] + (["#total_payments"] if V_key not in V_keys_percentage else []))
          for V_key in V_keys}


section = st.sidebar.radio("Section", ["ETF Savings Plan", "Real Estate Financing", "Documentation", "Code", "Alpha: Interest Triangle"])
st.title(section)

disclaimer = "**Disclaimer**: This is work in progress. I do not take any responsibility for the correctness of any provided data. 😅"
st.write(disclaimer)

if section == "ETF Savings Plan":
  col1, col2, col3 = st.beta_columns(3)

  V_0 = col1.number_input("Start Capital V_0", 0.0, 1000_000_000.00, 100_000.00, step=5_000.00)
  rate = col2.number_input("Monthly Rate", 0.0, 1000_000_000.00, 1_350.00, step=50.00)
  start = Month(1990, 1)
  runtime_years = col3.number_input("Runtime [years]", 1, 100, 30)
  tax_variants = list(TAX_INFO_STOCKS.keys())
  tax_variant = col1.selectbox("Tax Variant", tax_variants, index=tax_variants.index("married"))
  #basis_tax_rate = col2.number_input("Basis Tax Rate [%]", 0.0, 50.0, 0.5, step=0.1)

  etf_mode = st.radio("Interest Mode", ["Fixed Interest Rate", "Historical Performance"])
  if etf_mode == "Fixed Interest Rate":
    r"""
This mode simulates a stocks savings plan respecting German tax laws for different average interest rates.
The Plan with $p=0\%$ is a comparison to a savings strategy without any interest. 
    """

    etf_interest_rates = st.multiselect("ETF Interest Rates [%]", list(range(0, 21)), default=list(range(0, 10, 1)))

    @timer()
    def simulate_plans():
      return [StocksSavingsPlan("ETF_{:02d}%".format(p), V_0, rate, p_year=p, start=start, tax_info=tax_variant).year_steps(runtime_years)
              for p in etf_interest_rates]
    plans = simulate_plans()
  else:
    """
This mode simulates a stocks savings plan respecting German tax laws based on historical index performance.
It is intended to provide an overview of the interest spread when starting the investment at different times.
    """

    stocks_index = st.selectbox("Index", ["MSCI World"], index=0)

    possible_start_years = list(range(1970, 2021-runtime_years, 1))
    start_years = st.multiselect("Start years", possible_start_years, default=possible_start_years)
    start_months = st.multiselect("Start months", list(range(1, 13)), [1, 7])

    start_times = [Month(y, m) for y in start_years for m in start_months]

    @timer()
    def simulate_plans():
      progress_bar = st.progress(0.0)
      result = []
      for s in start_times:
        result.append(StocksSavingsPlanDataBased(f"ETF_{s.year:04d}-{s.month:02d}", V_0, rate, start=s, tax_info=tax_variant).year_steps(runtime_years))
        progress_bar.progress(len(result)/len(start_times))
      progress_bar.empty()
      return result
    plans = simulate_plans()

  V_keys = ["V_end", "V_net", "interest_cum", "tax_cum", "tax_sell", "interest_eff"]
  V_keys_selected = st.multiselect("Values to plot", V_keys, default=["V_end", "interest_eff"])
  """
* **V_end** is the expected value of the total depot at each time
* **V_net** is the expected value of the total depot **after sell tax** at each time
* **interest_cum** is the depot value excluding the starting capital and all monthly rates
* **tax_cum** are the cumulated tax payments at each year **without** selling
* **tax_sell** is the tax payment due on selling the whole depot content
* **interest_eff** is the interest rate of a savings plan with fixed interest rate that would have resulted in the same depot value at the given time (same $V_0$, average rate per month)
  """
  explanation = st.beta_expander("Detailed Explanation")
  with explanation:
    "**interest_eff**"
    "The effective interest rate uses the average rate obtained as follows:"
    explanation.latex(r"\text{rate\_avg} = \frac{1}{|\text{Months}|} \sum_{m = 1}^{|\text{Months}|} \text{rate}(\text{month}_m)")
    "Then it finds the monthly interest rate $q$ such that"
    explanation.latex(r"V_\text{end} = V_0 \cdot (1+q)^{|\text{Months}|} + \text{rate\_avg} \cdot \sum_{m = 1}^{|\text{Months}|} (1+q)^m.")
    "The final value $p$ is then obtained by $p = (1+q)^{12}$."
    "**Remark**: This approach is correct for plans where the monthly rate is fixed. If the monthly rate varies it becomes a (more or less accurate) approximation..."

  show_data = st.checkbox("Show raw data table")

  spread_df = get_spread(plans, V_keys=V_keys_selected)

  end_distribution_df = get_end_distribution(plans, V_keys_selected)

  for V_key in spread_df.keys():
    st.header(f"History of *{V_key}*")
    st.line_chart(spread_df[V_key])
    st.header(f"End Distribution of *{V_key}*")
    st.bar_chart(end_distribution_df[V_key])
    if show_data:
      st.dataframe(spread_df[V_key].iloc[::-1])

elif section == "Real Estate Financing":
  st.write("**Important**: This section still needs alot of love... Please be careful when interpreting the results!")

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

  show_data = st.checkbox("Show raw data table")

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
  V_keys_default = ["V_end", "V_net"]
  V_keys_selected = V_keys=st.multiselect("Values to plot", V_keys, default=V_keys)
  spread_df = get_spread(plans, V_keys_selected)

  for V_key in spread_df.keys():
    st.header(f"*{V_key}*")
    st.line_chart(spread_df[V_key])
    if show_data:
      st.dataframe(spread_df[V_key].iloc[::-1])


elif section == "Documentation":
  st.write("Here, random documentation of some implementation details is collected.")

  block = st.beta_expander("Effective Interest Rate")
  with block:
    # TODO: remove double definition of general explanations
    "This is the interest rate of a savings plan with fixed interest rate that would have resulted in the same depot value at the given time (same $V_0$, average rate per month)"
    "The effective interest rate uses the average rate obtained as follows:"
    block.latex(r"\text{rate\_avg} = \frac{1}{|\text{Months}|} \sum_{m = 1}^{|\text{Months}|} \text{rate}(\text{month}_m)")
    "Then it finds the monthly interest rate $q$ such that"
    block.latex(r"V_\text{end} = V_0 \cdot (1+q)^{|\text{Months}|} + \text{rate\_avg} \cdot \sum_{m = 1}^{|\text{Months}|} (1+q)^m.")
    "The final value $p$ is then obtained by $p = (1+q)^{12}$."
    "The right-hand side of the above formula can be transformed to"
    block.latex(r"""
        \begin{aligned}
               &V_0 \cdot (1+q)^{|\text{Months}|} + \text{rate\_avg} \cdot \sum_{m = 0}^{|\text{Months}|} (1+q)^m - \text{rate\_avg} \\
          = \, &V_0 \cdot (1+q)^{|\text{Months}|} + \text{rate\_avg} \cdot \frac{(1+q)^{|\text{Months}|+1}-(1+q)}{q} \\
          = \, &(1+q)^{|\text{Months}|} \cdot \left( V_0 + \frac{\text{rate\_avg} \cdot (1+q)}{q} \right) - \text{rate\_avg} \cdot \frac{(1+q)}{q}
        \end{aligned}
    """)
    r"As a result, multiplying the equation for $V_\text{end}$ by $q$ yields"
    block.latex(r"q V_\text{end} + \text{rate\_avg} \cdot (1+q) = (1+q)^{|\text{Months}|} \cdot \left( qV_0 + \text{rate\_avg} \cdot (1+q) \right)")
    "Solving this high-order polynomial for $q$ naively is numerically unstable. At least for $q>0$ we can take the logarithm of both sides to stabilize the computation:"
    block.latex(r"\log \big( V_\text{end} + \text{rate\_avg} \cdot (1+q) \big) = |\text{Months}| \cdot \log (1+q) + \log \big( qV_0 + \text{rate\_avg} \cdot (1+q) \big)")
    "TODO: $q=0$ is trivial solution; solving strategy for $q>0$, $q<0$"


elif section == "Alpha: Interest Triangle":
  st.write("**WARNING: This is untested and highly experimental! Only enjoy for visual purposes...**")

  st.write("The triangle below gives an overview how different start/sell times influence the interest rate. Move the mouse over a cell for easy reading.")

  V_0 = 0
  rate = 1000.00
  tax_variant = "married"
  start_year, end_year = 1980, 2020
  key_month = 1

  start_times = [Month(y, 1) for y in range(start_year, end_year + 1)]
  plans = [StocksSavingsPlanDataBased(f"ETF_{s.year:04d}-{s.month:02d}", V_0, rate, start=s, tax_info=tax_variant).year_steps(end_year - s.year)
           for s in start_times]

  triangle_cols = {"start_year": [],
                   "sell_year": [],
                   "runtime": [],
                   "interest": []}

  for plan in plans:
    year_summaries = plan.get_year_summaries()
    for entry in year_summaries:
      if entry["start"].month != key_month:
        continue
      cur_start_year, cur_sell_year = year_summaries[0]["start"].year, entry["start"].year
      triangle_cols["start_year"].append(cur_start_year)
      triangle_cols["sell_year"].append(cur_sell_year)
      triangle_cols["runtime"].append(cur_sell_year - cur_start_year)
      triangle_cols["interest"].append(entry["interest_eff"])
  triangle_cols_df = pd.DataFrame(triangle_cols)

  selection = alt.selection_single(on='mouseover', empty='none', fields=['runtime'])
  #selection = alt.selection_multi(fields=['runtime'])
  interest_color = alt.Color('interest:Q', scale=alt.Scale(domain=[-50, -30, -5, 0.0, 5, 15, 50],
                                                           range=["red", "red", "#FFAAAA", "#FFFFFF", "#BBFFBB", "#00AA00", "#006600"],
                                                           type="linear"))
  chart = alt.Chart(triangle_cols_df).mark_rect().encode(
    x='sell_year:O',
    y=alt.Y('start_year:O', sort=alt.EncodingSortField('start_year', order='descending')),
    color=alt.condition(selection, alt.value("black"), interest_color),
    #color=interest_color,
    tooltip=[
      alt.Tooltip(field="start_year", type="quantitative", title="Start Year"),
      alt.Tooltip(field="sell_year", type="quantitative", title="Sell Year"),
      alt.Tooltip(field="runtime", type="quantitative", title="Runtime [years]"),
      alt.Tooltip(field="interest", type="quantitative", title="Interest", format=".2f"),
    ]
  )

  if True:
    text = chart.mark_text(baseline='middle', fontSize=8, fontWeight=900).encode(
      text=alt.Text('interest:Q', format=".1f"),
      #color=alt.condition(selection, alt.value('#EEEE00'), alt.value('black')),
      color=alt.condition(selection, interest_color, alt.value("black"))
    )

    final_chart = (chart + text).properties(
      width=900,
      height=600
    ).add_selection(selection)
  else:
    final_chart = chart

  st.write(final_chart)


elif section == "Code":
  with open("app.py") as src_app:
    st.header("Code of streamlit app (app.py):")
    st.code(src_app.read())

  with open("interest2.py") as src_backend:
    st.header("Code of backend (interest2.py):")
    st.code(src_backend.read())