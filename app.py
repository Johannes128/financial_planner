import collections

from contexttimer import timer
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from interest2 import Date, StocksSavingsPlan, StocksSavingsPlanDataBased, AnnuityLoan, Parallel, TAX_INFO_STOCKS


def get_spread(plans, V_keys=("V_end",)):
  plan_dfs = [plan.to_year_dataframe() for plan in plans]

  spread_df = {V_key: pd.concat([plan_df[V_key] for plan_df in plan_dfs], axis=1,
                                keys=[plan.description for plan in plans]
                                ).sort_index()
                                 .interpolate(method="time", limit_direction="both", limit_area="inside")
               for V_key in V_keys}
  return spread_df


def get_end_distribution(plans, V_keys=("V_end",), V_keys_percentage=("interest_eff",)):
  total_payments = plans[0][-1]["rate_cum"]
  start_value = plans[0][0]["V_start"]

  return {V_key: pd.DataFrame({V_key: [plan[-1][V_key] for plan in plans] + ([start_value, total_payments] if V_key not in V_keys_percentage else [])},
                              index=[plan.description for plan in plans] + (["#start_value", "#total_payments"] if V_key not in V_keys_percentage else []))
          for V_key in V_keys}


def add_zero_column(df):
  df = df.copy()
  df.insert(0, "#zero", 0.0)
  return df

def show_spread_and_end_distribution(plans, V_keys_selected, show_data=False):
  spread_df = get_spread(plans, V_keys=V_keys_selected)
  end_distribution_df = get_end_distribution(plans, V_keys_selected)

  for V_key in spread_df.keys():
    st.header(f"History of *{V_key}*")
    st.line_chart(add_zero_column(spread_df[V_key]))
    st.header(f"End Distribution of *{V_key}*")
    st.bar_chart(end_distribution_df[V_key])
    if show_data:
      st.dataframe(spread_df[V_key].iloc[::-1])


section = st.sidebar.radio("Section", ["ETF Savings Plan", "Real Estate Financing", "ALPHA: Follow-Up Financing", "Interest Triangle", "Documentation", "Code"])
st.title(section)

disclaimer = "**Disclaimer**: This is work in progress. I do not take any responsibility for the correctness of any provided data. 😅"
st.write(disclaimer)

if section == "ETF Savings Plan":
  col1, col2, col3 = st.beta_columns(3)

  V_0 = col1.number_input("Start Capital V_0", 0.0, 1000_000_000.00, 100_000.00, step=5_000.00)
  rate = col2.number_input("Monthly Rate", 0.0, 1000_000_000.00, 1_350.00, step=50.00)
  start = Date(1990, 1)
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

    stocks_index = st.selectbox("Index", ["MSCI World"], index=0) # TODO: use this value

    possible_start_years = list(range(1970, 2021-runtime_years, 1))
    start_years = st.multiselect("Start years", possible_start_years, default=possible_start_years)
    start_months = st.multiselect("Start months", list(range(1, 13)), [1, 7])

    start_times = [Date(y, m) for y in start_years for m in start_months]

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
  show_spread_and_end_distribution(plans, V_keys_selected, show_data)


elif section == "Real Estate Financing":
  st.write("**Important**: This section still needs alot of love... Please be careful when interpreting the results!")

  with st.beta_container():
    st.subheader("Buying Parameters")
    col1, col2, col3 = st.beta_columns(3)

    price = col1.number_input("Price", 0.0, 1000_000_000.00, 450_000.00, step=10_000.00)
    total_capital = col2.number_input("Total capital", 0.0, 1000_000_000.00, 120_000.00, step=10_000.00)
    eigenkapital = col3.number_input("Total capital to use for buying", 0.0, 1000_000_000.00, 90_000.00, step=10_000.00)
    buy_costs_rate = col1.number_input("Buy costs rate", 0.0, 100.00, 6.5, step=0.5)

    buy_costs = price * buy_costs_rate / 100
    total_price = price + buy_costs
    loan_value = total_price - eigenkapital
    capital_remaining = total_capital - eigenkapital

    col1, col2, col3, col4 = st.beta_columns(4)
    col1.write("**Total price:**");       col2.write(f"**{total_price:_.2f}**")
    col3.write("**Buy costs:**");         col4.write(f"**{buy_costs:_.2f}**")
    col1.write("**Needed loan value:**"); col2.write(f"**{loan_value:_.2f}**")
    col3.write("**Capital remaining:**"); col4.write(f"**{capital_remaining:_.2f}**")

  with st.beta_container():
    st.subheader("Loan Parameters")
    col1, col2, col3 = st.beta_columns(3)
    loan_interest_rate = col3.number_input("Loan interest rate", 0.0, 100.00, 1.1, step=0.1)
    loan_runtime = col1.number_input("Loan run time", 5, 50, 20, step=5)
    loan_rate = col2.number_input("Loan monthly rate", 0.0, 100_000.00, 1350.00, step=100.00)

  start = Date(2021, 1)
  annuity_loan = AnnuityLoan("SPK", -loan_value, loan_rate, p_year=loan_interest_rate, start=start)

  sub_section = st.sidebar.radio("Real Estate Financing Mode", ["Annuity Loan", "Annuitiy Loan + ETF Savings Plan"])
  # TODO: "Annuity Loan + follow-up" - We need to implement the Chain simulation in the backend first

  if sub_section == "Annuity Loan":
    plans = [annuity_loan.year_steps(loan_runtime)]

  elif sub_section == "Annuitiy Loan + ETF Savings Plan":
    with st.beta_container():
      st.subheader("ETF Savings Options")
      total_monthly_budget = st.number_input("Total monthly budget", 0.0, 1000_000_000.00, 2_000.00, step=100.00)
      etf_rate = total_monthly_budget - loan_rate

      st.write(f"Will start with the remaining capital {capital_remaining:_.2f} and use the rate {etf_rate:_.2f}")

      etf_mode = st.radio("Interest Mode", ["Fixed Interest Rate", "Historical Performance"])
      if etf_mode == "Fixed Interest Rate":
        etf_interest_rates = st.multiselect("ETF interest rates", list(range(0,21)), default=list(range(0,10,2)))
        etf_plans = [StocksSavingsPlan("Plan_{:02d}%".format(p), capital_remaining, etf_rate, p_year=p, start=start) for p in etf_interest_rates]

        annuity_loans = [annuity_loan] * len(etf_plans)
        start_times = [start] * len(etf_plans)

      elif etf_mode == "Historical Performance":
        col1, col2 = st.beta_columns(2)
        stocks_index = col1.selectbox("Index", ["MSCI World"], index=0)  # TODO: use this value
        tax_variants = list(TAX_INFO_STOCKS.keys())
        tax_variant = col2.selectbox("Tax Variant", tax_variants, index=tax_variants.index("married"))

        possible_start_years = list(range(1970, 2021 - loan_runtime, 1))
        start_years = possible_start_years
        #start_years = st.multiselect("Start years", possible_start_years, default=possible_start_years)
        start_months = st.multiselect("Start months", list(range(1, 13)), [1])
        start_times = [Date(y, m) for y in start_years for m in start_months]

        annuity_loans = [AnnuityLoan("SPK", -loan_value, loan_rate, p_year=loan_interest_rate, start=s) for s in start_times]
        etf_plans = [StocksSavingsPlanDataBased(f"ETF_{s.year:04d}-{s.month:02d}", capital_remaining, etf_rate, start=s, tax_info=tax_variant)
                     for s in start_times]

      plans = [Parallel(etf_plan.description,
                        start=s,
                        plans=[annuity_loan,
                               etf_plan],
                        ).year_steps(loan_runtime)
               for s, annuity_loan, etf_plan in zip(start_times, annuity_loans, etf_plans)]

  show_data = st.checkbox("Show raw data table")

  V_keys = ["V_end", "V_net", "rate_cum"]
  V_keys_default = ["V_net"]
  V_keys_selected = V_keys=st.multiselect("Values to plot", V_keys, default=V_keys_default)

  show_spread_and_end_distribution(plans, V_keys_selected, show_data)


elif section == "ALPHA: Follow-Up Financing":
  col1, col2 = st.beta_columns(2)
  loan_debt = col1.number_input("Start loan debt", 0.0, 1000_000_000.00, 250_000.00, step=5000.00)
  loan_rate = col2.number_input("Loan rate", 0.0, 1000_000_000.00, 600.00, step=100.00)
  etf_value = col1.number_input("Start ETF value", 0.0, 1000_000_000.00, 150_000.00, step=5000.00)
  etf_rate = col2.number_input("ETF rate", 0.0, 1000_000_000.00, 650.00, step=50.00)

  runtime = col1.slider("Runtime", 0, 50, 10, 1)
  col2.write(f"**Runtime: {runtime} years**")

  loan_interest_rates = st.multiselect("Loan interest rates", list(range(0, 21)), default=list(range(0, 11, 1)))
  etf_interest_rates = st.multiselect("ETF interest rates", list(range(0, 21)), default=list(range(0, 11, 1)))

  show_tooltips = st.checkbox("Show tooltips", False)

  start = Date(2021, 1)

  interest_grid = {"loan_interest_rate": [],
                   "etf_interest_rate": [],
                   "loan_value": [],
                   "etf_V_net": [],
                   "total_V_net": [],
                   }

  for loan_interest_rate in loan_interest_rates:
    for etf_interest_rate in etf_interest_rates:
      loan = AnnuityLoan("SPK", -loan_debt, loan_rate, p_year=loan_interest_rate, start=start)
      etf_plan = StocksSavingsPlan("Plan_{:02d}%".format(etf_interest_rate), etf_value, etf_rate, p_year=etf_interest_rate, start=start)
      plan = Parallel(etf_plan.description,
                      start=start,
                      plans=[loan, etf_plan],
                      ).year_steps(runtime)

      interest_grid["loan_interest_rate"].append(loan_interest_rate)
      interest_grid["etf_interest_rate"].append(etf_interest_rate)
      interest_grid["loan_value"].append(loan[-1]["V_end"])
      interest_grid["etf_V_net"].append(etf_plan[-1]["V_net"])
      interest_grid["total_V_net"].append(plan[-1]["V_net"])

  interest_grid_df = pd.DataFrame(interest_grid)

  min_V_net, max_V_net = min(interest_grid["total_V_net"]), max(interest_grid["total_V_net"])
  domain, color_range = [0.0], ["white"]
  if min_V_net < 0:
    domain = [min_V_net] + domain
    color_range = ["red"] + color_range
  if max_V_net > 0:
    domain = domain + [max_V_net]
    color_range = color_range + ["#006600"]

  red_green_scale = alt.Scale(domain=domain,
                              range=color_range,
                              type="linear")

  selection_etf_p =  alt.selection_single(on='mouseover', empty='none', fields=['etf_interest_rate'])
  selection_loan_p = alt.selection_single(on='mouseover', empty='none', fields=['loan_interest_rate'])

  interest_grid_chart = alt.Chart(interest_grid_df).mark_rect().encode(
    x='loan_interest_rate:O',
    y=alt.Y('etf_interest_rate:O', sort=alt.EncodingSortField('etf_interest_rate', order='descending')),
    color='total_V_net:Q',
    #color=alt.condition(selection_etf_p|selection_loan_p, alt.value("black"),
    #                    alt.Color('total_V_net', scale=red_green_scale)),
    opacity=alt.condition(selection_etf_p|selection_loan_p, alt.value(0.6), alt.value(1.0)),
    tooltip=([] if not show_tooltips else [
      alt.Tooltip(field="loan_interest_rate", type="quantitative", title="Loan interest rate"),
      alt.Tooltip(field="etf_interest_rate", type="quantitative", title="ETF interest rate"),
      alt.Tooltip(field="loan_value", type="quantitative", title="Loan value at end", format=".2f"),
      alt.Tooltip(field="etf_V_net", type="quantitative", title="ETF V_net at end", format=".2f"),
      alt.Tooltip(field="total_V_net", type="quantitative", title="Total V_net at end", format=".2f"),
    ])
  )

  text = interest_grid_chart.mark_text(baseline='middle', fontSize=8, fontWeight=900).encode(
    text=alt.Text('total_V_net:Q', format=".1f"),
    #color=alt.condition(selection_etf_p|selection_loan_p, alt.value('#EEEE00'), alt.value('black')),
    color=alt.value("black")
  )

  interest_grid_chart = (interest_grid_chart + text).properties(
    width=700,
    height=300
  ).add_selection(selection_etf_p).add_selection(selection_loan_p)

  bar_chart_loan_p = alt.Chart(interest_grid_df).mark_bar(size=20).encode(
    y=alt.Y("total_V_net", scale=alt.Scale(domain=domain[:1] + domain[-1:])),
    x=alt.X("loan_interest_rate"),
    color=alt.Color('total_V_net', scale=red_green_scale, legend=None),
  ).transform_filter(
    selection_etf_p
  ).properties(
    width=350,
    height=300
  )

  bar_chart_etf_p = alt.Chart(interest_grid_df).mark_bar(size=20).encode(
    y=alt.Y("total_V_net", scale=alt.Scale(domain=domain[:1] + domain[-1:])),
    x=alt.X("etf_interest_rate"),
    color=alt.Color('total_V_net', scale=red_green_scale, legend=None)
  ).transform_filter(
    selection_loan_p
  ).properties(
    width=350,
    height=300
  )


  st.write(interest_grid_chart & (bar_chart_etf_p | bar_chart_loan_p))


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
    "The final value $p$ is then obtained by $p = (1+q)^{12} - 1$."
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


elif section == "Interest Triangle":
  st.write("**WARNING: This is untested and highly experimental! Only enjoy for visual purposes...**")

  st.write("The triangle below gives an overview how different start/sell times influence the interest rate. Move the mouse over a cell for easy reading.")
  st.write("Each line in the line plot represents the interests on selling *for all plans with the same runtime* started at all possible times in the data range. E.g., if the start year is 2000, the end year is 2012, and a plan with a 10 years runtime is hovered in the triangle, the line plot shows the year interest for the plans sold at the end of 2009, 2010 and 2011 which have been started at the beginning of 2000, 2001, 2002, respectively.")

  col1, col2 = st.beta_columns(2)

  V_0 = col1.number_input("Start Capital V_0", 0.0, 1000_000_000.00, 100_000.00, step=5_000.00)
  rate = col2.number_input("Monthly Rate", 0.0, 1000_000_000.00, 1_350.00, step=50.00)
  start_year = col1.number_input("Start Year", 1970, 2020, 1970, step=5)
  end_year = col2.number_input("End Year", start_year, 2020, 2020, step=5) + 1

  stocks_index = st.selectbox("Index", ["MSCI World"], index=0)  # TODO: use this value

  tax_variant = "married"
  key_month = 1 # TODO: make this clean!

  start_times = [Date(y, 1) for y in range(start_year, end_year + 1)]
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
      cur_start_year, cur_sell_year = year_summaries[0]["start"].next_month().year, entry["start"].year
      triangle_cols["start_year"].append(cur_start_year)
      triangle_cols["sell_year"].append(cur_sell_year)
      triangle_cols["runtime"].append(cur_sell_year - cur_start_year + 1)
      triangle_cols["interest"].append(entry["interest_eff"])
  triangle_cols_df = pd.DataFrame(triangle_cols)

  selection = alt.selection_single(on='mouseover', empty='none', fields=['runtime'])
  selector = alt.selection_single(on="mouseover", empty='all', fields=['runtime'])

  interest_color = alt.Color('interest:Q', scale=alt.Scale(domain=[-50, -30, -5, 0.0, 5, 15, 50],
                                                           range=["red", "red", "#FFAAAA", "#FFFFFF", "#BBFFBB", "#00AA00", "#006600"],
                                                           type="linear"))
  triangle_chart = alt.Chart(triangle_cols_df).mark_rect().encode(
    x='sell_year:O',
    y=alt.Y('start_year:O', sort=alt.EncodingSortField('start_year', order='descending'), axis=alt.Axis(orient='right')),
    #color=alt.condition(selection, alt.value("black"), interest_color),
    color=interest_color,
    opacity=alt.condition(selection, alt.value(0.5), alt.value(1.0)),
    tooltip=[
      alt.Tooltip(field="start_year", type="quantitative", title="Start Year"),
      alt.Tooltip(field="sell_year", type="quantitative", title="Sell Year"),
      alt.Tooltip(field="runtime", type="quantitative", title="Runtime [years]"),
      alt.Tooltip(field="interest", type="quantitative", title="Interest", format=".2f"),
    ]
  )

  text = triangle_chart.mark_text(baseline='middle', fontSize=8, fontWeight=900).encode(
    text=alt.Text('interest:Q', format=".1f"),
    #color=alt.condition(selection, alt.value('#EEEE00'), alt.value('black')),
    color=alt.value("black")
  )

  size_factor = (end_year-start_year)/50
  GRAPH_WIDTH = max(450, 900 * size_factor)

  final_triangle_chart_chart = (triangle_chart + text).properties(
    width=GRAPH_WIDTH,
    height=max(250, 500 * size_factor)
  ).add_selection(selection).add_selection(selector)

  time_range = min(triangle_cols["start_year"]), max(triangle_cols["sell_year"])

  timeseries_chart = alt.Chart(triangle_cols_df).mark_line().encode(
    y="interest",
    x=alt.X("sell_year", scale=alt.Scale(domain=time_range), axis=alt.Axis(format='4.0f')),
    color=alt.Color('runtime:O', scale=alt.Scale(scheme='dark2'), legend=alt.Legend(columns=3, symbolLimit=100)),
  ).transform_filter(
    selector
  ).properties(
    width=GRAPH_WIDTH,
    height=max(100, 200 * size_factor)
  )

  marker_value = max(5.0, triangle_cols_df[triangle_cols_df["runtime"] == max(triangle_cols_df["runtime"])]["interest"].mean() * 0.75)
  test_data = pd.DataFrame({
    "sell_year": time_range * 3,
    "value": [-marker_value, -marker_value] + [0.0, 0.0] + [marker_value, marker_value],
    "type": ["bad", "bad"] + ["zero", "zero"] + ["good", "good"]
  })

  lines = alt.Chart(test_data).mark_line().encode(
    y="value",
    x="sell_year",
    color=alt.Color('type:O', scale=alt.Scale(scheme='set1'))
  )

  st.write(final_triangle_chart_chart & (lines + timeseries_chart))


elif section == "Code":
  with open("app.py") as src_app:
    st.header("Code of streamlit app (app.py):")
    st.code(src_app.read())

  with open("interest2.py") as src_backend:
    st.header("Code of backend (interest2.py):")
    st.code(src_backend.read())