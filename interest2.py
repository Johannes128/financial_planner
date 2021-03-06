import collections
import copy
import csv
from datetime import datetime, date, timedelta
import functools
import itertools
from math import log
import numbers
import pprint
import time

from contexttimer import timer
import pandas as pd
import tabulate

import numpy as np
import scipy as sp
import scipy.optimize


TABLULATE_FLOAT_FMT = "6_.2f"

def cum_key(key):
  return key + "_cum"


one_day = timedelta(days=1)
year_timedelta = timedelta(days=365)
month_timedelta = timedelta(days=365/12)


class Date(date):
  def __new__(cls, year, month, day=1):
    return date.__new__(cls, year, month, day)

  @staticmethod
  def from_date(d : date):
    return Date(d.year, d.month, d.day)

  def to_date(self):
    return date(self.year, self.month, self.day)

  def prev_month(self):
    if self.month > 1:
      return Date(self.year, self.month-1)
    else:
      return Date(self.year-1, 12)

  def next_month(self):
    if self.month < 12:
      return Date(self.year, self.month+1)
    else:
      return Date(self.year+1, 1)

  def prev_day(self):
    return Date.from_date(self - one_day)

  def next_day(self):
    return Date.from_date(self + one_day)

  def last_day_of_month(self):
    d = self.next_month()
    return Date.from_date(Date(d.year, d.month, 1) - one_day)

  def __str__(self):
    return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

  def __repr__(self):
    return f"{self.__class__.__name__}({self.year}, {self.month}, {self.day})"


class Rate:
  def __init__(self, plan):
    self.plan = plan

  def __call__(self, start, limit=None):
    pass


class ConstantRate(Rate):
  def __init__(self, value, plan):
    super().__init__(plan)
    self.value = value

  def __call__(self, start, limit=float("inf")):
    if self.plan.finished:
      return 0.0
    return min(self.value, limit)



class MonthEntryBase(dict):
  FLOAT_ATTRIBUTES_GENERAL = ["V_start", "V_end", "V_net"]
  FLOAT_ATTRIBUTES_CUM = ["rate", "interest", "tax"]
  NON_NUM_ATTRIBUTES = ["start", "end"]

  @classmethod
  def get_show_attributes(cls, level="ALL"):
    if level == "ALL":
      return (["start", "end"] + cls.FLOAT_ATTRIBUTES_GENERAL
              + sorted(cls.FLOAT_ATTRIBUTES_CUM + [cum_key(attr) for attr in cls.FLOAT_ATTRIBUTES_CUM]))
    elif level == "MINIMAL":
      return cls.FLOAT_ATTRIBUTES_GENERAL
    else:
      raise ValueError(f"Unknown level={level}")

  def __init__(self, **kwargs):
    prev = kwargs.pop("prev", None)
    #kwargs.setdefault("next", None)

    super().__init__(**kwargs)

    if "start" not in self:
      self["start"] = Date(1, 1)
    if "end" not in self:
      self["end"] = self["start"].last_day_of_month()

    for key in self.FLOAT_ATTRIBUTES_GENERAL:
      if key not in self:
        self[key] = float("nan")

    for key in self.FLOAT_ATTRIBUTES_CUM:
      if key not in self:
        self[key] = 0.00


    if prev is None:
      for key in self.FLOAT_ATTRIBUTES_CUM:
        if cum_key(key) not in kwargs:
          self[cum_key(key)] = self[key]
    else:
      for key in self.FLOAT_ATTRIBUTES_CUM:
        if cum_key(key) not in kwargs:
          self[cum_key(key)] = self[key] + prev.get(cum_key(key), 0.00)

      #prev["next"] = self

  def __add__(self, other):
    keys = (set(self.keys()) | set(other.keys())) - set(self.NON_NUM_ATTRIBUTES)
    assert self["start"] == other["start"], f"start mismatch: {self['start']} != {other['start']}"
    assert self["end"] == other["end"], f"end mismatch: {self['end']} != {other['end']}"
    pairs = {key: self.get(key, 0.00) + other.get(key, 0.00) for key in keys}
    pairs["start"] = self["start"]
    pairs["end"] = self["end"]
    return MonthEntryBase(**pairs)


class MonthHistory(list):
  MonthEntry = MonthEntryBase

  def __init__(self, description, V_0, rate, start=None):
    super().__init__()
    self.description = description
    self.V_0 = V_0
    self.start = start
    self.finished = False

    if isinstance(rate, numbers.Number):
      self.rate_function = ConstantRate(rate, self)
    elif isinstance(rate, Rate):
      self.rate_function = rate
    else:
      self.rate_function = None

    if start is not None:
      self.restart(start)

  def rate(self, start, limit=float("inf")):
    return self.rate_function(start, limit)

  def restart(self, start):
    self.start = start
    self.clear()
    self.append(self.MonthEntry(start=self.start.prev_month(), V_start=self.V_0, V_end=self.V_0))
    return self[-1]

  def get_current_year_entries(self):
    year = self[-1]["start"].year
    year_entries = []
    for entry in reversed(self):
      if entry["start"].year == year:
        year_entries.append(entry)
      else:
        break
    return list(reversed(year_entries))

  def elapsed_years(self, start):
    return (start-self[0]["end"]) / year_timedelta

  def elapsed_months(self, start):
    #return (start-self[0]["start"]) / month_timedelta
    initial_start = self[0]["start"]
    return (start.year - initial_start.year) * 12 + start.month - initial_start.month

  def month_step(self, from_month, force_store_additional_info=False):
    raise NotImplementedError

  def month_steps(self, num_months, form_month=None):
    cur_month = (self.start if form_month is None else form_month)
    for m in range(num_months):
      self.month_step(cur_month, force_store_additional_info=(m+1 == num_months or (m+1)%12 == 0))
      cur_month = cur_month.next_month()
    return self

  def year_steps(self, num_years):
    return self.month_steps(12*num_years)

  def get_year_summaries(self):
    def merge_year_entries(entries):
      entry = copy.copy(entries[-1])
      entry["start"] = entries[0]["start"]
      entry["end"] = entries[-1]["end"]
      entry["V_start"] = entries[0]["V_start"]
      for attr in entry.FLOAT_ATTRIBUTES_CUM:
        entry[attr] = sum(e[attr] for e in entries)
      return entry

    years = sorted(set(e["start"].year for e in self[1:]))
    years_entries = {year: [] for year in years}
    for entry in self[1:]:
      years_entries[entry["start"].year].append(entry)

    first_entry = self[0].copy()
    first_entry["end"] = self[0]["start"].last_day_of_month()

    result = [first_entry]
    for year, year_entries in years_entries.items():
      result.append(merge_year_entries(year_entries))

    return result

  # vvvvvvvvvv output vvvvvvvvvv
  def table_header(self, level="ALL"):
    return self.MonthEntry.get_show_attributes(level)

  def to_table(self, entries=None, level="ALL", include_header=False):
    entries = (self if entries is None else entries)
    attributes = self.table_header(level)
    table = [[entry.get(attr, 0.00) for attr in attributes]
             for entry in entries]
    if include_header:
      table = [self.table_header(level)] + table
    return table

  def to_dataframe(self, entries=None, index="start", level="ALL"):
    table = self.to_table(entries)
    df = pd.DataFrame(table, columns=self.table_header(level)).set_index(index)
    df.index = pd.to_datetime(df.index)
    return df


  def to_string(self, entries=None, level="ALL"):
    table = self.to_table(entries, level)
    return "\n".join([
      f"*** {self.description}",
      tabulate.tabulate(table, floatfmt=TABLULATE_FLOAT_FMT,
                        headers=self.table_header(level))
    ])

  def to_year_dataframe(self, index="end", level="ALL"):
    return self.to_dataframe(self.get_year_summaries(), index=index, level=level)

  def print(self, level="ALL"):
    print(self.to_string(self, level))

  def print_years(self, level="ALL"):
    years_summaries = self.get_year_summaries()
    print(self.to_string(years_summaries, level))
  # ^^^^^^^^^^ output ^^^^^^^^^^

  def plot(self):
    import matplotlib.pyplot as plt
    #plt.style.use("fivethirtyeight")
    plt.xticks(rotation=45, ha='right')

    starts = [e["start"] for e in self]
    V_ends = [e["V_end"] for e in self]
    plt.bar(starts, V_ends, width=10)
    #plt.plot(starts, V_ends)
    plt.show()


class AnnuityLoan(MonthHistory):
  class MonthEntry(MonthEntryBase):
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest"]

  def __init__(self, *args, p_year, **kwargs):
    super().__init__(*args, **kwargs)
    self.p_year = p_year

  def grow_factor(self, start):
    return 1.0 + (self.p_year/100/12)

  def month_step(self, from_month, force_store_additional_info=False):
    if self.finished:
      self.append(copy.copy(self[-1]))

    if from_month < self.start:
      return

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start, -initial_value)

    value = initial_value + rate
    if value >= 0:
      rate = max(0.00, rate - value)
      value = 0.00
      self.finished = True

    value *= self.grow_factor(start)
    interest = value - initial_value - rate

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value,
                                V_end=value, V_net=value,
                                rate=rate, interest=interest,
                                prev=self[-1]))
    return self[-1]


class TaxInfo:
  def __init__(self, tax_rate, tax_free, fraction_to_tax=1.0):
    self.tax_rate = tax_rate
    self.tax_free = tax_free
    self.fraction_to_tax = fraction_to_tax
    self.effective_tax_rate = tax_rate * fraction_to_tax
    self.basis_interest_factor = 0.005 * 0.7 # TODO: make configurable


TAX_INFO_REGULAR = {
  "single": TaxInfo(0.26375, 801.00),
  "married": TaxInfo(0.26375, 1_602.00)
}
TAX_INFO_STOCKS = {
  "single": TaxInfo(0.26375, 801.00, 0.7),
  "married": TaxInfo(0.26375, 1_602.00, 0.7)
}


class SavingsPlan(MonthHistory):
  # TODO: * vorschüssig vs nachschüssig
  #       * Zinsperioden: jährlich, >monatlich<, ...

  class MonthEntry(MonthEntryBase):
    FLOAT_ATTRIBUTES_GENERAL = ["V_start", "V_end", "V_net", "interest_eff"]
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest", "tax"]

  TAX_INFO_MAP = TAX_INFO_REGULAR

  def __init__(self, *args, p_year=None, tax_info="married", **kwargs):
    if isinstance(tax_info, str):
      self.tax_info = self.TAX_INFO_MAP[tax_info]
    elif isinstance(tax_info, TaxInfo):
      self.tax_info = tax_info
    else:
      raise ValueError(f"Unknown tax info {tax_info}")

    self.p_year = p_year

    super().__init__(*args, **kwargs)

  def grow_factor(self, start):
    return 1.0 + (self.p_year/100/12)

  def effective_grow_percentage(self, V_0, rate_cum, V_E, num_months):
    avg_rate = rate_cum / num_months

    if False:
      # TODO: this implcitly assumes that the rate is paid at the end of each month
      def h_positive(q):
        return log(V_0*q + avg_rate) + num_months * log(1+q) - log( avg_rate + q*V_E)

      def h_negative(q):
        return ( q*V_0 + avg_rate ) * (1 + q) ** num_months - q*V_E - avg_rate
    else:
      # TODO: this implcitly assumes that the rate is paid at the beginning of each month
      def h_positive(q):
        return log(V_0 * q + avg_rate* (1+q)) + num_months * log(1+q) - log( q * V_E + avg_rate * (1+q) )

      def h_negative(q):
        return (q * V_0 + avg_rate * (1+q)) * (1+q) ** num_months  -  ( q * V_E + avg_rate * (1+q) )

    def find_positive_value(f, t, mult=10):
      while f(t) < 1e-5:
        t *= mult
      return t

    if V_0 + rate_cum < V_E:
      b = find_positive_value(h_positive, 1e-16)
      sol = sp.optimize.root_scalar(h_positive, method="brentq", bracket=[1e-16, b])
    elif h_negative(-1e-16) >= 0:
      return 0.0
    else:
      a = find_positive_value(h_negative, -1e-16)
      sol = sp.optimize.root_scalar(h_negative, method="brentq", bracket=[a, -1e-16])

    # the result is the interest rate per month, so convert it to a year interest rate
    return 100 * ((1+sol.root)**12 - 1)


  def month_step(self, from_month, force_store_additional_info=False):
    if self.finished:
      self.append(copy.copy(self[-1]))

    if from_month < self.start:
      return

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start) # TODO: how to model the rate?

    value = initial_value + rate # TODO: distinguish payment at start/end
    value *= self.grow_factor(start)
    interest = value - initial_value - rate

    # calculate tax
    cur_year_entries = [e for e in reversed(self) if e["start"].year == start.year]
    cur_year_interest = sum(e["interest"] for e in cur_year_entries)
    cur_year_interest += interest
    value_for_tax = min(max(0.00, cur_year_interest - self.tax_info.tax_free), interest)
    tax = -self.tax_info.effective_tax_rate * value_for_tax
    value += tax

    # TODO: refactor
    if start.month == 12 or force_store_additional_info:
      interest_effective = self.effective_grow_percentage(self[0]["V_start"],
                                                          self[-1]["rate_cum"]+rate,
                                                          value,
                                                          self.elapsed_months(start))
    else:
      interest_effective = float("nan")

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value,
                                V_end=value, V_net=value,
                                rate=rate, interest=interest,
                                interest_eff=interest_effective,
                                tax=tax,
                                prev=self[-1]))
    return self[-1]


class StocksSavingsPlan(SavingsPlan):
  class MonthEntry(SavingsPlan.MonthEntry):
    FLOAT_ATTRIBUTES_GENERAL = ["V_start", "V_end", "V_net", "tax_sell", "interest_eff"]
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest", "tax", "int_tax_paid"]

  TAX_INFO_MAP = TAX_INFO_STOCKS

  def grow_factor(self, start):
    return (1 + self.p_year / 100) ** (1 / 12)

  def month_step(self, from_month, force_store_additional_info=False):
    if self.finished:
      self.append(copy.copy(self[-1]))

    if from_month < self.start:
      return

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start)

    value = initial_value + rate  # TODO: distinguish payment at start/end
    value *= self.grow_factor(start)
    interest = value - initial_value - rate

    # calculate tax on sell
    tax_paid = self[-1]["tax_cum"]
    interest_tax_paid = self[-1]["int_tax_paid_cum"]
    interest_to_tax = interest + self[-1]["interest_cum"] - interest_tax_paid
    interest_to_tax = max(0.00, interest_to_tax * self.tax_info.fraction_to_tax - self.tax_info.tax_free)
    tax_on_sell = -interest_to_tax * self.tax_info.tax_rate
    V_after_tax = value + tax_on_sell + tax_paid

    if from_month.month == 12:
      # the year has finished - calculate the Vorabpauschale
      year_entries = self.get_current_year_entries()

      cur_year_interest = sum(e["interest"] for e in year_entries) + interest

      vorabpauschale_begin = ((12 - year_entries[0]["start"].month + 1) / 12 * year_entries[0]["V_start"]
                              * self.tax_info.basis_interest_factor)
      vorabpauschale_rates = (sum((12 - e["start"].month + 1) / 12 * e["rate"] for e in year_entries)
                              + 1 / 12 * rate
                             ) * self.tax_info.basis_interest_factor
      vorabpauschale = vorabpauschale_begin + vorabpauschale_rates

      interest_to_tax = min(vorabpauschale, max(0.00, cur_year_interest))
      interest_part_for_tax = interest_to_tax * self.tax_info.fraction_to_tax
      tax = -max(0.00, interest_part_for_tax - self.tax_info.tax_free) * self.tax_info.tax_rate
    else:
      interest_to_tax = 0.0
      tax = 0.0

    # TODO: refactor
    if start.month == 12 or force_store_additional_info:
      interest_effective = self.effective_grow_percentage(self[0]["V_start"],
                                                          self[-1]["rate_cum"]+rate,
                                                          value,
                                                          self.elapsed_months(start))
    else:
      interest_effective = float("nan")

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value,
                                V_end=value, V_net=V_after_tax,
                                #fac=value/(self[0]["V_start"]+self[-1]["rate_cum"]+rate),
                                rate=rate, interest=interest,
                                interest_eff=interest_effective,
                                tax=tax, tax_sell=tax_on_sell,
                                int_tax_paid=interest_to_tax,
                                prev=self[-1]))
    return self[-1]


class StocksSavingsPlanDataBased(StocksSavingsPlan):
  @staticmethod
  @functools.lru_cache(100)
  def load_data(filename : str):
    @timer()
    def _load_data():
      with open(filename) as f:
        def parse_line(l):
          parsed_date, value = datetime.strptime(l[0], "%Y-%m-%d").date(), float(l[1])
          return [Date.from_date(parsed_date).last_day_of_month().next_day(), value] # TODO: here we assume that the data times are end of months day, e.g. 2002-01-31
        chart_data = [parse_line(l) for i, l in enumerate(csv.reader(f, delimiter=";")) if i > 0]
      return {line[0]: line[1] for line in chart_data}
    return _load_data()


  def __init__(self, *args, **kwargs):
    self.chart_data = self.load_data("data/MSCI_World_Performance.csv")
    super().__init__(*args, **kwargs)

  def grow_factor(self, start):
    value_start, value_end = self.chart_data[start], self.chart_data[start.next_month()]
    return value_end/value_start


class Chain(MonthHistory):
  def __init__(self, description, start=None, plans=None):
    super().__init__(description, start)
    self.histories = (plans if plans is not None else [])


class Parallel(MonthHistory):
  def __init__(self, description, start=None, plans=None, fixed_budget=None):
    desc_counts = collections.Counter([plan.description for plan in plans])
    for desc, count in desc_counts.items():
      if count > 1:
        raise ValueError(f"The description '{desc}' occurs {count} times. Descriptions need to be unique.")

    super().__init__(description, None, start) # TODO: is rate=None a good idea?
    self.description = description
    self.histories = (plans if plans is not None else [])

    self.histories_with_rate = [history for history in self.histories if history.rate_function is not None]
    self.histories_without_rate = [history for history in self.histories if history.rate_function is None]

    self.fixed_budget = fixed_budget
    if fixed_budget is None and self.histories_without_rate:
      raise ValueError("Plans without rate only possible when fixed_budget is set.")

    history_starts = [history.start for history in self.histories]
    set_history_starts = [history_start for history_start in history_starts if history_start is not None]
    if start is not None:
      assert min(set_history_starts) >= start
      self.restart(start)
    elif all(history_start is not None for history_start in history_starts):
      self.restart(min(history_starts))


  def append_sum(self, entries):
    self.append(sum(entries[1:], entries[0]))
    return self[-1]

  def restart(self, start):
    self.start = start
    initial_entries = [(history.restart(start) if history.start is None
                        else history.restart(history.start))
                       for history in self.histories]
    first_start = min(entry["start"] for entry in initial_entries)
    return self.append_sum([entry for entry in initial_entries if entry["start"] == first_start])

  def clear(self):
    for history in self.histories:
      history.clear()

  def month_step(self, from_month, force_store_additional_info=False):
    entries_with_rate = [history.month_step(from_month, force_store_additional_info) for history in self.histories_with_rate]

    if self.fixed_budget is not None:
      used_rate = sum([0.0] + [entry["rate"] for entry in entries_with_rate])
      rate_for_others = (self.fixed_budget - used_rate) / len(self.histories_without_rate)
      for plan in self.histories_without_rate:
        plan.rate_function = ConstantRate(rate_for_others, plan)
      entries_without_rate = [history.month_step(from_month, force_store_additional_info) for history in self.histories_without_rate]
      entries = entries_with_rate + entries_without_rate
    else:
      entries = entries_with_rate

    entries = [entry for entry in entries if entry is not None]
    self.finished = all(history.finished for history in self.histories)
    if entries:
      return self.append_sum(entries)

  # vvvvvvvvvv output vvvvvvvvvv
  def print(self, level="ALL"):
    if level == "ALL":
      for history in self.histories:
        history.print(level)
        print()
    super().print(level)

  def print_years(self, level="ALL"):
    if level == "ALL":
      for history in self.histories:
        history.print_years(level)
        print()

    super().print_years(level)

    years_summaries = [history.get_year_summaries()
                       for history in self.histories]
    starts = sorted(set(entry["start"] for summary in years_summaries for entry in summary))
    m_index = [0 for _ in self.histories]
    values = []
    for start in starts:
      cur_values = [start]
      for i, summary in enumerate(years_summaries):
        if m_index[i] >= len(summary) or start < summary[m_index[i]]["start"]:
          cur_values.append(None)
        else:
          if m_index[i] >= len(summary):
            cur_values.append(None)
          else:
            entry = summary[m_index[i]]
            cur_values.append(entry["V_end"])
          m_index[i] += 1
      cur_values.append(sum(v for v in cur_values[1:] if v is not None))
      values.append(cur_values)

    print()
    headers = ["\nstart"] + [history.description + "\nV_end" for history in self.histories] + ["total\nV_end"]
    print(tabulate.tabulate(values, floatfmt=TABLULATE_FLOAT_FMT,
                            headers=headers))

  def print_summary(self):
    values = []
    for history in self.histories:
      values += [[history.description, history[-1]["V_end"], history[-1]["V_net"]]]

    print(f"*** Summary of {self.description}")
    values += [["SUM", sum(e[1] for e in values), sum(e[2] for e in values)]]
    print(tabulate.tabulate(values, floatfmt=TABLULATE_FLOAT_FMT,
                            headers=["description", "V_end", "V_net"]))
    print()
  # ^^^^^^^^^^ output ^^^^^^^^^^



if __name__ == "__main__":

  if True:
    Parallel(
      "parallel plan",
      start=Date(2020, 1),
      plans=[AnnuityLoan("SPK", -380_800.00, 1_311.00, p_year=1.15, start=Date(2021, 1)),
             StocksSavingsPlan("ETF", 57_000.00, 653.00, p_year=5.0, start=Date(2021, 1))],
    ).year_steps(32).print_years()
    #).year_steps(10).plot()


  if False:
    StocksSavingsPlan("savings", 100_000.00, 1100.00, p_year=5.0, start=Date(2020, 1)).year_steps(30).print_years()