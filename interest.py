import itertools
import copy
import csv
import pprint
import time

import tabulate


class HistoryEntry:
  def __init__(self, V_start, rate, interest, month, total_interest=0.00, total_rate=0.00,
               tax=0.00, total_tax=0.00, V_for_tax=0.00, V_to_tax=0.00, V_after_tax=0.00,
               vorabpauschale=0.00, tax_on_sell=0.00, tax_free_used=0.00, gain_after_tax=0.00,
               V_end=0.00):
    self.V_start = V_start
    self.V_end = V_end
    self.rate = rate
    self.total_rate = total_rate
    self.interest = interest
    self.total_interest = total_interest
    self.month = month
    self.year = month/12
    self.tax = tax
    self.total_tax = total_tax
    self.V_for_tax = V_for_tax
    self.V_to_tax = V_to_tax
    self.V_after_tax = V_after_tax
    self.vorabpauschale = vorabpauschale
    self.tax_on_sell = tax_on_sell
    self.tax_free_used = tax_free_used
    self.gain_after_tax = gain_after_tax

  def final_value(self):
    if self.V_after_tax != 0.00:
      return self.V_after_tax
    else:
      return self.V_end

  def __str__(self):
    return "HistoryEntry({})".format(", ".join(f"{k}={v}" for k,v in self.__dict__.items()))

  def __repr__(self):
    return str(self)


class HistoryBase(list):
  def __init__(self, V_0, month_offset=0, total_interest_start=0.00, total_rate_start=0.00):
    super().__init__()
    self.year_summaries = []
    self.restart(V_0, month_offset, total_interest_start, total_rate_start)

  def restart(self, V_0, month_offset=0, total_interest_start=0.00, total_rate_start=0.00):
    self.clear()
    self.year_summaries = []
    self.append(HistoryEntry(V_0, rate=0.00, interest=0.00,
                             total_interest=total_interest_start,
                             total_rate=total_rate_start,
                             month=month_offset,
                             V_end=V_0))

  def last(self):
    return self[-1]

  def append(self, entry):
    if len(self) == 0:
      super().append(entry)
    else:
      entry.total_rate = self[-1].total_rate + entry.rate
      entry.total_interest = self[-1].total_interest + entry.interest
      entry.total_tax = self[-1].total_tax + entry.tax
      super().append(entry)

  # vvvvv printing methods vvvvv
  def _get_table_header(self):
    return ["year", "month",
            "V_start", "rate", "total_rate",
            "interest", "total_interest",
            "tax", "total_tax",
            "V_end"]

  def _get_table_row(self, e):
    return [e.year, e.month,
            e.V_start, e.rate, e.total_rate,
            e.interest, e.total_interest,
            e.tax, e.total_tax,
            e.V_end]

  def _print_table(self, table):
    print(tabulate.tabulate(table, floatfmt="6_.2f",
                            headers=self._get_table_header()))
    print()

  def print(self):
    table = [self._get_table_row(e) for e in self]
    self._print_table(table)

  def print_years_old(self):
    table = [self._get_table_row(e)
             for i,e in enumerate(self)
             if i != 1 and round(e.year) == e.year or i == len(self)-1]
    self._print_table(table)

  def print_years(self):
    table = []
    last_start_entry = None
    for i, e in enumerate(self):
      if (i == 1 or round(e.year) != e.year) and i != len(self) - 1:
        continue

      e_cpy = copy.deepcopy(e)
      if last_start_entry is not None:
        e_cpy.V_start = last_start_entry.V_end
        e_cpy.interest = e.total_interest - last_start_entry.total_interest
        e_cpy.rate = e.total_rate - last_start_entry.total_rate

      table.append(self._get_table_row(e_cpy))
      last_start_entry = e

    self._print_table(table)

    table = [self._get_table_row(e) for e in self.year_summaries]
    self._print_table(table)


  def print_minimal(self):
    table = [self._get_table_row(e) for e in [self[0], self[-1]]]
    self._print_table(table)
  # ^^^^^ printing methods ^^^^^


class History(HistoryBase):
  def __init__(self, V_0, p_year, rate,
               years=None, month_offset = 0,
               total_interest_start=0.00, total_rate_start=0.00,
               tax_rate=0.26375, tax_free=1_602.00,
               payment_at_period_start=True):
    super().__init__(V_0, month_offset, total_interest_start, total_rate_start)

    self.V_0 = V_0
    self.p_year = p_year

    self.rate = rate if callable(rate) else (lambda *args: rate)

    self.years = years
    self.month_offset = month_offset

    self.total_interest_start = total_interest_start
    self.total_rate_start = total_rate_start

    self.tax_rate = tax_rate
    self.tax_free = tax_free

    self.payment_at_period_start = payment_at_period_start

    if self.V_0 is not None:
      self.calculate()

  def get_p_month(self, abs_month):
    #return ((1 + self.p_year / 100) ** (1 / 12) - 1) * 100
    return self.p_year/12

  def get_grow_factor(self, abs_month):
    return 1.0 + self.get_p_month(abs_month) / 100

  def continue_history(self, loan=None):
    if loan is not None:
      self.month_offset = loan.last().month
      self.V_0 = loan.last().V_end
      self.total_interest_start = loan.last().total_interest
      self.total_rate_start = loan.last().total_rate
      self.calculate()

  def get_current_year_entries(self, rel_month):
    cur_year_start_idx = (rel_month - 1) // 12 * 12 + 1  # TODO: respect start month in year
    return [self[i] for i in range(cur_year_start_idx, len(self))]

  def get_value_rate_interest(self, rel_month, abs_month):
    value = self.last().V_end
    rate = self.rate(rel_month, abs_month)

    intial_value = value
    if self.payment_at_period_start:
      value += rate
      value_start = value
      value *= self.get_grow_factor(abs_month)
      interest = value - value_start
    else:
      value_start = value
      value *= self.get_grow_factor(abs_month)
      interest = value - value_start
      value += rate

    return intial_value, value, rate, interest

  def update_year_summaries(self, rel_month, abs_month, finished):
    last_entry = self.last()
    if (rel_month != 1 and round(last_entry.year) == last_entry.year) or finished:
      year_entries = self.get_current_year_entries(rel_month)
      self.year_summaries.append(HistoryEntry(
        V_start=year_entries[0].V_start,
        V_end=year_entries[-1].V_end,
        rate=sum(e.rate for e in year_entries),
        total_rate=year_entries[-1].total_rate,
        interest=sum(e.interest for e in year_entries),
        total_interest=year_entries[-1].total_interest,
        tax=sum(e.tax for e in year_entries),
        total_tax=year_entries[-1].total_tax,
        month=year_entries[-1].month,
        V_to_tax=year_entries[-1].V_to_tax,
        V_for_tax=year_entries[-1].V_for_tax,
        V_after_tax=year_entries[-1].V_after_tax,
        tax_on_sell=year_entries[-1].tax_on_sell,
        gain_after_tax=year_entries[-1].gain_after_tax,
        vorabpauschale=year_entries[-1].vorabpauschale,
      ))
      return True
    return False

  def month_step(self, rel_month, abs_month, max_months):
    initial_value, value, rate, interest = self.get_value_rate_interest(rel_month, abs_month)

    if self.V_0 < 0: # this is a loan
      tax = 0.00
      if value >= 0:
        rate = -initial_value
        value = 0.00
    else: # this is a savings plan
      cur_year_interest = sum(e.interest for e in self.get_current_year_entries(rel_month))
      cur_year_interest += interest
      value_for_tax = min(max(0.00, cur_year_interest - self.tax_free), interest)
      tax = -self.tax_rate * value_for_tax
      value += tax

    self.append(HistoryEntry(0.00 if len(self) == 0 else self.last().V_end,
                             rate, interest, abs_month, tax=tax, V_end=value))

    finished = False
    if max_months is not None and abs_month - self.month_offset >= max_months:
      finished = True
    elif max_months is None and value == 0.00:
      finished = True

    self.update_year_summaries(rel_month, abs_month, finished)
    return not finished

  def calculate(self):
    if self.years is not None:
      max_months = 12*self.years
    else:
      max_months = None

    self.restart(self.V_0, self.month_offset, self.total_interest_start, self.total_rate_start)

    for rel_month in itertools.count(1):
      if not self.month_step(rel_month, rel_month+self.month_offset, max_months):
        break


class AnnuityLoan(History):
  pass


class SavingsPlan(History):
  pass


class StocksSavingsPlan(History):
  # TODO: also respect dividends

  def __init__(self, *args, basis_interest=0.5, **kwargs):
    self.part_relevant_for_tax = 0.7
    self.basis_interest = basis_interest
    self.basis_interest_factor = self.basis_interest / 100 * 0.7

    super().__init__(*args, **kwargs)

  def get_p_month(self, abs_month):
    return ( (1+self.p_year/100) ** (1/12) - 1 ) * 100

  def update_year_summaries(self, rel_month, abs_month, finished):
    # at the end of year, calculate tax that has to be paid during hold (vorabpauschale)
    year_entries = self.get_current_year_entries(rel_month)
    if len(year_entries) >= 12:
      cur_year_interest = sum(e.interest for e in year_entries)
      cur_year_rate = sum(e.rate for e in year_entries)
      value_at_beginning = year_entries[-1].V_end - cur_year_interest - cur_year_rate

      vorabpauschale_begin = value_at_beginning * self.basis_interest_factor
      vorabpauschale_rates = sum((12.0 - i) / 12 * e.rate for i, e in enumerate(year_entries, start=0)) * self.basis_interest_factor
      vorabpauschale = vorabpauschale_begin + vorabpauschale_rates

      V_for_tax = min(vorabpauschale, max(0.00, cur_year_interest))
      V_part_for_tax = V_for_tax * self.part_relevant_for_tax
      tax = -max(0.00, V_part_for_tax - self.tax_free) * self.tax_rate

      last_entry = self.pop()
      self.append(HistoryEntry(0.00 if len(self) == 0 else self.last().V_end,
                               last_entry.rate, last_entry.interest, abs_month,
                               V_end=last_entry.V_end,
                               tax=tax, V_for_tax=V_for_tax, vorabpauschale=vorabpauschale))
      self.add_sell_tax(self.last())

    super().update_year_summaries(rel_month, abs_month, finished)

  def add_sell_tax(self, entry):
    tax_paid = sum(e.tax for e in self[:-1])
    V_tax_paid = sum(e.V_for_tax for e in self[:-1])
    entry.V_to_tax = entry.V_end - self[0].V_end - entry.total_rate - V_tax_paid
    V_to_tax = max(0.00, entry.V_to_tax * self.part_relevant_for_tax - self.tax_free)
    entry.tax_on_sell = -V_to_tax * self.tax_rate
    entry.V_after_tax = entry.V_end + entry.tax_on_sell + tax_paid
    entry.gain_after_tax = entry.total_interest + entry.tax_on_sell + tax_paid

  def month_step(self, rel_month, abs_month, max_months):
    initial_value, value, rate, interest = self.get_value_rate_interest(rel_month, abs_month)

    self.append(HistoryEntry(0.00 if len(self) == 0 else self.last().V_end,
                             rate, interest, abs_month, V_end=value))

    self.add_sell_tax(self.last())

    finished = False
    if max_months is not None and abs_month - self.month_offset >= max_months:
      finished = True
    elif max_months is None and value == 0.00:
      finished = True

    self.update_year_summaries(rel_month, abs_month, finished)

    return not finished

  # vvvvv printing methods vvvvv
  def _get_table_header(self):
    return ["year", "m",
            "V_start", "rate", "tot_rate",
            "interest", "tot_interest",
            "V_end",
            "VAP", "V_for_tax",
            "tax", "tot_tax", "V_to_tax", "tax_sell",
            "V_aft_tax", "g_aft_tax",
            ]

  def _get_table_row(self, e):
    return [e.year, e.month,
            e.V_start, e.rate, e.total_rate,
            e.interest, e.total_interest,
            e.V_end,
            e.vorabpauschale, e.V_for_tax,
            e.tax, e.total_tax, e.V_to_tax, e.tax_on_sell,
            e.V_after_tax, e.gain_after_tax]
  # ^^^^^ printing methods ^^^^^


with open("data/MSCI_World_Performance.csv") as f:
  def parse_line(l):
    return [time.strptime(l[0], "%Y-%m-%d"), float(l[1])]
  msci_world_data = [parse_line(l) for i,l in enumerate(csv.reader(f, delimiter=";")) if i > 0]


class StocksSavingsPlanDataBased(StocksSavingsPlan):
  def get_p_month(self, abs_month):
    raise ValueError("Should not be called")

  def get_grow_factor(self, abs_month):
    abs_month += 12*30 # TODO: define offset somewhere else
    #abs_month += 420  # TODO: define offset somewhere else
    value_start, value_end = msci_world_data[abs_month][1], msci_world_data[abs_month+1][1]
    return value_end/value_start


class Chain:
  def __init__(self, *loans):
    self.loans = loans
    self.calculate()

  def calculate(self):
    last_l = None
    for l in self.loans:
      l.continue_history(last_l)
      last_l = l

  def last(self):
    return self.loans[-1].last()

  def print_years(self):
    for l in self.loans:
      l.print_years()

  def print(self):
    for l in self.loans:
      l.print()


class Rate:
  def __init__(self, rate, month_delay=0):
    self.rate = rate
    self.month_delay = month_delay

  def __call__(self, rel_month, abs_month):
    if rel_month > self.month_delay:
      return self.rate
    else:
      return 0.00


class FinancialPlan:
  def __init__(self, **kwargs):
    self.plans = kwargs

  def print(self, only_summary=False):
    values = []
    for name, plan in self.plans.items():
      if not only_summary:
        print("***", name)
        plan.print_years()
      values += [[name, plan.last().final_value()]]

    print("*** Summary")
    values += [["SUM", sum(e[1] for e in values)]]
    print(tabulate.tabulate(values, floatfmt="6_.2f",
                            headers=["plan", "value"]))
    print()


# Interhyp calculates their loans as follows:
# * use the standard interest (NOT the effective)
# * payment at end of month
# Example: AnnuityLoan(-296_200.00, 0.87, 1_300.00, 15, payment_at_period_start=False).print()

#Chain(AnnuityLoan(-345_450.00, 1.28, 938.00, 15),
#      AnnuityLoan(None, 1.01, 1001.00, 10)).print_years()

#Chain(SavingsPlan(10_000.00, 1.00, 100.00, 10),
#      SavingsPlan(None, 1.01, 1001.00, 10)).print_years()

#StocksSavingsPlanDataBased(45_000.00, 0.0, 1_100.00, 15, tax_free=1602.00).print_years()
StocksSavingsPlan(45_000.00, 3.0, 710.00, 15, tax_free=1602.00).print_years()

#AnnuityLoan(-100_000.00, 0.84, Rate(384.00, 12), 10, payment_at_period_start=True).print_years()

#AnnuityLoan(-296_200.00, 0.87, 1_300.00, 15, payment_at_period_start=False).print()

if False:
  month_budget = 2_000.00
  etf_p = 5.00
  only_summary = True

  print("*" * 30)
  print("*** 15a ***")
  if True:
    print("Capital: 99_000.00, run time: 15a, low rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-439_000.00, 1.06, 1_108.00, 15),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 5)),
      ETF=StocksSavingsPlan(45_000.00, etf_p, month_budget-1_108.00-384.00, 15, tax_free=1602.00)
    )
    plan.print(only_summary)

  if True:
    print("Capital: 99_000.00, run time: 15a, low rate to bank, 30_000.00 from somewhere")
    plan = FinancialPlan(
      bank=AnnuityLoan(-439_000.00, 1.06, 1_108.00, 15),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 5)),
      ETF=StocksSavingsPlan(45_000.00+30_000.00, etf_p, month_budget-1_108.00-384.00, 15, tax_free=1602.00)
    )
    plan.print(only_summary)

  if False:
    print("Capital: 129_800.00, run time: 15a, low rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-409_200.00, 0.97, 1_003.00, 15),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 5)),
      ETF=StocksSavingsPlan(45_000.00-30_800.00, etf_p, month_budget - 1_003.00 - 384.00, 15, tax_free=1602.00)
    )
    plan.print(only_summary)

  if False:
    print("Capital: 99_000.00, run time: 15a, high rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-440_000.00, 1.06, 1_569.00, 15),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 5)),
      ETF=StocksSavingsPlan(45_000.00, etf_p, month_budget-1_569.00-384.00, 15, tax_free=1602.00)
    )
    plan.print(only_summary)

  print()
  print("*" * 30)
  print("*** 20a ***")

  if True:
    print("Capital: 99_000.00, run time: 20a, low rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-440_000.00, 1.28, 1_104.00, 20),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 10)),
      ETF=StocksSavingsPlan(45_000.00, etf_p, month_budget-1_104.00-384.00, 20, tax_free=1602.00)
    )
    plan.print(only_summary)

  if True:
    print("Capital: 99_000.00, run time: 20a, low rate to bank, 30_000.00 from somewhere")
    plan = FinancialPlan(
      bank=AnnuityLoan(-440_000.00, 1.28, 1_104.00, 20),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 10)),
      ETF=StocksSavingsPlan(45_000.00+30_000.00, etf_p, month_budget-1_104.00-384.00, 20, tax_free=1602.00)
    )
    plan.print(only_summary)

  if False:
    print("Capital: 129_800.00, run time: 20a, low rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-409_200.00, 1.25, 1_098.00, 20),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 10)),
      ETF=StocksSavingsPlan(45_000.00-30_800.00, etf_p, month_budget-1_098.00-384.00, 20, tax_free=1602.00)
    )
    plan.print(only_summary)

  if False:
    print("Capital: 99_000.00, run time: 20a, high rate to bank")
    plan = FinancialPlan(
      bank=AnnuityLoan(-440_000.00, 1.29, 1_562.00, 20),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 10)),
      ETF=StocksSavingsPlan(45_000.00, etf_p, month_budget-1_562.00-384.00, 20, tax_free=1602.00)
    )
    plan.print(only_summary)

  if True:
    print("Capital: 99_000.00, run time: 15a + 5a with high interest, low rate to bank")
    plan = FinancialPlan(
      bank=Chain(AnnuityLoan(-439_000.00, 1.06, 1_108.00, 15),
                 AnnuityLoan(None, 3.00, 1_108.00, 5)),
      KfW=Chain(AnnuityLoan(-100_000.00, 0.84, 384.00, 9),
                AnnuityLoan(None, 1.84, 384.00, 10)),
      ETF=StocksSavingsPlan(45_000.00, etf_p, month_budget-1_108.00-384.00, 20, tax_free=1602.00)
    )
    plan.print(only_summary)