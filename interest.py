import math
import itertools
import tabulate

class HistoryEntry:
  def __init__(self, V, rate, interest, month, total_interest=0.00, total_rate=0.00,
               tax=0.00, total_tax=0.00, V_for_tax=0.00, V_to_tax=0.00, V_after_tax=0.00,
               vorabpauschale=0.00, tax_on_sell=0.00, tax_free_used=0.00, gain_after_tax=0.00):
    self.V = V
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


class HistoryBase(list):
  def __init__(self, V_0, month_offset=0, total_interest_start=0.00, total_rate_start=0.00):
    super().__init__()
    self.restart(V_0, month_offset, total_interest_start, total_rate_start)

  def restart(self, V_0, month_offset=0, total_interest_start=0.00, total_rate_start=0.00):
    self.clear()
    self.append(HistoryEntry(V_0, rate=0.00, interest=0.00,
                             total_interest=total_interest_start,
                             total_rate=total_rate_start,
                             month=month_offset))

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
            "V", "rate", "total_rate",
            "interest", "total_interest",
            "tax", "total_tax"]

  def _get_table_row(self, e):
    return [e.year, e.month,
            e.V, e.rate, e.total_rate,
            e.interest, e.total_interest,
            e.tax, e.total_tax]

  def _print_table(self, table):
    print(tabulate.tabulate(table, floatfmt="6_.2f",
                            headers=self._get_table_header()))

  def print(self):
    table = [self._get_table_row(e) for e in self]
    self._print_table(table)

  def print_years(self):
    table = [self._get_table_row(e)
             for i,e in enumerate(self)
             if i != 1 and round(e.year) == e.year or i == len(self)-1]
    self._print_table(table)

  def print_minimal(self):
    table = [self._get_table_row(e) for e in [self[0], self[-1]]]
    self._print_table(table)
  # ^^^^^ printing methods ^^^^^


class History(HistoryBase):
  def __init__(self, V_0, p_year, rate,
               years=None, month_offset = 0,
               total_interest_start=0.00, total_rate_start=0.00,
               tax_rate=0.26375, tax_free=1_602.00):
    super().__init__(V_0, month_offset, total_interest_start, total_rate_start)

    self.V_0 = V_0
    self.p_year = p_year

    #self.p_month = p_year/12 # TODO: different for savings plans (exponential growth during the year)
    self.p_month = ( (1+p_year/100) ** (1/12) - 1 ) * 100

    self.rate = rate

    self.years = years
    self.month_offset = month_offset

    self.total_interest_start = total_interest_start
    self.total_rate_start = total_rate_start

    self.tax_rate = tax_rate
    self.tax_free = tax_free

    if self.V_0 is not None:
      self.calculate()

  def continue_history(self, loan=None):
    if loan is not None:
      self.month_offset = loan.last().month
      self.V_0 = loan.last().V
      self.total_interest_start = loan.last().total_interest
      self.total_rate_start = loan.last().total_rate
      self.calculate()

  def get_current_year_entries(self, rel_month):
    cur_year_start_idx = (rel_month - 1) // 12 * 12 + 1  # TODO: respect start month in year
    return [self[i] for i in range(cur_year_start_idx, len(self))]

  def month_step(self, rel_month, abs_month, max_months):
    value = self.last().V
    current_interest = value * self.p_month / 100
    value += current_interest

    current_rate = self.rate
    value += self.rate
    if self.V_0 < 0: # this is a loan
      tax = 0.00
      if value + self.rate >= 0:
        current_rate = value
        value = 0.00
    else: # this is a savings plan
      cur_year_interest = sum(e.interest for e in self.get_current_year_entries(rel_month)) + current_interest
      value_for_tax = min(max(0.00, cur_year_interest - self.tax_free), current_interest)
      tax = -self.tax_rate * value_for_tax
      value += tax

    self.append(HistoryEntry(value, current_rate, current_interest, abs_month, tax=tax))

    if max_months is not None and abs_month - self.month_offset >= max_months:
      return False
    elif max_months is None and value == 0.00:
      return False

    return True

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


class StockSavingsPlan(History):
  # TODO: also respect dividends

  def month_step(self, rel_month, abs_month, max_months):
    # TODO: move somewhere else
    self.part_relevant_for_tax = 0.7
    self.basis_interest = 0.5 / 100 * 0.7


    # TODO: distinguish time of payment (beginning vs. end)
    value = self.last().V + self.rate
    current_interest = value * self.p_month / 100
    value += current_interest

    current_rate = self.rate
    #value += self.rate

    self.append(HistoryEntry(value, current_rate, current_interest, abs_month))

    # at the end of year, calculate tax
    year_entries = self.get_current_year_entries(rel_month)
    if len(year_entries) >= 12:
      cur_year_interest = sum(e.interest for e in year_entries)
      cur_year_rate = sum(e.rate for e in year_entries)
      value_at_beginning = year_entries[-1].V - cur_year_interest - cur_year_rate

      vorabpauschale_begin = value_at_beginning * self.basis_interest
      vorabpauschale_rates = sum((12.0-i)/12 * e.rate for i,e in enumerate(year_entries, start=0)) * self.basis_interest
      vorabpauschale = vorabpauschale_begin + vorabpauschale_rates

      V_for_tax = min(vorabpauschale, cur_year_interest)
      V_part_for_tax = V_for_tax * self.part_relevant_for_tax
      tax = -max(0.00, V_part_for_tax - self.tax_free) * self.tax_rate

      del self[-1]
      self.append(HistoryEntry(value, current_rate, current_interest, abs_month,
                               tax=tax, V_for_tax=V_for_tax, vorabpauschale=vorabpauschale))

    last = self.last()
    tax_paid = sum(e.tax for e in self[:-1])
    V_tax_paid = sum(e.V_for_tax for e in self[:-1])
    last.V_to_tax = value - self[0].V - last.total_rate - V_tax_paid
    V_to_tax = max(0.00, last.V_to_tax * self.part_relevant_for_tax - self.tax_free)
    last.tax_on_sell = -V_to_tax * self.tax_rate
    last.V_after_tax = last.V + last.tax_on_sell + tax_paid

    last.gain_after_tax = last.total_interest + last.tax_on_sell + tax_paid

    if max_months is not None and abs_month - self.month_offset >= max_months:
      return False
    elif max_months is None and value == 0.00:
      return False

    return True

  # vvvvv printing methods vvvvv
  def _get_table_header(self):
    return ["year", "m",
            "V", "rate", "tot_rate",
            "interest", "tot_interest",
            "VAP", "V_for_tax",
            "tax", "tot_tax", "V_to_tax", "tax_sell",
            "V_aft_tax", "g_aft_tax"]

  def _get_table_row(self, e):
    return [e.year, e.month,
            e.V, e.rate, e.total_rate,
            e.interest, e.total_interest,
            e.vorabpauschale, e.V_for_tax,
            e.tax, e.total_tax, e.V_to_tax, e.tax_on_sell,
            e.V_after_tax, e.gain_after_tax]
  # ^^^^^ printing methods ^^^^^


class Chain:
  def __init__(self, *loans):
    self.loans = loans
    self.calculate()

  def calculate(self):
    last_l = None
    for l in self.loans:
      l.continue_history(last_l)
      last_l = l

  def print_years(self):
    for l in self.loans:
      l.print_years()
      print()

  def print(self):
    for l in self.loans:
      l.print()
      print()



#Chain(AnnuityLoan(-345_450.00, 1.28, 938.00, 15),
#      AnnuityLoan(None, 1.01, 1001.00, 10)).print_years()

#Chain(SavingsPlan(10_000.00, 1.00, 100.00, 10),
#      SavingsPlan(None, 1.01, 1001.00, 10)).print_years()

StockSavingsPlan(100_000.00, 5.00, 1000.00, 15, tax_free=1602.00).print_years()