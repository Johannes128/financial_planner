import math
import tabulate

class LoanHistoryEntry:
  def __init__(self, V, rate, interest, month, total_interest=0.00, total_rate=0.00):
    self.V = V
    self.rate = rate
    self.total_rate = total_rate
    self.interest = interest
    self.total_interest = total_interest
    self.month = month
    self.year = month/12


class LoanHistory(list):
  def __init__(self, V_0, month_offset=0, interest_start=0.00, total_rate_start=0.00):
    super().__init__()
    self.append(LoanHistoryEntry(V_0, rate=0.00, interest=0.00,
                                 total_interest=interest_start,
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
      super().append(entry)

  # vvvvv printing methods vvvvv
  def _get_table_header(self):
    return ["year", "month",
            "V", "rate", "total_rate",
            "interest", "total_interest",
            "reduction", "reduction_percent"]

  def _get_table_row(self, e):
    return [e.year, e.month,
            e.V, e.rate, e.total_rate,
            e.interest, e.total_interest]

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


class AnnuityLoan:
  def __init__(self, S_0, p_year, rate,
               years=None, month_offset = 0,
               total_interest_start=0.00, total_rate_start=0.00):
    self.S_0 = S_0
    self.p_year = p_year
    self.p_month = p_year/12
    self.rate = rate

    self.years = years
    self.month_offset = month_offset

    self.total_interest_start = total_interest_start
    self.total_rate_start = total_rate_start

  def continue_history(self, history=None):
    if history is not None:
      self.month_offset = history.last().month
      self.S_0 = history.last().V
      self.total_interest_start = history.last().total_interest
      self.total_rate_start = history.last().total_rate


  def get_history(self):
    if self.years is not None:
      max_months = 12*self.years
    else:
      max_months = None

    history = LoanHistory(self.S_0, self.month_offset,
                          self.total_interest_start, self.total_rate_start)
    month = self.month_offset+1
    while True:
      value = history.last().V
      current_interest = value * self.p_month/100
      value += current_interest

      if self.S_0 < 0 and value + self.rate >= 0:
        current_rate = value
        value = 0.00
      else:
        current_rate = self.rate
        value += self.rate

      history.append(LoanHistoryEntry(value, current_rate, current_interest,
                                      month))

      if max_months is not None and month-self.month_offset >= max_months:
        break
      elif max_months is None and value == 0.00:
        break

      month += 1

    return history

  def print_years(self):
    self.get_history().print_years()


class LoanChain:
  def __init__(self, *loans):
    self.loans = loans
    self.histories = []
    self.calculate()

  def calculate(self):
    self.histories = []
    last_h = None
    for l in self.loans:
      l.continue_history(last_h)
      last_h = l.get_history()
      self.histories.append(last_h)

  def print_years(self):
    for l,h in zip(self.loans, self.histories):
      h.print_years()
      print()

  def print(self):
    for l,h in zip(self.loans, self.histories):
      h.print()
      print()



LoanChain(AnnuityLoan(-345_450.00, 1.28, 938.00, 15),
         AnnuityLoan(None, 1.01, 1001.00, 10)).print_years()

#LoanChain(AnnuityLoan(345_450.00, 1.28, 938.00, 15),
#          AnnuityLoan(None, 1.01, 1001.00, 10)).print_years()

