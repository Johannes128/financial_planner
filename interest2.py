import copy
from datetime import date
import itertools
import tabulate


TABLULATE_FLOAT_FMT = "6_.2f"

def cum_key(key):
  return key + "_cum"


class Month(date):
  def __new__(cls, year, month):
    return date.__new__(cls, year, month, 1)

  def prev_month(self):
    if self.month > 1:
      return Month(self.year, self.month-1)
    else:
      return Month(self.year-1, 12)

  def next_month(self):
    if self.month < 12:
      return Month(self.year, self.month+1)
    else:
      return Month(self.year+1, 1)

  def __str__(self):
    return f"{self.year:04d}-{self.month:02d}"

  def __repr__(self):
    return f"{self.__class__.__name__}({self.year}, {self.month})"


class MonthEntryBase(dict):
  FLOAT_ATTRIBUTES_GENERAL = ["V_start", "V_end"]
  FLOAT_ATTRIBUTES_CUM = ["rate", "interest", "tax"]
  NON_NUM_ATTRIBUTES = ["start"]

  @classmethod
  def get_show_attributes(cls, level="ALL"):
    if level == "ALL":
      return (["start"] + cls.FLOAT_ATTRIBUTES_GENERAL
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
      self["start"] = Month(1, 1)

    for key in itertools.chain(self.FLOAT_ATTRIBUTES_GENERAL, self.FLOAT_ATTRIBUTES_CUM):
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
    pairs = {key: self.get(key, 0.00) + other.get(key, 0.00) for key in keys}
    pairs["start"] = self["start"]
    return MonthEntryBase(**pairs)


class MonthHistory(list):
  MonthEntry = MonthEntryBase

  def __init__(self, description, V_0, start=None):
    super().__init__()
    self.description = description
    self.V_0 = V_0
    self.start = start
    self.finished = False

    if start is not None:
      self.restart(start)

  def restart(self, start):
    self.start = start
    self.clear()
    self.append(self.MonthEntry(start=self.start.prev_month(), V_start=self.V_0, V_end=self.V_0))
    return self[-1]

  def month_step(self, from_month):
    raise NotImplementedError

  def month_steps(self, num_months, form_month=None):
    cur_month = (self.start if form_month is None else form_month)
    for m in range(num_months):
      self.month_step(cur_month)
      cur_month = cur_month.next_month()
    return self

  def year_steps(self, num_years):
    return self.month_steps(12*num_years)

  def get_year_summaries(self):
    def merge_year_entries(entries):
      entry = copy.copy(entries[-1])
      entry["start"] = entries[0]["start"]
      entry["V_start"] = entries[0]["V_start"]
      for attr in entry.FLOAT_ATTRIBUTES_CUM:
        entry[attr] = sum(e[attr] for e in entries)
      return entry

    years = sorted(set(e["start"].year for e in self[1:]))
    years_entries = {year: [] for year in years}
    for entry in self[1:]:
      years_entries[entry["start"].year].append(entry)

    result = [self[0]]
    for year, year_entries in years_entries.items():
      result.append(merge_year_entries(year_entries))

    return result

  # vvvvvvvvvv output vvvvvvvvvv
  def table_header(self, level="ALL"):
    return self.MonthEntry.get_show_attributes(level)

  def to_table(self, entries, level="ALL"):
    attributes = self.table_header(level)
    return [[entry.get(attr, 0.00) for attr in attributes]
            for entry in entries]

  def to_string(self, entries, level="ALL"):
    table = self.to_table(entries, level)
    return "\n".join([
      f"*** {self.description}",
      tabulate.tabulate(table, floatfmt=TABLULATE_FLOAT_FMT,
                        headers=self.table_header(level))
    ])

  def print(self, level="ALL"):
    print(self.to_string(self, level))

  def print_years(self, level="ALL"):
    years_summaries = self.get_year_summaries()
    print(self.to_string(years_summaries, level))
  # ^^^^^^^^^^ output ^^^^^^^^^^


class AnnuityLoan(MonthHistory):
  class MonthEntry(MonthEntryBase):
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest"]

  def rate(self, start):
    return 1000.00

  def grow_factor(self, start):
    return 1.0 + (0.05/12)

  def month_step(self, from_month):
    if self.finished:
      self.append(copy.copy(self[-1]))

    if from_month < self.start:
      return

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start) # TODO: how to model the rate?

    value = initial_value + rate
    if value >= 0:
      rate = max(0.00, rate - value)
      value = 0.00
      self.finished = True

    value *= self.grow_factor(start)
    interest = value - initial_value - rate

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value, V_end=value,
                                rate=rate, interest=interest,
                                prev=self[-1]))
    return self[-1]


class TaxInfo:
  def __init__(self, tax_rate, tax_free, fraction_to_tax=1.0):
    self.tax_rate = tax_rate
    self.tax_free = tax_free
    self.fraction_to_tax = fraction_to_tax
    self.effective_tax_rate = tax_rate * fraction_to_tax


tax_info_simple = TaxInfo(0.26375, 1_602.00)
tax_info_stocks = TaxInfo(0.26375, 1_602.00, 0.7)


class SavingsPlan(MonthHistory):
  # TODO: * vorschüssig vs nachschüssig
  #       * Zinsperioden: jährlich, >monatlich<, ...

  class MonthEntry(MonthEntryBase):
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest", "tax"]

  def rate(self, start):
    return 1000.00

  def grow_factor(self, start):
    return 1.0 + (0.05/12)

  def month_step(self):
    if self.finished:
      self.append(copy.copy(self[-1]))

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start) # TODO: how to model the rate?

    value = initial_value + rate
    value *= self.grow_factor(start)
    interest = value - initial_value - rate

    # calculate tax
    self.tax_info = tax_info_simple # TODO: move to constructor
    cur_year_entries = [e for e in reversed(self) if e["start"].year == start.year]
    cur_year_interest = sum(e["interest"] for e in cur_year_entries)
    cur_year_interest += interest
    value_for_tax = min(max(0.00, cur_year_interest - self.tax_info.tax_free), interest)
    tax = -self.tax_info.effective_tax_rate * value_for_tax
    value += tax

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value, V_end=value,
                                rate=rate, interest=interest,
                                tax=tax,
                                prev=self[-1]))
    return self[-1]


class StocksSavingsPlan(MonthHistory):
  pass


class Chain(MonthHistory):
  def __init__(self, description, start=None, plans=None):
    super().__init__(description, start)
    self.histories = (plans if plans is not None else [])


class Parallel(MonthHistory):
  def __init__(self, description, start=None, plans=None):
    super().__init__(description, start)
    self.description = description
    self.histories = (plans if plans is not None else [])

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

  def month_step(self, from_month):
    entries = [history.month_step(from_month) for history in self.histories]
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
  # ^^^^^^^^^^ output ^^^^^^^^^^


if False:
  plan = Parallel(
    "Price: 550k, Capital: 91_800.00, run time: 15a",
    date(2021, 1),
    AnnuityLoan("bank", -394_868.00, 1.00, 981.00, 15),
    Chain("KfW", AnnuityLoan(-100_000.00, 0.67, 375.00, 12, 10),
                 AnnuityLoan(None, 1.67, 375.00, 5)),
  )

if False:
  a = AnnuityLoan("bank", 10_000.00, Month(2021, 1))
  a.year_steps(3)
  a.print()

Parallel(
  "parallel plan",
  start=Month(2020, 1),
  plans=[AnnuityLoan("bank", -200_000.00, start=Month(2021, 1)),
         AnnuityLoan("bank", -100_000.00)]
         #SavingsPlan("savings", 100_000.00, start=Month(2022, 1))],
).year_steps(5).print_years()