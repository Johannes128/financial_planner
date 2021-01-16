import copy
from datetime import date
import itertools
import tabulate


TABLULATE_FLOAT_FMT = "6_.2f"


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
  FLOAT_ATTRIBUTES_CUM = ["rate", "interest"]
  NON_NUM_ATTRIBUTES = ["start"]

  @classmethod
  def get_show_attributes(cls, level="ALL"):
    if level == "ALL":
      return (["start"]
              + cls.FLOAT_ATTRIBUTES_GENERAL + cls.FLOAT_ATTRIBUTES_CUM
              + ["cum_" + attr for attr in cls.FLOAT_ATTRIBUTES_CUM])
    elif level == "MINIMAL":
      return cls.FLOAT_ATTRIBUTES_GENERAL
    else:
      raise ValueError(f"Unknown level={level}")

  def __init__(self, **kwargs):
    prev = kwargs.get("prev", None)
    #kwargs.setdefault("next", None)

    super().__init__(**kwargs)

    if "start" not in self:
      self["start"] = Month(1, 1)

    for key in itertools.chain(self.FLOAT_ATTRIBUTES_GENERAL, self.FLOAT_ATTRIBUTES_CUM):
      if key not in self:
        self[key] = 0.00

    if prev is None:
      for key in self.FLOAT_ATTRIBUTES_CUM:
        if "cum_" + key not in kwargs:
          self["cum_" + key] = self[key]
    else:
      for key in self.FLOAT_ATTRIBUTES_CUM:
        if "cum_" + key not in kwargs:
          self["cum_" + key] = self[key] + prev.get("cum_" + key, 0.00)

      #prev["next"] = self

  def __add__(self, other):
    keys = (set(self.keys()) | set(other.keys())) - set(self.NON_NUM_ATTRIBUTES)
    assert self["start"] == other["start"]
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

  def month_step(self):
    raise NotImplementedError

  def month_steps(self, num_months):
    for m in range(num_months):
      self.month_step()
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

  # vvvvvvvvvvv output vvvvvvvvvvvvv
  def table_header(self, level="ALL"):
    return self.MonthEntry.get_show_attributes(level)

  def to_table(self, entries, level="ALL"):
    attributes = self.table_header(level)
    return [[entry[attr] for attr in attributes]
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
  # ^^^^^^^^^^^^ output ^^^^^^^^^^^^^^


class AnnuityLoan(MonthHistory):
  class MonthEntry(MonthEntryBase):
    FLOAT_ATTRIBUTES_CUM = ["rate", "interest"]

  def rate(self, start):
    return 100.00

  def get_grow_factor(self, start):
    return 1.0 + (0.05/12)

  def month_step(self):
    if self.finished:
      self.append(copy.copy(self[-1]))

    initial_value = self[-1]["V_end"]
    start = self[-1]["start"].next_month()
    rate = self.rate(start) # TODO: how to model the rate?

    value = initial_value + rate
    if value >= 0:
      rate = max(0.00, rate - value)
      value = 0.00
      self.finished = True

    value *= self.get_grow_factor(start)
    interest = value - initial_value - rate

    self.append(self.MonthEntry(start=start,
                                V_start=initial_value, V_end=value,
                                rate=rate, interest=interest,
                                prev=self[-1]))
    return self[-1]


class SavingsPlan(MonthHistory):
  pass


class StocksSavingsPlan(MonthHistory):
  pass


class Chain(MonthHistory):
  def __init__(self, description, start=None, plans=[]):
    super().__init__(description, start)
    self.histories = plans


class Parallel(MonthHistory):
  def __init__(self, description, start=None, plans=[]):
    super().__init__(description, start)
    self.description = description
    self.histories = plans
    self.restart(start)

  def append_sum(self, entries):
    self.append(sum(entries[1:], entries[0]))
    return self[-1]

  def restart(self, start):
    self.start = start
    return self.append_sum([history.restart(start) for history in self.histories])

  def clear(self):
    for history in self.histories:
      history.clear()

  def month_step(self):
    entries = [history.month_step() for history in self.histories]
    self.finished = all(history.finished for history in self.histories)
    return self.append_sum(entries)


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
  "Test",
  start=Month(2021, 1),
  plans=[AnnuityLoan("bank", -10_000.00),
         AnnuityLoan("bank", -10_000.00)],
).year_steps(11).print_years()