"""Utility for timing operations."""
import collections
import time

starts = collections.Counter()
totals = collections.Counter()
counts = collections.Counter()
current = ""


def start(key):
  global current
  current += "/" + key
  starts[current] = time.time()


def stop():
  global current
  counts[current] += 1
  totals[current] += time.time() - starts[current]
  current = current[:current.rindex("/")]


def print_times():
  total_time = sum(v if k.count("/") == 1 else 0 for k, v in totals.items())
  for k in sorted(totals):
    print((" " * (k.count("/") - 1)) + f"{k}: {100.0 * totals[k] / total_time:0.1f}% " +
          f"avg={avg_time(k):0.4f}s ({totals[k]:0.1f}s/{counts[k]})")


def total_time(key):
  return totals[key]


def avg_time(key):
  return totals[key] / counts[key]


def clear():
  global starts, totals, counts, current
  starts, totals, counts = collections.Counter(), collections.Counter(), collections.Counter()
  current = ""
