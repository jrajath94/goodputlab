"""GoodputLab open-loop load generator.

Public surface: arrival processes (Poisson, ON/OFF) and the per-workload
trace generators that build on top of them.  All randomness flows
through a single ``random.Random(seed)`` per generator instance, which
keeps replay byte-identical (LOAD-07).
"""

from loadgen.arrival import OnOffArrival, OpenLoopScheduler, PoissonArrival

__all__ = ["OnOffArrival", "OpenLoopScheduler", "PoissonArrival"]

