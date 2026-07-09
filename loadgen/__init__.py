"""GoodputLab open-loop load generator.

Public surface: arrival processes (Poisson, ON/OFF) and the per-workload
trace generators that build on top of them.  All randomness flows
through a single ``random.Random(seed)`` per generator instance, which
keeps replay byte-identical (LOAD-07).
"""

from loadgen.agentic import AgenticTraceGenerator, AgenticWorkloadConfig
from loadgen.arrival import OnOffArrival, OpenLoopScheduler, PoissonArrival
from loadgen.chat import ChatTraceGenerator, ChatWorkloadConfig
from loadgen.rag import RagTraceGenerator, RagWorkloadConfig

__all__ = [
    "AgenticTraceGenerator",
    "AgenticWorkloadConfig",
    "ChatTraceGenerator",
    "ChatWorkloadConfig",
    "OnOffArrival",
    "OpenLoopScheduler",
    "PoissonArrival",
    "RagTraceGenerator",
    "RagWorkloadConfig",
]


