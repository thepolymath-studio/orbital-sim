"""Week 2026-10-wk1 (The Martian): continuous-thrust scenarios.

Idealised: circular coplanar orbits for Earth (1 AU) and Mars
(1.5237 AU), constant a_T (no mass change), and planetary gravity
ignored. The runs stop when r reaches Mars's orbital radius. There is
no Mars phasing, capture or speed matching.

THRUST_TO_MARS : prograde thrust the whole way.
BURN_COAST     : burn, coast, burn. ILLUSTRATIVE ONLY: it shows
                 "long burns with coasts" (Burke's tables, F22) but
                 is not Burke's or Weir's actual burn schedule.
"""
from __future__ import annotations

from .constants import A_HERMES, DAY, R_EARTH_ORBIT, R_MARS_ORBIT
from .dynamics import circular_state
from .propagate import Arc, fly_schedule

THRUST_TO_MARS = (Arc(True, None),)

BURN_COAST = (
    Arc(True, 40 * DAY),     # burn 1
    Arc(False, 40 * DAY),    # coast
    Arc(True, None),         # burn 2, until Mars's radius
)


def to_mars(arcs, a_T=A_HERMES):
    """Fly from Earth's circular orbit to Mars's radius."""
    y0 = circular_state(R_EARTH_ORBIT)
    return fly_schedule(y0, arcs, a_T, R_MARS_ORBIT)
