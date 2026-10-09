"""Physical constants (SI) used across orbital-sim.

Sources
-------
MU_SUN : IAU 2009 / DE430 heliocentric gravitational constant GM_sun.
AU     : IAU 2012 Resolution B2 (exact definition).
R_MARS : Mars semi-major axis 1.5237 AU (NASA NSSDC Mars fact sheet,
         227.956e6 km), used as a circular-orbit radius.
A_HERMES : Hermes constant thrust acceleration in Andy Weir's novel,
         2 mm/s^2 (Burke, NASA Glenn, NTRS 20150019662).
"""

MU_SUN = 1.32712440018e20   # m^3 / s^2
AU = 1.495978707e11         # m
DAY = 86_400.0              # s
HOUR = 3_600.0              # s

R_EARTH_ORBIT = 1.0 * AU    # m, circular idealisation
R_MARS_ORBIT = 1.5237 * AU  # m, circular idealisation

A_HERMES = 2.0e-3           # m / s^2

# Earth, for the patched-conic flyby (week 2026-10-wk1 Sim post).
# MU_EARTH : GM_earth, 398600.4418 km^3/s^2 (IERS / DE430).
# R_EARTH  : mean radius 6371.0 km (IUGG), as in research S14.
MU_EARTH = 3.986004418e14   # m^3 / s^2
R_EARTH = 6.371e6           # m
