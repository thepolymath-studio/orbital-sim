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

# Earth-Moon system, for the 2026-10-wk2 post (Apollo 13).
# MU_MOON    : GM_moon, 4902.800066 km^3/s^2 (JPL DE430).
# R_MOON     : mean radius 1737.4 km (NASA NSSDC Moon fact sheet).
# R_EARTH_EQ : equatorial radius 6378.137 km (WGS-84).
# H_ENTRY    : Apollo entry interface, 400,000 ft altitude
#              (Apollo 13 Mission Report, MSC-02680, table 4-III).
MU_MOON = 4.902800066e12    # m^3 / s^2
R_MOON = 1.7374e6           # m
R_EARTH_EQ = 6.378137e6     # m

# NASA units in the Apollo record ("miles" = nautical miles).
FT = 0.3048                 # m, international foot (exact)
NMI = 1852.0                # m, nautical mile (exact)
H_ENTRY = 400_000 * FT      # m, 121.92 km

# D_MOON : Earth-Moon mean distance 384,400 km (NASA NSSDC Moon
#          fact sheet), used only as the "far point" of a fall home.
D_MOON = 3.844e8            # m
