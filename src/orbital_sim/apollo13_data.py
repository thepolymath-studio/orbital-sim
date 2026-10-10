"""Apollo 13 tracking numbers, as read out on the air-to-ground loop.

Every row is an Apollo Control (Public Affairs) status call in the
NASA Apollo Flight Journal, transcribed as spoken: distance in
nautical miles, speed in ft/s. The calls give "distance from the
Moon" (or Earth); the fits in apollo13.py show that this is an
ALTITUDE above the surface, not a distance from the centre.

GET is ground elapsed time, as the call states it (hh:mm). Where a
call states no time, the time is the one implied by the call
("one minute to ignition" ...); those rows are marked "inferred".

Mission Report rows come from NASA MSC-02680 (April 1970),
tables 4-II and 4-III (archive.org OCR of the scanned report; the
values used are legible and agree with the Flight Journal calls).
"""

AFJ = "https://apollojournals.org/afj/ap13fj/"

# (GET, altitude nmi, speed ft/s, leg, note). Moon-relative.
# Every coast reading between 75:06 and the PC+2 ignition at
# 79:27:38.95 (the burn changes the orbit, so later calls are a
# different hyperbola and are not used).
MOON_PASS = (
    ("075:06", 5828, 4497, "in", "12day4-approach-moon"),
    ("075:26", 4968, 4631, "in", "12day4-approach-moon"),
    ("075:58", 3573, 4943, "in", "12day4-approach-moon"),
    ("076:18", 2623, 5277, "in", "12day4-approach-moon"),
    ("076:49", 1184, 6218, "in", "12day4-approach-moon"),
    ("077:04", 526, 7122, "in", "12day4-approach-moon"),
    ("077:38", 566, 7062, "out", "13day4-leaving-moon"),
    ("077:49", 1007, 6399, "out", "13day4-leaving-moon"),
    ("078:04", 1704, 5768, "out", "13day4-leaving-moon"),
    ("078:20", 2461, 5343, "out", "13day4-leaving-moon"),
    ("078:38", 3268, 5030, "out", "13day4-leaving-moon"),
    ("078:46", 3664, 4913, "out", "13day4-leaving-moon"),
    ("078:57", 4146, 4791, "out", "13day4-leaving-moon"),
    ("079:10", 4717, 4675, "out", "13day4-leaving-moon"),
    ("079:13", 4839, 4651, "out", "13day4-leaving-moon"),
    ("079:17", 5039, 4616, "out", "13day4-leaving-moon"),
    ("079:23", 5263, 4578, "out", "13day4-leaving-moon"),
    ("079:25", 5349, 4564, "out", "13day4-leaving-moon"),
    ("079:26", 5383, 4560, "out",
     "13day4-leaving-moon; inferred (1 min to ignition)"),
    ("079:27", 5426, 4552, "out",
     "13day4-leaving-moon; inferred (<30 s to ignition)"),
)

# The CAPCOM line the film quotes (7,062 ft/s; NASA: 566 nmi).
FILM_LINE_GET = "077:38"

# (GET, altitude nmi, speed ft/s, note). Earth-relative.
# Every reading after the last course correction (137:39:51.5,
# 2.9 ft/s) that states both numbers, up to 142:17. Later calls
# give "range to go" (downrange), not altitude, and are skipped.
EARTH_RETURN = (
    ("138:02", 35729, 10400, "26day6-servicemoduleseparation"),
    ("138:15", 34350, 10607, "27day6-odysseyresurrected"),
    ("138:26", 33369, 10757,
     "27day6-odysseyresurrected; inferred"),
    ("139:19", 27698, 11779, "27day6-odysseyresurrected"),
    ("139:42", 25227, 12307, "27day6-odysseyresurrected"),
    ("139:59", 23196, 12798, "27day6-odysseyresurrected"),
    ("140:37", 18623, 14144, "27day6-odysseyresurrected"),
    ("140:47", 17225, 14629, "27day6-odysseyresurrected"),
    ("140:58", 15786, 15192, "28day6-farewellaquarius"),
    ("141:16", 13283, 16350, "28day6-farewellaquarius"),
    ("141:24", 12064, 16989, "28day6-farewellaquarius"),
    ("141:40", 9703, 18504, "29day6-returnhome"),
    ("141:44", 9102, 18952, "29day6-returnhome"),
    ("141:56", 7084, 20770, "29day6-returnhome"),
    ("142:00", 6613, 21227, "29day6-returnhome"),
    ("142:08", 5134, 23023, "29day6-returnhome"),
    ("142:14", 4075, 24619, "29day6-returnhome"),
    ("142:17", 3505, 25693, "29day6-returnhome"),
)

# Mission Report MSC-02680, table 4-III (a): pericynthion after
# the free-return burn (61:29:43.5 GET, DPS, 37.8 ft/s).
PERI_ALT_NMI = 136.2
PERI_SPEED_FPS = 8053.4
PERI_GET = "077:20:57"   # OCR "TTs20: ST"; treat as approximate

# Flight Journal / Mission Control: "closest approach to the
# surface of the Moon will be at 137 nautical miles, or 253.7 km"
# (064:39 GET) -- the brief's "about 254 km".
PERI_ALT_AFJ_KM = 253.7

# Table 4-II: PC+2 (transearth injection) ignition, Moon-relative.
PC2_GET = "079:27:38.95"
PC2_ALT_NMI = 5465.26
PC2_SPEED_FPS = 4547.7

# Table 4-III (b): entry interface (400,000 ft), after the fourth
# (final) midcourse correction.
ENTRY_SPEED_FPS = 36210.9
ENTRY_GET = "142:40:46"

# ---------------------------------------------------------------
# Alternate histories (2026-10-wk2 Sim post). Same sources.

# Free-return burn (Mission Report table 4-III: 61:29:43.5 GET,
# 37.8 ft/s; the PAD read up at 060:53:09 said 38.0 ft/s).
FR_BURN_GET = "061:29:43.5"
FR_BURN_DV_FPS = 37.8
FR_BURN_PAD_DV_FPS = 38.0

# Earth-relative calls either side of that burn
# (10day3-free-return): GET, distance nmi, speed ft/s.
EARTH_AROUND_FR_BURN = (
    ("060:44", 187006, 3089, "10day3-free-return"),
    ("061:46", 188863, 3084, "10day3-free-return"),
)

# PC+2 burn. Mission Report section 4.0: 263.8 s, 860.5 ft/s.
# Final PAD (Brand, 077:55:24 GET, 13day4-leaving-moon), Noun 81
# in LVLH, ft/s: x forward along the local horizontal, y across
# the orbit plane, z toward the Moon's centre. 861.5 ft/s total.
PC2_DV_FPS = 860.5
PC2_BURN_S = 263.8
PC2_PAD_LVLH_FPS = (833.0, -50.9, -213.9)

# Every Moon-relative call after PC+2 cutoff (079:32:05) on the
# 13day4-leaving-moon page that states both numbers (79:33,
# 79:51 and 80:27 give no speed). Altitude nmi, speed ft/s.
MOON_AFTER_PC2 = (
    ("079:40", 6003, 4980, "13day4-leaving-moon"),
    ("080:49", 9118, 4718, "13day4-leaving-moon"),
    ("080:56", 9420, 4701, "13day4-leaving-moon"),
    ("081:19", 10431, 4649, "13day4-leaving-moon"),
    ("081:27", 10776, 4633, "13day4-leaving-moon"),
    ("081:45", 11587, 4600, "13day4-leaving-moon"),
)

# Without PC+2 (sourced, not computed): Mission Report section
# 4.0, "would have caused the command module to nominally land in
# the Indian Ocean south of Mauritius Island at approximately 152
# hours"; PC+2 gave "an initial predicted earth landing point in
# the Pacific Ocean at 142:53:00. The transearth trip time was
# thus reduced by about 9 hours." Landing: 142:54:41 (section 1).
# The Flight Journal (10day3) gives the free-return plan's entry
# interface as 151:45 GET.
NO_PC2_LANDING_H = 152.0
PC2_PRED_LANDING_GET = "142:53:00"
SPLASHDOWN_GET = "142:54:41"
MR_HOURS_SAVED = 9
NO_PC2_EI_GET_AFJ = "151:45"

# ---- Earth-frame check of the Moon pass (feedback round 2) ----

# Earth-relative calls BEFORE the pass (Earth reference; PAO
# switched to the Moon frame after ~71 h). (GET, alt nmi, ft/s).
EARTH_PRE_PASS = (
    ("059:51", 185455, 3118, "10day3-free-return"),
    ("061:46", 188863, 3084, "10day3-free-return"),
    ("063:28", 191898, 3034, "10day3-free-return"),
    ("065:27", 195375, 2981, "11day3-minimise-power"),
    ("066:37", 197424, 2953, "11day3-minimise-power"),
    ("068:21", 200396, 2919, "11day3-minimise-power"),
    ("070:12", 203670, 2910, "11day3-minimise-power"),
)

# Earth-relative calls AFTER PC+2 and before MCC-5 (105:18).
EARTH_POST_PC2 = (
    ("086:01", 197899, 3740, "14day4-homeward; approx time"),
    ("092:54", 182390, 3920, "15day4-mailbox"),
    ("094:41", 178274, 3982, "16day4-checkingodyssey"),
    ("104:11", 155111, 4399, "19day5-themanualcoursecorrection"),
    ("104:37", 153971, 4421, "19day5-themanualcoursecorrection"),
)

# Moon-relative calls AFTER the PC+2 burn (alt nmi, ft/s).
MOON_POST_PC2 = (
    ("079:40", 6003, 4980, "13day4-leaving-moon"),
    ("080:49", 9118, 4718, "13day4-leaving-moon"),
    ("080:56", 9420, 4701, "13day4-leaving-moon"),
    ("081:19", 10431, 4649, "13day4-leaving-moon"),
    ("081:27", 10776, 4633, "13day4-leaving-moon"),
    ("081:45", 11587, 4600, "13day4-leaving-moon"),
    ("085:1x", 20400, 4396, "14day4-homeward; approx time"),
)

# PC+2 burn size, Mission Report table 4-III (b).
PC2_DV_FPS = 860.5

# Pericynthion position, table 4-III (a), read from the scanned
# page (archive.org PDF p. 20): 3.02 N, 179.29 W, 77:20:57.
PERI_LAT_DEG = 3.02
PERI_LON_EAST_DEG = -179.29

# JPL Horizons (DE441), 1970-04-15 00:34 UTC = 77:20:57 GET
# (launch 1970-04-11 19:13:00 UTC):
#   Moon geocentric speed 0.9668 km/s, distance 404,450 km,
#   flight-path angle +0.2 deg (Moon near apogee);
#   sub-Earth point on the Moon (MOON_ME, east-positive):
#   lon +0.94 deg, lat -3.50 deg.
V_MOON_HORIZONS = 966.8          # m/s
D_MOON_HORIZONS = 4.0445e8       # m
SUB_EARTH_LON_EAST_DEG = 0.94
SUB_EARTH_LAT_DEG = -3.50
