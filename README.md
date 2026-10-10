# orbital-sim

The tested orbital mechanics behind [@thepolymath.studio](https://instagram.com/thepolymath.studio)'s
space posts. Every number a post quotes is produced by a script in `examples/` and checked by
`tests/`.

```bash
pip install -e ".[test]"
pytest
```

Physics lives in `src/orbital_sim/`. Numerics come from scipy (`solve_ivp`, DOP853, rtol 1e-12),
and the equations of motion are written out by hand because they are what the posts explain.

| module | what |
|---|---|
| `constants.py` | GM☉, AU, orbit radii, the novel's Hermes thrust, Earth and Moon (all SI) |
| `dynamics.py` | equation of motion r̈ = −μ r/r³ + a_T, thrust laws, circular start state |
| `propagate.py` | DOP853 propagation, stop events, burn/coast schedules, uniform resampling |
| `hohmann.py` | the Hohmann Δv pair and transfer time (baseline) |
| `scenarios.py` | per-post scenarios |
| `ephemeris.py` | circular planet states, JPL mean longitudes (nominal phasing) |
| `flyby.py` | patched-conic flyby: turn angle δ = 2 arcsin(1/e), v∞ rotation, hyperbola |
| `primer.py` | minimum-time low-thrust rendezvous: Lawden's primer vector, batched scan + Levenberg–Marquardt |
| `lowthrust.py` | direct piecewise-linear steering law (optimality cross-check) |
| `race.py` | the 2026-10-wk1 Sim race: scenario and the three strategies |
| `search.py` | the race's search: seeds, continuation in phase and flyby turn |
| `visviva.py` | vis-viva v² = μ(2/r − 1/a), escape speed, v∞², the hyperbola form |
| `apollo13.py` | 2026-10-wk2: one-energy fits to tracking calls, the Moon pass, entry speed |
| `apollo13_data.py` | Apollo 13 Flight Journal calls and Mission Report values, with sources |
| `whatif.py` | 2026-10-wk2 Sim: alternate histories (other Moon passes, the PC+2 burn, a toy drift) |

## 2026-10-wk1: The Martian, option 02 (continuous thrust)

Hermes in Andy Weir's novel accelerates at a constant 2 mm/s² (7.2 m/s every hour). Starting on
Earth's circular orbit (1 AU), it thrusts prograde until it reaches Mars's orbital radius
(1.5237 AU):

- `thrust_to_mars`: the engine is on the whole way;
- `burn_coast`: a 40 d burn, a 40 d coast, then a burn to Mars's radius. This schedule is
  **illustrative** and shows the "long burns with coasts" pattern of NASA Glenn's
  burn tables. It is not Burke's or Weir's actual schedule.

```bash
python examples/2026-10-wk1_continuous_thrust.py --check
```

This writes `outputs/2026-10-wk1/{thrust_to_mars,burn_coast}.csv`, `continuous_thrust.npz` and a
JSON sidecar that gives the columns, units, frame and derived numbers.

Idealisations: constant a_T (no mass loss), circular coplanar planet orbits, and no planetary
gravity. The runs stop at Mars's radius, with no phasing and no capture. Prograde thrust all the
way arrives fast and is not bound to the Sun. The novel's Hermes flips and brakes.

Tests (`tests/test_dynamics.py`, `test_hohmann.py`, `test_continuous_thrust.py`):

- with thrust off, energy and angular momentum are conserved (e = 0.5, 10 orbits, < 1e-10);
- a circular orbit stays circular and returns after Kepler's period;
- dE/dt = v·a_T (finite differences) and E₁ − E₀ = ∫ v·a_T dt;
- Δv = a_T·t, in both the gravity-free limit and the accumulated budget over burn arcs;
- in the low-thrust limit, Δv approaches v_c(r₁) − v_c(a) (Edelbaum, coplanar);
- the integrated Hohmann ellipse reaches aphelion at r₂ after t_H, with Δv₁ = 2.94 and Δv₂ = 2.65 km/s.

## 2026-10-wk1 Sim post: The Martian, the race back to Mars

Hermes is inbound past Earth and has to get back to Mars. Same planets, same start epoch
(t = 0, the Earth encounter), same integrator (DOP853, rtol 1e-12), and the fastest
**rendezvous** with Mars (position and velocity) wins:

- **01 Hohmann**: an impulsive stop onto Earth's orbit, a wait for the phase angle
  α = π − n₂ t_H (eq. 04), then the two Hohmann burns;
- **02 Continuous thrust**: the ion drive alone (2 mm/s², the novel's value), always on and
  steered along Lawden's primer vector, which is Pontryagin's minimum-time law;
- **03 Purnell**: a patched-conic Earth flyby (|v∞| conserved, turn ≤ δ = 2 arcsin(1/e) at a
  300 km periapsis) followed by the same ion drive. A distant pass (ψ = 0) is 02, so 03 is never
  slower.

Set-up: circular coplanar Earth (1 AU) and Mars (1.5237 AU). Hermes coasts in from Mars's
orbit and meets Earth with v∞ = 6 km/s (NASA Glenn's upper limit). The nominal Mars phasing
comes from the JPL mean longitudes on the novel's Earth-flyby date (6 Jul 2036). No route may
pass inside 0.387 AU (Mercury's distance). The race is repeated for 24 start phases, 15° apart.

```bash
python examples/2026-10-wk1_race.py --check            # race + phase sweep (~5 min on 8 cores)
python examples/2026-10-wk1_race.py --extras --check   # floor sensitivity, direct-law gap
```

This writes `outputs/2026-10-wk1-sim/`: `ship_01/02/03.csv`, `planets.csv`,
`flyby_earth_frame.csv` and `sweep.csv`, all on one shared clock, plus `race.npz` and the
`race.json` sidecar. The sidecar documents every column, unit, frame and assumption.
Transit times are banned on the post's slides. They appear only in the sidecar, under
`SIDECAR_ONLY_not_for_slides`.

How the low-thrust optimum is found, and how far to trust it: every 02/03 trajectory is a
Pontryagin extremal (the tests check that the Hamiltonian is constant, and that p(0) ∥ v∞,out
at an interior best flyby turn). The search seeds from a scan, then continues solutions across
start phases and flyby turns. A direct 8-segment piecewise-linear law, warm-started from
the extremal, never beats it (it needs about 0.9% longer). The result is the best extremal found, not a proof of global
optimality. Where the 0.387 AU floor binds, the low-thrust times are upper bounds.

Tests: `test_flyby.py` (eq. 08/09 closed form and integrated through Earth's field),
`test_primer.py`, `test_lowthrust.py`, `test_race.py` (each strategy reaches Mars within
10 km and 1 cm/s, Hohmann against the closed form, Δv = a_th·t, 03 ≤ 02, transversality,
determinism).

## 2026-10-wk2: Apollo 13, the free return by vis-viva

Vis-viva, v² = μ(2/r − 1/a), is the energy form of the same two-body law. Here it meets
real telemetry, using patched conics with one body at a time:

- **Moon leg.** 20 Moon-relative (distance, speed) calls from the NASA Apollo Flight
  Journal, from 75:06 GET to the PC+2 ignition at 79:27:39, are fitted with **one**
  Moon-centred hyperbola (one v∞). The calls' "distance from the Moon" is an altitude:
  read that way, one v∞ = 1.046 km/s fits every call to within 0.21% in speed. Read as
  distance from the centre, no single conic fits. Vis-viva then predicts the speed at the
  Mission Report's pericynthion (136.2 nmi, 8,053.4 ft/s), and equation 09 gives e and the
  turn angle δ.
- **The pass, integrated.** The hyperbola is flown with `propagate.fly` (DOP853, Moon
  only) from pericynthion in both directions. The calls' clock times are an independent
  check, because they land on the integrated time of flight.
- **Earth leg.** 18 Earth-relative calls after the last course correction fix the return
  conic's energy. Vis-viva gives the speed at the entry interface (400,000 ft above the
  equatorial radius), which is compared with the Mission Report and with escape speed
  there.

```bash
python examples/2026-10-wk2_free_return.py --check     # < 1 s
```

This writes `outputs/2026-10-wk2/`: `moon_pass.csv` (the Moon-centred trajectory and
Moon-relative speed, uniform 20 s steps), `moon_readings.csv`, `earth_readings.csv`,
`visviva_curves.csv`, `free_return.npz` and the `free_return.json` sidecar (columns, units,
frame, time base and every derived number).

Idealisations: planar two-body legs. The Earth's pull on the Moon leg and the Moon's pull
on the Earth leg are ignored (the fits show that both are below the calls' scatter over
these windows). Lunar altitudes are measured from the mean radius of 1,737.4 km, while the
report measures from the landing-site radius (each km of datum shifts the pericynthion
speed by about 1.7 ft/s). The figure-eight and the three-body picture are week 3's.

Tests (`tests/test_visviva.py`, `tests/test_free_return.py`):

- vis-viva's limits: circular (a = r), parabola (escape speed), hyperbola (a = −μ/v∞²);
  the inverses for a and the energy;
- the integrator obeys vis-viva at 4,001 samples on an e = 0.6 ellipse and on a Moon
  hyperbola (to 1e-10), conserves energy, and reaches the apoapsis 2a − r_p to 1 m;
- the calls fit one hyperbola as altitudes (< 0.25%) and fail as centre distances
  (> 10%; five calls would even be bound); v∞ inbound = v∞ outbound (eq. 08);
- the pericynthion speed is within 0.05% of the report's 8,053.4 ft/s; the report's
  independent PC+2 state (table 4-II) lies on the fitted hyperbola;
- δ from eq. 09 equals the integrated turn far out; the integrated pass equals the
  closed-form conic (`flyby.hyperbola_state`) to 1 m;
- the calls' GETs fit the integrated time of flight to < 40 s, with the fitted
  pericynthion within 10 s of the report's 77:20:57;
- the entry speed is within 0.1% of the report's 36,210.9 ft/s and just below escape;
- the Earth-frame check: the patched-conic energy change V·Δv∞ equals the vector sum for any
  geometry. Apollo 13's pericynthion lay within 0.5° of the anti-Earth point (the report's
  179.29°W against JPL Horizons' sub-Earth point), so the pass alone changed the Earth-frame
  energy by at most 0.03 km²/s². The PC+2 burn, with its direction solved from the post-burn
  Moon calls, accounts for the 0.217 km²/s² that the Earth calls show to within 0.01.

## 2026-10-wk2 Sim post: Apollo 13, alternate histories

The concept post shows that the real numbers fit vis-viva. This one asks "what if?", using the same
fitted Moon hyperbola (v∞ = 1.046 km/s from 20 Flight Journal calls), the same model (patched
conics, planar) and the same integrator.

1. **A different Moon pass.** v∞ stays the same and the closest approach moves: grazing, 50, 150,
   252 (the real one), 500, 1,000 and 3,000 km. Equation 09 gives e and the turn δ, and vis-viva
   gives the speed at closest approach. Every pass is flown in one Moon-centred frame and arrives
   along +x, so the paths overlay. Lower passes turn more: 92.2° when grazing, 87.7° for the real
   pass, 58.2° at 3,000 km, which is about 1.7° per 100 km near the real pass. The model gives
   the exit direction in the Moon's frame only. It does not say where the Earth would be.
2. **No PC+2 burn.** The final PAD gives the burn in LVLH axes (x +833.0, y −50.9, z −213.9 ft/s;
   the Mission Report gives 860.5 ft/s in total). On the fitted hyperbola, at the burn's midpoint,
   the burn points 58.6° away from the velocity. Only 153 m/s of the 262 m/s adds speed, and
   vis-viva predicts v∞ = 1.241 km/s afterwards. The six Flight Journal calls after the burn fit
   one hyperbola with v∞ = 1.242 km/s (to within 0.03%), and the integrated path reaches their
   distances within 30 s of their clock times. Without the burn, v∞ would have stayed at
   1.046 km/s. For the entry speed, vis-viva bounds every coast home that is bound to the Earth
   and falls from the Moon's distance: 10.98 to 11.075 km/s (escape speed), whatever the path.
   NASA's 11.037 km/s lies inside that range. The "about 9 hours later" is the Mission Report's
   figure (152 h against 142:53:00). The model does **not** compute it, because that needs the
   Earth and the Moon together.
3. **No free-return burn.** This is not simulated, because a two-body model would mislead. The
   script gives the burn's size (11.52 m/s, which is 1.2% of the 940 m/s Earth-relative speed)
   and a labelled TOY: a straight-line drift with gravity ignored, which comes to 658 km by the
   Moon and 3,367 km by entry. The real miss needs the Earth and the Moon pulling together
   (week 3).

```bash
python examples/2026-10-wk2_alt_histories.py --check   # < 1 s
```

This writes `outputs/2026-10-wk2-sim/`: `passes.csv` (all seven passes in long format),
`passes_summary.csv`, `pc2.csv` (GET 75:06 to 82:00, with and without PC+2 on one clock),
`post_pc2_calls.csv`, `alt_histories.npz` and the `alt_histories.json` sidecar.

Tests (`tests/test_alt_histories.py`, 23 tests, < 1 s):

- the aim offset b is angular momentum (b v∞ = r_p v_p), and gravity focuses (b > r_p);
- the turn shrinks with altitude, and dδ/dr_p matches the analytic derivative;
- each pass, flown with DOP853, is the closed-form conic to 1 m, keeps energy to 1e-10, and far
  out arrives along +x and leaves at heading δ (to 0.01°);
- independent of eq. 09, an aim 1% inside the grazing offset hits the Moon and 1% outside misses;
- the burn algebra: no burn changes nothing, a burn along v adds, a sideways burn adds in
  quadrature;
- the PAD burn predicts the post-burn calls' v∞ to 0.1%. A flipped z axis, or counting the
  whole burn along the velocity, misses by more than 5%;
- integrated falls from 384,400 km at the same speed in different directions reach entry at the
  same speed (vis-viva: energy, not path);
- the toy drift's numbers and linearity; determinism.
