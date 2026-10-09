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
| `constants.py` | GM☉, AU, orbit radii, the novel's Hermes thrust, Earth (all SI) |
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
