
# Planet Nine Dynamical Exploration (v5.1)

### *Comparative Stability Study: Ammonite (2023 KQ14) vs. Known ETNO Ensemble*

This repository contains a publication-grade orbital integration pipeline designed to evaluate the dynamical stability of Extreme Trans-Neptunian Objects (ETNOs) under the influence of the hypothesized Planet Nine. The framework specifically contrasts the newly discovered ETNO **Ammonite (2023 KQ14)** against a baseline ensemble of known ETNOs (including Sedna, Goblin, and 2012 VP113).

---

## 🔬 Core Methodology & Protocol

The pipeline utilizes a high-precision, paired-clone statistical approach to isolate the dynamical fingerprint of Planet Nine:

1. **Paired Clone Ensembles:** Generates identical multivariate Gaussian clone distributions ($\sigma$ scaled to orbit-determination uncertainties) executed across parallel control groups (with and without Planet Nine).
2. **De-contaminated Jacobi Elements:** Computes osculating elements and orbit-pole angles relative to the shifting **interior massive-body COM** (Sun + Giants + P9) rather than simple heliocentric or global barycentric coordinates. This shields the target's orbital evolution metadata from solar reflex motion and Jovian wobble.
3. **Galactic Tide Tensor:** Integrates external Galactic disk/halo tides via custom `REBOUNDx` force callbacks, converting dynamically between J2000 ecliptic and Galactic coordinates (Heisler & Tremaine 1986; Levison et al. 2001).
4. **Pre-flight Validation:** Enforces strict energy conservation metrics ($|dE/E| \le 10^{-5}$) using a transient validation run before releasing production integrations.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **N-Body Engine:** `rebound` (utilizing the *Mercurius* hybrid symplectic integrator)
* **Extensible Physics:** `reboundx` (for external Galactic tidal forces)
* **Parallelism:** `concurrent.futures.ProcessPoolExecutor` (highly optimized multi-core processing)
* **Numerical Operations:** `numpy`

---

## 🚀 Quick Start

### 1. Installation

Ensure you have the required physical simulation packages installed:

```bash
pip install numpy rebound reboundx

```

### 2. Running a Quick Smoke Test

To verify the environment, integrator accuracy, and pickling serialization without waiting for a $10^9$-year simulation, run the code in **Quick Test Mode**:

```bash
export P9_QUICK_TEST=1
python stability_suite.py

```

*This scales down the integration to $10^3$ years and 2 clones per object for immediate validation.*

### 3. Production Simulation

For publication-grade data generation ($1\text{ Gyr}$ integration timeframe across 24 clones per ensemble):

```bash
python stability_suite.py

```

---

## 📦 Output Data Structure

Simulations are outputted directly to `./p9_v5_results/v5_stability_metrics.pkl`. The pickled payload contains exhaustive reproducibility metadata and time-series arrays for tracked elements:

* `a`, `e`, `inc` (Semi-major axis, eccentricity, inclination)
* `omega`, `varpi` (Argument of periapsis, longitude of perihelion)
* `pole_lon`, `pole_lat` (Orbit-pole orientation drift tracking)
* Full hardware, platform, and package version strings for exact replication.
