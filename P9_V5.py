"""
==============================================================================
  PLANET NINE DYNAMICAL EXPLORATION — v5.1 (Publication-Grade)
  Ammonite (2023 KQ14) vs. Known ETNO Ensemble — Comparative Stability Study
==============================================================================
Protocol:
  1. Paired clone ensembles (identical ICs) integrated with/without Planet Nine.
  2. Jacobi elements relative to interior massive-body COM (Sun+giants+P9).
  3. Tracks a, e, inc, omega, varpi, and orbit-pole drift (ecliptic lon/lat).
  4. Pre-flight energy validation + full reproducibility metadata in output.
==============================================================================
"""

import os
import sys
import time
import pickle
import platform
from datetime import datetime, timezone

import numpy as np
import rebound
import reboundx
from concurrent.futures import ProcessPoolExecutor

# ════════════════════════════════════════════════════════════════════════════
# SIMULATION PARAMETERS
# ════════════════════════════════════════════════════════════════════════════
PROTOCOL_VERSION = "5.3"
RANDOM_SEED = 42
CLONE_SCALE = 1e-3

# Galactic tide (Heisler & Tremaine 1986; Levison et al. 2001 tensor).
# Oort constants in km/s/kpc; local midplane density in Msun/pc^3.
GALACTIC_RHO_MSUN_PC3 = 0.10   # Levison et al. (2006); standard ETNO/Oort-cloud value
OORT_A_KMS_KPC = 15.3          # IAU 2019 local standard of rest
OORT_B_KMS_KPC = -11.9         # IAU 2019 (negative = retrograde shear)

# Ecliptic J2000 directions for the galactic basis (Batygin & Brown 2019 convention).
_ECL_NGP_LON = np.radians(96.38)
_ECL_NGP_LAT = np.radians(29.81)
_ECL_GC_LON = np.radians(266.57)
_ECL_GC_LAT = np.radians(5.53)

_PC_TO_AU = 206264.80624709636
_KM_TO_AU = 1.0 / 1.495978707e8
_SEC_PER_YR = 31557600.0
_KMS2_TO_AU_YR2 = _KM_TO_AU / (_SEC_PER_YR ** 2)
_KMS_KPC_TO_YR = (1000.0 / 3.0856775814913673e16) * _SEC_PER_YR
_AU_TO_KPC = 1.0 / (_PC_TO_AU * 1000.0)
_GALACTIC_RHO_MSUN_AU3 = GALACTIC_RHO_MSUN_PC3 / (_PC_TO_AU ** 3)
_OORT_A_YR = OORT_A_KMS_KPC * _KMS_KPC_TO_YR
_OORT_B_YR = OORT_B_KMS_KPC * _KMS_KPC_TO_YR
_TIDE_COEFF_Z = -(
    4.0 * np.pi * (4.0 * np.pi ** 2) * _GALACTIC_RHO_MSUN_AU3
    + 2.0 * (_OORT_A_YR ** 2 - _OORT_B_YR ** 2)
)
_OORT_AB_KMS_KPC = OORT_A_KMS_KPC - OORT_B_KMS_KPC
_OORT_2B_KMS_KPC = -2.0 * OORT_B_KMS_KPC

_z_gal = np.array([
    np.cos(_ECL_NGP_LAT) * np.cos(_ECL_NGP_LON),
    np.cos(_ECL_NGP_LAT) * np.sin(_ECL_NGP_LON),
    np.sin(_ECL_NGP_LAT),
])
_x_gal = np.array([
    np.cos(_ECL_GC_LAT) * np.cos(_ECL_GC_LON),
    np.cos(_ECL_GC_LAT) * np.sin(_ECL_GC_LON),
    np.sin(_ECL_GC_LAT),
])
_x_gal -= np.dot(_x_gal, _z_gal) * _z_gal
_x_gal /= np.linalg.norm(_x_gal)
_y_gal = np.cross(_z_gal, _x_gal)
_R_ECL_TO_GAL = np.vstack([_x_gal, _y_gal, _z_gal])
_R_GAL_TO_ECL = _R_ECL_TO_GAL.T

T_END = 1.0e9          # Integration end time (yr). Use 4.6e9 for full Solar System age.
N_STEPS = 2000         # Temporal checkpoints
DT_BASE = 0.5          # Mercurius fixed step (yr)
N_CLONES = 24          # Clones per ETNO (increase for publication statistics)
SAVE_DIR = "./p9_v5_results"

VALIDATE_BEFORE_RUN = True
VALIDATION_T_YR = 1.0e5
MAX_ENERGY_DRIFT = 1e-5   # Relative |dE/E| over VALIDATION_T_YR (Mercurius-typical)

# Set environment variable P9_QUICK_TEST=1 for a fast smoke test (~minutes).
if os.environ.get("P9_QUICK_TEST", "").lower() in ("1", "true", "yes"):
    T_END = 1.0e3
    N_STEPS = 10
    N_CLONES = 2
    VALIDATION_T_YR = 100.0

os.makedirs(SAVE_DIR, exist_ok=True)

# Per-worker cache (child processes only).
_WORKER_BASE_SIMS = {}

# ETNO catalog: observed elements in degrees (J2000 ecliptic; see metadata note on
# Jacobi conversion applied at clone insertion).
ETNO_CATALOG = {
    "Ammonite":   {"a": 251.9, "e": 0.7383, "inc": 24.1, "Omega": 122.4, "omega": 301.2, "M": 12.4},
    "Sedna":      {"a": 506.8, "e": 0.8496, "inc": 11.9, "Omega": 144.3, "omega": 311.5, "M": 358.1},
    "2012_VP113": {"a": 266.2, "e": 0.7050, "inc": 24.1, "Omega": 90.8,  "omega": 293.8, "M": 5.2},
    "Goblin":     {"a": 1010., "e": 0.9350, "inc": 11.6, "Omega": 301.0, "omega": 118.0, "M": 355.0},
    "2013_SY99":  {"a": 733.0, "e": 0.9310, "inc": 4.2,  "Omega": 29.5,  "omega": 32.1,  "M": 359.4},
    "2015_BP519": {"a": 433.0, "e": 0.9200, "inc": 54.1, "Omega": 135.1, "omega": 348.2, "M": 1.1},
}

# Batygin & Brown (2021) nominal Planet Nine parameters.
P9_BASELINE = {
    "m": 1.5e-5,
    "a": 380.0,
    "e": 0.15,
    "inc": np.radians(16.0),
    "Omega": np.radians(95.0),
    "omega": np.radians(300.0),
    "M": 0.0,
}

SUN_MASS = 1.00000598  # Sun + folded Mercury–Mars mass (Msun)

# J2000.0 heliocentric fallback (radians for angles). Source: JPL DE440.
GIANT_PLANET_FALLBACK = {
    "Jupiter": {
        "m": 0.0009543, "a": 5.20336301, "e": 0.04839266,
        "inc": np.radians(1.30529793), "Omega": np.radians(100.55629091),
        "omega": np.radians(14.75308576), "M": np.radians(20.02026099),
    },
    "Saturn": {
        "m": 0.0002858, "a": 9.53707032, "e": 0.05415060,
        "inc": np.radians(2.48446976), "Omega": np.radians(113.66242448),
        "omega": np.radians(92.59887871), "M": np.radians(317.02045047),
    },
    "Uranus": {
        "m": 0.0000436, "a": 19.19126393, "e": 0.04716771,
        "inc": np.radians(0.76964716), "Omega": np.radians(74.01692503),
        "omega": np.radians(170.95427630), "M": np.radians(142.59024484),
    },
    "Neptune": {
        "m": 0.0000515, "a": 30.06896348, "e": 0.00858587,
        "inc": np.radians(1.76995259), "Omega": np.radians(131.78405702),
        "omega": np.radians(44.62925369), "M": np.radians(260.24714907),
    },
}

GIANT_PLANETS = ["Jupiter", "Saturn", "Uranus", "Neptune"]

TRACKED_ELEMENTS = ("a", "e", "inc", "omega", "varpi", "pole_lon", "pole_lat")


# ════════════════════════════════════════════════════════════════════════════
# ORBITAL / STATISTICAL HELPERS
# ════════════════════════════════════════════════════════════════════════════
def lon_peri(orbit):
    """Longitude of perihelion (varpi = Omega + omega), radians."""
    return np.mod(orbit.Omega + orbit.omega, 2.0 * np.pi)


def orbit_pole_angles_deg(particle, primary):
    """
    Ecliptic longitude/latitude (deg) of the orbit pole from COM-relative h.

    Uses r_rel = r_clone - r_COM and v_rel = v_clone - v_COM so that pole
    drift is not polluted by SSB/Jovian wobble in the global frame.
    """
    rx = particle.x - primary.x
    ry = particle.y - primary.y
    rz = particle.z - primary.z
    vx = particle.vx - primary.vx
    vy = particle.vy - primary.vy
    vz = particle.vz - primary.vz

    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h_norm = np.sqrt(hx * hx + hy * hy + hz * hz)
    if h_norm == 0.0 or not np.isfinite(h_norm):
        return np.nan, np.nan
    pole_lat = np.degrees(np.arcsin(np.clip(hz / h_norm, -1.0, 1.0)))
    pole_lon = np.degrees(np.arctan2(hy, hx)) % 360.0
    return pole_lon, pole_lat


def _particle_index(sim, particle_name):
    for i in range(sim.N):
        if sim.particles[i].name == particle_name:
            return i
    raise KeyError(particle_name)


def _interior_com_particle(sim, particle_index):
    """COM particle for all bodies with index < particle_index (Jacobi primary)."""
    total_m = 0.0
    sx = sy = sz = 0.0
    svx = svy = svz = 0.0
    for i in range(particle_index):
        body = sim.particles[i]
        total_m += body.m
        sx += body.m * body.x
        sy += body.m * body.y
        sz += body.m * body.z
        svx += body.m * body.vx
        svy += body.m * body.vy
        svz += body.m * body.vz
    if total_m <= 0.0:
        return None
    inv_m = 1.0 / total_m
    return rebound.Particle(
        m=total_m,
        x=sx * inv_m, y=sy * inv_m, z=sz * inv_m,
        vx=svx * inv_m, vy=svy * inv_m, vz=svz * inv_m,
    )


def add_clone_with_jacobi_ic(sim, a, e, inc, Omega, omega, M, name):
    """
    Insert a massless clone whose Keplerian elements are Jacobi-relative to the
    interior massive-body COM. Passing primary explicitly prevents REBOUND from
    defaulting to the Sun and distorting the covariance cloud at t = 0.
    """
    primary = _interior_com_particle(sim, sim.N)
    if primary is None:
        raise RuntimeError("Cannot add clone: no interior massive bodies found.")
    sim.add(
        primary=primary,
        m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M,
    )
    sim.particles[-1].name = name


def get_jacobi_elements(sim, particle_name):
    """
    Jacobi osculating elements for a clone relative to the interior massive-body COM.

    Explicit primary avoids heliocentric contamination from solar reflex motion.
    """
    try:
        p = sim.particles[particle_name]
        idx = _particle_index(sim, particle_name)
        primary = _interior_com_particle(sim, idx)
        if primary is None:
            return None
        return p.orbit(primary=primary)
    except (KeyError, IndexError, ValueError):
        return None


def orbit_is_trackable(orb, max_distance=2500.0):
    if orb is None:
        return False
    if not np.isfinite(orb.a) or not np.isfinite(orb.e):
        return False
    if orb.a <= 0.0 or orb.e < 0.0 or orb.e >= 1.0:
        return False
    if orb.a > max_distance:
        return False
    return True


def generate_covariance_clones(central_elements, num_clones, scale=CLONE_SCALE, rng=None):
    """
    Diagonal Gaussian clone ensemble around a nominal ETNO orbit.
    Sigmas mimic modest orbit-determination uncertainties (see metadata).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    mean = np.array([
        central_elements["a"], central_elements["e"],
        np.radians(central_elements["inc"]),
        np.radians(central_elements["Omega"]),
        np.radians(central_elements["omega"]),
        np.radians(central_elements["M"]),
    ], dtype=float)

    sigma = np.array([
        max(0.005 * mean[0], 0.5),
        0.01,
        np.radians(0.5),
        np.radians(0.5),
        np.radians(0.5),
        np.radians(1.0),
    ], dtype=float)

    cov = np.diag(sigma ** 2) * scale
    clones = rng.multivariate_normal(mean, cov, num_clones)
    clones[:, 0] = np.clip(clones[:, 0], 1e-6, None)
    clones[:, 1] = np.clip(clones[:, 1], 0.0, 0.999)
    clones[:, 2] = np.clip(clones[:, 2], 0.0, np.pi)
    clones[:, 3:] = np.mod(clones[:, 3:], 2.0 * np.pi)
    return clones


def generate_all_clone_fields(seed=RANDOM_SEED, scale=CLONE_SCALE):
    """Generate paired clone ICs once; shared by with-P9 and without-P9 suites."""
    rng = np.random.default_rng(seed)
    return {
        name: generate_covariance_clones(elements, N_CLONES, scale=scale, rng=rng)
        for name, elements in ETNO_CATALOG.items()
    }


# ════════════════════════════════════════════════════════════════════════════
# GALACTIC TIDE (REBOUNDx — attached per-worker after sim.copy())
# ════════════════════════════════════════════════════════════════════════════
def _galactic_tide_update_accelerations(reb_sim, rebx_force, particles, N):
    """
    Heisler & Tremaine (1986) galactic tidal tensor in a galactic frame,
    applied to all particles via REBOUNDx create_force callback.
    """
    for i in range(N):
        pos_ecl = np.array([particles[i].x, particles[i].y, particles[i].z])
        pos_gal = _R_ECL_TO_GAL @ pos_ecl
        xg_kpc, yg_kpc, zg_kpc = pos_gal * _AU_TO_KPC

        ax_kms2 = _OORT_AB_KMS_KPC * xg_kpc
        ay_kms2 = _OORT_2B_KMS_KPC * xg_kpc
        az_kms2 = _TIDE_COEFF_Z * pos_gal[2]

        acc_gal = np.array([
            ax_kms2 * _KMS2_TO_AU_YR2,
            ay_kms2 * _KMS2_TO_AU_YR2,
            az_kms2,
        ])
        acc_ecl = _R_GAL_TO_ECL @ acc_gal
        particles[i].ax += acc_ecl[0]
        particles[i].ay += acc_ecl[1]
        particles[i].az += acc_ecl[2]


def _attach_galactic_tide(sim):
    """
    Attach the standard galactic tide through REBOUNDx.

    Must be called inside each worker process (and validation) after sim.copy()
    so reboundx pointers survive ProcessPoolExecutor pickling.
    """
    rebx = reboundx.Extras(sim)
    gt = rebx.create_force("galactic_tide")
    gt.force_type = "pos"
    gt.update_accelerations = _galactic_tide_update_accelerations
    rebx.add_force(gt)
    return rebx


# ════════════════════════════════════════════════════════════════════════════
# SIMULATION BUILD
# ════════════════════════════════════════════════════════════════════════════
def _configure_solar_system_from_horizons(sim):
    sim.add("Sun")
    sim.particles[0].m = SUN_MASS
    sim.particles[0].name = "Sun"
    for planet in GIANT_PLANETS:
        sim.add(planet)
        sim.particles[-1].name = planet


def _configure_solar_system_from_fallback(sim):
    sim.add(m=SUN_MASS)
    sim.particles[0].name = "Sun"
    sun = sim.particles[0]
    for planet in GIANT_PLANETS:
        fb = GIANT_PLANET_FALLBACK[planet]
        sim.add(
            primary=sun,
            m=fb["m"], a=fb["a"], e=fb["e"], inc=fb["inc"],
            Omega=fb["Omega"], omega=fb["omega"], M=fb["M"],
        )
        sim.particles[-1].name = planet


def build_base_solar_system(p9_dict=None):
    """
    Build Sun + giants (+ optional P9). Returns (simulation, build_info dict).
    """
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    build_info = {
        "ephemeris_source": "horizons",
        "horizons_error": None,
        "reference_plane": "ecliptic J2000",
    }

    try:
        _configure_solar_system_from_horizons(sim)
    except Exception as exc:
        build_info["ephemeris_source"] = "j2000_fallback"
        build_info["horizons_error"] = repr(exc)
        sim = rebound.Simulation()
        sim.units = ("yr", "AU", "Msun")
        _configure_solar_system_from_fallback(sim)

    if p9_dict is not None:
        sim.add(
            m=p9_dict["m"], a=p9_dict["a"], e=p9_dict["e"], inc=p9_dict["inc"],
            Omega=p9_dict["Omega"], omega=p9_dict["omega"], M=p9_dict["M"],
        )
        sim.particles[-1].name = "PlanetNine"

    sim.move_to_com()
    sim.integrator = "mercurius"
    sim.dt = DT_BASE
    sim.testparticle_type = 1
    sim.exit_max_distance = 2500.0
    return sim, build_info


def _get_base_simulation_copy(p9_presence):
    cache_key = "with_p9" if p9_presence else "without_p9"
    if cache_key not in _WORKER_BASE_SIMS:
        p9_dict = P9_BASELINE if p9_presence else None
        sim, _ = build_base_solar_system(p9_dict=p9_dict)
        _WORKER_BASE_SIMS[cache_key] = sim
    return _WORKER_BASE_SIMS[cache_key].copy()


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════════════════════
def validate_integrator(t_test_yr=VALIDATION_T_YR, max_rel_drift=MAX_ENERGY_DRIFT):
    """
    Pre-flight check: short integration with P9 + one Ammonite clone.
    Returns (passed: bool, relative_energy_drift: float).
    """
    sim, _ = build_base_solar_system(p9_dict=P9_BASELINE)
    _attach_galactic_tide(sim)  # reboundx: attach after base sim, before clone insertion
    clones = generate_covariance_clones(ETNO_CATALOG["Ammonite"], 1)
    a, e, inc, Om, om, M = clones[0]
    add_clone_with_jacobi_ic(sim, a, e, inc, Om, om, M, "validation_clone")
    sim.move_to_com()

    e0 = sim.energy()
    if not np.isfinite(e0) or e0 == 0.0:
        return False, float("nan")

    sim.integrate(t_test_yr, exact_finish_time=0)
    rel_drift = abs((sim.energy() - e0) / e0)
    return rel_drift <= max_rel_drift, float(rel_drift)


# ════════════════════════════════════════════════════════════════════════════
# WORKER & PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def _worker_integrate_clone_ensemble(args):
    object_name, clone_idx, clone_elements, p9_presence, t_end, n_steps = args
    sim = _get_base_simulation_copy(p9_presence)
    _attach_galactic_tide(sim)  # reboundx: attach inside worker after sim.copy()

    a, e, inc, Omega, omega, M = clone_elements
    p_name = f"clone_{object_name}_{clone_idx}"
    add_clone_with_jacobi_ic(sim, a, e, inc, Omega, omega, M, p_name)
    sim.move_to_com()

    times = np.linspace(0.0, t_end, n_steps)
    tracks = {key: np.full(n_steps, np.nan) for key in TRACKED_ELEMENTS}

    is_alive = True
    for idx, t in enumerate(times):
        if not is_alive:
            break
        try:
            sim.integrate(t, exact_finish_time=0)
        except rebound.Escape:
            is_alive = False
            break

        orb = get_jacobi_elements(sim, p_name)
        if not orbit_is_trackable(orb):
            is_alive = False
            break

        p = sim.particles[p_name]
        p_idx = _particle_index(sim, p_name)
        primary = _interior_com_particle(sim, p_idx)
        plon, plat = orbit_pole_angles_deg(p, primary)

        tracks["a"][idx] = orb.a
        tracks["e"][idx] = orb.e
        tracks["inc"][idx] = np.degrees(orb.inc)
        tracks["omega"][idx] = np.mod(np.degrees(orb.omega), 360.0)
        tracks["varpi"][idx] = np.mod(np.degrees(lon_peri(orb)), 360.0)
        tracks["pole_lon"][idx] = plon
        tracks["pole_lat"][idx] = plat

    return object_name, clone_idx, tracks


def run_stability_suite(p9_presence, clone_fields):
    """
    Integrate a pre-generated clone field. Same clone_fields must be used for
    both with-P9 and without-P9 runs to enable paired Delta-varpi analysis.
    """
    p9_mode = "with_P9" if p9_presence else "without_P9"
    print(f"\n  Execution mode: [{p9_mode}]")

    work_tasks = []
    for target_name in ETNO_CATALOG:
        clone_field = clone_fields[target_name]
        for c_idx in range(N_CLONES):
            work_tasks.append(
                (target_name, c_idx, clone_field[c_idx], p9_presence, T_END, N_STEPS)
            )

    compiled = {
        name: {key: [] for key in TRACKED_ELEMENTS}
        for name in ETNO_CATALOG
    }

    worker_count = min(len(work_tasks), os.cpu_count() or 1)
    print(f"  Tasks: {len(work_tasks)} | Workers: {worker_count}")
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for obj_name, clone_idx, tracks in executor.map(_worker_integrate_clone_ensemble, work_tasks):
            for key in TRACKED_ELEMENTS:
                compiled[obj_name][key].append(tracks[key])

    for obj_name in compiled:
        for key in TRACKED_ELEMENTS:
            compiled[obj_name][key] = np.array(compiled[obj_name][key])

    core_hours = (time.time() - t0) * worker_count / 3600.0
    print(f"  Completed in {core_hours:.3f} core-hours.")
    return compiled


def build_run_metadata(build_info, clone_fields, validation_result):
    """Full reproducibility block stored in the output pickle."""
    p9_record = {
        "m_msun": P9_BASELINE["m"],
        "a_au": P9_BASELINE["a"],
        "e": P9_BASELINE["e"],
        "inc_deg": float(np.degrees(P9_BASELINE["inc"])),
        "Omega_deg": float(np.degrees(P9_BASELINE["Omega"])),
        "omega_deg": float(np.degrees(P9_BASELINE["omega"])),
        "M_deg": float(np.degrees(P9_BASELINE["M"])),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "clone_scale": CLONE_SCALE,
        "t_end_yr": T_END,
        "n_steps": N_STEPS,
        "n_clones": N_CLONES,
        "dt_yr": DT_BASE,
        "integrator": "mercurius",
        "timeline_yr": np.linspace(0.0, T_END, N_STEPS),
        "rebound_version": rebound.__version__,
        "reboundx_version": reboundx.__version__,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "ephemeris_source": build_info["ephemeris_source"],
        "horizons_error": build_info["horizons_error"],
        "reference_plane": build_info["reference_plane"],
        "sun_mass_msun": SUN_MASS,
        "inner_planets": "Mercury-Mars mass folded into Sun",
        "giants": GIANT_PLANETS,
        "clone_frame_note": (
            "Catalog elements are J2000 ecliptic. Clones are inserted and tracked "
            "as Jacobi elements relative to the interior massive-body COM. "
            "sim.add() uses explicit primary=interior_COM for IC conversion; "
            "orbit poles use COM-relative r and v (not SSB/global coordinates)."
        ),
        "tracked_quantities": list(TRACKED_ELEMENTS),
        "p9_baseline": p9_record,
        "galactic_tide": {
            "enabled": True,
            "model": "Heisler & Tremaine (1986) tidal tensor; Levison et al. (2001) ETNO standard",
            "implementation": "reboundx create_force('galactic_tide') — attached per-worker after sim.copy()",
            "rho_msun_pc3": GALACTIC_RHO_MSUN_PC3,
            "Oort_A_km_s_kpc": OORT_A_KMS_KPC,
            "Oort_B_km_s_kpc": OORT_B_KMS_KPC,
            "ecliptic_to_galactic_basis": {
                "NGP_lon_deg": float(np.degrees(_ECL_NGP_LON)),
                "NGP_lat_deg": float(np.degrees(_ECL_NGP_LAT)),
                "GC_lon_deg": float(np.degrees(_ECL_GC_LON)),
                "GC_lat_deg": float(np.degrees(_ECL_GC_LAT)),
            },
            "note": (
                "Tidal tensor applied in galactic coordinates and rotated to ecliptic J2000. "
                "sim.energy() excludes the external tidal potential; validation drift "
                "measures N-body energy only."
            ),
        },
        "etno_catalog": ETNO_CATALOG,
        "clone_initial_conditions": {
            name: field.tolist() for name, field in clone_fields.items()
        },
        "validation": {
            "enabled": VALIDATE_BEFORE_RUN,
            "t_test_yr": VALIDATION_T_YR,
            "max_rel_energy_drift": MAX_ENERGY_DRIFT,
            "passed": validation_result[0],
            "measured_rel_energy_drift": validation_result[1],
        },
    }


# Backward-compatible alias used in earlier revisions.
get_barycentric_elements = get_jacobi_elements


if __name__ == "__main__":
    print("=" * 70)
    print("  PLANET NINE COMPARATIVE STABILITY PROTOCOL v5.1")
    print("=" * 70)

    print("\n[1/5] Probing solar-system ephemeris configuration...")
    _, build_info = build_base_solar_system(p9_dict=P9_BASELINE)
    print(f"      Ephemeris source: {build_info['ephemeris_source']}")

    validation_result = (True, 0.0)
    if VALIDATE_BEFORE_RUN:
        print(f"\n[2/5] Pre-flight integrator validation ({VALIDATION_T_YR:.0e} yr)...")
        validation_result = validate_integrator()
        status = "PASS" if validation_result[0] else "FAIL"
        print(f"      Energy drift: {validation_result[1]:.2e}  [{status}]")
        if not validation_result[0]:
            raise RuntimeError(
                "Integrator validation failed. Do not proceed to production run. "
                "Check dt, integrator settings, or initial conditions."
            )
    else:
        print("\n[2/5] Pre-flight validation skipped (VALIDATE_BEFORE_RUN=False).")

    print(f"\n[3/5] Generating paired clone fields (seed={RANDOM_SEED})...")
    clone_fields = generate_all_clone_fields()

    print("\n[4/5] Running comparative integrations...")
    results_with_p9 = run_stability_suite(p9_presence=True, clone_fields=clone_fields)
    results_without_p9 = run_stability_suite(p9_presence=False, clone_fields=clone_fields)

    print("\n[5/5] Writing checkpoint...")
    metadata = build_run_metadata(build_info, clone_fields, validation_result)
    output_checkpoint = {
        "metadata": metadata,
        "with_p9": results_with_p9,
        "without_p9": results_without_p9,
    }

    save_path = os.path.join(SAVE_DIR, "v5_stability_metrics.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(output_checkpoint, f)

    print(f"\n[OK] Checkpoint saved: {save_path}")
    print("     Run Results_P9_V5.py for Delta-varpi, pole drift, and publication figures.")
