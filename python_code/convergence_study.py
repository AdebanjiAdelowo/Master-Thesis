"""
Resolution study for the LTD optimal-mixing experiment.

Stages (run from python_code/):

  python convergence_study.py run      --N 32 64 128 256 512 --workers 6
  python convergence_study.py analyze  --tag sin-orig
  python convergence_study.py compare  --tags sin-orig sin-dealias
  python convergence_study.py figures  --tag sin-orig --compare sin-orig_vs_sin-dealias

`run` writes raw data (large, not committed) to <root>/raw/<tag>/.
`analyze` and `compare` write compact JSON summaries to <root>/summary/, and
`figures` reads only those summaries, so every figure can be regenerated
from committed data.  The defaults reproduce the thesis configuration:
sine-bump data, a = 0.5 : 1/16 : 15/16, F = 1, t = 0 : 0.05 : 10,
L^p tolerance 1e-3, RK45 rtol 1e-6 / atol 1e-8, no dealiasing.
"""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mixing  # noqa: E402
import convergence as cv  # noqa: E402

DEFAULT_ROOT = os.path.join(os.path.dirname(HERE), 'results')
IDATA = {'sin': mixing.idata_sin, 'diag': mixing.idata_diag,
         'strip': mixing.idata_strip, 'trigpoly': mixing.idata_trigpoly}
THESIS_A = np.arange(0.5, 15 / 16 + 1e-9, 1 / 16)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_tag(args):
    tag = f"{args.ic}-{'dealias' if args.dealias else 'orig'}"
    if args.kappa:
        tag += f'-kappa{args.kappa:g}'
    if (args.rtol, args.atol) != (1e-6, 1e-8):
        tag += f'-rtol{args.rtol:g}'
    if args.tol != 1e-3:
        tag += f'-tol{args.tol:g}'
    return tag


def run_dir(root, tag, N):
    return os.path.join(root, 'raw', tag, f'N{N}')


def case_stem(a):
    return f'a{a:.4f}'


def t_grid(t_end, dt):
    # np.arange(0, 10.05, 0.05) in the thesis; built the same way here.
    return np.arange(0.0, t_end + dt, dt)


def software_versions():
    import scipy
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(),
            'machine': platform.machine()}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_case(cfg):
    """Integrate one (N, a) case and write raw diagnostics to disk."""
    N, a = cfg['N'], cfg['a']
    ops = mixing.build_operators(N)
    t_eval = t_grid(cfg['t_end'], cfg['dt_out'])
    idata_fn = IDATA[cfg['ic']]

    t0 = time.perf_counter()
    sol, theta0, l4i, l8i = mixing.integrate(
        a, idata_fn, ops, F=cfg['F'], t_eval=t_eval, tol=cfg['tol'],
        rtol=cfg['rtol'], atol=cfg['atol'], dealias=cfg['dealias'],
        stop_on_res_loss=cfg['stop'], kappa=cfg.get('kappa', 0.0))
    runtime = time.perf_counter() - t0

    n = N * N
    T = len(sol.t)
    dx = ops['dx']
    K = int(np.rint(np.sqrt(2) * (N // 2))) + 1
    out = {k: np.zeros(T) for k in
           ('norm_l2', 'l4_raw', 'l8_raw', 'hm1_raw', 'frac_N4', 'frac_N3')}
    spectrum = np.zeros((T, K))
    theta_store = np.zeros((T, N, N), dtype=np.float32)
    for i in range(T):
        th_hat = (sol.y[:n, i] + 1j * sol.y[n:, i]).reshape(N, N)
        th = np.real(np.fft.ifft2(th_hat))
        out['norm_l2'][i] = np.linalg.norm(th_hat.ravel()) / N**2
        out['l4_raw'][i] = np.linalg.norm(th.ravel(), 4) * np.sqrt(dx)
        out['l8_raw'][i] = np.linalg.norm(th.ravel(), 8) * dx**0.25
        out['hm1_raw'][i] = np.linalg.norm((ops['LAMBDA_INV'] * th_hat).ravel()) / N**2
        out['frac_N4'][i] = cv.band_fraction(th_hat, N // 4)
        out['frac_N3'][i] = cv.band_fraction(th_hat, N // 3)
        s = cv.shell_spectrum(th_hat)
        spectrum[i, :len(s)] = s
        theta_store[i] = th
    del sol.y

    # Which conserved norm triggered the stop (evaluated at the event state).
    trigger = None
    if sol.status == 1 and len(sol.y_events[0]):
        ye = sol.y_events[0][0]
        th = np.real(np.fft.ifft2((ye[:n] + 1j * ye[n:]).reshape(N, N)))
        d = {'L2': abs(np.linalg.norm(th.ravel()) * dx - 1),
             'L4': abs(np.linalg.norm(th.ravel(), 4) * np.sqrt(dx) / l4i - 1),
             'L8': abs(np.linalg.norm(th.ravel(), 8) * dx**0.25 / l8i - 1)}
        trigger = max(d, key=d.get)

    meta = dict(cfg)
    meta.update({
        'runtime_s': runtime, 'nfev': int(sol.nfev), 'status': int(sol.status),
        'stopped_by_event': bool(sol.status == 1),
        't_event': float(sol.t_events[0][0]) if sol.status == 1 else None,
        'trigger': trigger, 'n_frames': T, 't_last': float(sol.t[-1]),
        'l4_init': float(l4i), 'l8_init': float(l8i),
        'versions': software_versions(),
    })

    d = run_dir(cfg['root'], cfg['tag'], N)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, case_stem(a))
    np.savez_compressed(
        stem + '.npz', t=sol.t,
        norm_l2=out['norm_l2'],
        norm_l4=out['l4_raw'] / out['l4_raw'][0],
        norm_l8=out['l8_raw'] / out['l8_raw'][0],
        norm_hm1=out['hm1_raw'] / out['hm1_raw'][0],
        hm1_raw=out['hm1_raw'], frac_N4=out['frac_N4'], frac_N3=out['frac_N3'],
        spectrum=spectrum)
    if cfg['save_fields']:
        np.save(stem + '_theta.npy', theta_store)
    with open(stem + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    return N, a, runtime, meta['t_last'], trigger


def cmd_run(args):
    if args.kappa and not args.no_stop:
        raise SystemExit('--kappa > 0 needs --no-stop: L^p norms are not conserved with diffusion, '
                         'so the L^p resolution check is not a valid stopping rule')
    tag = args.tag or default_tag(args)
    a_values = np.array(args.a) if args.a else THESIS_A
    cases = []
    for N in sorted(args.N, reverse=True):          # most expensive first
        for a in a_values:
            stem = os.path.join(run_dir(args.root, tag, N), case_stem(a))
            if args.skip_existing and os.path.exists(stem + '.json'):
                continue
            cases.append(dict(
                N=int(N), a=float(a), ic=args.ic, F=args.F, t_end=args.t_end,
                dt_out=args.dt_out, tol=args.tol, rtol=args.rtol, atol=args.atol,
                dealias=args.dealias, kappa=args.kappa, stop=not args.no_stop,
                save_fields=not args.no_fields, root=args.root, tag=tag))
    print(f'tag={tag}: {len(cases)} case(s), {args.workers} worker(s)', flush=True)
    if args.workers <= 1:
        for c in cases:
            print('  N=%d a=%.4f  %.1fs  t_last=%.2f  trigger=%s' % run_case(c), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_case, c) for c in cases]
        for f in as_completed(futs):
            print('  N=%d a=%.4f  %.1fs  t_last=%.2f  trigger=%s' % f.result(), flush=True)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def load_tag(root, tag):
    """runs[N][a] = dict of arrays + 'meta'.  Fields are memory-mapped lazily."""
    base = os.path.join(root, 'raw', tag)
    runs = {}
    for d in sorted(os.listdir(base)):
        if not d.startswith('N'):
            continue
        N = int(d[1:])
        for fn in sorted(os.listdir(os.path.join(base, d))):
            if not fn.endswith('.json'):
                continue
            stem = os.path.join(base, d, fn[:-5])
            with open(stem + '.json') as f:
                meta = json.load(f)
            arc = np.load(stem + '.npz')
            r = {k: arc[k] for k in arc.files}
            r['meta'] = meta
            r['theta_path'] = stem + '_theta.npy'
            runs.setdefault(N, {})[round(meta['a'], 6)] = r
    return runs


def theta_frames(run):
    return np.load(run['theta_path'], mmap_mode='r')


def _check_comparable(runs):
    keys = ('ic', 'F', 't_end', 'dt_out', 'tol', 'rtol', 'atol', 'dealias', 'stop', 'kappa')
    ref = None
    for N in runs:
        for a in runs[N]:
            m = {k: runs[N][a]['meta'].get(k, 0.0 if k == 'kappa' else None) for k in keys}
            if ref is None:
                ref = m
            elif m != ref:
                raise ValueError(f'runs in one tag differ in configuration: {m} vs {ref}')
    return ref


def _power_law(a_list, r_list):
    if len(a_list) < 3 or any(r <= 0 for r in r_list):
        return None
    return cv.fit_power_law(a_list, r_list)


def rnd(x, sig=7):
    """Round floats (recursively) for compact JSON."""
    if isinstance(x, float):
        return float(f'{x:.{sig}g}') if np.isfinite(x) else None
    if isinstance(x, (np.floating,)):
        return rnd(float(x), sig)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), sig)
    if isinstance(x, dict):
        return {str(k): rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    return x


def cmd_analyze(args):
    runs = load_tag(args.root, args.tag)
    config = _check_comparable(runs)
    Ns = sorted(runs)
    a_all = sorted(set.intersection(*[set(runs[N]) for N in Ns]))
    summary = {'tag': args.tag, 'config': config, 'resolutions': Ns, 'a_values': a_all}
    ts = {'tag': args.tag, 'runs': {}, 'pairs': {}, 'vs_reference': {}, 'spectra': {}}

    # ---- per-run fits and diagnostics -----------------------------------
    per_run = {}
    windows = {}          # windows[N][a] = (t0, t1) of the original fit window
    for N in Ns:
        per_run[N] = {}
        windows[N] = {}
        for a in a_all:
            r = runs[N][a]
            t, h = r['t'], r['norm_hm1']
            sl = cv.original_window(t)
            fits = {
                'original_last2of3': cv.fit_decay(t, h, sl),
                'full_trajectory': cv.fit_decay(t, h, slice(0, len(t))),
                'last_half': cv.fit_decay(t, h, slice(len(t) // 2, len(t))),
            }
            windows[N][a] = (float(t[sl][0]), float(t[-1]))
            m = r['meta']
            per_run[N][a] = {
                'a': a, 't_stop': float(t[-1]), 'stopped_by_event': m['stopped_by_event'],
                't_event': m['t_event'], 'trigger': m['trigger'],
                'runtime_s': m['runtime_s'], 'nfev': m['nfev'],
                'hm1_stop': float(h[-1]),
                'max_drift': {p: float(np.max(np.abs(r[f'norm_{p}'] - 1)))
                              for p in ('l2', 'l4', 'l8')},
                'frac_N4_stop': float(r['frac_N4'][-1]), 'frac_N3_stop': float(r['frac_N3'][-1]),
                'frac_N4_init': float(r['frac_N4'][0]), 'frac_N3_init': float(r['frac_N3'][0]),
                'fits': fits,
            }
            ts['runs'][f'{N}/{a}'] = {
                't': t, 'norm_hm1': h, 'hm1_raw': r['hm1_raw'],
                'drift_l2': np.abs(r['norm_l2'] - 1), 'drift_l4': np.abs(r['norm_l4'] - 1),
                'drift_l8': np.abs(r['norm_l8'] - 1),
                'r_loc': cv.local_rate(t, h), 'frac_N4': r['frac_N4'], 'frac_N3': r['frac_N3'],
            }
            # spectra at common physical times, plus the stopping time
            want = [x for x in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0) if x <= t[-1] + 1e-9]
            idx = sorted(set([int(np.argmin(abs(t - x))) for x in want] + [len(t) - 1]))
            ts['spectra'][f'{N}/{a}'] = {'t': t[idx], 'E': r['spectrum'][idx, :N // 2 + 1]}
    summary['runs'] = {str(N): list(per_run[N].values()) for N in Ns}
    summary['runtime_total_s'] = {str(N): sum(per_run[N][a]['runtime_s'] for a in a_all) for N in Ns}

    # ---- exponent per resolution, several window conventions ------------
    alpha = {}
    for N in Ns:
        alpha[str(N)] = {}
        for w in ('original_last2of3', 'full_trajectory', 'last_half'):
            alpha[str(N)][w] = _power_law(a_all, [per_run[N][a]['fits'][w]['r'] for a in a_all])
    summary['alpha'] = alpha
    orig_seq = [alpha[str(N)]['original_last2of3']['alpha'] for N in Ns]
    summary['alpha_original_sequence'] = {'N': Ns, **cv.richardson_assessment(orig_seq)}

    # ---- fixed-time exponent α(t) ----------------------------------------
    # Local rate r(t; a) = OLS decay rate of log‖θ‖_{H⁻¹} on [t − h, t + h],
    # then α(t) from r(t; a) ∝ a^(−α) across all a whose run covers t + h.
    h = args.half_width
    fixed_time = {}
    for N in Ns:
        rows = []
        for tc in np.arange(h, 10.0, 0.25):
            aa, rr = [], []
            for a in a_all:
                t = runs[N][a]['t']
                if t[-1] >= tc + h - 1e-9:
                    aa.append(a)
                    rr.append(cv.fit_decay(t, runs[N][a]['norm_hm1'],
                                           cv.window_slice(t, tc - h, tc + h))['r'])
            if len(aa) < 4 or min(rr) <= 0:
                continue
            p = cv.fit_power_law(aa, rr)
            rows.append({'t': float(tc), 'n_a': len(aa), 'a_max': max(aa), 'alpha': p['alpha'],
                         'se_alpha': p['se_alpha'], 'rms_resid_log': p['rms_resid_log'],
                         'r': dict(zip([str(a) for a in aa], rr))})
        fixed_time[str(N)] = rows
    # like-for-like resolution check: α(t) at N and 2N on the a-set available at N
    ft_pairs = {}
    for Nc, Nf in zip(Ns[:-1], Ns[1:]):
        rows = []
        for rc in fixed_time[str(Nc)]:
            aa = [float(a) for a in rc['r']]
            rf = [cv.fit_decay(runs[Nf][a]['t'], runs[Nf][a]['norm_hm1'],
                               cv.window_slice(runs[Nf][a]['t'], rc['t'] - h, rc['t'] + h))['r']
                  for a in aa]
            rows.append({'t': rc['t'], 'n_a': rc['n_a'], 'alpha_coarse': rc['alpha'],
                         'alpha_fine_same_a': cv.fit_power_law(aa, rf)['alpha']})
        ft_pairs[f'{Nc}-{Nf}'] = rows
    summary['fixed_time_alpha'] = {'half_width': h, 'by_N': fixed_time, 'pairs': ft_pairs}

    # ---- matched windows: same time window, different N ------------------
    matched = {}
    for Nw in Ns:
        row = {}
        for N in [n for n in Ns if n >= Nw]:
            rs, ok = [], True
            fits = {}
            for a in a_all:
                t0, t1 = windows[Nw][a]
                t = runs[N][a]['t']
                if t[-1] < t1 - 1e-9:
                    ok = False
                    break
                f = cv.fit_decay(t, runs[N][a]['norm_hm1'], cv.window_slice(t, t0, t1))
                fits[a] = f
                rs.append(f['r'])
            if ok:
                row[str(N)] = {'power_law': _power_law(a_all, rs),
                               'r': rs, 'fits': list(fits.values())}
            else:
                row[str(N)] = None
        seq = [row[str(N)]['power_law']['alpha'] for N in Ns if N >= Nw and row[str(N)]]
        conv = cv.richardson_assessment(seq)
        matched[str(Nw)] = {'window_from_N': Nw,
                            'windows': {str(a): windows[Nw][a] for a in a_all},
                            'by_N': row, 'alpha_sequence': seq, 'convergence': conv}
    summary['matched_windows'] = matched

    # ---- cross-resolution field / observable errors ----------------------
    have_fields = all(os.path.exists(runs[N][a]['theta_path']) for N in Ns for a in a_all)
    pairs = {}
    if have_fields:
        for Nc, Nf in zip(Ns[:-1], Ns[1:]):
            pairs[f'{Nc}-{Nf}'] = {}
            for a in a_all:
                rc, rf = runs[Nc][a], runs[Nf][a]
                T = min(len(rc['t']), len(rf['t']))
                thc, thf = theta_frames(rc), theta_frames(rf)
                E = np.array([cv.relative_l2_difference(np.asarray(thc[i], float),
                                                        np.asarray(thf[i], float)) for i in range(T)])
                LB = np.array([cv.unrepresentable_fraction(np.asarray(thf[i], float), Nc)
                               for i in range(T)])
                Eh = np.abs(rc['hm1_raw'][:T] - rf['hm1_raw'][:T]) / rf['hm1_raw'][:T]
                t = rc['t'][:T]
                w0, w1 = windows[Nc][a]
                inw = (t >= w0 - 1e-9) & (t <= w1 + 1e-9)
                fc = per_run[Nc][a]['fits']['original_last2of3']
                ff_same = cv.fit_decay(rf['t'], rf['norm_hm1'], cv.window_slice(rf['t'], w0, w1))
                pairs[f'{Nc}-{Nf}'][str(a)] = {
                    'E_field_t0': float(E[0]), 'E_field_at_tstop_coarse': float(E[-1]),
                    'E_field_max_in_window': float(E[inw].max()),
                    'E_hm1_max_in_window': float(Eh[inw].max()),
                    'E_hm1_at_tstop_coarse': float(Eh[-1]),
                    'lower_bound_at_tstop_coarse': float(LB[-1]),
                    'r_coarse': fc['r'], 'r_fine_same_window': ff_same['r'],
                    'r_rel_diff_same_window': abs(fc['r'] - ff_same['r']) / ff_same['r'],
                    't_stop_coarse': float(rc['t'][-1]), 't_stop_fine': float(rf['t'][-1]),
                    't_last_E_below_1e-2': float(t[E <= 1e-2][-1]) if np.any(E <= 1e-2) else None,
                    't_last_E_below_1e-3': float(t[E <= 1e-3][-1]) if np.any(E <= 1e-3) else None,
                }
                ts['pairs'][f'{Nc}-{Nf}/{a}'] = {'t': t, 'E_field': E, 'E_hm1': Eh, 'lower_bound': LB}
        # every resolution against the finest one
        Nref = Ns[-1]
        for N in Ns[:-1]:
            for a in a_all:
                rc, rf = runs[N][a], runs[Nref][a]
                T = min(len(rc['t']), len(rf['t']))
                thc, thf = theta_frames(rc), theta_frames(rf)
                E = np.array([cv.relative_l2_difference(np.asarray(thc[i], float),
                                                        np.asarray(thf[i], float)) for i in range(T)])
                ts['vs_reference'][f'{N}/{a}'] = {'t': rc['t'][:T], 'E_field': E}
    summary['pairs'] = pairs
    summary['fields_available'] = have_fields

    out = os.path.join(args.root, 'summary', args.tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(rnd(summary), f, indent=1)
    with open(os.path.join(out, 'timeseries.json'), 'w') as f:
        json.dump(rnd(ts, 6), f, separators=(',', ':'))
    write_tables(summary, os.path.join(out, 'tables.md'))
    print(f'wrote {out}/summary.json, timeseries.json, tables.md')


def write_tables(s, path):
    Ns = s['resolutions']
    L = [f"# Resolution study: {s['tag']}", '',
         'Generated by `convergence_study.py analyze`. Do not edit by hand.', '',
         '## Configuration', '', '```', json.dumps(s['config'], indent=1), '```', '']

    L += ['## Exponent per resolution (original fit protocol)', '',
          'Fit of log‖θ‖_{H⁻¹} over the last 2/3 of each run, then r = C a^(−α).',
          'The OLS interval is a regression statistic only: it does not include',
          'discretisation error or the choice of fit window.', '',
          '| N | α_N | OLS s.e. | 95%% CI (OLS, n=%d) | rms log resid | runtime, all a (s) |' % len(s['a_values']),
          '|---:|---:|---:|---|---:|---:|']
    for N in Ns:
        p = s['alpha'][str(N)]['original_last2of3']
        L.append(f"| {N} | {p['alpha']:.4f} | {p['se_alpha']:.4f} | "
                 f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | {p['rms_resid_log']:.2e} | "
                 f"{s['runtime_total_s'][str(N)]:.1f} |")
    L += ['', '### Sensitivity to the fit-window convention', '',
          '| N | last 2/3 (thesis) | last half | full trajectory |', '|---:|---:|---:|---:|']
    for N in Ns:
        p = s['alpha'][str(N)]
        L.append(f"| {N} | {p['original_last2of3']['alpha']:.4f} | {p['last_half']['alpha']:.4f} | "
                 f"{p['full_trajectory']['alpha']:.4f} |")

    L += ['', '## Per-run results', '',
          '| N | a | t_stop | trigger | r | s.e.(r) OLS | fit window | n | lag-1 resid. corr. | '
          '‖θ‖_{H⁻¹}(t_stop) | max L² / L⁴ / L⁸ drift | frac > N/4 at t_stop | runtime (s) |',
          '|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---|---:|---:|']
    for N in Ns:
        for r in s['runs'][str(N)]:
            f = r['fits']['original_last2of3']
            d = r['max_drift']
            L.append(f"| {N} | {r['a']:.4f} | {r['t_stop']:.2f}{'' if r['stopped_by_event'] else ' (no stop)'} | "
                     f"{r['trigger']} | {f['r']:.4f} | {f['se_slope']:.1e} | "
                     f"[{f['x0']:.2f}, {f['x1']:.2f}] | {f['n']} | {f['lag1_autocorr']:.2f} | "
                     f"{r['hm1_stop']:.4f} | {d['l2']:.1e} / {d['l4']:.1e} / {d['l8']:.1e} | "
                     f"{r['frac_N4_stop']:.1e} | {r['runtime_s']:.1f} |")

    ft = s['fixed_time_alpha']
    L += ['', '## Fixed-time exponent α(t)', '',
          f"Local rate from an OLS fit on [t − {ft['half_width']}, t + {ft['half_width']}] for every a whose",
          'run is still resolved there; α(t) from r(t; a) ∝ a^(−α). n_a < 8 means the largest-a',
          'runs only are left, so late-time values rest on fewer and larger a.', '',
          '| t | ' + ' | '.join(f'α (N={N}) [n_a]' for N in Ns) + ' |',
          '|---:|' + '---:|' * len(Ns)]
    times = sorted({r['t'] for N in Ns for r in ft['by_N'][str(N)]})
    for tc in times:
        cells = []
        for N in Ns:
            m = [r for r in ft['by_N'][str(N)] if abs(r['t'] - tc) < 1e-9]
            cells.append(f"{m[0]['alpha']:.3f} [{m[0]['n_a']}]" if m else '')
        L.append(f'| {tc:.2f} | ' + ' | '.join(cells) + ' |')

    L += ['', 'Like-for-like check: max |α_N(t) − α_2N(t)| over all t, same a-set (the one available at N):', '']
    for pk, rows in ft['pairs'].items():
        d = [abs(r['alpha_coarse'] - r['alpha_fine_same_a']) for r in rows]
        if d:
            L.append(f"- {pk}: max difference {max(d):.2e} (over {len(d)} times, up to t = {rows[-1]['t']:.2f})")

    L += ['', '## Matched windows', '',
          'Row: fit windows taken from the original-protocol windows of resolution N_w.',
          'Column: resolution actually used to evaluate r on those windows.', '',
          '| window from N_w | ' + ' | '.join(f'N={N}' for N in Ns) + ' | observed orders | Richardson |',
          '|---:|' + '---:|' * len(Ns) + '---|---|']
    for Nw in Ns:
        m = s['matched_windows'][str(Nw)]
        cells = []
        for N in Ns:
            v = m['by_N'].get(str(N))
            cells.append(f"{v['power_law']['alpha']:.4f}" if v else '')
        conv = m['convergence']
        orders = ', '.join(f'{p:.2f}' for p in conv['orders'])
        rich = (f"{conv['extrapolated']:.4f} (±{conv['error_estimate']:.1e})" if conv['justified']
                else 'not justified: ' + conv['reason'])
        L.append(f'| {Nw} | ' + ' | '.join(cells) + f' | {orders} | {rich} |')
    q = s['alpha_original_sequence']
    L += ['', 'Thesis-protocol sequence α_N (own window at each N): differences '
          + ', '.join(f'{d:+.4f}' for d in q['differences']) + '; Richardson '
          + (f"{q['extrapolated']:.4f}" if q['justified'] else 'not justified: ' + q['reason']) + '.']

    if s['pairs']:
        L += ['', '## Cross-resolution errors', '',
              'E = ‖I θ_N − θ_2N‖/‖θ_2N‖ (spectral interpolation, exact via Parseval); '
              'LB = fraction of θ_2N outside the N-grid band (rigorous lower bound on E).', '',
              '| pair | a | E(0) | max E in window | E(t_stop,N) | LB(t_stop,N) | max E_H⁻¹ in window | '
              'r_N | r_2N same window | rel. diff |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for pk, d in s['pairs'].items():
            for a, v in d.items():
                L.append(f"| {pk} | {float(a):.4f} | {v['E_field_t0']:.1e} | {v['E_field_max_in_window']:.1e} | "
                         f"{v['E_field_at_tstop_coarse']:.1e} | {v['lower_bound_at_tstop_coarse']:.1e} | "
                         f"{v['E_hm1_max_in_window']:.1e} | {v['r_coarse']:.4f} | "
                         f"{v['r_fine_same_window']:.4f} | {v['r_rel_diff_same_window']:.1e} |")
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------------------
# compare two tags (e.g. original vs dealiased) at equal N
# ---------------------------------------------------------------------------

def cmd_compare(args):
    A, B = args.tags
    ra, rb = load_tag(args.root, A), load_tag(args.root, B)
    ref_tag, ref_N = (args.reference.split(':') if args.reference else (A, None))
    rr = ra if ref_tag == A else (rb if ref_tag == B else load_tag(args.root, ref_tag))
    ref_N = int(ref_N) if ref_N else max(rr)
    Ns = sorted(set(ra) & set(rb))
    a_all = sorted(set.intersection(*[set(ra[N]) & set(rb[N]) for N in Ns]))
    res = {'tags': [A, B], 'reference': f'{ref_tag}:{ref_N}', 'by_N': {}}
    ts = {}
    for N in Ns:
        rows = []
        ra_r, rb_r = [], []
        for a in a_all:
            x, y, z = ra[N][a], rb[N][a], rr[ref_N][a]
            fx = cv.fit_decay(x['t'], x['norm_hm1'], cv.original_window(x['t']))
            fy = cv.fit_decay(y['t'], y['norm_hm1'], cv.original_window(y['t']))
            # same window for both schemes: the shorter run's original window
            t0, t1 = max(fx['x0'], fy['x0']), min(fx['x1'], fy['x1'])
            ra_r.append(fx['r'])
            rb_r.append(fy['r'])
            row = {'a': a, 't_stop': [float(x['t'][-1]), float(y['t'][-1])],
                   'r_own_window': [fx['r'], fy['r']], 'common_window': None}
            if t1 - t0 >= 2 * x['meta']['dt_out'] - 1e-9:      # at least 3 samples
                fx_c = cv.fit_decay(x['t'], x['norm_hm1'], cv.window_slice(x['t'], t0, t1))
                fy_c = cv.fit_decay(y['t'], y['norm_hm1'], cv.window_slice(y['t'], t0, t1))
                fz_c = cv.fit_decay(z['t'], z['norm_hm1'], cv.window_slice(z['t'], t0, t1))
                row.update({'common_window': [t0, t1], 'r_common_window': [fx_c['r'], fy_c['r']],
                            'r_reference_common_window': fz_c['r']})
            if os.path.exists(x['theta_path']) and os.path.exists(y['theta_path']) and \
                    os.path.exists(z['theta_path']):
                T = min(len(x['t']), len(y['t']), len(z['t']))
                tx, ty, tz = theta_frames(x), theta_frames(y), theta_frames(z)
                E_ab = [cv.relative_l2_difference(np.asarray(tx[i], float), np.asarray(ty[i], float))
                        for i in range(T)]
                E_a = [cv.relative_l2_difference(np.asarray(tx[i], float), np.asarray(tz[i], float))
                       for i in range(T)]
                E_b = [cv.relative_l2_difference(np.asarray(ty[i], float), np.asarray(tz[i], float))
                       for i in range(T)]
                ts[f'{N}/{a}'] = {'t': x['t'][:T], 'E_A_vs_B': E_ab, 'E_A_vs_ref': E_a, 'E_B_vs_ref': E_b}
                row['E_A_vs_ref_at_T'] = E_a[-1]
                row['E_B_vs_ref_at_T'] = E_b[-1]
                row['E_A_vs_B_at_T'] = E_ab[-1]
                row['t_common_end'] = float(x['t'][T - 1])
            rows.append(row)
        res['by_N'][str(N)] = {'rows': rows,
                               'alpha': [_power_law(a_all, ra_r)['alpha'], _power_law(a_all, rb_r)['alpha']]}
    out = os.path.join(args.root, 'summary', f'{A}_vs_{B}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'compare.json'), 'w') as f:
        json.dump(rnd(res), f, indent=1)
    with open(os.path.join(out, 'timeseries.json'), 'w') as f:
        json.dump(rnd(ts, 6), f, separators=(',', ':'))
    L = [f'# {A} vs {B} (reference {res["reference"]})', '',
         'Generated by `convergence_study.py compare`. Do not edit by hand.', '',
         '| N | a | t_stop A / B | r A / B (own windows) | common window | r A / B / ref (common window) | '
         'E(A,ref) / E(B,ref) at end of common span |',
         '|---:|---:|---|---|---|---|---|']
    for N in Ns:
        for r in res['by_N'][str(N)]['rows']:
            e = (f"{r['E_A_vs_ref_at_T']:.1e} / {r['E_B_vs_ref_at_T']:.1e} (t={r['t_common_end']:.2f})"
                 if 'E_A_vs_ref_at_T' in r else '')
            if r['common_window']:
                cw = (f"[{r['common_window'][0]:.2f}, {r['common_window'][1]:.2f}] | "
                      f"{r['r_common_window'][0]:.4f} / {r['r_common_window'][1]:.4f} / "
                      f"{r['r_reference_common_window']:.4f}")
            else:
                cw = 'none (windows do not overlap) | '
            L.append(f"| {N} | {r['a']:.4f} | {r['t_stop'][0]:.2f} / {r['t_stop'][1]:.2f} | "
                     f"{r['r_own_window'][0]:.4f} / {r['r_own_window'][1]:.4f} | {cw} | {e} |")
    L += ['', '| N | α (A, own windows) | α (B, own windows) |', '|---:|---:|---:|']
    for N in Ns:
        al = res['by_N'][str(N)]['alpha']
        L.append(f'| {N} | {al[0]:.4f} | {al[1]:.4f} |')
    with open(os.path.join(out, 'tables.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {out}/compare.json, timeseries.json, tables.md')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=DEFAULT_ROOT, help='output root (default: ../results)')
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='integrate cases and save raw data')
    r.add_argument('--N', type=int, nargs='+', default=[32, 64, 128, 256])
    r.add_argument('--a', type=float, nargs='+', help='a values (default 0.5:1/16:15/16)')
    r.add_argument('--ic', choices=sorted(IDATA), default='sin')
    r.add_argument('--F', type=float, default=1.0)
    r.add_argument('--t-end', type=float, default=10.0)
    r.add_argument('--dt-out', type=float, default=0.05)
    r.add_argument('--tol', type=float, default=1e-3, help='L^p resolution-check tolerance')
    r.add_argument('--rtol', type=float, default=1e-6)
    r.add_argument('--atol', type=float, default=1e-8)
    r.add_argument('--dealias', action='store_true', help='2/3-rule dealiasing')
    r.add_argument('--kappa', type=float, default=0.0,
                   help='diffusivity κ in θ_t + u·∇θ = κΔθ (0 = thesis model; needs --no-stop)')
    r.add_argument('--no-stop', action='store_true', help='disable the L^p stopping event')
    r.add_argument('--no-fields', action='store_true', help='do not store θ snapshots')
    r.add_argument('--tag', help='output tag (default derived from options)')
    r.add_argument('--workers', type=int, default=1)
    r.add_argument('--skip-existing', action='store_true')
    r.set_defaults(func=cmd_run)

    an = sub.add_parser('analyze', help='fits, exponents, cross-resolution errors')
    an.add_argument('--tag', required=True)
    an.add_argument('--half-width', type=float, default=0.25,
                    help='half-width of the local fit window for α(t)')
    an.set_defaults(func=cmd_analyze)

    c = sub.add_parser('compare', help='compare two tags at equal N')
    c.add_argument('--tags', nargs=2, required=True)
    c.add_argument('--reference', help='TAG:N used as reference solution (default: finest N of the first tag)')
    c.set_defaults(func=cmd_compare)

    fg = sub.add_parser('figures', help='make figures from summary JSON')
    fg.add_argument('--tag', required=True)
    fg.add_argument('--compare', help='name of a compare summary directory')
    fg.add_argument('--families', nargs='+', help='other tags whose α(t) is overlaid on the main tag')
    fg.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'images', 'convergence'))
    fg.set_defaults(func=lambda a: __import__('convergence_figures').make_all(a))

    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
