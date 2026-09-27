"""
Figures for the resolution study, built only from the committed summaries
written by `convergence_study.py analyze` / `compare`.

    python convergence_study.py figures --tag sin-orig --compare sin-orig_vs_sin-dealias
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE_ORDINAL = ['#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b', '#081f40']
CAT = ['#2a78d6', '#eb6834', '#1baf7a']            # categorical slots 1-3
SHOW_A = (0.5, 0.75, 0.9375)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 1.6, 'figure.dpi': 150, 'savefig.bbox': 'tight',
    'text.color': INK, 'legend.frameon': False,
})


def n_colors(Ns):
    return {N: BLUE_ORDINAL[i + max(0, len(BLUE_ORDINAL) - len(Ns) - 1)] for i, N in enumerate(Ns)}


def a_colors(a_values):
    lo, hi = np.array([0xf5, 0xa3, 0x7f]) / 255, np.array([0x7a, 0x2a, 0x08]) / 255
    return {a: tuple(lo + (hi - lo) * i / max(1, len(a_values) - 1)) for i, a in enumerate(a_values)}


def load(root, name, fn):
    with open(os.path.join(root, 'summary', name, fn)) as f:
        return json.load(f)


def akey(a):
    return str(float(a))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + '.png'))
    fig.savefig(os.path.join(out, name + '.pdf'), metadata={'CreationDate': None})   # byte-stable for a given Matplotlib version
    plt.close(fig)
    print('  ', name)


# ---------------------------------------------------------------------------

def fig_hm1_histories(s, ts, out):
    Ns, col = s['resolutions'], n_colors(s['resolutions'])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, a in zip(axes, SHOW_A):
        for N in Ns:
            r = ts['runs'][f'{N}/{akey(a)}']
            t, h = np.array(r['t']), np.array(r['norm_hm1'])
            ax.plot(t, np.log(h), color=col[N], label=f'N={N}')
            ax.plot(t[-1], np.log(h[-1]), 'o', ms=4, color=col[N])
        ax.set_title(f'a = {a:g}')
        ax.set_xlabel('t')
    axes[0].set_ylabel(r'$\log(\|\theta\|_{H^{-1}}/\|\theta_0\|_{H^{-1}})$')
    axes[-1].legend(loc='lower left')
    fig.suptitle(r'$H^{-1}$ mix-norm histories; dots mark the $L^p$ stopping time $t_{\rm stop}(N)$',
                 x=0.02, y=1.04, ha='left', fontsize=10)
    save(fig, out, 'hm1_histories')


def fig_cross_error(s, ts, out):
    Ns = s['resolutions']
    pairs = [f'{a}-{b}' for a, b in zip(Ns[:-1], Ns[1:])]
    col = {p: c for p, c in zip(pairs, n_colors(Ns[:-1]).values())}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex='col', sharey='row')
    for j, a in enumerate(SHOW_A):
        for p in pairs:
            d = ts['pairs'][f'{p}/{akey(a)}']
            t = np.array(d['t'])
            axes[0, j].semilogy(t, d['E_field'], color=col[p], label=f'N={p.replace("-", " vs ")}')
            axes[0, j].semilogy(t, d['lower_bound'], color=col[p], ls=':', lw=1.2)
            axes[1, j].semilogy(t, np.maximum(d['E_hm1'], 1e-12), color=col[p])
        axes[0, j].set_title(f'a = {a:g}')
        axes[1, j].set_xlabel('t')
    axes[0, 0].set_ylabel(r'$E_N=\|I\theta_N-\theta_{2N}\|_{L^2}/\|\theta_{2N}\|_{L^2}$')
    axes[1, 0].set_ylabel(r'$|\,\|\theta_N\|_{H^{-1}}-\|\theta_{2N}\|_{H^{-1}}| / \|\theta_{2N}\|_{H^{-1}}$')
    h = [Line2D([], [], color=col[p], label=f'N = {p.replace("-", " vs ")}') for p in pairs]
    h.append(Line2D([], [], color=INK2, ls=':', label='lower bound: part of $\\theta_{2N}$ outside the N band'))
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Successive-resolution differences; each curve ends at $t_{\\rm stop}$ of the coarser grid',
                 x=0.02, y=1.0, ha='left', fontsize=10)
    save(fig, out, 'cross_resolution_error')


def fig_spectra(s, ts, out):
    Ns, col = s['resolutions'], n_colors(s['resolutions'])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    a = 0.5
    # (1) all N at a common time
    t_common = 1.0
    for N in Ns:
        sp = ts['spectra'][f'{N}/{akey(a)}']
        tt = np.array(sp['t'])
        i = int(np.argmin(abs(tt - t_common)))
        if abs(tt[i] - t_common) > 1e-6:
            continue
        E = np.array(sp['E'][i])
        k = np.arange(len(E))
        axes[0].loglog(k[1:], E[1:], color=col[N], label=f'N={N}')
    axes[0].set_title(f'a = {a:g}, t = {t_common:g} (runs alive at t)')
    # (2) finest N at several times
    Nf = Ns[-1]
    sp = ts['spectra'][f'{Nf}/{akey(a)}']
    ac = a_colors(list(range(len(sp['t']))))
    for i, tt in enumerate(sp['t']):
        E = np.array(sp['E'][i])
        axes[1].loglog(np.arange(1, len(E)), E[1:], color=ac[i], label=f't = {tt:.2f}')
    axes[1].set_title(f'a = {a:g}, N = {Nf}: evolution')
    # (3) every run at its own stopping time, k scaled by N/2
    for N in Ns:
        sp = ts['spectra'][f'{N}/{akey(a)}']
        E = np.array(sp['E'][-1])
        k = np.arange(len(E))
        axes[2].loglog(k[1:] / (N / 2), E[1:], color=col[N], label=f'N={N}, t={sp["t"][-1]:.2f}')
    axes[2].set_title(f'a = {a:g}: spectrum at $t_{{\\rm stop}}(N)$')
    axes[2].set_xlabel(r'$|k| / (N/2)$')
    for ax in axes[:2]:
        ax.set_xlabel(r'shell $|k|$')
    axes[0].set_ylabel(r'$E(|k|)=\sum_{\mathrm{round}|k|}|\hat\theta_k|^2/N^4$')
    for ax in axes:
        ax.legend(fontsize=6.5, loc='lower left')
    save(fig, out, 'scalar_spectra')

    # spectral crowding vs the L^p criterion
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5), sharex=True)
    for N in Ns:
        r = ts['runs'][f'{N}/{akey(a)}']
        t = np.array(r['t'])
        drift = np.max(np.array([r['drift_l2'], r['drift_l4'], r['drift_l8']]), axis=0)
        axes[0].semilogy(t, np.maximum(drift, 1e-12), color=col[N], label=f'N={N}')
        axes[1].semilogy(t, np.maximum(r['frac_N4'], 1e-30), color=col[N])
    axes[0].axhline(1e-3, color=INK2, lw=1, ls='--')
    axes[0].text(0.02, 1.25e-3, 'stopping tolerance $10^{-3}$', color=INK2, fontsize=7.5,
                 transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel(r'max$_p$ $|\|\theta\|_{L^p}/\|\theta_0\|_{L^p}-1|$')
    axes[1].set_ylabel('variance fraction with\n' + r'max$(|k_x|,|k_y|) > N/4$')
    axes[1].set_xlabel('t')
    axes[0].legend(loc='lower right', ncol=2)
    axes[0].set_title(f'a = {a:g}: $L^p$ drift and high-wavenumber content')
    save(fig, out, 'spectral_crowding_vs_lp')


def fig_rates(s, out):
    Ns, col = s['resolutions'], n_colors(s['resolutions'])
    a = np.array(s['a_values'])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for N in Ns:
        r = [x['fits']['original_last2of3']['r'] for x in s['runs'][str(N)]]
        p = s['alpha'][str(N)]['original_last2of3']
        axes[0].loglog(a, r, 'o', ms=4, color=col[N])
        axes[0].loglog(a, p['C'] * a**(-p['alpha']), color=col[N], lw=1.1,
                       label=f"N={N}: $\\alpha$={p['alpha']:.2f}")
        m = s['matched_windows'][str(Ns[0])]['by_N'][str(N)]
        if m:
            pm = m['power_law']
            axes[1].loglog(a, m['r'], 'o', ms=4, color=col[N])
            axes[1].loglog(a, pm['C'] * a**(-pm['alpha']), color=col[N], lw=1.1,
                           label=f"N={N}: $\\alpha$={pm['alpha']:.3f}")
    axes[0].set_title('own fit window (last 2/3 of $[0,t_{\\rm stop}(N)]$)')
    axes[1].set_title(f'common fit window (the N={Ns[0]} windows) at every N')
    for ax in axes:
        ax.set_xlabel('a')
        ax.legend(fontsize=7)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        ax.set_xticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'])
        ax.set_yticks([0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(['0.1', '0.15', '0.2', '0.3', '0.4', '0.5'])
        ax.minorticks_off()
    axes[0].set_ylabel(r'decay rate $r$  ($\|\theta\|_{H^{-1}}\sim e^{-rt}$)')
    save(fig, out, 'rate_vs_a')


def fig_alpha(s, out):
    Ns = s['resolutions']
    col = n_colors(Ns)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    al = [s['alpha'][str(N)]['original_last2of3'] for N in Ns]
    y = [p['alpha'] for p in al]
    err = np.array([[p['alpha'] - p['ci95'][0] for p in al], [p['ci95'][1] - p['alpha'] for p in al]])
    ax.errorbar(Ns, y, yerr=err, fmt='o-', color=INK, ms=5, capsize=3, lw=1.6, zorder=5,
                label='own window at each N (thesis protocol); bars: OLS 95% CI')
    for Nw in Ns[:-1]:
        m = s['matched_windows'][str(Nw)]
        xs = [N for N in Ns if m['by_N'].get(str(N))]
        ys = [m['by_N'][str(N)]['power_law']['alpha'] for N in xs]
        ax.plot(xs, ys, 's--', color=col[Nw], ms=3.5, lw=1.1, label=f'fit windows of the N={Nw} runs')
    ax.axhline(1.0, color=INK2, lw=1, ls=':')
    ax.text(Ns[-1], 1.02, r'$\alpha=1$: $a^{-1}$ scaling of the IKX lower bound (benchmark, not a prediction for LTD)',
            fontsize=7, color=INK2, ha='right')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.minorticks_off()
    ax.set_xlabel('resolution N (evaluation grid)')
    ax.set_ylabel(r'fitted exponent $\alpha$ in $r\propto a^{-\alpha}$')
    ax.set_ylim(0.9, None)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    save(fig, out, 'alpha_vs_N')


def fig_local_rate(s, ts, out):
    Ns = s['resolutions']
    Nf = Ns[-1]
    a_vals = s['a_values']
    ac, col = a_colors(a_vals), n_colors(Ns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for a in a_vals:
        r = ts['runs'][f'{Nf}/{akey(a)}']
        axes[0].plot(r['t'], r['r_loc'], color=ac[a], label=f'a={a:g}')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'$r_{\rm loc}(t)=-\,d\log\|\theta\|_{H^{-1}}/dt$')
    axes[0].set_title(f'N = {Nf}: instantaneous decay rate')
    axes[0].legend(fontsize=6.5, ncol=2)
    ft = s['fixed_time_alpha']
    n_all = len(a_vals)
    for N in Ns:
        rows = ft['by_N'][str(N)]
        t = np.array([r['t'] for r in rows])
        al = np.array([r['alpha'] for r in rows])
        full = np.array([r['n_a'] == n_all for r in rows])
        axes[1].plot(t, np.where(full, al, np.nan), color=col[N], label=f'N={N}')
        axes[1].plot(t, al, color=col[N], alpha=0.35, lw=1.1)
    axes[1].axhline(1.0, color=INK2, lw=1, ls=':')
    axes[1].text(0.02, 1.03, r'$\alpha=1$: IKX lower-bound benchmark', color=INK2, fontsize=7,
                 transform=axes[1].get_yaxis_transform())
    axes[1].set_xlabel('t')
    axes[1].set_ylabel(r'$\alpha(t)$ in $r(t;a)\propto a^{-\alpha(t)}$')
    axes[1].set_title(f'fixed-time exponent (local fits on $t\\pm${ft["half_width"]:g})')
    axes[1].text(0.98, 0.97, 'faded: not all a values\nstill resolved at t', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=7, color=INK2)
    axes[1].legend(fontsize=7, loc='center right')
    save(fig, out, 'local_decay_rate')


def fig_lp(s, ts, out):
    Ns = s['resolutions']
    a = 0.5
    fig, axes = plt.subplots(1, len(Ns), figsize=(2.3 * len(Ns), 2.9), sharey=True)
    for ax, N in zip(axes, Ns):
        r = ts['runs'][f'{N}/{akey(a)}']
        t = np.array(r['t'])
        for c, p in zip(CAT, ('drift_l2', 'drift_l4', 'drift_l8')):
            ax.semilogy(t, np.maximum(r[p], 1e-14), color=c, lw=1.3)
        ax.axhline(1e-3, color=INK2, lw=0.9, ls='--')
        ax.set_title(f'N={N}')
        ax.set_xlabel('t')
    axes[0].set_ylabel(r'$|\|\theta\|_{L^p}/\|\theta_0\|_{L^p}-1|$')
    axes[-1].legend(handles=[Line2D([], [], color=c, label=l) for c, l in zip(CAT, ('$L^2$', '$L^4$', '$L^8$'))],
                    loc='lower right')
    fig.suptitle(f'Conserved-norm drift, a = {a:g} (dashed: stopping tolerance)', x=0.02, y=1.04, ha='left', fontsize=10)
    save(fig, out, 'lp_drift')


def fig_compare(c, ts, out):
    Ns = sorted(int(n) for n in c['by_N'])
    col = n_colors(Ns)
    A, B = c['tags']
    ref_tag, ref_N = c['reference'].split(':')
    a = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for N in Ns:
        d = ts.get(f'{N}/{akey(a)}')
        if not d:
            continue
        t = np.array(d['t'])
        axes[0].semilogy(t, d['E_A_vs_ref'], color=col[N])
        if not (B == ref_tag and str(N) == ref_N):          # the reference itself has zero error
            axes[0].semilogy(t, d['E_B_vs_ref'], color=col[N], ls='--')
    axes[0].set_title(f'a = {a:g}: difference from the {ref_tag} N={ref_N} solution')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel(r'relative $L^2$ difference')
    h = [Line2D([], [], color=col[N], label=f'N={N}') for N in Ns]
    h += [Line2D([], [], color=INK2, label=A), Line2D([], [], color=INK2, ls='--', label=B)]
    axes[0].legend(handles=h, fontsize=7)

    # exponent: own windows (differ between schemes because their stopping times differ)
    # versus common windows (the same time windows for both schemes)
    own = np.array([c['by_N'][str(N)]['alpha'] for N in Ns])
    axes[1].plot(Ns, own[:, 0], 'o-', color=CAT[0], label=f'{A}, own windows')
    axes[1].plot(Ns, own[:, 1], 's-', color=CAT[1], label=f'{B}, own windows')
    Nc, com = [], []
    for N in Ns:
        rows = c['by_N'][str(N)]['rows']
        if all(r['common_window'] for r in rows):
            aa = np.log([r['a'] for r in rows])
            com.append([-np.polyfit(aa, np.log([r['r_common_window'][k] for r in rows]), 1)[0]
                        for k in (0, 1)])
            Nc.append(N)
    com = np.array(com)
    axes[1].plot(Nc, com[:, 0], 'o:', color=CAT[0], mfc='none', label=f'{A}, common windows')
    axes[1].plot(Nc, com[:, 1], 's:', color=CAT[1], mfc='none', ms=8, label=f'{B}, common windows')
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(Ns)
    axes[1].set_xticklabels([str(N) for N in Ns])
    axes[1].minorticks_off()
    axes[1].set_xlabel('N')
    axes[1].set_ylabel(r'fitted exponent $\alpha$')
    axes[1].set_title('own windows: differ because the stopping times differ\n'
                      'common windows: overlap of both schemes\' windows at each N', fontsize=8.5)
    axes[1].legend(fontsize=7)
    save(fig, out, 'dealias_comparison')


def fig_families(summaries, out):
    """α(t) at the finest resolution of several initial-data families."""
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for c, s in zip(CAT, summaries):
        Nf = s['resolutions'][-1]
        rows = s['fixed_time_alpha']['by_N'][str(Nf)]
        n_all = len(s['a_values'])
        t = np.array([r['t'] for r in rows])
        al = np.array([r['alpha'] for r in rows])
        full = np.array([r['n_a'] == n_all for r in rows])
        ax.plot(t, np.where(full, al, np.nan), color=c, label=f"{s['config']['ic']} data, N={Nf}")
        ax.plot(t, al, color=c, alpha=0.35, lw=1.1)
    ax.axhline(1.0, color=INK2, lw=1, ls=':')
    ax.text(0.02, 1.03, r'$\alpha=1$: IKX lower-bound benchmark', color=INK2, fontsize=7,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\alpha(t)$ in $r(t;a)\propto a^{-\alpha(t)}$')
    ax.set_title('fixed-time exponent for two initial-data families (faded: not all a resolved)')
    ax.legend()
    save(fig, out, 'alpha_t_families')


def make_all(args):
    root = args.root
    s = load(root, args.tag, 'summary.json')
    ts = load(root, args.tag, 'timeseries.json')
    out = args.out
    print('figures ->', out)
    fig_hm1_histories(s, ts, out)
    if s['pairs']:
        fig_cross_error(s, ts, out)
    fig_spectra(s, ts, out)
    fig_rates(s, out)
    fig_alpha(s, out)
    fig_local_rate(s, ts, out)
    fig_lp(s, ts, out)
    if getattr(args, 'families', None):
        fig_families([s] + [load(root, t, 'summary.json') for t in args.families], out)
    if args.compare:
        fig_compare(load(root, args.compare, 'compare.json'),
                    load(root, args.compare, 'timeseries.json'), out)
