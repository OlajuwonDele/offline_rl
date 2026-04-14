import os
import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12})

EXP_DIR    = os.path.join(os.path.dirname(__file__), '..', 'exp')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Parsing helpers ───────────────────────────────────────────────────────────

ENV_KEYS = ['cube-single', 'antsoccer-arena', 'antmaze-medium']

def get_env(folder):
    for key in ENV_KEYS:
        if key in folder:
            return key
    return 'unknown'

def get_alpha(folder):
    """Return the last _a<value> in the folder name (works for SAC+BC, IQL, FQL)."""
    matches = re.findall(r'_a([\d.]+)', folder)
    return float(matches[-1]) if matches else None

def load_csv(folder_path, filename):
    path = os.path.join(folder_path, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found")
    return None

def scan_dir(q_dir):
    """Return list of (folder_name, folder_path, env, alpha) for all run dirs."""
    if not os.path.exists(q_dir):
        print(f"  WARNING: {q_dir} does not exist")
        return []
    entries = []
    for folder in sorted(os.listdir(q_dir)):
        path = os.path.join(q_dir, folder)
        if os.path.isdir(path):
            entries.append((folder, path, get_env(folder), get_alpha(folder)))
    return entries

def best_run(runs):
    """Return the run with the highest peak eval success rate."""
    def peak(run):
        df = load_csv(run[1], 'eval.csv')
        if df is None or 'eval/success_rate' not in df.columns:
            return -1.0
        return df['eval/success_rate'].max()
    return max(runs, key=peak)

# ── Plot helpers ──────────────────────────────────────────────────────────────

def finalize(ax, xlabel, ylabel, title, output_path):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved → {output_path}")
    plt.close()

def plot_success_sweep(runs, title, output_path):
    """Success rate vs steps for a list of runs, one line per alpha."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for _, path, _, alpha in sorted(runs, key=lambda x: x[3] or 0):
        df = load_csv(path, 'eval.csv')
        if df is None or 'eval/success_rate' not in df.columns:
            continue
        label = f'α={alpha:.0f}' if alpha is not None else path
        ax.plot(df['step'], df['eval/success_rate'], label=label, linewidth=2)
        plotted = True
    if not plotted:
        plt.close()
        return
    ax.set_ylim(bottom=0)
    finalize(ax, 'Environment Steps', 'Success Rate', title, output_path)

def plot_train_metric(runs, col, ylabel, title, output_path):
    """Plot a train.csv metric vs steps for a list of runs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for _, path, _, alpha in sorted(runs, key=lambda x: x[3] or 0):
        df = load_csv(path, 'train.csv')
        if df is None or col not in df.columns:
            continue
        label = f'α={alpha:.0f}' if alpha is not None else path
        ax.plot(df['step'], df[col], label=label, linewidth=1.5, alpha=0.85)
        plotted = True
    if not plotted:
        plt.close()
        return
    finalize(ax, 'Training Steps', ylabel, title, output_path)

# ── Q1: SAC+BC ────────────────────────────────────────────────────────────────

def plot_q1():
    print("\n=== Q1: SAC+BC ===")
    runs = scan_dir(os.path.join(EXP_DIR, 'q1'))

    cube_runs     = [(f, p, e, a) for f, p, e, a in runs if e == 'cube-single']
    antsoccer_runs = [(f, p, e, a) for f, p, e, a in runs if e == 'antsoccer-arena']
    antmaze_runs  = [(f, p, e, a) for f, p, e, a in runs if e == 'antmaze-medium']

    # ── cube-single: alpha sweep (required by assignment) ──
    if cube_runs:
        plot_success_sweep(
            cube_runs,
            'SAC+BC — cube-single, α sweep',
            os.path.join(ASSETS_DIR, 'q1_cube_alpha_sweep.png'),
        )
        # MSE vs alpha (useful sanity check mentioned in assignment)
        plot_train_metric(
            cube_runs,
            'actor/mse',
            'MSE (policy vs dataset actions)',
            'SAC+BC — cube-single, policy MSE by α',
            os.path.join(ASSETS_DIR, 'q1_cube_mse.png'),
        )
        # Q-value bounds
        plot_train_metric(
            cube_runs,
            'critic/q_max',
            'Q-max',
            'SAC+BC — cube-single, Q-max by α',
            os.path.join(ASSETS_DIR, 'q1_cube_qmax.png'),
        )
        plot_train_metric(
            cube_runs,
            'critic/q_min',
            'Q-min',
            'SAC+BC — cube-single, Q-min by α',
            os.path.join(ASSETS_DIR, 'q1_cube_qmin.png'),
        )

    # ── antsoccer-arena: alpha sweep ──
    if antsoccer_runs:
        plot_success_sweep(
            antsoccer_runs,
            'SAC+BC — antsoccer-arena, α sweep',
            os.path.join(ASSETS_DIR, 'q1_antsoccer_alpha_sweep.png'),
        )

    # ── antmaze-medium (debugging task, optional) ──
    if antmaze_runs:
        plot_success_sweep(
            antmaze_runs,
            'SAC+BC — antmaze-medium, α sweep',
            os.path.join(ASSETS_DIR, 'q1_antmaze_alpha_sweep.png'),
        )

# ── Q2: IQL ───────────────────────────────────────────────────────────────────

def plot_q2():
    print("\n=== Q2: IQL ===")
    runs = scan_dir(os.path.join(EXP_DIR, 'q2'))

    cube_runs      = [(f, p, e, a) for f, p, e, a in runs if e == 'cube-single']
    antsoccer_runs = [(f, p, e, a) for f, p, e, a in runs if e == 'antsoccer-arena']

    # ── cube-single: alpha sweep (required) ──
    if cube_runs:
        plot_success_sweep(
            cube_runs,
            'IQL — cube-single, α sweep',
            os.path.join(ASSETS_DIR, 'q2_cube_alpha_sweep.png'),
        )
        plot_train_metric(
            cube_runs,
            'actor/mse',
            'MSE (policy vs dataset actions)',
            'IQL — cube-single, policy MSE by α',
            os.path.join(ASSETS_DIR, 'q2_cube_mse.png'),
        )

    # ── antsoccer-arena ──
    if antsoccer_runs:
        plot_success_sweep(
            antsoccer_runs,
            'IQL — antsoccer-arena, α sweep',
            os.path.join(ASSETS_DIR, 'q2_antsoccer_alpha_sweep.png'),
        )

# ── Q3: FQL ───────────────────────────────────────────────────────────────────

def plot_q3():
    print("\n=== Q3: FQL ===")
    runs = scan_dir(os.path.join(EXP_DIR, 'q3'))

    cube_runs      = [(f, p, e, a) for f, p, e, a in runs if e == 'cube-single']
    antsoccer_runs = [(f, p, e, a) for f, p, e, a in runs if e == 'antsoccer-arena']

    if cube_runs:
        plot_success_sweep(
            cube_runs,
            'FQL — cube-single, α sweep',
            os.path.join(ASSETS_DIR, 'q3_cube_alpha_sweep.png'),
        )
        plot_train_metric(
            cube_runs,
            'actor/mse',
            'MSE (one-step policy vs dataset actions)',
            'FQL — cube-single, one-step policy MSE by α',
            os.path.join(ASSETS_DIR, 'q3_cube_mse.png'),
        )

    if antsoccer_runs:
        plot_success_sweep(
            antsoccer_runs,
            'FQL — antsoccer-arena, α sweep',
            os.path.join(ASSETS_DIR, 'q3_antsoccer_alpha_sweep.png'),
        )

# ── Algorithm comparison ──────────────────────────────────────────────────────

def plot_comparison():
    print("\n=== Algorithm comparison ===")
    algo_map = {'q1': 'SAC+BC', 'q2': 'IQL', 'q3': 'FQL'}

    for env_key in ['cube-single', 'antsoccer-arena']:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for q, algo in algo_map.items():
            runs = scan_dir(os.path.join(EXP_DIR, q))
            env_runs = [(f, p, e, a) for f, p, e, a in runs if e == env_key]
            if not env_runs:
                continue
            run = best_run(env_runs)
            df = load_csv(run[1], 'eval.csv')
            if df is None or 'eval/success_rate' not in df.columns:
                continue
            ax.plot(df['step'], df['eval/success_rate'],
                    label=f'{algo} (α={run[3]:.0f})', linewidth=2)
            plotted = True

        if plotted:
            ax.set_ylim(bottom=0)
            slug = env_key.replace('-', '_')
            finalize(ax, 'Environment Steps', 'Success Rate',
                     f'Algorithm comparison — {env_key}',
                     os.path.join(ASSETS_DIR, f'comparison_{slug}.png'))
        else:
            plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    plot_q1()
    plot_q2()
    plot_q3()
    plot_comparison()
    print("\nDone! All plots saved to assets/")