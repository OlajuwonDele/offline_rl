# Offline Reinforcement Learning: SAC+BC, IQL, and FQL

Implementation of three offline RL algorithms:SAC+BC, IQL, and FQL, for continuous control on simulated robotic tasks from the [OGBench](https://github.com/seohongpark/ogbench) suite. Evaluated on cube manipulation and ant soccer.

---

## Overview

Offline RL learns policies entirely from static datasets, without any environment interaction. This poses a key challenge: *distributional shift*, where the learned policy may query the Q-function on out-of-distribution (OOD) actions not covered by the dataset, leading to overestimated values and unstable training.

The three algorithms explored here tackle this with different strategies:

| Algorithm | Policy Class | OOD Mitigation Strategy |
|-----------|-------------|--------------------------|
| SAC+BC | Gaussian | Explicit BC loss term on actor |
| IQL | Gaussian | Expectile regression + advantage-weighted BC |
| FQL | Flow (expressive) | One-step policy distillation from behavioral flow |

---

##  SAC+BC - Behavioral Regularization

SAC+BC (Fujimoto & Gu, 2021; Haarnoja et al., 2018) augments the standard SAC actor loss with a behavioral cloning (BC) term that penalizes deviation from dataset actions.

### Actor Loss

The policy minimizes:

$$\mathcal{L}(\pi) = \mathbb{E}\left[ -\frac{1}{2}\sum_{i=1}^{2} Q_i(s, a^\pi) + \alpha \cdot \frac{1}{|A|}\|a - a^\pi\|_2^2 + \beta \cdot \log \pi(a^\pi|s) \right]$$

The first term maximizes Q-values; the second is the BC regularizer; the third maximizes policy entropy. The coefficient α controls the strength of behavioral regularization and is the key hyperparameter to tune.

### Bellman Backup

Q-functions are trained with:

$$y = r + \frac{\gamma}{2}\sum_{j=1}^{2} \bar{Q}_j(s', a')$$

The average of the two target Q-values is used, and the entropy is ommitted from the backup, both as empirical choices that tend to work better in offline settings.

### Results

![SAC+BC Training Curves](assets/sacbc_curves.png)

| Task | α | Final Success Rate |
|------|---|-------------------|
| cube-single | 100 | >75% |
| antsoccer-arena | 3 | >5% |

**Effect of α on cube-single:**

| α | Behavior |
|---|---------|
| 10 | Insufficient regularization — Q-values diverge, policy ignores dataset |
| 100 | Best balance — MSE to dataset actions is low but non-zero |
| 1000 | Policy collapses to behavioral cloning — no RL improvement |

**Sanity checks:**
- `q_max` should converge to ~0 (reward is 0 or −1)
- `q_min` should converge to ~−100 = −1/(1−γ) on maze tasks; ~−50 to −70 on cube-single (short-horizon)
- MSE between policy and dataset actions should decrease as α increases

---

## Part 2: IQL — Implicit Q-Learning

IQL (Kostrikov et al., 2022) avoids OOD action queries entirely. Rather than computing a max over actions in the Bellman backup, it uses **expectile regression** on a separate value function V(s), which implicitly approximates the maximum without querying Q on unseen actions.

### Value Loss (Expectile Regression)

$$\mathcal{L}(V) = \mathbb{E}_{(s,a)\sim D}\left[ \ell_2^\tau\left(V(s) - \min_{i=1,2} \bar{Q}_i(s,a)\right) \right]$$

where the asymmetric expectile loss $\ell_2^\tau(x) = |\tau - \mathbf{1}(x > 0)| \cdot x^2$ approximates the max operator as τ → 1. We use τ = 0.9 throughout.

### Q Loss

$$\mathcal{L}(Q) = \sum_{i=1}^{2}\mathbb{E}_{(s,a,r,s')\sim D}\left[ \left(Q_i(s,a) - r - \gamma V(s')\right)^2 \right]$$

No policy actions appear in the Bellman backup — all backups use in-dataset transitions only.

### Policy Loss (Advantage-Weighted Regression)

$$\mathcal{L}(\pi) = \mathbb{E}_{(s,a)\sim D}\left[ -\min\left(e^{\alpha A(s,a)}, M\right) \log \pi(a|s) \right]$$

where $A(s,a) = \min_i Q_i(s,a) - V(s)$ and the weight is clipped at M = 100 for stability. This is a form of weighted behavioral cloning — actions with positive advantage are upweighted, negative advantage actions are downweighted.

### Results

![IQL Training Curves](assets/iql_curves.png)

| Task | α | Final Success Rate |
|------|---|-------------------|
| cube-single | 3 | >60% |
| antsoccer-arena | 10 | >5% |

### SAC+BC vs. IQL

**Best performance:** SAC+BC tends to achieve higher peak performance on cube-single (>75% vs >60%), while IQL is more competitive on harder tasks like antsoccer-arena.

**Sensitivity to α:** IQL is noticeably more robust to the choice of α. Because the policy is extracted via a fixed advantage-weighted regression step (rather than by gradient ascent on Q), large α values don't cause the same kind of Q-function exploitation/divergence seen in SAC+BC. In practice, IQL often works well with α ∈ {1, 3, 10} across diverse tasks, whereas SAC+BC requires per-task tuning over a much wider range.

---

## Part 3: FQL — Flow Q-Learning with Expressive Policies

FQL (Park et al., 2025) addresses a limitation of Gaussian policies: when the behavioral distribution is **multimodal**, a unimodal policy cannot represent all useful modes in the dataset. FQL trains a **flow policy** (an ODE-based generative model) for behavioral cloning, but decouples it from Q-maximization using an auxiliary **one-step policy**.

### Why Not Just Use Flow Policies Directly?

The naive extension of SAC+BC to flows (called FBRAC) directly maximizes Q through the flow ODE. This requires **backpropagation through time (BPTT)** — computing gradients through all F=10 ODE integration steps — which is both unstable and expensive.

### FQL's Key Idea

Train two separate policies:
- **Behavioral flow policy** πᵥ — trained purely with flow-matching BC loss, no Q-gradient
- **One-step policy** π_ω — a standard feedforward network distilled from πᵥ, but also trained to maximize Q

This avoids BPTT entirely while preserving the multimodality of the flow policy through distillation.

### Losses

**Behavioral flow policy** (pure BC via flow matching):

$$\mathcal{L}(v) = \mathbb{E}\left[ \frac{1}{|A|}\|v(s, \tilde{a}, t) - (a - z)\|_2^2 \right], \quad \tilde{a} = (1-t)z + ta$$

**One-step policy** (Q-maximization + distillation from flow):

$$\mathcal{L}(\pi_\omega) = \mathbb{E}\left[ -\frac{1}{2}\sum_{i=1}^{2} Q_i(s, \pi_\omega(s,z)) + \frac{\alpha}{|A|}\|\pi_\omega(s,z) - \pi_v(s,z)\|_2^2 \right]$$

Gradients do **not** flow through πᵥ in the distillation term. The Bellman backup uses π_ω (not πᵥ), and evaluation at test time also uses π_ω.

### Results

![FQL Training Curves](assets/fql_curves.png)

| Task | α | Final Success Rate |
|------|---|-------------------|
| cube-single | 300 | >80% |
| antsoccer-arena | 10 | >30% |

FQL significantly outperforms SAC+BC and IQL on antsoccer-arena, where behavioral multimodality matters most. The expressive flow policy captures multiple valid locomotion modes present in the dataset, which a unimodal Gaussian cannot represent.

**Implementation notes:**
- Clip flow policy actions to [−1, 1] before feeding to Q-functions (out-of-bounds actions are physically invalid)
- Do **not** clip one-step policy actions in the distillation loss (clipping prevents gradient correction of OOD outputs)
- Use the **average** (not minimum) of the two target Q-values in the Bellman backup — critical for antsoccer performance

---

## Training Commands

### SAC+BC

```bash
# cube-single
uv run src/scripts/run.py --run_group=q1 --base_config=sacbc \
  --env_name=cube-single-play-singletask-task1-v0 \
  --seed=0 --alpha=100

# antsoccer-arena
uv run src/scripts/run.py --run_group=q1 --base_config=sacbc \
  --env_name=antsoccer-arena-navigate-singletask-task1-v0 \
  --seed=0 --alpha=3
```

### IQL

```bash
# cube-single
uv run src/scripts/run.py --run_group=q2 --base_config=iql \
  --env_name=cube-single-play-singletask-task1-v0 \
  --seed=0 --alpha=3

# antsoccer-arena
uv run src/scripts/run.py --run_group=q2 --base_config=iql \
  --env_name=antsoccer-arena-navigate-singletask-task1-v0 \
  --seed=0 --alpha=10
```

### FQL

```bash
# cube-single
uv run src/scripts/run.py --run_group=q3 --base_config=fql \
  --env_name=cube-single-play-singletask-task1-v0 \
  --seed=0 --alpha=300

# antsoccer-arena
uv run src/scripts/run.py --run_group=q3 --base_config=fql \
  --env_name=antsoccer-arena-navigate-singletask-task1-v0 \
  --seed=0 --alpha=10
```

**Tip:** Run 4 agents in parallel on a single GPU for hyperparameter sweeps — see `README.md` in the starter code for the parallel launch helper.

---

## Results Summary

| Algorithm | cube-single | antsoccer-arena |
|-----------|------------|-----------------|
| SAC+BC | >75% | >5% |
| IQL | >60% | >5% |
| FQL | >80% | >30% |

FQL's expressive flow policy yields the best overall performance, particularly on harder tasks requiring multimodal behavior. IQL is the most stable and easiest to tune. SAC+BC is the simplest to implement but most sensitive to α.

