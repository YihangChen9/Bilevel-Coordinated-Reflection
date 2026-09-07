# Resource Contest (RC) — Formal Scenario Definition

Self-contained formulation of the RC environment: the game, the bilevel
decision problem, all formulas, and how the algorithm's abstractions
(m_o, m_e, J) instantiate here.

---

## 1. The game

**Resource Contest** is a repeated cap-allocation game between one
**orchestrator** (leader) and **N workers** (followers), played for a
horizon of **T rounds** against a hidden environment.

The environment hides a vector of **caps**:

$$
M = (M_1, \dots, M_N), \qquad M_i \in \{0, 1, \dots, x_{\max}\},\ x_{\max}=10 .
$$

Nobody — neither orchestrator nor worker — observes $M$ directly. It can
only be probed through interaction. In the *non-stationary* variant the
caps are a function of time, $M_i(t)$, with a scheduled flip.

### One round (timestep $t = 0, \dots, T-1$)

**Step 1 — Leader allocates.** The orchestrator commits a budget allocation
over the probability simplex:

$$
\tau^t = (\tau_1^t, \dots, \tau_N^t), \qquad \tau_i^t \ge 0,\quad \sum_{i=1}^{N} \tau_i^t = C
$$

with total budget $C = 1$. ($\tau$ is produced by the strategy LLM
conditioned on the strategy memory $m_o$; malformed outputs are projected
back onto the simplex: clip to $\ge 0$, renormalize; empty → uniform.)

**Step 2 — Followers act.** Each worker $i$ independently submits an integer
**action** $x_i^t \in \{0, \dots, 10\}$ (up to 3 attempts per round; the last
one counts). The environment **silently clips** it to the hidden cap:

$$
\tilde{x}_i^t = \min\!\left(x_i^t,\; M_i(t)\right) .
$$

The worker's tool feedback reveals *whether* clipping occurred — never the
exact value of $M_i$ beyond what the clip itself discloses.

**Step 3 — Reward.** The team receives the shared scalar reward

$$
R_t \;=\; \sum_{i=1}^{N} \tau_i^t \,\cdot\, \tilde{x}_i^t
\;=\; \sum_{i=1}^{N} \tau_i^t \cdot \min\!\left(x_i^t, M_i(t)\right).
$$

---

## 2. Optimal play and regret

### The follower's subproblem is closed-form

Fix $\tau_i$. The worker's contribution $\tau_i \cdot \min(x_i, M_i)$ is
**monotone non-decreasing in $x_i$** — strictly increasing on
$[0, M_i]$, flat beyond. Hence *any* $x_i \ge M_i$ is optimal, e.g.
$x_i = x_{\max}$:

$$
\arg\max_{x_i} \ \tau_i \cdot \min(x_i, M_i) \;\supseteq\; \{M_i, M_i+1, \dots, x_{\max}\} .
$$

Under optimal worker play, $\tilde{x}_i = M_i$ — i.e. the effective action
reveals the cap.

### The leader's subproblem is a linear program over the simplex

Given $\tilde{x} = M$, the reward $R_t = \sum_i \tau_i M_i$ is **linear in
$\tau$**, so the maximum sits on a vertex of the simplex: put the whole
budget on the largest cap.

$$
\tau^\star = e_{i^\star}, \quad i^\star = \arg\max_i M_i(t),
\qquad
R^\star_t = C \cdot \max_i M_i(t) = \max_i M_i(t).
$$

What makes the problem non-trivial is that $M$ is **unknown**: the leader
must *infer* $\arg\max_i M_i$ from the history of $(\tau, \tilde{x}, R)$
observations — a best-arm-identification problem with the twist that arm
values are only revealed when workers probe high enough, and (in `drift`)
they change mid-episode.

### Metric

$$
\text{Regret}(T) \;=\; \sum_{t=0}^{T-1}\bigl(R^\star_t - R_t\bigr)
\qquad (\text{lower is better; } 0 = \text{oracle}).
$$

Both terms are computed from environment ground truth ($\tilde{x}$ is read
from env state, never parsed from worker text).

---

## 3. Bilevel structure — mapping to the algorithm

RC instantiates the canonical algorithm (Bilevel Coordinated Reflection,
see CLAUDE.md) as follows:

| Algorithm abstraction | RC instantiation |
|---|---|
| query $q$ | "round $t$" (the game itself is the task) |
| decomposition $\tau \sim p_{\text{LLM}}(\cdot \mid q, m_o)$ | strategy LLM emits the budget vector $\tau$ from orch_log history |
| worker output $x_i \sim p_{\text{LLM}}(\cdot \mid \tau_i, m_e)$ | integer commit via `commit_action`, ReAct loop ≤8 steps |
| reflection $c_i$ | `submit_reflection` text, e.g. "agent_2 discovered M₂=8" (Group D) |
| shared workboard $m_e$ | markdown file: per-agent `action=…` lines + reflection sections |
| strategy memory $m_o$ | orch_log: one structured record per round, `reward= tau= x= x_input=` |
| utility $U(x)$ | round reward $R_t$ |
| $J(m_e)$ surrogate | **# agents whose reflection contains a concrete cap claim** (text-computable, no extra LLM call); monotone in information revealed |
| gated write (Phase II) | reflection committed iff $J$ strictly increases, else rolled back |
| meta-reflection → $m_o$ (Phase III) | driver appends the round record to orch_log |
| slow / fast timescales | orch_log persists across episodes (slow); workboard turns over within a round by default (fast) |

### The two information channels (and where drift breaks them)

$$
\underbrace{M_i \xrightarrow{\;\text{clip}\;} \tilde{x}_i}_{\text{env reveals}}
\xrightarrow{\;\text{record}\;}
\underbrace{m_o:\ \texttt{best\_observed\_x}_i = \max_{s \le t} \tilde{x}_i^s}_{\text{leader's belief}}
\qquad\quad
\underbrace{M_i \to \text{scratch}_i:\ \texttt{M=K}}_{\text{worker's belief}}
$$

Both beliefs are **cumulative / write-once**: `best_observed_x` is a running
max that never decreases; scratch caches `M=K` and the worker re-uses it
without re-probing. Under stationary caps both are exact after one
exploration round. Under `drift` (caps change at $t=10$) both become stale
simultaneously — the leader keeps believing the old argmax *and* the worker
never re-probes to generate contradicting evidence. This double staleness
is the measured failure mode (regret ≈ 52 vs ≈ 3 stationary, 11 variants
unable to fix it; see ABLATION_FINDINGS Finding 7).

---

## 4. Configurations

| config | T | N | caps $M$ | $R^\star_t$ | $\sum_t R^\star_t$ | what it stresses |
|---|--:|--:|---|--:|--:|---|
| `easy` | 15 | 3 | (3, 5, 8) | 8 | 120 | wide gaps — trivial identification |
| `hard` | 20 | 3 | (6, 7, 8) | 8 | 160 | near-ties — uniform already gets 7, needs precision |
| `many` | 20 | 6 | (2, 4, 5, 6, 7, 9) | 9 | 180 | N=6 — exploration cost, long histories |
| `drift` | 20 | 3 | (3,5,8) → at $t{=}10$ → (9,5,4) | 8 → 9 | 170 | argmax flips agent₂ → agent₀ mid-episode |

Note `drift`'s adversarial construction: the *old* argmax (agent 2) drops
from 8 to 4, while the *previously weakest* agent (agent 0: 3 → 9) becomes
the new argmax. A system that trusts cached beliefs keeps paying
$\tau$ to a capped-at-4 agent while the cap-9 agent sits at zero allocation.

---

## 5. Baselines (non-LLM reference policies)

| policy | $\tau$ rule | knowledge |
|---|---|---|
| **oracle** | $e_{\arg\max_i M_i(t)}$ every round | sees true caps (upper bound) |
| **uniform** | $\tau_i = 1/N$ always | none |
| **random** | random simplex point each round | none |
| **ε-greedy (0.1)** | round 0 uniform probe; then $1{-}\varepsilon$ on empirical argmax, $\varepsilon$ spread | learns from $\tilde{x}$ history |

All baselines assume workers play the closed-form optimum
($x_i = x_{\max}$), so they isolate the *leader's* learning problem.
ε-greedy is the honest non-LLM competitor: it re-estimates every round, so
it tracks `drift` (regret ≈ 1 post-flip) precisely where the memory-based
bilevel stack does not — re-estimation vs. accumulation is the entire
story of the drift column.

---

## 6. Why RC is a good probe (and what it is not)

**Good because:** (i) ground truth is exact — regret is computable to the
decimal, no LLM judging; (ii) the follower problem is closed-form, so any
observed "worker learning" is attributable and auditable (scratch text);
(iii) the leader problem is a clean bandit, so memory-policy effects
(windowing, gating, resets) show up as crisp regret differences;
(iv) episodes are cheap (~50–170 LLM calls), enabling 11-variant × 2-model
sweeps.

**Not:** a coordination benchmark. Workers' optima are independent — no
congestion, no shared resources, no spatial conflict. The workboard is
provably non-load-bearing under default play (ablation-confirmed:
`workboard_off` = no effect). Coordination pressure is what Overcooked
contributes; RC contributes *identifiability*.

---

## 7. Pointers

- Results & analysis: `experiments/MASTER_RESULTS.md` (Part II),
  `experiments/rc/ABLATION_FINDINGS.md`
- Algorithm: `CLAUDE.md` § Canonical Algorithm
- Raw episodes: `experiments/rc/ablation_results/<model>/<variant>/<config>.json`
- Forensic traces: `workspace/rc_ablation/.../seed_42/llm_trace.jsonl`
