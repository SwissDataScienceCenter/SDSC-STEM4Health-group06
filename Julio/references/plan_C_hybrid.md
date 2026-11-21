# Plan C – Hybrid model

## Idea
Keep the mass balances (the structure we trust: conservation, feeds, dilution), but do not impose the kinetic laws.
Learn the specific rates (growth, death, substrate uptake, byproduct production) from the data, as functions of the state.
Answers the question: what really drives growth and death in this process?

## Key trick: work with amounts
Amount = concentration × volume. Feeds add volume but no cells, so the amount of cells changes only by growth and death (sampling removes <1%, negligible).

---

## C1. Observed rates from data
**What:** for each training batch and each interval between two samples (k → k+1):

```
mu_net = ln( (Xv·V)_k+1 / (Xv·V)_k ) / dt               net growth rate
kd     = Δ(Xd·V) / ( mean(Xv·V) · dt )                   death rate
qP     = Δ(P·V)  / ( mean(Xv·V) · dt )                   byproduct rate
qS     = ( S added by feeds − Δ(S·V) ) / ( mean(Xv·V) · dt )   substrate uptake
```

qS needs the substrate content of feed B and feed_substrate: estimate it, or take it from Plan A fit.
**Out:** table of rates + state at the start of each interval (S, P, Xv, viability, t)
**Code:** `src/rates.py`
**Done when:** rate table for all training batches.

## C2. Look at the rates
**What:** plot each rate against S, P, Xv, t.
Questions: does growth fall when P rises? does death start after the peak? is uptake constant per cell?
**Out:** short list of findings. This is the core of the interpretability story, and can suggest Plan A v2.

## C3. Rate models
**What:** for each rate, a simple model on 1–3 state variables (linear / polynomial, or small tree / GP).
Prefer simple, readable forms that stay bounded outside the data.
**Code:** `src/models/hybrid.py`

## C4. Simulate
**What:** `HybridModel` with the common interface. Plug the learned rates into the shared simulator (`src/simulator.py`, written in Plan A2) and simulate each batch from day 0.
**Done when:** one batch simulated with realistic curves.

## C5. Validate
**What:** shared `cross_validate`, compare with baseline and Plan A.

## C6. Optional refinement
Fine-tune the rate models on the full trajectories (like Plan A3), not only on the observed rates.

## Pitfalls
- daily sampling → noisy rates: smooth, or use log differences
- unknown substrate content of the feeds → affects qS
- learned rates used outside the training range → simulation drifts; keep rate models simple and bounded
