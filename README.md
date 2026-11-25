# Causal Arithmetic: Auditing LLM Addition Mechanisms

This repository contains the official code for auditing how Large Language Models (LLMs) internally perform multi-digit addition. While models can output the correct sum, it is unclear *how* they compute it internally. 

We use **Causal Abstraction** and **Distributed Alignment Search (DAS)** to test three competing high-level causal models of addition:

1. **Model A (Carry):** The model computes leading digits independently, detects if the ones-column produces a carry, and adds it. 
2. **Model B (Leading Digits):** The model computes the leading digits (hundreds/tens) completely separately from the ones digit.
3. **Model C (Ones-Sum):** The model computes the full ones-column sum (0–18) as a single unit, deriving both the final ones digit and the carry from it.

## Methodology
We employ **DAS** (via `pyvene`) to discover distributed subspaces within the model's hidden states that encode these intermediate variables (e.g., the carry bit). By performing **interchange interventions** (swapping the learned subspaces between base and source inputs) we measure **Interchange Intervention Accuracy (IIA)**. 

The high-level model with the highest IIA is the one the network actually implements. 

