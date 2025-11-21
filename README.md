# Economic Coherence Dynamics

**Economic Coherence Dynamics** is a field-theoretic framework for macroeconomic systems,
based on interacting density–phase fields. Each subsystem (sector, region, cluster) is
described by a density field \(\rho_i(x,t)\) and a phase field \(\theta_i(x,t)\).
The interplay of density, phase, and interdependence gives rise to coherent structures,
stable exchange cycles, and emergent macro-organizational patterns.

This repository focuses on the economic domain and is conceptually linked to the
separate project:

> **Unified Theory of Motion**
> (a continuum field-theoretic framework for organized motion in physics).

The present project can be read independently: no prior knowledge of the physical theory
is required in order to follow the economic formulation.

---

## Core paper

The foundational document of this repository is:

> **A Lagrangian Framework for Economic Coherence:  
> Field Dynamics, Phase Alignment, and Emergent Structure**  
> *Ivan Salines — Independent Researcher* (2025)

It introduces:

- a Lagrangian density for multi-component economic systems,
- coupled evolution equations for density and phase,
- the emergence of coherence clusters and stable exchange cycles,
- a detailed two-subsystem model as a minimal illustration.

The LaTeX source is in:

```text
docs/18_Lagrangian_Economic_Model/main.tex
```

---

## Repository structure

```text
docs/
    18_Lagrangian_Economic_Model/
        main.tex                    # main LaTeX source of the core paper
        figures/
            phase_alignment_diagram.tikz.tex
            coherence_cluster_network.tikz.tex

code/
    toy_models/
        two_agent_dynamics.ipynb    # (optional) numerical experiments

README.md
LICENSE
CITATION.cff
.gitignore
```

- `docs/` collects all formal documents (papers, notes, appendices).
- `code/` hosts small numerical or conceptual models illustrating the dynamics.
- `figures/` contains TikZ-based figures for fully reproducible diagrams.

Future extensions may include:

- coherence clusters in heterogeneous networks,
- phase cycles and macroeconomic regimes,
- solitonic macro-structures and persistent flows.

---

## Relation to the Unified Theory of Motion

This repository extends the conceptual foundation of the **Unified Theory of Motion**
project by applying density–phase field dynamics to economic organization.

The guiding idea is that:

- coherence, stability, and persistent exchange structures
- can be understood as low-energy configurations of an interacting field,
- where misalignment of phase drives transfers and tensions,
- and phase alignment creates long-lived, efficient patterns of resource circulation.

While the mathematical logic mirrors the physical theory, the present project is
entirely focused on economic systems and can be developed as an autonomous line of research.

---

## Building the documents

To compile the main paper:

```bash
cd docs/18_Lagrangian_Economic_Model
pdflatex main.tex
bibtex main   # if/when a bibliography is added
pdflatex main.tex
pdflatex main.tex
```

(Or simply use Overleaf with the contents of `main.tex` and the TikZ figures.)

---

## Citation

If you use this work in academic or research contexts, please cite the core paper
and/or this repository (see `CITATION.cff`):

> I. Salines, *A Lagrangian Framework for Economic Coherence:
> Field Dynamics, Phase Alignment, and Emergent Structure*, 2025.
