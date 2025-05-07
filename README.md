# Agony

**Authors:** [James Bannon](https://github.com/jbannon), [Julia Nguyen](https://github.com/julianguyn), [Matthew Boccalon](https://github.com/mattbocc), [Jermiah Joseph](https://github.com/jjjermiah)

**Contact:** [bhklab.jermiahjoseph@gmail.com](mailto:bhklab.jermiahjoseph@gmail.com)

**Description:** Agony implementation

--------------------------------------

[![pixi-badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square)](https://github.com/prefix-dev/pixi)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![Built with Material for MkDocs](https://img.shields.io/badge/mkdocs--material-gray?logo=materialformkdocs&style=flat-square)](https://github.com/squidfunk/mkdocs-material)

![GitHub last commit](https://img.shields.io/github/last-commit/bhklab/agony?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/bhklab/agony?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bhklab/agony?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/bhklab/agony?style=flat-square)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/bhklab/agony?style=flat-square)

## Set Up

### Prerequisites

Pixi is required to run this project.
If you haven't installed it yet, [follow these instructions](https://pixi.sh/latest/)

### Installation

1. Clone this repository to your local machine
2. Navigate to the project directory
3. Set up the environment using Pixi:

```bash
pixi install
```

### Usage

agon bindings are available in the `agony` module.

I've setup two tasks to showcase this

```console
pixi run help
```

```console
pixi run cycle_dfs
```

## Documentation

Click [here](https://bhklab.github.io/agony) to view the full documentation.

## References

### 📄 Annotated List of Papers

--------------------------------------

#### 1. [Inferring Genome-Wide Interaction Networks Using the Phi-Mixing Coefficient](https://pmc.ncbi.nlm.nih.gov/articles/PMC7731978/pdf/nihms-1539098.pdf)

This paper introduces a novel algorithm based on the φ-mixing coefficient from probability theory to infer genome-wide interaction networks. The method allows for the construction of weighted and directed networks that can contain cycles, and it has been applied to study subtypes of lung cancer, including small cell (SCLC) and non-small cell (NSCLC), as well as normal lung tissue.

--------------------------------------

#### 2. [Finding Hierarchy in Directed Online Social Networks](https://dl.acm.org/doi/10.1145/1963405.1963484)

^[The authors define a measure of hierarchy within directed online social networks and present an efficient algorithm to compute this measure. This work provides insights into the structural organization of social networks and how hierarchical relationships can be identified and quantified.]({"attribution":{"attributableIndex":"717-1"}}) [oai_citation:0‡ACM Digital Library](https://dl.acm.org/doi/10.1145/1963405.1963484?utm_source=chatgpt.com)

--------------------------------------

#### 3. [Faster Way to Agony: Discovering Hierarchies in Directed Graphs](https://arxiv.org/pdf/1902.01477)

^[This paper presents an improved algorithm for computing "agony," a metric used to quantify the hierarchical structure in directed graphs. The proposed method reduces the computational complexity from O(nm²) to O(m²), making it more practical for analyzing large-scale networks.]({"attribution":{"attributableIndex":"1127-1"}}) [oai_citation:1‡arXiv](https://arxiv.org/abs/1902.01477?utm_source=chatgpt.com)

--------------------------------------

#### 4. [Tiers for Peers: A Practical Algorithm for Discovering Hierarchy in Weighted Networks](https://arxiv.org/pdf/1903.02999)

^[Building upon the concept of agony, this study extends the approach to weighted networks and introduces constraints on the number of hierarchical levels. The authors connect the problem to the capacitated circulation problem and provide both exact and heuristic solutions, demonstrating efficiency in handling large datasets.]({"attribution":{"attributableIndex":"1522-1"}})

--------------------------------------

#### 5. [Reconstructing Directed Gene Regulatory Network by Only Gene Expression Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5001240/)

^[The paper proposes the Context-Based Dependency Network (CBDN) method to reconstruct directed gene regulatory networks using solely gene expression data. This approach addresses the challenge of inferring regulatory directions without additional data, such as eQTLs or gene knock-out experiments, which are often unavailable for human tissue samples.]({"attribution":{"attributableIndex":"1987-1"}})
