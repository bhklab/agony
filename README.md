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

## Introduction

The notion of **graph agony** was developed out of a desire to find hierarchies in directed networks [[2]](#2). Intuitively, a hierarchy -- in an organization or social network, for example -- involves assigning *levels* or *ranks* to people such that the ranks are ordered. For example, the CEO of a company should rank higher than an intern. 

Agony works by trying to *find* a hierarchy that best reflects that structure of a given directed network. At a very high level the algorithms in [[2]](#2), [[3]](#3), and [[4]](#4) operate on the same way:

- **Input:**  A directed graph $G=(V,E)$.
- **Output:** A rank function $r^*:V\to \mathbb{N}$, and a value $A(G)$ that is the agony of the graph $G$.

The values of $r^*$ and $A(G)$ are found by solving the following minimization problem:

$$\underset{r:V\to\mathbb{N}}{\min}\sum_{(u,v)\in E}\max(0,r(u)-r(v)+1)$$

where $r^*$ is the function that minimizes the above sum and $A(G)$ is the value achieved.

## Immediate Next Steps

The most immediate next steps involve implementing the ability to compute the weighted and unweighted versions of agony as described in [[3]](#3) and [[4]](#4), respectively. We should do this at least in first in pure `python.` Below are some next steps along with tentative assignments to project members.

**TO DO:**
- Read over Tiers for Peers [[4]](#4) and define the most general form of the agony problem. (James)
- Re-write the linear program agony code such that it's more readable. (James)
- Write the `C++` code in `python` likely using the `heapq` module for the priority queue. `R` also has a [priority queue implemented](https://www.rdocumentation.org/packages/collections/versions/0.1.5/topics/PriorityQueue). (James, Jermiah, Matthew)
- De-bug and benchmark the `python` implementation against its `C++` counterpart. (Jermiah, Matthew)
- Investigate the weighted agony algorithm defined in [[4]](#4). (James, Jermiah, Matthew). The author of [[4]](#4) has made a [C++ implementation available](http://users.ics.aalto.fi/ntatti/agony.zip). The files are also in this repo in the `agony_cpp` folder. 


## Project Ideas

### Software Paper

If we can write a decent `python` and/or `R` package we can release it as its own software paper. It might be worth talking to Ben about this because it's not clear if re-writing publicly available code is a worthy paper. That said, making a library available publicly in pure `python` or `R` is maybe worth circulating/publishing as a matter of record. 

### Pharmacogenomics Papers

The way I see it, there are two ways that we can proceed with a PGx style paper. One way involves using the agony itself as a biomarker. The second involves using agony for rank aggregation.

##### Agony As A Biomarker

In the unweighted case, the agony is bounded above by $|E|$ and 
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

agony bindings are available in the `agony` module.

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

#### <a id="1">[1.]</a> [Inferring Genome-Wide Interaction Networks Using the Phi-Mixing Coefficient](https://pmc.ncbi.nlm.nih.gov/articles/PMC7731978/pdf/nihms-1539098.pdf)

^[This paper introduces a novel algorithm based on the φ-mixing coefficient from probability theory to infer genome-wide interaction networks. The method allows for the construction of weighted and directed networks that can contain cycles, and it has been applied to study subtypes of lung cancer, including small cell (SCLC) and non-small cell (NSCLC), as well as normal lung tissue. There is a [github repo](https://github.com/nitinksingh/phixer/tree/master) with an implementation that is a combination of C and Matlab.]

--------------------------------------

#### <a id = "2">[2.]</a>  [Finding Hierarchy in Directed Online Social Networks](https://dl.acm.org/doi/10.1145/1963405.1963484)

^[The authors define a measure of hierarchy within directed online social networks and present an efficient algorithm to compute this measure. This work provides insights into the structural organization of social networks and how hierarchical relationships can be identified and quantified.]({"attribution":{"attributableIndex":"717-1"}}) [oai_citation:0‡ACM Digital Library](https://dl.acm.org/doi/10.1145/1963405.1963484?utm_source=chatgpt.com)

--------------------------------------

#### <a id="3">[3.]</a> [Faster Way to Agony: Discovering Hierarchies in Directed Graphs](https://arxiv.org/pdf/1902.01477)

^[This paper presents an improved algorithm for computing "agony," a metric used to quantify the hierarchical structure in directed graphs. The proposed method reduces the computational complexity from O(nm²) to O(m²), making it more practical for analyzing large-scale networks.]({"attribution":{"attributableIndex":"1127-1"}}) [oai_citation:1‡arXiv](https://arxiv.org/abs/1902.01477?utm_source=chatgpt.com)

--------------------------------------

#### <a id="4">[4.]</a> [Tiers for Peers: A Practical Algorithm for Discovering Hierarchy in Weighted Networks](https://arxiv.org/pdf/1903.02999)

^[Building upon the concept of agony, this study extends the approach to weighted networks and introduces constraints on the number of hierarchical levels. The authors connect the problem to the capacitated circulation problem and provide both exact and heuristic solutions, demonstrating efficiency in handling large datasets.]({"attribution":{"attributableIndex":"1522-1"}})

--------------------------------------

#### <a id="5">[5.]</a> [Reconstructing Directed Gene Regulatory Network by Only Gene Expression Data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5001240/)

^[The paper proposes the Context-Based Dependency Network (CBDN) method to reconstruct directed gene regulatory networks using solely gene expression data. This approach addresses the challenge of inferring regulatory directions without additional data, such as eQTLs or gene knock-out experiments, which are often unavailable for human tissue samples.]({"attribution":{"attributableIndex":"1987-1"}})


--------------------------------------

#### <a id="6">[6.]</a> [Resolution of Ranking Hierarchies in Directed Networks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191604)
^[ranked stochastic block models]

--------------------------------------


#### <a id="7">[7.]</a> [SCENIC: single-cell regulatory network inference and clustering](https://www.nature.com/articles/nmeth.4463)
^[A paper that infers single cell gene regulatory networks. ]


--------------------------------------

#### <a id="8">[8.]</a> [Inferring Regulatory Networks from Expression Data Using Tree-Based Methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0012776)
^[The GENIE3 algorithm for inferring directed gene regulatory networks. There exists a [github repo](https://github.com/vahuynh/GENIE3) with implementations in C, R, Matlab, and python. This is used in the SCENIC pipeline.]

--------------------------------------

#### <a id="9">[9.]</a> [bLARS: An Algorithm to Infer Gene Regulatory Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7138615)
^[Another algorithm for inferring directed gene regulatory networks from gene expression.]

#### <a id="10">[10.]</a> [Integrate Any Omics: Towards genome-wide
data integration for patient stratification](https://arxiv.org/pdf/2401.07937)
^[]

#### <a id="11">[11.]</a> [Screening cell–cell communication in
spatial transcriptomics via collective
optimal transport](https://www.nature.com/articles/s41592-022-01728-4)