# DIU-MKG: A Multimodal Knowledge Graph Dataset for Dynamic Instructional Understanding

> **Anonymous Repository for ACM MM 2026 Dataset Track Submission**

## 📖 Introduction

**DIU-MKG** is a **multimodal knowledge graph dataset** designed for **dynamic instructional understanding**. It represents **authentic classroom videos** as structured graphs composed of **disciplinary concepts**, **visual resources**, **learner-state cues**, and **pedagogical relations**, explicitly capturing **instructional phase progression** and **multimodal instructional organization**.

The dataset is built for **real classroom instructional understanding** rather than generic action recognition or question answering. It provides **structured supervision** for studying **phase-aware pedagogy**, **cross-modal instructional organization**, and **graph-grounded multimodal learning** in educational scenarios.

## 📁 File Organization

The dataset is organized by **subject**, with all **multimodal features** and **transcript data** integrated directly into the graph JSON files.

<pre>
DIU-MKG/
├── <b>dataset_graph/</b>              # Contains the <b>1,852 episode graphs</b>
│   ├── <b>biology/</b>                # Each JSON integrates graph topology, transcripts, and visual features
│   ├── <b>physics/</b>
│   ├── <b>geography/</b>
│   └── ... (<b>10 K-12 subjects</b> total)
├── <b>benchmarks/</b>
│   ├── <b>retrieval/</b>              # Query-candidate pairs for <b>Zero-Shot Retrieval</b>
│   └── <b>intent_recognition/</b>     # Data for the <b>intent-relation ablation study</b>
├── 🔴 **Supplementary Material.pdf**   # <b>Supplementary documentation</b> covering <b>dataset schema</b>, <b>graph construction</b>, <b>cross-modal refinement</b>, <b>annotation examples</b>, <b>expert validation</b>, <b>error analysis</b>, <b>workflow</b>, and <b>evaluation settings</b>
└── <b>README.md</b>
</pre>

The **Supplementary Material.pdf** complements the main paper by providing additional details on the **DIU-MKG dataset**, including its **schema**, **construction prompt**, **cross-modal refinement process**, **annotation examples**, **expert validation**, **error analysis**, **workflow**, and **task settings**.

## 🛠️ Construction Pipeline

To ensure **high graph quality** under **noisy classroom conditions**, DIU-MKG is built through a **three-stage cross-modal refinement pipeline**:

1. **Multimodal Information Extraction**  
   Extracting instructional signals from **speech** (using **FunASR**) and **visual channels**.

2. **Cross-Modal Semantic Refinement**  
   Using **teacher speech as a semantic anchor** (via **GPT-4o**) to correct visual parsing errors caused by **board occlusion**, **handwritten ambiguity**, or **low resolution**.

3. **Pedagogical Graph Construction**  
   Instantiating the refined data into **structured episode graphs** with **explicit pedagogical relations**, followed by **expert-based quality assurance**.

## 📊 Dataset Statistics

DIU-MKG is constructed from **463 K-12 teaching competition videos** taught by **frontline teachers**, covering **10 subjects** and approximately **95 hours** of classroom recordings.

The released dataset contains:

- **1,852 episode graphs**
- **22,280 multimodal nodes**
- **18,367 pedagogical relations**

Each lesson is organized using a **lesson–phase–episode graph structure**, where instructional content is segmented into four **instructional phases**:

- **Introduction**
- **Exposition**
- **Interaction**
- **Conclusion**

## 🔍 Benchmark Tasks

DIU-MKG currently supports two validation settings used in the main paper:

### 1. Zero-Shot Instructional Segment Retrieval
This task evaluates whether **structured pedagogical graphs** help identify the correct instructional segment beyond transcript similarity alone.

### 2. Pedagogical Intent Recognition with Relation Ablation
This task evaluates whether the gains from DIU-MKG mainly come from **explicit pedagogical relation semantics**, rather than from graph structure alone.

## ✅ Quality Control

To assess graph quality, we conduct **expert validation** on sampled episode graphs. The current release reports:

- **300 sampled episode graphs**
- **3 reviewers with educational backgrounds**
- **93.3% overall acceptance rate**
- **92.7% expert consistency**

These results indicate that the automatically constructed graphs are generally well aligned with the source classroom segments and preserve useful pedagogical structure under real classroom conditions.

## 📌 Notes

- The released graph labels are in **English**.
- The source classroom context, including transcripts, is in **Chinese**.
- Depending on privacy constraints, **raw classroom videos** may be **restricted or controlled** rather than fully redistributed.
- The released package is centered on **graph-structured supervision** and **aligned multimodal context**.
