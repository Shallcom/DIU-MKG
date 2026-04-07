# DIU-MKG: A Multimodal Knowledge Graph Dataset for Dynamic Instructional Understanding

> **Anonymous Repository for ACM MM 2026 Dataset Track Submission**

## 📖 Introduction
**DIU-MKG** is a multimodal knowledge graph dataset designed for dynamic instructional understanding. It represents authentic classroom videos as structured graphs composed of disciplinary concepts, visual resources, learner-state cues, and pedagogical relations, explicitly capturing stage progression and multimodal instructional organization.

## 📁 File Organization
The dataset is organized by subject, with all multimodal features and transcript data integrated directly into the graph JSON files.

```text
DIU-MKG/
├── dataset_graph/              # Contains the 1,852 episode graphs
│   ├── biology/                # Each JSON integrates graph topology, transcripts, and visual features
│   ├── physics/
│   ├── geography/
    └── ... (10 K-12 subjects total)
├── benchmarks/
│   ├── retrieval/              # Query-candidate pairs for Zero-Shot Retrieval
│   └── intent_recognition/     # Data for the intent-relation ablation study
├── Supplementary Material.pdf  # Supplementary documentation covering dataset schema, graph construction, cross-modal refinement, annotation examples, validation, and evaluation settings
└── README.md
```

This supplementary material complements the main paper by providing additional details on the DIU-MKG dataset, including its schema, construction prompt, cross-modal refinement process, annotation examples, expert validation, error analysis, workflow, and task settings.

## 🛠️ Construction Pipeline
To ensure high graph quality under noisy classroom conditions, DIU-MKG is built through a three-stage cross-modal refinement pipeline:

1.  **Multimodal Information Extraction**: Extracting instructional signals from speech (using FunASR) and visual channels.
2.  **Cross-Modal Semantic Refinement**: Using teacher speech as a semantic anchor (via GPT-4o) to correct visual parsing errors caused by board occlusion or low resolution . 
3.  **Pedagogical Graph Construction**: Instantiating the refined data into structured episode graphs with explicit pedagogical relations, followed by expert-based quality assurance.
