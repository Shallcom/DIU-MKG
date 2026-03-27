没问题，既然你觉得前面的版本太长，那我们就来个**“精简干练版”**。保留核心的介绍、你要求的目录结构，以及数据集构建流程。这样审稿人一眼就能看清仓库的内容和数据的组织方式。
# DIU-MKG: A Multimodal Knowledge Graph Dataset for Dynamic Instructional Understanding

> **Anonymous Repository for ACM MM 2026 Dataset Track Submission**

## 📖 Introduction
[cite_start]**DIU-MKG** is a multimodal knowledge graph dataset designed for dynamic instructional understanding[cite: 188]. [cite_start]It represents authentic classroom videos as structured graphs composed of disciplinary concepts, visual resources, learner-state cues, and pedagogical relations, explicitly capturing stage progression and multimodal instructional organization[cite: 189].

## 📁 File Organization
The dataset is organized by subject, with all multimodal features and transcript data integrated directly into the graph JSON files.

```text
DIU-MKG/
[cite_start]├── dataset_graph/              # Contains the 1,852 episode graphs [cite: 190]
│   ├── biology/                # Each JSON integrates graph topology, transcripts, and visual features
│   ├── physics/
│   ├── geography/
[cite_start]│   └── ... (10 K-12 subjects total) [cite: 546]
├── benchmarks/
│   ├── retrieval/              # Query-candidate pairs for Zero-Shot Retrieval
│   └── intent_recognition/     # Data for the intent-relation ablation study
└── README.md
```

## 🛠️ Construction Pipeline
[cite_start]To ensure high graph quality under noisy classroom conditions, DIU-MKG is built through a three-stage cross-modal refinement pipeline[cite: 191, 334]:

1.  [cite_start]**Multimodal Information Extraction**: Extracting instructional signals from speech (using FunASR) and visual channels (using InternVL-3) [cite: 488, 495-498].
2.  [cite_start]**Cross-Modal Semantic Refinement**: Using teacher speech as a semantic anchor (via GPT-4o) to correct visual parsing errors caused by board occlusion or low resolution [cite: 489, 500-504]. 
3.  [cite_start]**Pedagogical Graph Construction**: Instantiating the refined data into structured episode graphs with explicit pedagogical relations, followed by expert-based quality assurance [cite: 489-490, 507].