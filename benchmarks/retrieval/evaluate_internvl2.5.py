# 文件名：eval_internvl2_text_retrieval.py
# 作用：用 InternVL2.5-8B 做“纯文本/图谱片段检索”，评测范式与 qwen3 脚本严格一致

import os
import json
import random
import re
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

# ================= 1. 配置区域 =================
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph"  # 与 qwen3 一致
MODEL_PATH = '/home/workspace/jxk/sourcecode/InternVL-all/InternVL2.5/Intern2-5-8B'

NUM_TEST_FILES = None      # None 表示全量
NUM_QUERIES = 200          # 与 qwen3 一致
NUM_CANDIDATES = 5
NUM_RUNS = 10               # 先跑 1 次观察

# ===============================================
# 与 qwen3 基本相同的数据加载 & 查询构造
# ===============================================

def load_data(data_root, limit=None):
    documents = []
    filepaths = []

    if not os.path.exists(data_root):
        print(f"❌ 找不到目录: {data_root}")
        return []

    subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        video_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]

        for video_id in video_dirs:
            video_path = os.path.join(subject_path, video_id)
            for filename in os.listdir(video_path):
                if filename.endswith("_label.json"):
                    filepaths.append({
                        'subject': subject,
                        'video_id': video_id,
                        'phase': filename.split('_')[1],
                        'path': os.path.join(video_path, filename)
                    })

    if limit:
        filepaths = filepaths[:limit]

    print(f"📦 扫描到 {len(filepaths)} 个教学微阶段 JSON 文件。正在解析...")

    for file_info in tqdm(filepaths, desc="Parsing PedagogyGraphs"):
        try:
            with open(file_info['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)

            transcript = data.get('transcript', "")
            graph = data.get('pedagogy_graph', {})
            nodes = graph.get('nodes', [])
            edges = graph.get('edges', [])

            concepts = [n['label'] for n in nodes if n.get('type') == 'Concept']

            graph_text_parts = []
            node_map = {n['id']: n['label'] for n in nodes}

            for edge in edges:
                src = node_map.get(edge['source'], "Unknown")
                tgt = node_map.get(edge['target'], "Unknown")
                intent = edge.get('intent', 'relates to')
                graph_text_parts.append(f"Concept '{src}' {intent} '{tgt}'")

            if not graph_text_parts:
                graph_text_parts = concepts

            graph_text = ". ".join(graph_text_parts)

            if not transcript and not graph_text:
                continue

            doc_id = f"{file_info['subject']}_{file_info['video_id']}_{file_info['phase']}"

            documents.append({
                'id': doc_id,
                'transcript': transcript if transcript else "No transcript.",
                'graph_text': graph_text if graph_text else "No graph.",
                'concepts': concepts
            })

        except Exception:
            continue

    return documents


def generate_synthetic_queries(documents, num_queries=200):
    queries = []
    print(f"🎯 正在从全量知识库中生成 {num_queries} 道「意图驱动」多模态查询题...")

    valid_docs = [d for d in documents if len(d['concepts']) > 0 and len(d['graph_text']) > 15]

    intent_templates = {
        "visualizes": [
            "Find the segment where the teacher uses visual aids to show {concept}.",
            "Show me the part where {concept} is visually illustrated on the screen."
        ],
        "scaffolds": [
            "Where does the teacher provide step-by-step support for understanding {concept}?",
            "Find the segment scaffolding the students' knowledge about {concept}."
        ],
        "emphasizes": [
            "At what point does the teacher highlight or emphasize {concept}?",
            "Locate the segment stressing the importance of {concept}."
        ],
        "prompts_thinking": [
            "Where does the teacher ask questions to trigger thinking about {concept}?",
            "Find the exact moment the teacher prompts students to reflect on {concept}."
        ],
        "summarizes": [
            "Show me the conclusion or summary phase regarding {concept}."
        ],
        "default": [
            "How does the teacher explain {concept} in this pedagogical segment?"
        ]
    }

    for _ in range(num_queries):
        target_doc = random.choice(valid_docs)
        concept = random.choice(target_doc['concepts'])

        chosen_intent = "default"
        for intent_key in intent_templates.keys():
            if intent_key in target_doc['graph_text']:
                chosen_intent = intent_key
                break

        query_text = random.choice(intent_templates[chosen_intent]).format(concept=concept)
        queries.append({
            'query': query_text,
            'ground_truth_id': target_doc['id']
        })
    return queries

# ===============================================
# InternVL2.5 作为“文本 reranker”的调用
# ===============================================

def internvl_chat_text(model, tokenizer, prompt: str) -> str:
    """
    用 InternVL2.5 的 chat 接口做纯文本对话。
    不传 image，不改你的 prompt 逻辑。
    """
    # InternVL2.5 的 chat 接口支持 pixel_values=None 用纯文本
    response, _ = model.chat(
        tokenizer,
        None,                          # 纯文本场景，不提供图像
        prompt,
        generation_config=dict(
            max_new_tokens=64,
            do_sample=False
        ),
        num_patches_list=None,
        history=None,
        return_history=True
    )
    return response.strip()


def evaluate_with_internvl(documents, queries, model, tokenizer, mode='text'):
    """
    与 qwen3 evaluate_with_llm 一致：只改调用大模型的函数。
    mode = 'text' 用 transcript，mode = 'graph' 用 graph_text
    """
    correct_hits = 0
    results = []

    for q_data in tqdm(queries, desc=f"Eval InternVL2.5 ({mode.upper()})", leave=False):
        question = q_data['query']
        gt_id = q_data['ground_truth_id']

        distractors = [d for d in documents if d['id'] != gt_id]
        candidates = random.sample(distractors, NUM_CANDIDATES - 1)
        candidates.append(next(d for d in documents if d['id'] == gt_id))
        random.shuffle(candidates)

        correct_segment_idx = next(i for i, d in enumerate(candidates) if d['id'] == gt_id)

        # prompt 构造保持与你 qwen3 版本一样
        prompt = (
            "You are an expert pedagogical AI assistant. Your task is to accurately retrieve the correct instructional segment.\n"
            f"USER QUERY: \"{question}\"\n\n"
            "CANDIDATE SEGMENTS:\n"
        )
        for idx, doc in enumerate(candidates):
            content = doc['transcript'] if mode == 'text' else doc['graph_text']
            if len(content) > 1000:
                content = content[:1000] + "..."
            prompt += f"[Segment {idx}]\n{content}\n\n"

        prompt += "Based on the pedagogical intent, return ONLY the exact segment label (e.g., '[Segment 2]') that best matches the query. No other words."

        llm_response = internvl_chat_text(model, tokenizer, prompt)

        hit = False
        chosen_id = "Unknown"
        chosen_idx = -1

        match = re.search(r"Segment\s*(\d+)", llm_response, re.IGNORECASE)
        if match:
            chosen_idx = int(match.group(1))
            if 0 <= chosen_idx < len(candidates):
                chosen_id = candidates[chosen_idx]['id']
                if chosen_id == gt_id:
                    hit = True

        if hit:
            correct_hits += 1

        print(f"\n[{'TEXT' if mode=='text' else 'GRAPH'} 模式] 评测结果追踪:")
        print(f"🧑‍🏫 查询目标(Query): {question}")
        print(f"🎯 正确答案编号应该为: [Segment {correct_segment_idx}] (ID: {gt_id})")
        print(f"🤖 模型原始输出(Raw): {llm_response}")
        print(f"🧠 提取到的选项: [Segment {chosen_idx if chosen_idx != -1 else 'None'}] (匹配ID: {chosen_id})")
        print(f"➡️ 判定结果: {'✅ 正确 (HIT)' if hit else '❌ 错误 (MISS)'}")
        print("-" * 60)

        results.append({
            'query': question,
            'gt_id': gt_id,
            'chosen_id': chosen_id,
            'hit': hit
        })

    acc = (correct_hits / len(queries)) * 100 if queries else 0.0
    return acc, results

# ===============================================
# 主程序
# ===============================================

if __name__ == "__main__":
    print("📥 Loading InternVL2.5-8B Model (for text retrieval)...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    print("✅ Model Loaded.")

    docs = load_data(DATA_ROOT, limit=NUM_TEST_FILES)
    queries = generate_synthetic_queries(docs, num_queries=NUM_QUERIES)

    MODELS_TO_TEST = ["InternVL2.5-8B"]
    final_stats = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n▶ 正在评测大模型: {model_name}")
        text_accs, graph_accs = [], []

        for run_idx in range(NUM_RUNS):
            print(f"  - 运行第 {run_idx+1}/{NUM_RUNS} 次...")
            text_acc, _ = evaluate_with_internvl(docs, queries, model, tokenizer, mode='text')
            graph_acc, _ = evaluate_with_internvl(docs, queries, model, tokenizer, mode='graph')
            text_accs.append(text_acc)
            graph_accs.append(graph_acc)

        final_stats[model_name] = {
            'text_mean': float(np.mean(text_accs)),
            'text_std': float(np.std(text_accs)),
            'graph_mean': float(np.mean(graph_accs)),
            'graph_std': float(np.std(graph_accs))
        }

    print("\n\n📊 Table 1: Multi-LLM Zero-Shot Retrieval Accuracy")
    for model_name, stats in final_stats.items():
        gain = stats['graph_mean'] - stats['text_mean']
        print(f"{model_name:<15} | Text: {stats['text_mean']:.1f}±{stats['text_std']:.1f}% | "
              f"Graph: {stats['graph_mean']:.1f}±{stats['graph_std']:.1f}% | Gain: +{gain:.1f}%")
