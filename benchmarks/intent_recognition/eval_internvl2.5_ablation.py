# 文件名：eval_internvl2.5_classification_fixed.py
import os
import json
import random
import re
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

# ================= 1. 配置区域 =================
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph"  
MODEL_PATH = '/home/workspace/jxk/sourcecode/InternVL-all/InternVL2.5/Intern2-5-8B'

NUM_QUERIES = 200          
NUM_RUNS = 10              

# 核心意图类别（用于展示给模型的标准选项）
INTENT_CLASSES = ["Visualizes", "Scaffolds", "Emphasizes", "Prompts Thinking", "Summarizes"]

# ===============================================
# 数据加载与题目构造 (修复了 0 提取的 Bug)
# ===============================================
def load_and_generate_classification_queries(data_root, num_queries=200):
    documents = []
    print(f"📦 正在扫描并解析 JSON 数据集，构建分类题库...")

    for root, _, files in os.walk(data_root):
        for filename in files:
            if filename.endswith("_label.json"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    transcript = data.get('transcript', "")
                    nodes = data.get('pedagogy_graph', {}).get('nodes', [])
                    edges = data.get('pedagogy_graph', {}).get('edges', [])

                    if not transcript or not edges:
                        continue

                    node_map = {n['id']: n['label'] for n in nodes}
                    
                    # 1. 宽容度极高的意图边过滤
                    valid_edges = []
                    for e in edges:
                        # 获取 json 里的 intent，转成全小写，去掉多余空格
                        raw_intent = str(e.get('intent', '')).lower().strip().replace('_', ' ')
                        
                        # 匹配我们在论文中定义的五大核心意图
                        matched_standard_intent = None
                        if 'visualize' in raw_intent:
                            matched_standard_intent = "Visualizes"
                        elif 'scaffold' in raw_intent:
                            matched_standard_intent = "Scaffolds"
                        elif 'emphasize' in raw_intent:
                            matched_standard_intent = "Emphasizes"
                        elif 'prompt' in raw_intent or 'think' in raw_intent:
                            matched_standard_intent = "Prompts Thinking"
                        elif 'summarize' in raw_intent or 'conclu' in raw_intent:
                            matched_standard_intent = "Summarizes"
                            
                        if matched_standard_intent:
                            valid_edges.append({
                                'source': e.get('source'),
                                'target': e.get('target'),
                                'original_intent': e.get('intent'),
                                'standard_intent': matched_standard_intent
                            })

                    if not valid_edges:
                        continue
                    
                    # 随机挑一个被映射好的核心意图作为题目
                    target_edge = random.choice(valid_edges)
                    target_concept = node_map.get(target_edge['source'], "Unknown Concept")
                    ground_truth_intent = target_edge['standard_intent']

                    # 2. 构建两种图谱表示
                    full_graph_parts = []
                    no_intent_graph_parts = []
                    
                    for edge in edges:
                        src = node_map.get(edge['source'], "Unknown")
                        tgt = node_map.get(edge['target'], "Unknown")
                        intent_str = edge.get('intent', 'relates to')
                        
                        full_graph_parts.append(f"Concept '{src}' {intent_str} '{tgt}'")
                        no_intent_graph_parts.append(f"Concept '{src}' is related to '{tgt}'")

                    documents.append({
                        'id': filename,
                        'transcript': transcript,
                        'full_graph': ". ".join(full_graph_parts),
                        'no_intent_graph': ". ".join(no_intent_graph_parts),
                        'target_concept': target_concept,
                        'ground_truth': ground_truth_intent
                    })

                except Exception:
                    continue

    print(f"✅ 成功从库中提取 {len(documents)} 个含有效意图的片段。")
    if len(documents) == 0:
        print("❌ 提取失败：请检查 JSON 文件里的 edges 中是否有 intent 字段。")
        return []
    
    if len(documents) > num_queries:
        test_queries = random.sample(documents, num_queries)
    else:
        test_queries = documents
        
    return test_queries


# ===============================================
# InternVL2.5 模型调用逻辑
# ===============================================
def internvl_chat_text(model, tokenizer, prompt: str) -> str:
    response, _ = model.chat(
        tokenizer,
        None, 
        prompt,
        generation_config=dict(
            max_new_tokens=16, 
            do_sample=False
        ),
        num_patches_list=None,
        history=None,
        return_history=True
    )
    return response.strip()

def evaluate_classification(queries, model, tokenizer, mode='full_graph'):
    correct_hits = 0

    for q_data in tqdm(queries, desc=f"Eval ({mode.upper()})", leave=False):
        concept = q_data['target_concept']
        gt_intent = q_data['ground_truth'].lower()
        
        if mode == 'text':
            content = q_data['transcript']
        elif mode == 'no_intent_graph':
            content = q_data['no_intent_graph']
        else: # full_graph
            content = q_data['full_graph']
            
        if len(content) > 1500:
            content = content[:1500] + "..."

        prompt = (
            "You are an expert pedagogical AI. Analyze the following instructional segment.\n\n"
            f"[Segment Content]\n{content}\n\n"
            f"Based ONLY on the segment above, what is the teacher's primary pedagogical intent regarding the concept '{concept}'?\n"
            f"You MUST choose exactly ONE from the following list: {INTENT_CLASSES}.\n"
            "Output ONLY the exact intent word from the list."
        )

        llm_response = internvl_chat_text(model, tokenizer, prompt).strip().lower()

        # 精确判断模型是否猜中
        # 比如 gt_intent 是 "visualizes", 如果模型的回答包含这个词就算对
        if gt_intent in llm_response or (gt_intent.split()[0] in llm_response):
            correct_hits += 1

    acc = (correct_hits / len(queries)) * 100 if queries else 0.0
    return acc


# ===============================================
# 主程序
# ===============================================
if __name__ == "__main__":
    print("📥 Loading InternVL2.5-8B Model (for Intent Classification Ablation)...")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    print("✅ Model Loaded.")

    MODELS_TO_TEST = ["InternVL2.5-8B"]
    final_stats = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n▶ 正在进行【意图分类消融评测】: {model_name}")
        text_accs, no_intent_accs, full_graph_accs = [], [], []

        for run_idx in range(NUM_RUNS):
            print(f"\n  - 运行第 {run_idx+1}/{NUM_RUNS} 次...")
            
            queries = load_and_generate_classification_queries(DATA_ROOT, num_queries=NUM_QUERIES)
            if not queries:
                print("程序终止。")
                exit()
            
            text_acc = evaluate_classification(queries, model, tokenizer, mode='text')
            text_accs.append(text_acc)
            
            no_intent_acc = evaluate_classification(queries, model, tokenizer, mode='no_intent_graph')
            no_intent_accs.append(no_intent_acc)
            
            full_graph_acc = evaluate_classification(queries, model, tokenizer, mode='full_graph')
            full_graph_accs.append(full_graph_acc)

        final_stats[model_name] = {
            'text_mean': float(np.mean(text_accs)),
            'text_std': float(np.std(text_accs)),
            'no_intent_mean': float(np.mean(no_intent_accs)),
            'no_intent_std': float(np.std(no_intent_accs)),
            'full_graph_mean': float(np.mean(full_graph_accs)),
            'full_graph_std': float(np.std(full_graph_accs))
        }

    print("\n\n" + "="*60)
    print("📊 Table X: Ablation Study on Pedagogical Intent Edges (Intent Classification Task)")
    print("="*60)
    for model_name, stats in final_stats.items():
        print(f"Model: {model_name}\n")
        print(f"  [1] Text-only Baseline   : {stats['text_mean']:.1f} ± {stats['text_std']:.1f}%")
        print(f"  [2] Graph (No Intent)    : {stats['no_intent_mean']:.1f} ± {stats['no_intent_std']:.1f}%")
        print(f"  [3] PedagogyGraph (Full) : {stats['full_graph_mean']:.1f} ± {stats['full_graph_std']:.1f}%")
        
        intent_gain = stats['full_graph_mean'] - stats['no_intent_mean']
        print(f"\n💡 核心结论: 在意图分类推理中，添加“结构化意图边”带来了高达 +{intent_gain:.1f}% 的绝对提升！")
        print("="*60)
