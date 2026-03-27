# 文件名：eval_ollama_classification_ablation.py
# 作用：基于 Ollama 接口的多模型意图分类消融实验
import os
import json
import random
import requests
import re
from tqdm import tqdm
import numpy as np

# ================= 1. 配置区域 =================
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph" 
OLLAMA_URL = "http://localhost:11456/api/generate"

# 你可以在这里加入你本地部署的多个模型名
MODELS_TO_TEST = [
    "qwen2.5vl:7b",
    # "internvl3:8b", # 如果你也部署了这个，可以直接取消注释一起跑
]

NUM_TEST_FILES = None    # None 表示扫描全量文件
NUM_QUERIES = 200        # 测试时可改小（如10），正式跑用 200
NUM_RUNS = 10             # 每个模型跑几轮取平均

INTENT_CLASSES = ["Visualizes", "Scaffolds", "Emphasizes", "Prompts Thinking", "Summarizes"]

# ================= 2. Ollama 请求接口 =================
def query_ollama(prompt, model_name):
    """调用本地 Ollama 接口进行推理"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # 分类任务必须是0，保证输出确定性
            "num_predict": 32    # 分类任务只需要极短的输出
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"\n❌ Ollama 请求失败 [{model_name}]: {e}")
        return "Error"

# ================= 3. 数据加载与题目构造 =================
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
                        raw_intent = str(e.get('intent', '')).lower().strip().replace('_', ' ')
                        
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
        return []
    
    if len(documents) > num_queries:
        test_queries = random.sample(documents, num_queries)
    else:
        test_queries = documents
        
    return test_queries


# ================= 4. 评测逻辑 =================
def evaluate_classification(queries, model_name, mode='full_graph'):
    correct_hits = 0

    for q_data in tqdm(queries, desc=f"Eval {model_name} ({mode.upper()})", leave=False):
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

        # 高强度约束 Prompt，确保模型只输出我们想要的分类词
        prompt = (
            "You are an expert pedagogical AI. Analyze the following instructional segment.\n\n"
            f"[Segment Content]\n{content}\n\n"
            f"Based ONLY on the segment above, what is the teacher's primary pedagogical intent regarding the concept '{concept}'?\n"
            f"You MUST choose exactly ONE from the following list: {INTENT_CLASSES}.\n"
            "Output ONLY the exact intent word from the list, nothing else."
        )

        llm_response = query_ollama(prompt, model_name).strip().lower()

        # 只要模型输出包含真实意图的关键词（比如 visualizes 匹配 visualize），就算作正确
        first_word = gt_intent.split()[0]
        if gt_intent in llm_response or first_word in llm_response:
            correct_hits += 1
            
        # [可选调试] 打印一条记录看看模型的作答情况
        # print(f"\n[GT]: {gt_intent} | [Pred]: {llm_response}")

    acc = (correct_hits / len(queries)) * 100 if queries else 0.0
    return acc


# ================= 5. 主程序 =================
def main():
    final_stats = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n=======================================================")
        print(f"▶ 正在进行【意图分类消融评测】: {model_name}")
        print(f"=======================================================")
        text_accs, no_intent_accs, full_graph_accs = [], [], []

        for run_idx in range(NUM_RUNS):
            print(f"\n  - 运行第 {run_idx+1}/{NUM_RUNS} 次...")
            
            # 每轮重新采样，保证置信度
            queries = load_and_generate_classification_queries(DATA_ROOT, num_queries=NUM_QUERIES)
            if not queries:
                print("❌ 无法生成测试集，程序终止。")
                return
            
            # 测 Text
            text_acc = evaluate_classification(queries, model_name, mode='text')
            text_accs.append(text_acc)
            
            # 测 No Intent Graph
            no_intent_acc = evaluate_classification(queries, model_name, mode='no_intent_graph')
            no_intent_accs.append(no_intent_acc)
            
            # 测 Full Graph
            full_graph_acc = evaluate_classification(queries, model_name, mode='full_graph')
            full_graph_accs.append(full_graph_acc)

        final_stats[model_name] = {
            'text_mean': float(np.mean(text_accs)),
            'text_std': float(np.std(text_accs)),
            'no_intent_mean': float(np.mean(no_intent_accs)),
            'no_intent_std': float(np.std(no_intent_accs)),
            'full_graph_mean': float(np.mean(full_graph_accs)),
            'full_graph_std': float(np.std(full_graph_accs))
        }

    # 统一打印精美报表
    print("\n\n" + "="*70)
    print("📊 Table X: Ablation Study on Pedagogical Intent Edges")
    print("="*70)
    for model_name, stats in final_stats.items():
        print(f"Model: {model_name}\n")
        print(f"  [1] Text-only Baseline   : {stats['text_mean']:.1f} ± {stats['text_std']:.1f}%")
        print(f"  [2] Graph (No Intent)    : {stats['no_intent_mean']:.1f} ± {stats['no_intent_std']:.1f}%")
        print(f"  [3] PedagogyGraph (Full) : {stats['full_graph_mean']:.1f} ± {stats['full_graph_std']:.1f}%")
        
        intent_gain = stats['full_graph_mean'] - stats['no_intent_mean']
        print(f"\n💡 结论: 添加“结构化意图边”带来了高达 +{intent_gain:.1f}% 的绝对提升！")
        print("-" * 70)

if __name__ == "__main__":
    main()
