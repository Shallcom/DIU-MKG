import os
import json
import random
import requests
import re
from tqdm import tqdm
import numpy as np

# ================= 1. 配置区域 =================
# 你真实的数据集根目录
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph" 
OLLAMA_URL = "http://localhost:11456/api/generate"

# 建议包含 Qwen, Llama, Mistral 等不同家族的模型，证明图谱的"模型无关性"
MODELS_TO_TEST = [
    #"qwen2.5vl:7b",      # 阿里系，中文强
    #"gemma3:12b",     # Meta系，推理强
    #"minicpm-v:8b",    # 如果有需要可以取消注释加入更多
    "qwen3-vl:8b",
    #"llava:7b"
]



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

MODELS_TO_TEST = [
    "qwen3-vl:8b",
    #"qwen2.5vl:7b",      # 阿里系，中文强
    #"gemma3:12b",     # Meta系，推理强
    #"minicpm-v:8b",    # 如果有需要可以取消注释加入更多
    #"llava:7b"
]

NUM_TEST_FILES = None   
NUM_QUERIES = 200         # ⚠️ 测试观察时建议先改小（如5个），确认没问题后再改回 200
NUM_CANDIDATES = 5      
# ===============================================

def query_ollama(prompt, model_name):
    """调用本地 Ollama 接口进行推理"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_predict": 1024   
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except Exception as e:
        print(f"\n❌ Ollama 请求失败 [{model_name}]: {e}")
        return "Error"

def load_data(data_root, limit=None):
    """加载数据逻辑保持不变"""
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
        
    print(f"📦 扫描到 {len(filepaths)} 个教学微阶段 JSON 文件 (来自 {len(subjects)} 个学科)。正在解析...")
    
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
            
        except Exception as e:
            continue
            
    return documents

def generate_synthetic_queries(documents, num_queries=200):
    """自动生成查询逻辑保持不变"""
    queries = []
    print(f"🎯 正在从全量知识库中生成 {num_queries} 道「意图驱动」多模态查询题...")
    
    valid_docs = [d for d in documents if len(d['concepts']) > 0 and len(d['graph_text']) > 15]
    
    intent_templates = {
        "visualizes": ["Find the segment where the teacher uses visual aids to show {concept}.", "Show me the part where {concept} is visually illustrated on the screen."],
        "scaffolds": ["Where does the teacher provide step-by-step support for understanding {concept}?", "Find the segment scaffolding the students' knowledge about {concept}."],
        "emphasizes": ["At what point does the teacher highlight or emphasize {concept}?", "Locate the segment stressing the importance of {concept}."],
        "prompts_thinking": ["Where does the teacher ask questions to trigger thinking about {concept}?", "Find the exact moment the teacher prompts students to reflect on {concept}."],
        "summarizes": ["Show me the conclusion or summary phase regarding {concept}."],
        "default": ["How does the teacher explain {concept} in this pedagogical segment?"]
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

def evaluate_with_llm(documents, queries, model_name, mode='text'):
    """零样本 Reranker 评测核心函数 - 【已增加结果可视化打印】"""
    correct_hits = 0
    results = []
    
    for q_data in tqdm(queries, desc=f"Eval {model_name} ({mode.upper()})", leave=False):
        question = q_data['query']
        gt_id = q_data['ground_truth_id']
        
        distractors = [d for d in documents if d['id'] != gt_id]
        candidates = random.sample(distractors, NUM_CANDIDATES - 1)
        candidates.append(next(d for d in documents if d['id'] == gt_id))
        random.shuffle(candidates) 
        
        # 记录真实答案被洗牌到了哪个序号，方便比对
        correct_segment_idx = next(i for i, d in enumerate(candidates) if d['id'] == gt_id)
        
        prompt = (
            "You are an expert pedagogical AI assistant. Your task is to accurately retrieve the correct instructional segment.\n"
            f"USER QUERY: \"{question}\"\n\n"
            "CANDIDATE SEGMENTS:\n"
        )
        for idx, doc in enumerate(candidates):
            content = doc['transcript'] if mode == 'text' else doc['graph_text']
            if len(content) > 1000: content = content[:1000] + "..." 
            prompt += f"[Segment {idx}]\n{content}\n\n"
            
        prompt += "Based on the pedagogical intent, return ONLY the exact segment label (e.g., '[Segment 2]') that best matches the query. No other words."

        llm_response = query_ollama(prompt, model_name)
        
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
                    
        if hit: correct_hits += 1
            
        # =================【新增打印区域】=================
        print(f"\n[{'TEXT' if mode=='text' else 'GRAPH'} 模式] 评测结果追踪:")
        print(f"🧑‍🏫 查询目标(Query): {question}")
        print(f"🎯 正确答案编号应该为: [Segment {correct_segment_idx}] (ID: {gt_id})")
        print(f"🤖 模型原始输出(Raw): {llm_response}")
        print(f"🧠 提取到的选项: [Segment {chosen_idx if chosen_idx != -1 else 'None'}] (匹配ID: {chosen_id})")
        print(f"➡️ 判定结果: {'✅ 正确 (HIT)' if hit else '❌ 错误 (MISS)'}")
        print("-" * 60)
        # ===============================================

        results.append({'query': question, 'gt_id': gt_id, 'chosen_id': chosen_id, 'hit': hit})
        
    return (correct_hits / len(queries)) * 100 if queries else 0, results

def main():
    docs = load_data(DATA_ROOT, limit=NUM_TEST_FILES)
    queries = generate_synthetic_queries(docs, num_queries=NUM_QUERIES)
    
    NUM_RUNS = 5 # ⚠️ 测试观察答案阶段，建议先只跑 1 轮
    final_stats = {}
    
    for model in MODELS_TO_TEST:
        print(f"\n▶ 正在评测大模型: {model}")
        text_accs, graph_accs = [], []
        
        for run_idx in range(NUM_RUNS):
            print(f"  - 运行第 {run_idx+1}/{NUM_RUNS} 次...")
            text_acc, _ = evaluate_with_llm(docs, queries, model, mode='text')
            graph_acc, _ = evaluate_with_llm(docs, queries, model, mode='graph')
            text_accs.append(text_acc)
            graph_accs.append(graph_acc)
            
        final_stats[model] = {
            'text_mean': np.mean(text_accs),
            'text_std': np.std(text_accs),
            'graph_mean': np.mean(graph_accs),
            'graph_std': np.std(graph_accs)
        }
        
    print("\n\n📊 Table 1: Multi-LLM Zero-Shot Retrieval Accuracy")
    for model, stats in final_stats.items():
        gain = stats['graph_mean'] - stats['text_mean']
        print(f"{model:<15} | Text: {stats['text_mean']:.1f}±{stats['text_std']:.1f}% | Graph: {stats['graph_mean']:.1f}±{stats['graph_std']:.1f}% | Gain: +{gain:.1f}%")

if __name__ == "__main__":
    main()
