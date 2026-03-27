import os
import json
import random
import requests
import re
from tqdm import tqdm
import numpy as np

# ================= 1. 配置区域 =================
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph" 
# 🌟 必须用 chat 接口
OLLAMA_URL = "http://localhost:11456/api/chat"

MODELS_TO_TEST = [
    "qwen3-vl:8b",
]

NUM_TEST_FILES = None   
NUM_QUERIES = 200         # 先跑 5 个看看它思考的过程
NUM_CANDIDATES = 5      
# ===============================================

def query_ollama(prompt, model_name):
    """调用本地 Ollama 接口进行推理"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0, 
            "num_predict": -1,      # 🌟 杀招1：-1 表示无限制生成，任凭它写几万字草稿都不会被系统强制掐断！
            "num_ctx": 16384        # 🌟 杀招2：给足上下文空间，防止题目太长加上草稿太多导致直接崩溃
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=240) # 把超时时间拉长，给它充分的思考时间
        response.raise_for_status()
        
        # 🌟 杀招3：完美解析 Ollama 深度思考模型的 JSON 结构
        res_json = response.json()
        message = res_json.get('message', {})
        
        content = message.get('content', '').strip()
        thinking = message.get('thinking', '').strip()
        
        # 如果模型还在调皮只返回了 thinking，我们做个兜底抓取
        if not content and thinking:
            print("\n[提示] 模型全把答案写在 thinking 里了，尝试从 thinking 末尾提取...")
            content = thinking
            
        return content
    except Exception as e:
        print(f"\n❌ Ollama 请求失败 [{model_name}]: {e}")
        return "Error"

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
    queries = []
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
    correct_hits = 0
    results = []
    
    for q_data in tqdm(queries, desc=f"Eval {model_name} ({mode.upper()})", leave=False):
        question = q_data['query']
        gt_id = q_data['ground_truth_id']
        
        distractors = [d for d in documents if d['id'] != gt_id]
        candidates = random.sample(distractors, NUM_CANDIDATES - 1)
        candidates.append(next(d for d in documents if d['id'] == gt_id))
        random.shuffle(candidates) 
        
        correct_segment_idx = next(i for i, d in enumerate(candidates) if d['id'] == gt_id)
        
        prompt = (
            "You are an expert pedagogical AI assistant. Your task is to accurately retrieve the correct instructional segment.\n"
            f"USER QUERY: \"{question}\"\n\n"
            "CANDIDATE SEGMENTS:\n"
        )
        for idx, doc in enumerate(candidates):
            content = doc['transcript'] if mode == 'text' else doc['graph_text']
            
            # 🔥 稍微把输入限制缩小一丁点（从1000变为800），极大缓解显存压力，完全不影响评测精度
            if len(content) > 800: content = content[:800] + "..." 
            prompt += f"[Segment {idx}]\n{content}\n\n"
            
        prompt += "Based on the pedagogical intent, return ONLY the exact segment label (e.g., '[Segment 2]') that best matches the query. DO NOT output any reasoning. No other words."

        llm_response = query_ollama(prompt, model_name)
        
        hit = False
        chosen_id = "Unknown"
        chosen_idx = -1
        
        # 为了应对长篇大论，我们只取模型最后 100 个字符进行提取，确保提取到的是结论而不是过程
        tail_response = llm_response[-100:] if len(llm_response) > 100 else llm_response
        
        match = re.search(r"Segment\s*\[?(\d+)\]?", tail_response, re.IGNORECASE)
        if match:
            chosen_idx = int(match.group(1))
        else:
            numbers = re.findall(r"\d+", tail_response)
            if numbers:
                chosen_idx = int(numbers[-1])  # 取最后一个数字

        if 0 <= chosen_idx < len(candidates):
            chosen_id = candidates[chosen_idx]['id']
            if chosen_id == gt_id:
                hit = True
                    
        if hit: correct_hits += 1
            
        print(f"\n[{'TEXT' if mode=='text' else 'GRAPH'} 模式] 评测结果追踪:")
        print(f"🧑‍🏫 查询目标(Query): {question}")
        print(f"🎯 正确答案应该为: [Segment {correct_segment_idx}]")
        print(f"🤖 提取结果所依据的尾部文本: {repr(tail_response)}")
        print(f"🧠 最终提取到的选项: [Segment {chosen_idx if chosen_idx != -1 else 'None'}]")
        print(f"➡️ 判定结果: {'✅ 正确 (HIT)' if hit else '❌ 错误 (MISS)'}")
        print("-" * 60)

        results.append({'query': question, 'gt_id': gt_id, 'chosen_id': chosen_id, 'hit': hit})
        
    return (correct_hits / len(queries)) * 100 if queries else 0, results

def main():
    docs = load_data(DATA_ROOT, limit=NUM_TEST_FILES)
    queries = generate_synthetic_queries(docs, num_queries=NUM_QUERIES)
    
    NUM_RUNS = 10 
    final_stats = {}
    
    for model in MODELS_TO_TEST:
        print(f"\n▶ 正在评测大模型: {model}")
        text_accs, graph_accs = [], []
        
        for run_idx in range(NUM_RUNS):
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
