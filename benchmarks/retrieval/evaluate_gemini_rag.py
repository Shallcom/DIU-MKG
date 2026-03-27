import os
import json
import random
import re
import time
from tqdm import tqdm
import numpy as np
from openai import OpenAI  
import httpx # 引入httpx用于设置强制超时

# ================= 1. 配置区域 =================
DATA_ROOT = "/home/workspace/jxk/paperwork/PedagogyGraph/dataset_graph" 

# --- API 配置 ---
API_KEY = "sk-Kg201cYgxnFBIBqtHDFSvNoaN05WYEmwPkOuTu8zghj5GNa9"  
BASE_URL = "https://api.chatanywhere.tech/v1"

# 【防卡死修改 1】：配置 httpx 客户端的超时时间为 15 秒
timeout_settings = httpx.Timeout(60.0, connect=5.0)
http_client = httpx.Client(timeout=timeout_settings)

# 将自定义的 http_client 传入 OpenAI
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL,
    http_client=http_client,
    max_retries=0 # 关闭默认重试，由我们自己代码控制
)

#MODELS_TO_TEST = ["gemini-3-flash-preview"]

MODELS_TO_TEST = ["gpt-5.1"]


NUM_TEST_FILES = None   
NUM_QUERIES = 200       
NUM_CANDIDATES = 5      
NUM_RUNS = 3           
# ===============================================

def query_openai_api(prompt, model_name):
    messages = [
        {"role": "system", "content": "You are a strict evaluation bot. You must follow the user's output format instructions perfectly."},
        {"role": "user", "content": prompt}
    ]
    
    # 【防卡死修改 2】：只重试 1 次，如果连续两次都超时，直接返回 Error 跳过
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0
                # max_tokens=100 注释掉，防止影响格式
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 1:
                return "Error"
            time.sleep(2) 

def load_data(data_root, limit=None):
    documents = []
    filepaths = []
    if not os.path.exists(data_root): return []
        
    subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    for subject in subjects:
        subject_path = os.path.join(data_root, subject)
        video_dirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        for video_id in video_dirs:
            video_path = os.path.join(subject_path, video_id)
            for filename in os.listdir(video_path):
                if filename.endswith("_label.json"):
                    filepaths.append({
                        'subject': subject, 'video_id': video_id,
                        'phase': filename.split('_')[1], 'path': os.path.join(video_path, filename)
                    })
                    
    if limit: filepaths = filepaths[:limit]
        
    for file_info in filepaths: 
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
            
            graph_text = ". ".join(graph_text_parts) if graph_text_parts else ". ".join(concepts)
            if not transcript and not graph_text: continue
                
            doc_id = f"{file_info['subject']}_{file_info['video_id']}_{file_info['phase']}"
            documents.append({
                'id': doc_id, 'transcript': transcript if transcript else "No transcript.",
                'graph_text': graph_text if graph_text else "No graph.", 'concepts': concepts
            })
        except Exception: continue
    return documents

def generate_synthetic_queries(documents, num_queries=20):
    queries = []
    valid_docs = [d for d in documents if len(d['concepts']) > 0 and len(d['graph_text']) > 15]
    intent_templates = {
        "visualizes": ["Find the segment where the teacher uses visual aids to show {concept}."],
        "scaffolds": ["Where does the teacher provide step-by-step support for understanding {concept}?"],
        "emphasizes": ["At what point does the teacher highlight or emphasize {concept}?"],
        "prompts_thinking": ["Where does the teacher ask questions to trigger thinking about {concept}?"],
        "summarizes": ["Show me the conclusion or summary phase regarding {concept}."],
        "default": ["How does the teacher explain {concept} in this pedagogical segment?"]
    }

    for _ in range(num_queries):
        target_doc = random.choice(valid_docs)
        concept = random.choice(target_doc['concepts'])
        chosen_intent = next((k for k in intent_templates.keys() if k in target_doc['graph_text']), "default")
        query_text = random.choice(intent_templates[chosen_intent]).format(concept=concept)
        queries.append({'query': query_text, 'ground_truth_id': target_doc['id']})
    return queries

def evaluate_with_llm(documents, queries, model_name, mode='text'):
    correct_hits = 0
    results = []
    
    # 加入 tqdm 进度条，这样你就不用看刷屏的 DEBUG 了，也能知道它跑到哪里了
    for idx_q, q_data in tqdm(enumerate(queries), total=len(queries), desc=f"Evaluating {mode.upper()}"):
        question = q_data['query']
        gt_id = q_data['ground_truth_id']
        
        distractors = [d for d in documents if d['id'] != gt_id]
        candidates = random.sample(distractors, NUM_CANDIDATES - 1)
        candidates.append(next(d for d in documents if d['id'] == gt_id))
        random.shuffle(candidates)
        
        prompt = (
            "You are evaluating instructional segments.\n"
            f"USER QUERY: \"{question}\"\n\n"
            "CANDIDATE SEGMENTS:\n"
        )
        for idx, doc in enumerate(candidates):
            content = doc['transcript'] if mode == 'text' else doc['graph_text']
            if len(content) > 1000: content = content[:1000] + "..." 
            prompt += f"[Segment {idx}]\n{content}\n\n"
            
        prompt += (
            "Based on the pedagogical intent, which segment best matches the query?\n"
            "You MUST output exactly and ONLY the label (e.g., [Segment 2]).\n"
            "DO NOT OUTPUT ANY OTHER TEXT. NO EXPLANATIONS."
        )

        llm_response = query_openai_api(prompt, model_name)
        
        # 【防卡死修改 3】：如果接口超时返回了 "Error"，直接跳过这题算错，绝不卡死
        if llm_response == "Error":
            continue
        
        hit = False
        chosen_id = "Unknown"
        
        match = re.search(r"Segment\s*:?\s*\]?\s*(\d+)", llm_response, re.IGNORECASE)
        if match:
            chosen_idx = int(match.group(1))
        else:
            nums = re.findall(r"\b([0-4])\b", llm_response)
            if len(nums) == 1:
                chosen_idx = int(nums[0])
            else:
                chosen_idx = -1 
                
        if 0 <= chosen_idx < len(candidates):
            chosen_id = candidates[chosen_idx]['id']
            if chosen_id == gt_id:
                hit = True
                    
        if hit: correct_hits += 1
            
        results.append({'query': question, 'gt_id': gt_id, 'chosen_id': chosen_id, 'hit': hit})
        
    return (correct_hits / len(queries)) * 100 if queries else 0, results

def main():
    docs = load_data(DATA_ROOT, limit=NUM_TEST_FILES)
    queries = generate_synthetic_queries(docs, num_queries=NUM_QUERIES)
    
    final_stats = {}
    
    for model in MODELS_TO_TEST:
        print(f"\n▶ 正在评测大模型: {model}")
        text_accs, graph_accs = [], []
        
        for run_idx in range(NUM_RUNS):
            text_acc, _ = evaluate_with_llm(docs, queries, model, mode='text')
            time.sleep(1) 
            graph_acc, _ = evaluate_with_llm(docs, queries, model, mode='graph')
            time.sleep(1)
            
            text_accs.append(text_acc)
            graph_accs.append(graph_acc)
            
        final_stats[model] = {
            'text_mean': np.mean(text_accs), 'text_std': np.std(text_accs),
            'graph_mean': np.mean(graph_accs), 'graph_std': np.std(graph_accs)
        }
        
    print("\n\n📊 Table 1: Multi-LLM Zero-Shot Retrieval Accuracy")
    for model, stats in final_stats.items():
        gain = stats['graph_mean'] - stats['text_mean']
        print(f"{model:<15} | Text: {stats['text_mean']:.1f}±{stats['text_std']:.1f}% | Graph: {stats['graph_mean']:.1f}±{stats['graph_std']:.1f}% | Gain: +{gain:.1f}%")

if __name__ == "__main__":
    main()
