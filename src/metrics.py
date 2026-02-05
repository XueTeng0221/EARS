# evaluate.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from pipeline import EARSPipeline
from processor import AudioProcessor
from models import LocalLLM, extract_json

"""
Expected test set format (JSON):
[
    {
        "query": "Did the speaker sound angry when discussing the budget?",
        "audio_file": "meeting_01.mp3",
        "ground_truth_segments": [
            {"start": 120.5, "end": 135.2, "tags": ["angry speech"]}
        ],
        "expected_answer": "Yes, the speaker exhibited frustration at 2:00-2:15."
    },
    ...
]
"""

def calculate_ess(retrieved_units: List, ground_truth_segments: List[Dict]) -> float:
    """
    Evidence Support Score: 检索到的证据是否覆盖 Ground Truth 时间段
    使用 IoU (Intersection over Union) 计算
    """
    if not ground_truth_segments or not retrieved_units:
        return 0.0
    
    total_iou = 0.0
    for gt in ground_truth_segments:
        gt_start, gt_end = gt['start'], gt['end']
        max_iou = 0.0
        
        for unit in retrieved_units:
            # 计算时间段重叠
            overlap_start = max(gt_start, unit.start_time)
            overlap_end = min(gt_end, unit.end_time)
            overlap = max(0, overlap_end - overlap_start)
            
            union = (gt_end - gt_start) + (unit.end_time - unit.start_time) - overlap
            iou = overlap / union if union > 0 else 0
            max_iou = max(max_iou, iou)
        
        total_iou += max_iou
    
    return total_iou / len(ground_truth_segments)

def calculate_mrr(retrieved_units: List, ground_truth_segments: List[Dict]) -> float:
    """
    Mean Reciprocal Rank: 第一个正确证据的排名倒数
    """
    for rank, unit in enumerate(retrieved_units, start=1):
        for gt in ground_truth_segments:
            # 判断是否重叠超过 50%
            overlap = max(0, min(gt['end'], unit.end_time) - max(gt['start'], unit.start_time))
            gt_duration = gt['end'] - gt['start']
            if overlap / gt_duration > 0.5:
                return 1.0 / rank
    return 0.0

def evaluate_answer_with_llm(query: str, generated_answer: str, expected_answer: str) -> Tuple[float, bool]:
    """
    使用 LLM 判断生成的答案是否正确 (Answer Accuracy) 和是否存在幻觉 (RHR)
    返回: (accuracy_score, has_hallucination)
    """
    llm = LocalLLM.get_instance()
    system_prompt = """You are an Answer Evaluator. Compare the generated answer with the expected answer.
Output JSON:
{
    "is_correct": true/false,
    "has_hallucination": true/false,
    "explanation": "brief reason"
}
"""
    prompt = f"""
Question: {query}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Task:
1. Is the generated answer factually correct? (check if key facts match)
2. Does it contain hallucinations (information not supported by the question/context)?
"""
    
    response = llm.generate(prompt, system_prompt)
    result = extract_json(response)
    
    is_correct = result.get('is_correct', False)
    has_hallucination = result.get('has_hallucination', False)
    
    return (1.0 if is_correct else 0.0), has_hallucination

def run_evaluation(test_set_path: str, audio_dir: str) -> Dict:
    """
    对测试集运行完整评估
    """
    # 加载测试集
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # 初始化 AudioProcessor
    processor = AudioProcessor(whisper_size="base") # 可根据资源调整
    
    results = []
    ess_scores = []
    mrr_scores = []
    accuracy_scores = []
    rhr_flags = []
    
    for idx, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {idx+1}/{len(test_cases)}")
        print(f"{'='*60}")
        
        query = case['query']
        audio_file = os.path.join(audio_dir, case['audio_file'])
        ground_truth = case['ground_truth_segments']
        expected_ans = case.get('expected_answer', '')
        
        # Step 1: 处理音频 (如果尚未处理，可缓存结果)
        print(f"Processing audio: {audio_file}")
        evidence_db = processor.process_audio(audio_file)
        
        # Step 2: 运行 Pipeline
        pipeline = EARSPipeline(evidence_db)
        result = pipeline.run(query)
        
        if result['status'] == 'success':
            retrieved_units = [ev['raw_unit'] for ev in result['evidence']]
            generated_answer = result['answer']
            
            # Step 3: 计算指标
            ess = calculate_ess(retrieved_units, ground_truth)
            mrr = calculate_mrr(retrieved_units, ground_truth)
            
            # Step 4: 评估答案 (使用 LLM)
            if expected_ans:
                acc, has_hal = evaluate_answer_with_llm(query, generated_answer, expected_ans)
            else:
                acc, has_hal = 0.0, False # 如果没有标注，跳过
            
            ess_scores.append(ess)
            mrr_scores.append(mrr)
            accuracy_scores.append(acc)
            rhr_flags.append(1.0 if has_hal else 0.0)
            
            results.append({
                "query": query,
                "ess": ess,
                "mrr": mrr,
                "accuracy": acc,
                "has_hallucination": has_hal,
                "answer": generated_answer
            })
            
            print(f"✓ ESS: {ess:.2f} | MRR: {mrr:.2f} | Accuracy: {acc:.2f} | Hallucination: {has_hal}")
        else:
            print("✗ Pipeline failed to find sufficient evidence.")
            results.append({
                "query": query,
                "ess": 0.0,
                "mrr": 0.0,
                "accuracy": 0.0,
                "has_hallucination": False,
                "answer": "N/A"
            })
    
    summary = {
        "avg_ess": np.mean(ess_scores) if ess_scores else 0.0,
        "avg_mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
        "avg_accuracy": np.mean(accuracy_scores) if accuracy_scores else 0.0,
        "rhr": np.mean(rhr_flags) if rhr_flags else 0.0, # Reasoning Hallucination Rate
        "details": results
    }
    
    return summary

def visualize_metrics(summary: Dict, save_path: str = "results/metrics.png"):
    """
    绘制评估指标对比图
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    queries = [f"Q{i+1}" for i in range(len(summary['details']))]
    ess = [d['ess'] for d in summary['details']]
    mrr = [d['mrr'] for d in summary['details']]
    acc = [d['accuracy'] for d in summary['details']]
    rhr = [1.0 if d['has_hallucination'] else 0.0 for d in summary['details']]
    x = np.arange(len(queries))
    width = 0.2
    _, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width*1.5, ess, width, label='ESS (Evidence Support)', color='skyblue')
    ax.bar(x - width*0.5, mrr, width, label='MRR (Retrieval Rank)', color='lightgreen')
    ax.bar(x + width*0.5, acc, width, label='Accuracy (Answer Quality)', color='gold')
    ax.bar(x + width*1.5, rhr, width, label='RHR (Hallucination)', color='salmon')
    ax.set_xlabel('Test Cases')
    ax.set_ylabel('Score')
    ax.set_title(f'EARS Evaluation Results\n(Avg ESS: {summary["avg_ess"]:.2f}, MRR: {summary["avg_mrr"]:.2f}, Acc: {summary["avg_accuracy"]:.2f}, RHR: {summary["rhr"]:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.legend()
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Metrics visualization saved to {save_path}")

if __name__ == "__main__":
    test_set_example = [
        {
            "query": "Was there any angry speech during the meeting?",
            "audio_file": "35282289930-1-192.mp4",
            "ground_truth_segments": [
                {"start": 900.0, "end": 3600.0, "tags": ["angry speech"]}
            ],
            "expected_answer": "No, there was no angry speech during the meeting."
        }
    ]
    
    os.makedirs("data", exist_ok=True)
    with open("data/test_set.json", "w") as f:
        json.dump(test_set_example, f, indent=2)
    
    summary = run_evaluation(
        test_set_path="data/test_set.json",
        audio_dir="data/raw"
    )
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average ESS (Evidence Support):     {summary['avg_ess']:.3f}")
    print(f"Average MRR (Mean Reciprocal Rank): {summary['avg_mrr']:.3f}")
    print(f"Average Answer Accuracy:            {summary['avg_accuracy']:.3f}")
    print(f"Reasoning Hallucination Rate (RHR): {summary['rhr']:.3f}")
    with open("results/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    visualize_metrics(summary)
