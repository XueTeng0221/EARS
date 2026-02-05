# models.py

import json
import os
import torch
import chromadb
import re
from chromadb.utils import embedding_functions
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from processor import EvidenceUnit

class LocalLLM:
    r"""
    本地化大模型推理封装（单例模式）
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path="./local-qwen"):
        if cls._instance is None:
            print(f"Loading Local LLM: {model_path} ...")
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.1, # 低温以保证 JSON 格式稳定
                do_sample=False  # 确定性输出
            )
        
        # 解码并去除 prompt 部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class LocalVLM:
    """本地多模态模型封装（可选）。

    仅在提供 frame_path 时启用；加载失败会在上层回退到文本 LLM。
    """

    _instance = None

    @classmethod
    def get_instance(cls, model_path: str):
        if cls._instance is None or getattr(cls._instance, "model_path", None) != model_path:
            print(f"Loading Local VLM: {model_path} ...")
            cls._instance = cls(model_path)
        return cls._instance

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device_id = 0 if torch.cuda.is_available() else -1

        # 采用 pipeline 以适配不同 VLM（Qwen-VL/InternVL 等）。
        # 注：如果模型需要 trust_remote_code，可通过环境变量开启。
        trust_remote_code = os.getenv("EARS_TRUST_REMOTE_CODE", "0") == "1"
        self.pipe = pipeline(
            task="image-text-to-text",
            model=model_path,
            device=self.device_id,
            trust_remote_code=trust_remote_code,
        )

    def generate(self, prompt: str, frame_path: str) -> str:
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError(f"PIL/Pillow is required for VLM image input: {e}")

        image = Image.open(frame_path).convert("RGB")

        # 不同 transformers 版本/不同 pipeline 的调用签名略有差异，这里做兼容尝试。
        last_err = None
        for attempt in range(3):
            try:
                if attempt == 0:
                    out = self.pipe(image, prompt=prompt, max_new_tokens=512)
                elif attempt == 1:
                    out = self.pipe({"image": image, "text": prompt}, max_new_tokens=512)
                else:
                    out = self.pipe(image, prompt, max_new_tokens=512)
                break
            except Exception as e:
                last_err = e
                out = None

        if out is None:
            raise RuntimeError(f"VLM pipeline call failed: {last_err}")

        if isinstance(out, list) and out:
            item = out[0]
            if isinstance(item, dict):
                return item.get("generated_text") or item.get("text") or json.dumps(item, ensure_ascii=False)
            return str(item)
        if isinstance(out, dict):
            return out.get("generated_text") or out.get("text") or json.dumps(out, ensure_ascii=False)
        return str(out)

# 辅助函数：清洗 JSON
def extract_json(text: str) -> Dict:
    try:
        # 尝试直接解析
        return json.loads(text)
    except:
        # 尝试提取 Markdown 代码块中的 JSON
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
            return json.loads(clean_text)
        
        # 尝试找第一个 { 到最后一个 }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}

class QueryPlanner:
    r"""
    将用户查询转换为结构化的检索计划
    """
    
    def __init__(self):
        self.llm = LocalLLM.get_instance()

    def plan(self, user_query: str) -> Dict:
        system_prompt = """You are a Query Planner. Your task is to convert a user question into a structured search plan JSON.
Rules:
1. Extract 'semantic_query' for vector search.
2. If the user mentions tone/emotion (angry, happy, sad), set 'filters.tone'.
3. If the user mentions background events (laughter, music, applause), set 'filters.event'.
4. Output valid JSON only."""

        prompt = f"""User Query: "{user_query}"

Output JSON format:
{{
    "semantic_query": "keywords",
    "filters": {{
        "tone": "optional_tone",
        "event": "optional_event"
    }},
    "focus": "short description"
}}
"""
        response_text = self.llm.generate(prompt, system_prompt)
        print(f"[DEBUG] Planner Output: {response_text}")
        
        plan = extract_json(response_text)
        if not plan:
            return {"semantic_query": user_query, "filters": {}}
        return plan


class MultiScaleRetriever:
    r"""
    多尺度检索模块
    """
    
    def __init__(self, evidence_db: Dict[str, List[EvidenceUnit]]):
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("audio_evidence_local")
        except:
            pass
            
        self.collection = self.client.create_collection(
            name="audio_evidence_local",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self.evidence_map = {} 
        
        ids = []
        documents = []
        metadatas = []
        for idx, unit in enumerate(evidence_db.get('L2', [])):
            uid = f"L2_{idx}"
            ids.append(uid)
            documents.append(unit.transcript)
            tags_str = ",".join(unit.audio_tags).lower() if unit.audio_tags else ""
            metadatas.append({
                "start": unit.start_time,
                "audio_id": unit.audio_id,
                "tags": tags_str,
                "level": "L2"
            })
            self.evidence_map[uid] = unit
            
        if ids:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, filters: Dict = None, top_k: int = 5, coeff: int = 3) -> List[EvidenceUnit]:
        fetch_k = top_k * coeff
        
        # 简单的向量搜索
        results = self.collection.query(
            query_texts=[query], 
            n_results=fetch_k
        )
        
        if not results['ids']:
            return []

        candidates = []
        ids_list = results['ids'][0]
        metadatas_list = results['metadatas'][0]
        
        for i, uid in enumerate(ids_list):
            meta = metadatas_list[i]
            unit = self.evidence_map[uid]
            
            # Metadata 过滤逻辑
            if filters:
                if 'tone' in filters and filters['tone']:
                    req_tone = filters['tone'].lower()
                    if req_tone not in meta['tags']: 
                        continue
                if 'event' in filters and filters['event']:
                    req_event = filters['event'].lower()
                    if req_event not in meta['tags']:
                        continue
            
            candidates.append(unit)
            if len(candidates) >= top_k:
                break
                
        return candidates

class EvidenceReranker:
    r"""
    证据重排序模块
    """
    
    def rerank(self, query: str, candidates: List[EvidenceUnit], filters: Dict = None, 
               transc_coeff: float = 0.5, tone_coeff: float = 0.5) -> List[dict]:
        ranked = []
        query_words = query.lower().split()
        
        for unit in candidates:
            score = 0.5 
            transcript_lower = unit.transcript.lower()
            score += float(any(k in transcript_lower for k in query_words)) * transc_coeff
            if filters and 'tone' in filters:
                req_tone = filters['tone'].lower()
                score += float(any(req_tone in tag.lower() for tag in unit.audio_tags)) * tone_coeff
            
            evidence_pack = {
                "timestamp": f"{unit.start_time:.1f} - {unit.end_time:.1f}",
                "transcript": unit.transcript,
                "tags": unit.audio_tags,
                "relevance_score": score,
                "raw_unit": unit
            }
            ranked.append(evidence_pack)
        
        return sorted(ranked, key=lambda x: x['relevance_score'], reverse=True)

class Verifier:
    r"""
    证据充分性验证模块
    """
    
    def __init__(self):
        self.llm = LocalLLM.get_instance()

    def verify(self, query: str, evidence_packs: List[dict]) -> Tuple[bool, str]:
        if not evidence_packs:
            return False, "No evidence found."
            
        top = evidence_packs[0]
        system_prompt = "You are an Evidence Verifier. Check if the transcript and audio tags (tone/event) \
                        are sufficient to answer the user query."
        prompt = f"""User Question: "{query}"

Retrieved Evidence:
- Transcript: "{top['transcript']}"
- Detected Tags: {top['tags']}

Task:
1. Is this sufficient? (true/false)
2. Explain why. If user asks about tone, check tags.

Output JSON:
{{
    "is_sufficient": boolean,
    "reason": "explanation"
}}
"""
        response_text = self.llm.generate(prompt, system_prompt)
        print(f"[DEBUG] Verifier Output: {response_text}")
        
        res = extract_json(response_text)
        return res.get('is_sufficient', False), res.get('reason', 'Unknown reason')

def generate_final_answer(query: str, evidence: dict, frame_path: str | None = None) -> str:
    r"""
    生成最终回答的辅助函数
    """
    system_prompt = "You are a helpful assistant. Answer the user question based on the provided evidence."
    base_prompt = f"""Question: {query}

Evidence (Audio):
- Text: "{evidence['transcript']}"
- Context/Tone: {evidence['tags']}
- Time: {evidence['timestamp']}
"""

    # 有视频帧时：优先走多模态；失败则回退到文本
    if frame_path:
        vlm_path = os.getenv("EARS_VLM_PATH", "./local-qwen-vl")
        try:
            vlm = LocalVLM.get_instance(vlm_path)
            mm_prompt = base_prompt + "\nEvidence (Video Frame): Use the image to verify/clarify the audio evidence.\n"
            return vlm.generate(mm_prompt, frame_path)
        except Exception as e:
            print(f"[WARN] VLM unavailable, falling back to text-only LLM: {e}")

    llm = LocalLLM.get_instance()
    prompt = base_prompt + "\nPlease answer concisely.\n"
    return llm.generate(prompt, system_prompt)
