# pipeline.py

from termcolor import colored
from models import QueryPlanner, MultiScaleRetriever, EvidenceReranker, Verifier, generate_final_answer
from videotools import export_video_frame

class EARSPipeline:
    def __init__(self, dataset):
        self.planner = QueryPlanner()
        self.retriever = MultiScaleRetriever(dataset)
        self.reranker = EvidenceReranker()
        self.verifier = Verifier()
        
    def run(self, user_query: str, max_retries=2):
        print(colored(f"User Query: {user_query}", 'blue'))
        current_query = user_query
        for attempt in range(max_retries + 1):
            print(f"\n--- Attempt {attempt+1} ---")
            
            # 1. Plan
            plan = self.planner.plan(current_query)
            search_q = plan.get('semantic_query', current_query)
            print(f"Generated Plan: {plan}")
            
            # 2. Retrieve
            candidates = self.retriever.search(search_q, top_k=5)
            print(f"Retrieved {len(candidates)} candidates.")
            
            # 3. Rerank
            ranked_evidence = self.reranker.rerank(search_q, candidates)
            if not ranked_evidence:
                print(colored("No evidence found.", 'red'))
                break
                
            top_evidence = ranked_evidence[:3] # 取前3
            print(f"Top Evidence: {top_evidence[0]['transcript']} (Score: {top_evidence[0]['relevance_score']})")
            
            # 4. Verify
            is_sufficient, reason = self.verifier.verify(user_query, top_evidence)
            
            if is_sufficient:
                # 5. Generate Answer
                print("Generating final answer with Qwen...")
                frame_path = None
                raw_unit = top_evidence[0].get("raw_unit")
                try:
                    if raw_unit is not None and getattr(raw_unit, "video_path", None):
                        ts = float((raw_unit.start_time + raw_unit.end_time) / 2)
                        frame_path = export_video_frame(raw_unit.video_path, ts, prefix="evidence_center")
                except Exception as e:
                    print(colored(f"[WARN] Failed to export frame: {e}", "yellow"))

                answer = generate_final_answer(user_query, top_evidence[0], frame_path=frame_path)
                return {
                    "status": "success",
                    "answer": answer,
                    "evidence": top_evidence
                }
            else:
                # 6. Refine (Self-Correction)
                print(colored(f"Verification Failed: {reason}", 'yellow'))
                current_query = current_query + " (look for broader context)" 
                
        return {"status": "fail", "answer": "Unable to find sufficient evidence."}
