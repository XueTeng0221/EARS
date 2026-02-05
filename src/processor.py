# dataset.py

import os
import librosa
import whisper
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from videotools import export_video_frame, extract_audio_from_video, _is_audio_file, _is_video_file
from transformers import ClapModel, ClapProcessor, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from utils import _register_summarization_pipeline_for_transformers_v5

_register_summarization_pipeline_for_transformers_v5()

@dataclass
class EvidenceUnit:
    """结构化证据单元"""
    audio_id: str
    start_time: float
    end_time: float
    transcript: str
    level: str  # 'L1' or 'L2'
    embedding: np.ndarray = None 
    audio_tags: List[str] = field(default_factory=list)
    confidence: float = 0.0 # ASR 或 Tag 的置信度
    video_path: Optional[str] = None
    extracted_audio_path: Optional[str] = None

class AudioProcessor:
    def __init__(self, whisper_size: str = "base", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing AudioProcessor on {self.device}...")
        print(f"Loading Whisper ({whisper_size})...")
        
        self.asr_model = whisper.load_model(whisper_size, device=self.device)
        print("Loading CLAP (laion/clap-htsat-unfused)...")
        
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.clap_labels = [
            "neutral speech", "angry speech", "happy speech/laughter", 
            "applause", "silence", "background noise", "music", "typing on keyboard"
        ]
        
        inputs = self.clap_processor(text=self.clap_labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            label_out = self.clap_model.get_text_features(**inputs)
            self.label_embeddings = self._clap_out_to_tensor(label_out, prefer="pool")
            self.label_embeddings = self.label_embeddings / self.label_embeddings.norm(p=2, dim=-1, keepdim=True)

        print("Loading Summarizer...")
        _register_summarization_pipeline_for_transformers_v5()
        self._summarizer_mode = "disabled"
        self.summarizer = None
        self.summarizer_tokenizer = None
        self.summarizer_model = None
        # 多语种摘要模型（可通过环境变量覆盖）
        # 说明：mT5_multilingual_XLSum 支持多语种新闻摘要，在 ko/zh/en 上通常比英文 BART 更稳。
        summarizer_model_name = os.getenv("EARS_SUMMARIZER_MODEL", "csebuetnlp/mT5_multilingual_XLSum")
        self._summarizer_is_t5 = "t5" in summarizer_model_name.lower()
        device_id = 0 if device == "cuda" else -1
        supported_tasks = None
        try:
            from transformers.pipelines import SUPPORTED_TASKS  # type: ignore

            supported_tasks = SUPPORTED_TASKS
        except Exception:
            supported_tasks = None

        can_use_pipeline_summarization = (
            isinstance(supported_tasks, dict) and "summarization" in supported_tasks
        )

        if can_use_pipeline_summarization:
            try:
                self.summarizer = pipeline("summarization", model=summarizer_model_name, device=device_id)
                self._summarizer_mode = "pipeline_summarization"
            except Exception:
                self.summarizer = None
                self._summarizer_mode = "disabled"

        if self._summarizer_mode != "pipeline_summarization":
            print("Summarizer pipeline not available. Falling back to manual Seq2Seq...")
            try:
                self.summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
                self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name).to(self.device)
                self.summarizer_model.eval()
                try:
                    # 针对部分 BART 系列 checkpoint 的兼容设置；T5/mT5 不需要。
                    name_l = summarizer_model_name.lower()
                    if ("bart" in name_l or "distilbart" in name_l) and hasattr(self.summarizer_model, "generation_config"):
                        self.summarizer_model.generation_config.forced_bos_token_id = 0

                    if hasattr(self.summarizer_model, "config") and ("bart" in name_l or "distilbart" in name_l):
                        if getattr(self.summarizer_model.config, "tie_word_embeddings", None) is True:
                            self.summarizer_model.config.tie_word_embeddings = False
                            
                except Exception as cfg_e:
                    print(f"[WARN] Failed to patch summarizer generation config: {cfg_e}")
                
                self._summarizer_mode = "manual_seq2seq"
            except Exception as e2:
                print(f"Summarizer disabled (failed to load model): {e2}")
                self._summarizer_mode = "disabled"

    def _clap_out_to_tensor(self, out, prefer: str = "pool") -> torch.Tensor:
        """兼容 transformers 不同版本的 CLAP 输出结构，统一拿到 (B, D) 特征张量。

        prefer:
            - "pool": 优先拿 pooler/text_embeds/audio_embeds
            - "hidden": 可回退到 last_hidden_state 做均值池化
        """
        if isinstance(out, torch.Tensor):
            return out

        # dict-like
        if isinstance(out, dict):
            for k in ("text_embeds", "audio_embeds", "pooler_output"):
                if k in out and isinstance(out[k], torch.Tensor):
                    return out[k]
                
            if prefer == "hidden" and "last_hidden_state" in out and isinstance(out["last_hidden_state"], torch.Tensor):
                return out["last_hidden_state"].mean(dim=1)

        # attribute-like (e.g. BaseModelOutputWithPooling)
        for attr in ("text_embeds", "audio_embeds", "pooler_output"):
            if hasattr(out, attr):
                value = getattr(out, attr)
                if isinstance(value, torch.Tensor):
                    return value
                
        if prefer == "hidden" and hasattr(out, "last_hidden_state"):
            value = getattr(out, "last_hidden_state")
            if isinstance(value, torch.Tensor):
                return value.mean(dim=1)

        raise TypeError(f"Unsupported CLAP output type for feature tensor extraction: {type(out)}")

    def export_frame_for_unit(
        self,
        unit: EvidenceUnit,
        where: str = "center",
        out_dir: str = "data/processed/extracted_frames",
    ) -> Optional[str]:
        """对某个 EvidenceUnit 导出对齐的视频帧（需要 unit.video_path）。

        where:
            - "center": 片段中心点
            - "start": 片段起点
            - "end": 片段终点
        """
        if not unit.video_path:
            return None
        if where == "start":
            ts = float(unit.start_time)
        elif where == "end":
            ts = float(unit.end_time)
        else:
            ts = float((unit.start_time + unit.end_time) / 2)
        return export_video_frame(unit.video_path, ts, out_dir=out_dir, prefix=f"{unit.level}_{where}")

    def _get_clap_tags(self, audio_fragment: np.ndarray, sr: int, k: int = 2) -> List[str]:
        """
        对音频片段进行 Zero-shot 分类
        """
        if len(audio_fragment) < 1000: # 太短忽略
            return []

        inputs = self.clap_processor(audio=audio_fragment, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            audio_out = self.clap_model.get_audio_features(**inputs)
            audio_embeds = self._clap_out_to_tensor(audio_out, prefer="pool")
            audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 计算相似度: (1, D) @ (N_labels, D).T = (1, N_labels)
            similarity = (audio_embeds @ self.label_embeddings.T).softmax(dim=-1)
            
        # 获取 Top-k 标签
        scores, indices = similarity[0].topk(k)
        tags = []
        for score, idx in zip(scores, indices):
            if score > 0.3: # 阈值过滤
                tags.append(self.clap_labels[idx])
        return tags

    def _summarize_text(self, text: str, max_text_length: Tuple[int, int], length_range: Tuple[int, int]) -> str:
        """
        对长文本进行摘要
        """
        def _clean(s: str) -> str:
            return " ".join((s or "").split()).strip()

        if len(text.split()) < 30: # 太短不摘要
            return _clean(text)
        
        assert max_text_length[0] > max_text_length[1], "max_text_length input must be greater than output"
        assert length_range[1] > length_range[0], "length_range max must be greater than min"
        input_text = text[:max_text_length[0]]
        # T5/mT5 通常需要任务前缀
        if getattr(self, "_summarizer_is_t5", False):
            input_text = "summarize: " + input_text

        # 1) pipeline 模式
        if self._summarizer_mode == "pipeline_summarization" and self.summarizer is not None:
            try:
                summary = self.summarizer(
                    input_text,
                    max_length=length_range[1],
                    min_length=length_range[0],
                    do_sample=False,
                )
                # transformers 的输出 key 可能是 summary_text 或 generated_text
                item = summary[0] if isinstance(summary, list) and summary else {}
                out_text = item.get("summary_text") or item.get("generated_text")
                return _clean(out_text) if out_text else _clean(text[:max_text_length[1]]) + "..."
            except Exception as e:
                print(f"Summarization error (pipeline): {e}")
                return _clean(text[:max_text_length[1]]) + "..."

        # 2) 手动 Seq2Seq 模式
        if self._summarizer_mode == "manual_seq2seq" and self.summarizer_model is not None and self.summarizer_tokenizer is not None:
            try:
                inputs = self.summarizer_tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(self.device)
                with torch.no_grad():
                    out_ids = self.summarizer_model.generate(
                        **inputs,
                        max_length=length_range[1],
                        min_length=length_range[0],
                        num_beams=4,
                        do_sample=False,
                    )
                return _clean(self.summarizer_tokenizer.decode(out_ids[0], skip_special_tokens=True))
            except Exception as e:
                print(f"Summarization error (manual): {e}")
                return _clean(text[:max_text_length[1]]) + "..."

        # 3) 禁用摘要：回退到截断
        return _clean(text[:max_text_length[1]]) + "..."

    def process_audio(self, audio_path: str, chunk_len_l1: int = 30, sample_rate: int = 48000,
                      max_text_length: Tuple[int, int] = (1024, 200), length_range: Tuple[int, int] = (18, 36)) -> Dict[str, List[EvidenceUnit]]:
        """
        全流程处理：加载 -> ASR -> 切片 -> CLAP特征 -> L2构建 -> L1聚合 -> 摘要
        Parameters:
            audio_path: 音频/视频文件路径（若为视频，会先抽取音频）
            chunk_len_l1: L1 聚合的时间窗口长度（秒）
            sample_rate: 加载音频的采样率
            max_text_length: 摘要前的最大文本长度截断 (input_max, output_max)
            length_range: 摘要生成的长度范围 (min_length, max_length)
        Returns:
            Dict 包含 L1 和 L2 级别的 EvidenceUnit 列表
        """
        input_path = audio_path
        input_is_video = _is_video_file(input_path)
        if not (input_is_video or _is_audio_file(input_path)):
            raise ValueError(f"Unsupported media type: {input_path}")

        # 0. 如果是视频：先抽音频
        extracted_audio_path: Optional[str] = None
        asr_input_path = input_path
        librosa_input_path = input_path
        if input_is_video:
            extracted_audio_path = extract_audio_from_video(input_path, sample_rate=sample_rate)
            asr_input_path = extracted_audio_path
            librosa_input_path = extracted_audio_path

        # 1. 加载完整音频
        # Whisper 内部会重采样到 16k，但 CLAP 最好用 48k。
        # 为了方便，我们这里加载一次高采样率，ASR 时 whisper 会处理音频文件，CLAP 用 librosa 数组
        y, sr = librosa.load(librosa_input_path, sr=sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. ASR 转录 (使用 Whisper 的原生逻辑处理文件，这样最稳)
        print(f"  - Running ASR on {os.path.basename(input_path)}...")
        asr_result = self.asr_model.transcribe(asr_input_path, verbose=False)
        # 保存本次音频语言，供摘要策略判断
        self._current_audio_language = asr_result.get("language")
        segments = asr_result['segments']
        
        l2_units = []
        l1_units = []
        
        # 3. 构建 L2 (细粒度) 并提取 CLAP 特征
        print(f"  - Extracting CLAP features for {len(segments)} segments...")
        bar_seg = tqdm(segments, desc="L2 Segments", ncols=80)
        for seg in bar_seg:
            start_sec, end_sec = seg['start'], seg['end']
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            end_sample = min(end_sample, len(y))
            audio_slice = y[start_sample:end_sample]
            tags = self._get_clap_tags(audio_slice, sr)
            l2_units.append(EvidenceUnit(
                audio_id=os.path.basename(input_path),
                start_time=start_sec,
                end_time=end_sec,
                transcript=seg['text'].strip(),
                level="L2",
                audio_tags=tags,
                confidence=np.exp(seg.get('avg_logprob', -1.0)), # Whisper logprob 转概率近似
                video_path=input_path if input_is_video else None,
                extracted_audio_path=extracted_audio_path,
            ))
            
            bar_seg.set_postfix({"start_sec": f"{start_sec:.1f}", "end_sec": f"{end_sec:.1f}", "tags": ",".join(tags)})

        # 4. 构建 L1 (粗粒度) - 基于时间窗口聚合 L2
        print(f"  - Aggregating L1 context windows...")
        num_windows = int(np.ceil(duration / chunk_len_l1))
        bar_wnd = tqdm(range(num_windows), desc="L1 Windows", ncols=80)
        for wnd in bar_wnd:
            w_start = wnd * chunk_len_l1
            w_end = min((wnd + 1) * chunk_len_l1, duration)
            
            # 找到落在该窗口内的所有 L2 片段
            # 判定标准：L2 的中心点落在 L1 窗口内
            current_window_l2s = [u for u in l2_units if w_start <= (u.start_time + u.end_time) / 2 < w_end]
            if not current_window_l2s:
                continue
                
            # 聚合文本
            full_text = " ".join([u.transcript for u in current_window_l2s])
            
            # 聚合标签 (简单的频率统计，取 Top 2)
            all_tags = []
            for u in current_window_l2s:
                all_tags.extend(u.audio_tags)
                
            unique_tags = list(set(all_tags)) # 简单去重
            
            # 生成摘要
            summary_text = self._summarize_text(full_text, max_text_length, length_range)
            l1_units.append(EvidenceUnit(
                audio_id=os.path.basename(input_path),
                start_time=w_start,
                end_time=w_end,
                transcript=f"[SUMMARY] {summary_text}", # 加上标记以便区分
                level="L1",
                audio_tags=unique_tags,
                confidence=1.0, # 聚合层级暂定为1
                video_path=input_path if input_is_video else None,
                extracted_audio_path=extracted_audio_path,
            ))
            
            bar_wnd.set_postfix({"start_sec": f"{w_start:.1f}", "end_sec": f"{w_end:.1f}", "tags": ",".join(unique_tags)})
            
        return {"L1": l1_units, "L2": l2_units}

if __name__ == "__main__":
    files = os.listdir("data/raw")
    file = np.random.choice(files)
    try:
        test_file = os.path.join("data/raw", file)
        print(f"\nProcessing file: {test_file}")
        processor = AudioProcessor(whisper_size="tiny")
        result = processor.process_audio(test_file)
        print("\n" + "="*40)
        print("PROCESSING RESULT")
        print("="*40)
        print(f"Generated {len(result['L1'])} L1 units and {len(result['L2'])} L2 units.")
        if result['L2']:
            print("\n--- Sample L2 Unit (Fine-grained) ---")
            u = np.random.choice(result['L2'])
            print(f"Time: {u.start_time:.1f} - {u.end_time:.1f}s")
            print(f"Text: {u.transcript}")
            print(f"Tags: {u.audio_tags}")
            
        if result['L1']:
            print("\n--- Sample L1 Unit (Coarse Summary) ---")
            u = np.random.choice(result['L1'])
            print(f"Time: {u.start_time:.1f} - {u.end_time:.1f}s")
            print(f"Summary: {u.transcript}")
            print(f"Aggregated Tags: {u.audio_tags}")
    except Exception as e:
        print(f"Error processing file {file}: {e}")