# dataset.py

import os
import librosa
import whisper
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from transformers import ClapModel, ClapProcessor, pipeline

@dataclass
class EvidenceUnit:
    """结构化证据单元"""
    audio_id: str
    start_time: float
    end_time: float
    transcript: str
    level: str  # 'L1' or 'L2'
    embedding: np.ndarray = None 
    audio_tags: List[str] = field(default_factory=list) # 存储 CLAP 识别出的标签
    confidence: float = 0.0 # ASR 或 Tag 的置信度

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
            self.label_embeddings = self.clap_model.get_text_features(**inputs)
            self.label_embeddings = self.label_embeddings / self.label_embeddings.norm(p=2, dim=-1, keepdim=True)

        print("Loading Summarizer...")
        
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if device=="cuda" else -1)

    def _get_clap_tags(self, audio_fragment: np.ndarray, sr: int) -> List[str]:
        """
        对音频片段进行 Zero-shot 分类
        """
        if len(audio_fragment) < 1000: # 太短忽略
            return []

        inputs = self.clap_processor(audios=audio_fragment, sampling_rate=sr, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            audio_embeds = self.clap_model.get_audio_features(**inputs)
            audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 计算相似度: (1, D) @ (N_labels, D).T = (1, N_labels)
            similarity = (audio_embeds @ self.label_embeddings.T).softmax(dim=-1)
            
        # 获取 Top-2 标签
        scores, indices = similarity[0].topk(2)
        tags = []
        for score, idx in zip(scores, indices):
            if score > 0.3: # 阈值过滤
                tags.append(self.clap_labels[idx])
        return tags

    def _summarize_text(self, text: str, max_text_length: Tuple[int, int] = (1024, 200), length_range: Tuple[int, int] = (360, 7200)) -> str:
        """
        对长文本进行摘要
        注意：议会开头有 pledge of allegiance 等仪式环节，截断前几分钟
        """
        if len(text.split()) < 30: # 太短不摘要
            return text
        
        assert max_text_length[0] > length_range[1], "Max text length must be greater than summary max length."
        try:
            input_text = text[:max_text_length[0]] 
            summary = self.summarizer(input_text, max_length=length_range[1], min_length=length_range[0], do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:max_text_length[1]] + "..."

    def process_audio(self, audio_path: str, chunk_len_l1: int = 60, sample_rate: int = 48000) -> Dict[str, List[EvidenceUnit]]:
        """
        全流程处理：加载 -> ASR -> 切片 -> CLAP特征 -> L2构建 -> L1聚合 -> 摘要
        """
        # 1. 加载完整音频
        # Whisper 内部会重采样到 16k，但 CLAP 最好用 48k。
        # 为了方便，我们这里加载一次高采样率，ASR 时 whisper 会自己处理路径，CLAP 用 librosa 数组
        y, sr = librosa.load(audio_path, sr=sample_rate) 
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. ASR 转录 (使用 Whisper 的原生逻辑处理文件，这样最稳)
        print(f"  - Running ASR on {os.path.basename(audio_path)}...")
        asr_result = self.asr_model.transcribe(audio_path, verbose=False)
        segments = asr_result['segments']
        
        l2_units = []
        l1_units = []
        
        # 3. 构建 L2 (细粒度) 并提取 CLAP 特征
        print(f"  - Extracting CLAP features for {len(segments)} segments...")
        for seg in segments:
            start_sec, end_sec = seg['start'], seg['end']
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            end_sample = min(end_sample, len(y))
            audio_slice = y[start_sample:end_sample]
            tags = self._get_clap_tags(audio_slice, sr)
            l2_units.append(EvidenceUnit(
                audio_id=os.path.basename(audio_path),
                start_time=start_sec,
                end_time=end_sec,
                transcript=seg['text'].strip(),
                level="L2",
                audio_tags=tags,
                confidence=np.exp(seg.get('avg_logprob', -1.0)) # Whisper logprob 转概率近似
            ))

        # 4. 构建 L1 (粗粒度) - 基于时间窗口聚合 L2
        print(f"  - Aggregating L1 context windows...")
        num_windows = int(np.ceil(duration / chunk_len_l1))
        for i in range(num_windows):
            w_start = i * chunk_len_l1
            w_end = min((i + 1) * chunk_len_l1, duration)
            
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
            summary_text = self._summarize_text(full_text)
            l1_units.append(EvidenceUnit(
                audio_id=os.path.basename(audio_path),
                start_time=w_start,
                end_time=w_end,
                transcript=f"[SUMMARY] {summary_text}", # 加上标记以便区分
                level="L1",
                audio_tags=unique_tags,
                confidence=1.0 # 聚合层级暂定为1
            ))
            
        return {"L1": l1_units, "L2": l2_units}

if __name__ == "__main__":
    files = os.listdir("data/raw")
    for file in files:
        if not (file.endswith(".wav") or file.endswith(".mp3")):
            continue
            
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
            u = result['L2'][0]
            print(f"Time: {u.start_time:.1f}-{u.end_time:.1f}s")
            print(f"Text: {u.transcript}")
            print(f"Tags: {u.audio_tags}")
            
        if result['L1']:
            print("\n--- Sample L1 Unit (Coarse Summary) ---")
            u = result['L1'][0]
            print(f"Time: {u.start_time:.1f}-{u.end_time:.1f}s")
            print(f"Summary: {u.transcript}")
            print(f"Aggregated Tags: {u.audio_tags}")
