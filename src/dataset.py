import os
import json
import librosa
import whisper
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
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
    def __init__(self, whisper_size="base", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing AudioProcessor on {self.device}...")
        
        # 1. 加载 Whisper (ASR)
        print(f"Loading Whisper ({whisper_size})...")
        self.asr_model = whisper.load_model(whisper_size, device=self.device)
        
        # 2. 加载 CLAP (音频事件/情感检测)
        # 使用 LAION 的官方 Checkpoint
        print("Loading CLAP (laion/clap-htsat-unfused)...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
        self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        
        # 定义 CLAP 的候选标签（Zero-shot 分类用）
        # 可以根据具体场景扩展，例如加入 "angry speech", "sad tone" 等
        self.clap_labels = [
            "neutral speech", "angry speech", "happy speech/laughter", 
            "applause", "silence", "background noise", "music", "typing on keyboard"
        ]
        # 预计算标签的文本 Embeddings 以加速
        inputs = self.clap_processor(text=self.clap_labels, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.label_embeddings = self.clap_model.get_text_features(**inputs)
            # 归一化以便计算余弦相似度
            self.label_embeddings = self.label_embeddings / self.label_embeddings.norm(p=2, dim=-1, keepdim=True)

        # 3. 加载 Summarizer (用于 L1 摘要)
        # 使用 distilbart-cnn-12-6 比较轻量，适合 CPU/单卡
        print("Loading Summarizer...")
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if device=="cuda" else -1)

    def _get_clap_tags(self, audio_fragment, sr) -> List[str]:
        """
        对音频片段进行 Zero-shot 分类
        """
        # CLAP 模型通常需要 48kHz，如果不是则需要重采样 (这里简化处理，假设输入会被 processor 处理)
        # 注意：transformers 的 CLAP processor 需要 raw audio data
        if len(audio_fragment) < 1000: # 太短忽略
            return []

        # 处理音频输入
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

    def _summarize_text(self, text: str) -> str:
        """对长文本进行摘要"""
        if len(text.split()) < 30: # 太短不摘要
            return text
        try:
            # 限制输入长度，防止显存溢出
            input_text = text[:1024] 
            summary = self.summarizer(input_text, max_length=60, min_length=10, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:200] + "..."

    def process_audio(self, audio_path: str, chunk_len_l1=60) -> Dict[str, List[EvidenceUnit]]:
        """
        全流程处理：加载 -> ASR -> 切片 -> CLAP特征 -> L2构建 -> L1聚合 -> 摘要
        """
        # 1. 加载完整音频
        # Whisper 内部会重采样到 16k，但 CLAP 最好用 48k。
        # 为了方便，我们这里加载一次高采样率，ASR 时 whisper 会自己处理路径，CLAP 用 librosa 数组
        y, sr = librosa.load(audio_path, sr=48000) 
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
            
            # 提取对应的音频切片用于 CLAP
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            # 边界保护
            end_sample = min(end_sample, len(y))
            audio_slice = y[start_sample:end_sample]
            
            # 获取音频标签 (例如: ["angry speech", "background noise"])
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
            current_window_l2s = [
                u for u in l2_units 
                if w_start <= (u.start_time + u.end_time)/2 < w_end
            ]
            
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

# ---------------------------------------------------------
# 测试/演示代码
# ---------------------------------------------------------
if __name__ == "__main__":
    # 创建模拟数据目录和文件（如果不存在）
    test_dir = "data/test_raw"
    os.makedirs(test_dir, exist_ok=True)
    
    # 尝试生成一个正弦波音频文件用于测试（如果没有真实文件）
    test_file = os.path.join(test_dir, "test_tone.wav")
    if not os.path.exists(test_file):
        print("Generating dummy audio file for testing...")
        import soundfile as sf
        sr = 48000
        t = np.linspace(0, 10, sr*10) # 10秒
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440Hz 音
        sf.write(test_file, audio, sr)

    # 初始化处理器
    processor = AudioProcessor(whisper_size="tiny") # 使用 tiny 模型以加快演示速度
    
    # 处理
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
