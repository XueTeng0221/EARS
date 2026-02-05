import torch
from transformers import AutoModelForSeq2SeqLM

def _register_summarization_pipeline_for_transformers_v5() -> None:
    """在 transformers==5.0.0 里手动注册 summarization pipeline。
    transformers v5 默认移除了 "summarization" task（SUPPORTED_TASKS 里没有），
    但项目代码仍希望能用 pipeline("summarization", ...)。
    这里通过向 transformers.pipelines.SUPPORTED_TASKS 注入一个最小实现来兼容。
    """
    try:
        from transformers import pipelines
        from transformers.pipelines import Pipeline

        if "summarization" in getattr(pipelines, "SUPPORTED_TASKS", {}):
            return

        class _LocalSummarizationPipeline(Pipeline):
            def _sanitize_parameters(self, **kwargs):
                forward_params = {}
                for k in (
                    "max_length",
                    "min_length",
                    "do_sample",
                    "num_beams",
                    "temperature",
                ):
                    if k in kwargs and kwargs[k] is not None:
                        forward_params[k] = kwargs[k]
                return {}, forward_params, {}

            def preprocess(self, inputs):
                if isinstance(inputs, dict):
                    text = inputs.get("text") or inputs.get("inputs")
                else:
                    text = inputs
                if not isinstance(text, str):
                    raise TypeError(f"summarization expects str input, got {type(text)}")

                encoded = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                return {k: v.to(self.device) for k, v in encoded.items()}

            def _forward(self, model_inputs, **generate_kwargs):
                with torch.no_grad():
                    output_ids = self.model.generate(**model_inputs, **generate_kwargs)
                return output_ids

            def postprocess(self, model_outputs):
                summary = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
                return [{"summary_text": summary}]

        pipelines.SUPPORTED_TASKS["summarization"] = {
            "impl": _LocalSummarizationPipeline,
            "pt": (AutoModelForSeq2SeqLM,),
            # 多语种摘要默认模型（可在 dataset.py 里通过 EARS_SUMMARIZER_MODEL 覆盖）
            "default": {"model": ("csebuetnlp/mT5_multilingual_XLSum", "main")},
            "type": "text",
        }
        print("[INFO] Registered custom transformers pipeline task: summarization")
    except Exception as e:
        # 不要在 import 阶段 hard-fail；后面还有 manual fallback
        print(f"[WARN] Failed to register summarization pipeline: {e}")