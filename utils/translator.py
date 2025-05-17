import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

class Translator:
    def __init__(self, ckpt_dir="ai4bharat/indictrans2-en-indic-1B", quantization=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer, self.model = self._init_model(ckpt_dir, quantization)
        self.processor = IndicProcessor(inference=True)
        self.batch_size = 4

    def _init_model(self, ckpt_dir, quantization):
        config = None
        if quantization in ["4-bit", "8-bit"]:
            from transformers import BitsAndBytesConfig
            config = BitsAndBytesConfig(**{
                "4-bit": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                "8-bit": {
                    "load_in_8bit": True,
                    "bnb_8bit_use_double_quant": True,
                    "bnb_8bit_compute_dtype": torch.bfloat16,
                }
            }[quantization])
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            quantization_config=config,
            low_cpu_mem_usage=True
        )
        if not config:  # Full precision
            model = model.to(self.device)
            if self.device == "cuda":
                model.half()
        model.eval()
        return tokenizer, model

    def translate_batch(self, sentences, src_lang="eng_Latn", tgt_lang="hin_Deva"):
        translations = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            processed_batch = self.processor.preprocess_batch(batch, src_lang, tgt_lang)
            
            inputs = self.tokenizer(
                processed_batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_length=256, num_beams=5)
            
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            translations += self.processor.postprocess_batch(decoded, tgt_lang)
            
            del inputs
            torch.cuda.empty_cache()
        
        return translations