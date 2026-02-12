import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
# Note: eval_model from run_llava.py is designed for CLI use.
# For API usage, we'll reimplement the core logic here to be more flexible.

import requests
from PIL import Image
from io import BytesIO
import base64

class ModelService:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None

    def load_model(self):
        print(f"Loading model from {self.model_path}...")
        
        # 4-bit Quantization Config for 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

        try:
            # Use LLaVA's loader with our config
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.model_path),
                load_8bit=False,
                load_4bit=False, # We provide quantization_config via bnb_config
                device_map="auto",
                device=self.device,
                quantization_config=bnb_config
            )
            print("Model loaded successfully in 4-bit precision!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def process_image(self, image_data):
        # Handle base64 or raw bytes
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        return Image.open(BytesIO(image_bytes)).convert('RGB')

    def predict(self, image_input, prompt="Describe this ECG image.", conv_mode="llava_v1"):
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # 1. Process Image
            image = self.process_image(image_input)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            # 2. Build Prompt / Conversation
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

            if self.model.config.mm_use_im_start_end:
                 # Depending on model config slightly adjust prompt structure
                 # For PULSE (LLaVA-v1.6), usually just <image>\nPrompt is fine
                 pass

            qs = prompt
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            # Use specified conversation mode
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()

            # 3. Tokenize
            input_ids = tokenizer_image_token(prompt_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            # 4. Generate
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2, # Low temp for more deterministic output
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # 5. Decode
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            
            return {"inference": outputs.strip()}

        except Exception as e:
            print(f"Error during inference: {e}")
            return {"error": str(e)}
