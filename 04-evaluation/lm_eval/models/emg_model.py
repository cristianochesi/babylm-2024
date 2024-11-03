import torch
import re
from typing import List, Tuple

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from transformers import PreTrainedTokenizerFast
from .emg_config import EMGLanguageModel 

@register_model("emg_lm")
class EMGLM(LM):
    def __init__(self, pretrained='emg_model', device=None, batch_size=1):
        super().__init__()
        self.pretrained = pretrained
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._batch_size = batch_size
        self._create_model()

    def _create_model(self):
        # Adjuct to match the saved model
        self.model = EMGLanguageModel(11381, 650, 650, 1).to(self.device)
        # Remove the "module." prefix from state dict keys
        loaded_state_dict = torch.load(f"{self.pretrained}/model.pt", map_location=self.device)
    
        # Remove the "module." prefix from state dict keys
        new_state_dict = {k.replace("module.", ""): v for k, v in loaded_state_dict.items()}
    
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.pretrained)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        # print("DEBUGGING\nEvaluated sentence\tScore assigned (loglikelihood)")   # debugging
        for instance in requests:
            context, continuation = instance.args
            full_text = context + continuation
            full_text = re.sub(r'([.,!?])', r' \1 ', full_text.lower())  # the following 3 lines remove any metalinguistic information included in BLiMP, set everything to lowercase and space relevant punctuation
            full_text = re.sub(r'a:|b:|david:|sarah:', '', full_text)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            input_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
            target_ids = input_ids.clone()
            num_tokens = input_ids.shape[1]
            
            with torch.no_grad():
                output, _ = self.model(input_ids)  # this is for standard output
                # output, _, __ = self.model(input_ids) # this is for EMG overt nested output (EMG_nest_02)
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1), reduction='sum')
                loglikelihood = -loss.item()/num_tokens

            # print(full_text + "\t" + str(loglikelihood))                # debugging
            results.append((loglikelihood, False))                      # False indicates it's not a greedy decode
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return self.loglikelihood(requests)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for instance in requests:
            context, until = instance.args
            max_length = 512  # Or any other suitable value
            input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            
            generated = input_ids
            for _ in range(max_length - len(input_ids[0])):
                with torch.no_grad():
                    output, _ = self.model(generated)  # this is for standard output
                    # output, _, __ = self.model(generated) # this is for EMG overt nested output (EMG_nest_02)
                    next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
                    generated = torch.cat([generated, next_token], dim=1)
                
                decoded = self.tokenizer.decode(generated[0])
                if until in decoded:
                    break
            
            results.append(decoded)
        return results

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 512  # You may want to adjust this based on your model's capabilities

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value