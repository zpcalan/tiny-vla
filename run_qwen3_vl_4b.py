from typing import Any


from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from transformers import BitsAndBytesConfig
from modules import TimeStepEmbed, TransformerBlock, AdaLN

import argparse

parser = argparse.ArgumentParser(description='Input file paths')
parser.add_argument('--model-path', type=str, default='/data/k8s/zpc/tiny-vla/qwen3-vl-2b-instruct')
parser.add_argument('--image-file-path', type=str, default='/data/k8s/zpc/tiny-vla/random_shot.jpg')

args = parser.parse_args()

class Qwen3VLProvider:
    def __init__(self, model_path):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # 推理用fp16
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # 推荐
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
        )
        self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def one_step_infer(self):
        # print(self.model)
        input_text = "Describe this image"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": args.image_file_path,
                    },
                    # {
                    #     "type": "image",
                    #     "image": "E:/我的项目/tiny-vla/cat1.jpg",
                    # },
                    {
                        "type": "text",
                        "text": "Describe this image",
                    },
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info([messages])  # 关键步骤
        # print(
        #     self.processor.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True
        #     )
        # )
        inputs = self.processor(
            text=[
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            ],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # pic_1_list = [
        #     inputs.pixel_values[i].reshape(3, 2, 16, 16)[:, 0, :, :] # shape是[c, temporal, h, w],所以在第一维重复。
        #     for i in range(9600)
        # ]
        # all_pixels = (
        #     (
        #         torch.stack(pic_1_list)
        #         .reshape(40, 60, 2, 2, 3, 16, 16) #因为是分2x2 block-wise进行patch和flatten的，所以这里不是row-major。
        #         .permute(0, 2, 1, 3, 4, 5, 6)
        #     )
        #     .reshape(80, 120, 3, 16, 16)
        #     .permute(2, 0, 3, 1, 4)
        #     .reshape(3, 1280, 1920)
        #     .permute(1, 2, 0)
        #     .contiguous()
        # )
        # Image.fromarray((all_pixels.numpy() * 255).astype(np.uint8)).save("output.jpg")
        inputs = dict[Any, Any](inputs)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        # with torch.no_grad():
        #     response = ""
        #     for j in range(222):
        #         logits = self.model(**inputs, use_cache=True).logits
        #         next_token_logits = logits[:, -1, :]
        #         values, indices = torch.topk(next_token_logits, 3, dim=-1)
        #         # for v, i in zip(values[0], indices[0]):
        #         #     print("Try to sample for top3 tokens:", i, v, self.tokenizer.decode(i))
        #         if j != 0:
        #             inputs["input_ids"] = torch.cat(
        #                 [inputs["input_ids"], indices[0][1].view(1, 1)], dim=-1
        #             )
        #             response += self.tokenizer.decode(indices[0][1])
        #             print("====response:", response)
        #         else:
        #             inputs["input_ids"] = torch.cat(
        #                 [inputs["input_ids"], indices[0][0].view(1, 1)], dim=-1
        #             )
        #         inputs["attention_mask"] = torch.cat(
        #             [inputs["attention_mask"], Tensor([[1]])], dim=-1
        #         )
        #         # inputs["attention_mask"] = torch.cat(
        #         #     [inputs["attention_mask"], Tensor([[1]]).cuda()], dim=-1
        #         # )
        # print(self.tokenizer.decode(inputs["input_ids"][0]))
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=500,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        # )
        # print(outputs[0])
        # print(self.tokenizer.decode(outputs[0]))
        hidden_states = self.model(**inputs, use_cache=True, output_hidden_states=True).hidden_states
        return hidden_states[-1]
        # print(self.model.parameters)
        # total = 0
        # for name, param in self.model.named_parameters():
        #     print(
        #         name,
        #         param.dtype,
        #         param.numel(),
        #         param.element_size(),
        #         param.element_size() * param.numel(),
        #     )
        #     total += param.element_size() * param.numel()
        # print(total)
        # print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        # print(
        #     f"Reserved (by PyTorch pool): {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        # )
        # import time

        # time.sleep(10000)




class DiTProvider(nn.Module):
    def __init__(self, time_dim, hidden_dim, action_dim, num_layers):
        super().__init__()
        # For current version, we do not add state, for this is simple action scenario
        # 1.Encode for timestep as embedding vector. This timestep embedding will be applied to AdaLN for each norm op.
        self.timestep_encoder = TimeStepEmbed(time_dim)
        # 2.Project x_t noise to hidden_state.
        self.noise_proj = nn.Linear(action_dim, hidden_dim)
        # 3.Transformer blocks. Because there's vlm encoder hidden states, use cross attn.
        blocks = [
            TransformerBlock(hidden_dim, 4, time_dim)
            for _ in range(num_layers)
        ]
        self.transformer_blocks = nn.ModuleList(blocks)

        # 4.Get shift and scale for AdaLN of the last hidden_state.
        print("after block:", time_dim, hidden_dim)
        self.output_ada_ln = AdaLN(time_dim, hidden_dim)

        # 5.Project action output.
        self.action_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t, encoder_hidden_state, time_step):
        # x is projection of noise epsilon.
        # The time_step should be a scalar.
        time_emb = self.timestep_encoder(time_step)
        hidden_states = self.noise_proj(x_t)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_state, time_emb)
        hidden_states = self.output_ada_ln(hidden_states, time_emb)
        output = self.action_proj(hidden_states)
        return output




if __name__ == "__main__":
    action_dim = 2
    qwen3_vl_provider = Qwen3VLProvider(args.model_path)
    dit_provider = DiTProvider(time_dim=1024, hidden_dim=2048, action_dim=action_dim, num_layers=5)
    dit_provider.to(torch.cuda.current_device())
    dit_provider = dit_provider.to(torch.bfloat16)
    print(dit_provider.timestep_encoder.mlp.linear_1.weight)
    vlm_last_hidden_state = qwen3_vl_provider.one_step_infer()
    noise = torch.normal(mean=0.0, std=1.0, size=[1, action_dim], dtype=torch.bfloat16).cuda()
    output = dit_provider(x_t=noise, encoder_hidden_state=vlm_last_hidden_state, time_step=Tensor([2]).cuda())
    print(output)
