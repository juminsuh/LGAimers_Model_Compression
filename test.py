import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 경로
model_path = "./submit/model"

# 1. 토크나이저 & 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 채팅 형식으로 메시지 구성
messages = [
    {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": "인공지능 경량화란 무엇인가요?"}
]

# 3. 채팅 템플릿 적용
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# 4. 추론 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 5. 결과 출력
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))