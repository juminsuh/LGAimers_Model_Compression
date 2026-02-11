import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import Dataset, load_dataset

# ==========================================
# 1. 기본 설정
# ==========================================
MODEL_ID = "./base_model"
OUTPUT_DIR = "./submit_be/model"

# ==========================================
# 2. 데이터셋 준비 (사용자 맞춤형)
# ==========================================
# 사용자님께서 준비하신 고품질 데이터셋을 아래 리스트에 채워주세요.
# 텍스트가 다양하고 길수록(최대 2048토큰) 오차 보상이 정교해집니다.
# (최소 128개 이상의 문장을 권장합니다.)
my_custom_data = [
    "인공지능 경량화는 모델의 크기를 줄이는 기술입니다.",
    "딥러닝 모델의 파라미터를 줄이면 추론 속도가 빨라집니다.",
    "LG AI Research의 EXAONE 모델은 뛰어난 성능을 보여줍니다.",
    "양자화는 정밀도를 FP16에서 INT8로 낮추는 과정입니다.",
    "해커톤에서 좋은 성과를 거두기 위해 최적화를 수행합니다.",
    "vLLM은 매우 빠른 추론 속도를 제공하는 라이브러리입니다.",
    "데이터셋을 사용하여 모델의 활성화 분포를 분석합니다.",
    "파이썬과 파이토치를 사용하여 모델을 학습시킵니다."
] * 16 
# llmcompressor가 읽을 수 있도록 Dataset 객체로 변환
#dataset = Dataset.from_dict({"text": my_custom_data})
raw_dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
subset_dataset = raw_dataset.take(1000)
def format_data(example):
    # 'question' 내용만 'text'라는 키로 반환
    return {"text": example['question']}
dataset_stream = subset_dataset.map(format_data, remove_columns=['system_prompt', 'question', 'response', 'id'])
data_list = list(dataset_stream)
dataset = Dataset.from_list(data_list)


# ==========================================
# 3. 모델 로드
# ==========================================
print("원본 16비트 모델을 불러오는 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto" 
)

# ==========================================
# 4. 양자화 레시피 설정 (사용자 아이디어 집약)
# ==========================================
recipe = [
    GPTQModifier(
        targets="Linear",     # 모든 행렬 곱셈 층 타겟
        scheme="W4A16",       # 가중치는 4비트로 깎고, 연산은 16비트로 수행 (vLLM 최고 효율)
        ignore=["lm_head"],   # 최종 출력층은 16비트 유지 (정확도 보호)
        
        # [핵심 1] 오차 보상 연산 (Hessian 기반)
        sequential_update=True, 
        
        # [핵심 2] 튀는 값을 무시하고 밀집된 곳을 잡기 위한 댐핑 비율
        # (이전에는 에러가 났지만 GPTQModifier에서는 정상 작동하는 고급 옵션입니다)
        dampening_frac=0.01   
    )
]

# ==========================================
# 5. 양자화 실행 (오랜 시간 소요)
# ==========================================
# 각 층마다 16비트와 4비트를 비교하며 오차를 보상하는 연산이 들어갑니다.
# RTX 3060 환경에서 15분 ~ 30분 정도 걸릴 수 있습니다.
print(f"GPTQ 4비트 양자화 시작! (데이터 {len(my_custom_data)}개 기준)")
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=len(my_custom_data),
    max_seq_length=1024 # 메모리가 부족하면 512로 줄이세요
)

# ==========================================
# 6. 최종 파일 저장
# ==========================================
print("양자화 완료! 파일을 저장합니다...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"모든 과정이 끝났습니다. 최종 모델이 '{OUTPUT_DIR}'에 저장되었습니다.")