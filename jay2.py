import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import Dataset
import shutil
import os

# 1. 설정
MODEL_ID = "/mnt/aix7101/joo/model/base_model"
OUTPUT_DIR = "./submit/model"

# 2. 데이터셋 (리스트를 정식 Dataset 객체로 변환)
calibration_text_list = [
    # 한국어
    "인공지능 경량화는 모델의 크기를 줄이는 기술입니다.",
    "딥러닝 모델의 파라미터를 줄이면 추론 속도가 빨라집니다.",
    "LG AI Research의 EXAONE 모델은 뛰어난 성능을 보여줍니다.",
    "양자화는 정밀도를 FP16에서 INT8로 낮추는 과정입니다.",
    "해커톤에서 좋은 성과를 거두기 위해 최적화를 수행합니다.",
    "vLLM은 매우 빠른 추론 속도를 제공하는 라이브러리입니다.",
    "데이터셋을 사용하여 모델의 활성화 분포를 분석합니다.",
    "파이썬과 파이토치를 사용하여 모델을 학습시킵니다.",
    # 영어 추가
    "The transformer architecture uses self-attention mechanisms.",
    "Quantization reduces memory and computational costs.",
    "Large language models are changing how we use technology.",
    "Explain the difference between pruning and quantization.",
] * 16

dataset = Dataset.from_dict({"text": calibration_text_list})

# 3. 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto"
)

# 4. 양자화 레시피
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="W8A8", 
        ignore=["lm_head"]
    )
]

# 5. 양자화 실행
print("양자화 시작...")
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=len(calibration_text_list),
    max_seq_length=128
)

# 6. 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"완료! {OUTPUT_DIR}에 저장되었습니다.")

# ============================================================
# 제출 파일 생성
# ============================================================
print("[INFO] submit.zip 생성 중...")
shutil.make_archive("submit", "zip", root_dir="./submit")

zip_size = os.path.getsize("submit.zip") / (1024**3)
print(f"[INFO] submit.zip: {zip_size:.2f} GB")
print("[INFO] ✅ 완료!")


