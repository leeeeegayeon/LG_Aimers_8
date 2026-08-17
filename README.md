# 🚀 EXAONE-4.0-1.2B Model Optimization

> **LG Aimers 8기 Online Hackathon (Phase 2)**
> EXAONE-4.0-1.2B 모델의 성능을 최대한 유지하면서 추론 효율을 개선하기 위한 LLM 경량화 프로젝트입니다.

---

## 📌 Project Overview

본 프로젝트는 LG AI Research의 `EXAONE-4.0-1.2B`를 대상으로 다양한 경량화 전략을 실험하고, **모델 성능과 추론 속도 사이의 최적 trade-off**를 탐색한 프로젝트입니다.

온라인 해커톤에서는 `vLLM` 라이브러리 자체 수정이 허용되지 않았기 때문에, Hugging Face 표준 방식으로 저장되는 **모델 가중치 및 config 수준의 최적화**를 중심으로 실험했습니다.

### 핵심 목표

* Base model 대비 성능 최대한 유지
* Token당 추론 시간 감소
* NVIDIA L4 환경에서 효율적인 Quantization 탐색
* vLLM과 호환되는 Hugging Face 모델 생성

---

# 🏆 Hackathon Result

최종적으로 다양한 Quantization 및 fine-tuning 전략을 실험했습니다.

| 항목                 |                 결과 |
| ------------------ | -----------------: |
| Base Model         |    EXAONE-4.0-1.2B |
| Best Public Score  |         **0.618+** |
| Final Public Score |       **0.62 후반대** |
| Main Optimization  |       Quantization |
| Inference Engine   |               vLLM |
| Evaluation GPU     | NVIDIA L4 22.4 GiB |

> Public Score는 모델 성능과 token당 추론 시간 개선 정도를 함께 반영한 해커톤 평가 지표입니다.

---

# 🧠 Base Model

## EXAONE-4.0-1.2B

LG AI Research에서 공개한 Small-scale EXAONE 모델을 사용했습니다.

```text
LGAI-EXAONE/EXAONE-4.0-1.2B
```

단순히 parameter 수를 줄이는 방식은 정확도 손실이 커질 수 있기 때문에, 모델 구조 자체를 변경하기보다 **Quantization을 통해 weight precision을 낮추는 방향**을 중심으로 접근했습니다.

---

# 🖥️ Evaluation Environment

해커톤 평가 서버 환경은 다음과 같습니다.

| Component  | Specification      |
| ---------- | ------------------ |
| OS         | Ubuntu 22.04.5 LTS |
| CPU        | 6 vCPU             |
| RAM        | 28 GB              |
| GPU        | NVIDIA L4          |
| VRAM       | 22.4 GiB           |
| Python     | 3.11.14            |
| CUDA       | 12.8               |
| Inference  | vLLM               |
| Time Limit | 20 min             |

제출물은 운영진이 제공하는 동일한 vLLM 환경에서 실행되며, 모델 로드부터 전체 평가 과정은 20분 이내에 완료되어야 합니다.

평가 점수에서는 전체 실행 시간이 아니라 실제 benchmark inference 과정에서 측정되는 **token당 inference time**이 사용됩니다.

---

# 🔬 Optimization Experiments

프로젝트에서는 다음과 같은 경량화 전략을 실험했습니다.

```text
EXAONE-4.0-1.2B
        │
        ├── GPTQ
        │     └── W4A16
        │
        ├── AWQ
        │     ├── INT8
        │     └── INT4
        │
        ├── Mixed Precision
        │     └── Sensitive Layer FP16 Protection
        │
        └── LoRA / QLoRA
              └── NF4 + LoRA
```

각 방법의 정확도와 inference efficiency를 비교하면서 최종 모델을 탐색했습니다.

---

# 1️⃣ GPTQ Quantization

첫 번째 주요 접근은 **GPTQ 기반 Weight Quantization**이었습니다.

대표적으로 다음 설정을 실험했습니다.

```text
Weight : INT4
Activation : FP16

→ W4A16
```

Weight precision을 FP16에서 INT4로 낮춰 모델 크기와 memory bandwidth를 감소시키는 것을 목표로 했습니다.

### 관찰

GPTQ를 통해 모델을 상당히 압축할 수 있었지만, 단순히 모든 Linear Layer를 동일하게 INT4로 변환하는 경우 정확도 손실이 발생할 수 있었습니다.

따라서 이후에는 모든 Layer를 동일하게 처리하기보다 **Quantization에 민감한 Layer를 보호하는 전략**도 함께 실험했습니다.

---

# 2️⃣ AWQ Quantization

두 번째 주요 접근은 **AWQ(Activation-aware Weight Quantization)**였습니다.

실제 제출 모델 생성 과정에서 `llmcompressor`의 `AWQModifier`를 사용했습니다.

## AWQ INT4 Configuration

```python
{
    "targets": ["Linear"],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "strategy": "group",
        "group_size": 128,
        "symmetric": False
    }
}
```

즉,

```text
Weight Precision     : INT4
Quantization Strategy: Group-wise
Group Size           : 128
Symmetric            : False
Target               : Linear Layers
Ignored Layer        : lm_head
```

의 설정으로 모델을 Quantization했습니다.

---

# 3️⃣ Calibration Dataset

AWQ Quantization을 위해 EXAONE 학습 데이터인 `MANTA-1M` 일부를 calibration dataset으로 사용했습니다.

```text
Dataset:
LGAI-EXAONE/MANTA-1M

Calibration Samples:
64

Maximum Sequence Length:
256
```

실제 설정은 다음과 같습니다.

```python
NUM_CALIBRATION_SAMPLES = 64
MAX_SEQUENCE_LENGTH = 256
```

MANTA-1M의 conversation 데이터를 EXAONE의 chat template에 맞게 변환한 후 calibration에 사용했습니다.

```python
tokenizer.apply_chat_template(
    ex["conversations"],
    add_generation_prompt=True,
    tokenize=False
)
```

---

# 4️⃣ AWQ Auto Mapping

EXAONE 구조에 맞는 AWQ mapping을 직접 고정하기보다 먼저 `llmcompressor`의 **automatic mapping inference**를 사용했습니다.

```python
AWQModifier(
    mappings=None,
    ignore=["lm_head"],
    config_groups=AWQ_CONFIG_GROUPS
)
```

실행 결과 automatic mapping을 통한 AWQ calibration이 정상적으로 완료되었습니다.

```text
[AWQ] Trying auto mappings...
...
Compression lifecycle finalized
[AWQ] Auto-mapping AWQ done ✅
```

다만 calibration 과정에서 EXAONE의 일부 `v_proj → o_proj` 조합에 대해 incompatible balance layer warning이 발생했습니다.

```text
skipping AWQ for model.layers.*.self_attn.v_proj
because found incompatible balance layers
```

이는 EXAONE 구조와 llmcompressor가 자동 추론한 AWQ mapping 사이에 일부 구조적 차이가 존재했음을 보여줍니다.

---

# 5️⃣ Mixed Precision Strategy

모든 Layer를 동일한 precision으로 Quantization하는 방식뿐만 아니라 **Quantization-sensitive Layer를 높은 precision으로 유지하는 전략**도 실험했습니다.

기본 아이디어는 다음과 같습니다.

```text
Quantization-sensitive Layers
           │
           ▼
       FP16 유지

Less-sensitive Layers
           │
           ▼
      INT8 / INT4
```

이를 통해 전체 모델을 공격적으로 INT4로 변환했을 때 발생할 수 있는 정확도 하락을 줄이고자 했습니다.

### 핵심 아이디어

```text
Maximum Compression
        ↓
높은 inference efficiency
        ↓
하지만 accuracy degradation 증가

Selective Quantization
        ↓
일부 compression 포기
        ↓
Accuracy 보존
```

따라서 단순히 가장 작은 모델을 만드는 것이 아니라 **score 관점에서 최적의 precision allocation을 찾는 것**을 목표로 했습니다.

---

# 6️⃣ LoRA / QLoRA Experiment

Quantization 이후 성능을 다시 회복하기 위한 방법으로 **LoRA 기반 fine-tuning**도 실험했습니다.

주요 설정 중 하나는 다음과 같습니다.

```text
Quantization : NF4
LoRA Rank    : 16
```

그러나 실험 결과 LoRA fine-tuning이 항상 leaderboard 성능 향상으로 연결되지는 않았습니다.

추가 학습 과정에서 기존 모델이 가지고 있던 일반적인 reasoning 능력이 영향을 받을 수 있었으며, 최종적으로는 **Quantization 자체를 최적화하는 방향이 더 효과적**이라고 판단했습니다.

---

# 📊 Experiment Summary

전체적인 실험 흐름을 정리하면 다음과 같습니다.

| Experiment | Method                    | Precision       | 주요 결과                 |
| ---------- | ------------------------- | --------------- | --------------------- |
| Baseline   | Original Model            | FP16/BF16       | 기준 모델                 |
| Exp. 1     | GPTQ                      | W4A16           | 모델 압축 및 속도 개선 실험      |
| Exp. 2     | AWQ                       | INT8            | 정확도 보존 중심             |
| Exp. 3     | AWQ                       | INT4            | 높은 압축률 및 inference 효율 |
| Exp. 4     | Mixed Precision           | FP16 + INT4     | 민감 Layer 보호           |
| Exp. 5     | LoRA/QLoRA                | NF4 + LoRA r=16 | 성능 하락 관찰              |
| Final      | Quantization Optimization | Mixed           | **Public 0.62 후반대**   |

※ 개별 제출별 정확한 leaderboard score는 현재 보존된 notebook에 기록되어 있지 않아 임의로 작성하지 않았습니다.

---

# 🔍 AWQ Experiment Environment

실제 AWQ 실험 notebook에서 사용한 환경입니다.

```text
Python       : 3.12.12
PyTorch      : 2.9.0+cu128
Transformers : 4.57.3
NumPy        : 2.2.6
GPU          : Tesla T4
```

주의할 점은 **개발 환경과 실제 평가 환경이 다르다는 것**입니다.

```text
Development
    ↓
Tesla T4

Evaluation
    ↓
NVIDIA L4
```

따라서 로컬/Colab에서의 실행 속도가 실제 leaderboard의 token당 inference time과 반드시 동일하지는 않습니다.

---

# 📦 Model Export

Quantization이 끝난 모델은 Hugging Face 표준 형식으로 저장했습니다.

```python
model.save_pretrained(
    OUT_DIR,
    save_compressed=True
)

tokenizer.save_pretrained(OUT_DIR)
```

제출 파일은 다음과 같은 구조를 가지도록 생성했습니다.

```text
submit.zip
└── model/
    ├── config.json
    ├── generation_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── model.safetensors
    └── ...
```

실제로 notebook에서 zip 내부의 최상위 구조까지 검증했습니다.

```python
assert top == ["model"]
```

이를 통해 제출 파일이 반드시 `model/` directory만을 최상위에 포함하도록 구성했습니다.

---

# 💡 Key Findings

## 1. 낮은 bit가 항상 높은 점수를 의미하지 않는다

INT4는 모델 크기를 크게 줄일 수 있지만 accuracy degradation도 함께 발생할 수 있습니다.

따라서

```text
Smaller Model ≠ Better Score
```

였습니다.

---

## 2. Quantization은 Layer마다 영향이 다르다

모든 Linear Layer가 Quantization에 동일하게 반응하지 않았습니다.

일부 Layer는 precision 감소에 상대적으로 민감하기 때문에 중요한 Layer를 높은 precision으로 보호하는 **Mixed Precision 전략**을 고려했습니다.

---

## 3. LoRA가 반드시 성능을 복구하지 않는다

Quantization으로 감소한 성능을 fine-tuning으로 복구하려 했지만, LoRA/QLoRA 실험에서는 오히려 leaderboard 성능이 감소하는 경우가 있었습니다.

따라서 fine-tuning을 추가하는 것보다 **원본 모델의 능력을 최대한 보존하는 Quantization configuration 탐색**이 중요했습니다.

---

## 4. 모델 크기와 inference speed는 다르다

모델 파일이 작아졌다고 해서 반드시 token당 inference time이 같은 비율로 감소하는 것은 아닙니다.

실제 속도는

* GPU architecture
* Quantization kernel
* Memory bandwidth
* vLLM 지원 여부
* Weight packing 방식

등의 영향을 함께 받습니다.

따라서 모델 크기만 비교하기보다 **실제 vLLM 평가 환경에서의 inference efficiency**를 고려해야 했습니다.

---

## 5. Accuracy와 Speed의 Trade-off가 핵심이다

이번 해커톤의 핵심은 가장 작은 모델을 만드는 것이 아니라,

```text
Accuracy Preservation
        +
Inference Acceleration
        ↓
Best Leaderboard Score
```

를 만드는 것이었습니다.

Quantization 강도를 높일수록 inference efficiency는 좋아질 수 있지만 accuracy가 감소할 수 있기 때문에 두 요소의 균형점을 찾는 것이 가장 중요했습니다.

---

# 📁 Repository Structure

```text
EXAONE-4.0-Optimization/
│
├── README.md
│
├── notebooks/
│   ├── test1.ipynb
│   └── test2.ipynb
│
├── docs/
│   └── LG_Aimers_Hackathon.pdf
│
└── results/
    └── experiment_results.md
```

### `test1.ipynb`

AWQ INT4 Quantization을 적용하고 EXAONE 모델과 `llmcompressor` 간의 compatibility를 검증한 실험입니다.

* MANTA-1M calibration
* 64 calibration samples
* sequence length 256
* AWQ INT4
* group size 128
* automatic AWQ mapping
* compressed model export

### `test2.ipynb`

AWQ INT4 설정을 정리하여 실제 제출 모델을 생성한 notebook입니다.

* AWQ INT4
* group size 128
* asymmetric quantization
* `lm_head` 제외
* Hugging Face compressed model 저장
* 제출용 zip 생성
* zip directory structure 검증

---

# 🏁 Conclusion

본 프로젝트에서는 EXAONE-4.0-1.2B를 대상으로 GPTQ, AWQ, Mixed Precision, LoRA/QLoRA 등 다양한 LLM 경량화 전략을 실험했습니다.

단순히 모델의 bit-width를 낮추는 것보다 **어떤 Layer를 어떤 precision으로 유지할 것인지**, 그리고 해당 Quantization 방식이 **실제 vLLM 및 NVIDIA L4 환경에서 얼마나 효율적으로 동작하는지**가 중요하다는 점을 확인했습니다.

최종적으로 모델의 성능과 inference efficiency 사이의 trade-off를 조정하며 **Public Score 0.62 후반대**까지 도달했습니다.

이 프로젝트를 통해 단순 모델 Quantization뿐만 아니라 **LLM architecture, GPU hardware, inference engine을 함께 고려해야 실제 inference optimization으로 이어진다**는 점을 경험했습니다.

---

## 👩‍💻 Author

**이가연**

Sungshin Women's University
Department of AI
