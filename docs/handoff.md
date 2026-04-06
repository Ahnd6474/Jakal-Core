# Jakal-Core Handoff

## 목적

이 문서는 Jakal-Core의 최종 목표와, 다음 작업자가 바로 이어서 구현할 수 있도록 하기 위한 범위 정의 초안이다.

이 프로젝트의 목표는 저사양 노트북 같은 환경에서도 LLM 추론을 실제로 돌릴 수 있는 이기종 실행 런타임을 만드는 것이다. 핵심은 GPU 의존성을 강화하는 것이 아니라, CPU를 다시 주요 계산 자원으로 끌어올리고 CPU와 GPU를 함께 써서 현실적인 추론 경로를 만드는 데 있다.

성능 우위의 완전한 증명은 당장 1순위가 아니다. 먼저 실제 장비에서 설치하고 실행할 수 있는 완성품을 만들고, 그 위에서 CPU와 GPU의 새로운 분업 방식을 검증하는 것이 우선이다.

## 현재 전제

- 개발 환경은 `CPU + NVIDIA GPU + Intel GPU` 조합이다.
- 벤더 종속적인 단일 스택보다는, 여러 장치를 함께 다룰 수 있는 구조를 지향한다.
- 이 레포는 현재 구조 그래프, planner, execution report, direct execution 실험 코드까지는 갖고 있다.
- 아직 LLM 전용 runtime이나 모델 로더, KV cache 실구현, 실사용 inference loop는 없다.

## 최종 목표

1. 저사양 장비에서도 작은 LLM이 실제로 응답을 생성하게 만든다.
2. CPU를 단순 fallback이 아니라 주요 계산 자원으로 다시 활용한다.
3. CPU, NVIDIA GPU, Intel GPU를 섞어서 연산을 분담하는 실행 방식을 만든다.
4. 특정 회사 SDK에 완전히 묶이지 않는 범용 구조를 유지한다.
5. 최고 성능보다도 실제로 끝까지 돌아가는 완성품을 우선한다.

## 핵심 문제 정의

이 프로젝트는 아래 질문들에 답하는 방향으로 진행한다.

1. LLM 추론에서 CPU가 실제로 맡아야 이득인 연산은 무엇인가?
2. GPU가 항상 유리하지 않은 구간은 어디인가?
3. prefill과 decode를 서로 다른 장치 조합으로 분리하는 것이 유효한가?
4. 하나의 matmul 또는 attention 경로를 CPU와 GPU가 나눠 계산하는 방식이 실제로 가능한가?
5. Intel GPU와 NVIDIA GPU를 같은 planner 아래에서 다룰 수 있는가?

## 초기 제품 정의

초기 목표는 범용 딥러닝 프레임워크가 아니다. 다음 조건을 만족하는 작은 추론 실행기를 먼저 만든다.

- 단일 모델
- 단일 사용자 세션
- 배치 1
- CLI 실행
- 토큰 스트리밍 출력
- 실패 시 CPU fallback

즉, "빠른 범용 엔진"보다 "끝까지 답을 생성하는 작동 예제"가 먼저다.

## MVP 범위

### 지원 환경

- `CPU only`
- `CPU + Intel GPU`
- `CPU + NVIDIA GPU`
- 가능하면 이후 `CPU + Intel GPU + NVIDIA GPU`

### 첫 대상 작업

- small LLM 1종
- 양자화 포맷 1종
- 단일 프롬프트 입력
- 단일 응답 생성

### 반드시 필요한 기능

- 모델 로드
- 토큰 생성 loop
- prefill / decode 분리
- 장치 선택 옵션
- 실행 로그 출력
- fallback 경로

## CPU 활용 가설

이 프로젝트의 핵심은 CPU를 다시 계산 자원으로 쓰는 것이다. 아래 항목을 우선 실험 대상으로 둔다.

1. decode 단계의 작은 matmul은 CPU가 더 유리할 수 있다.
2. dequantization, repacking, sampling, logits postprocess는 CPU가 맡는 편이 단순하고 안정적일 수 있다.
3. KV cache 관리와 메모리 압박 대응은 CPU가 맡는 쪽이 현실적일 수 있다.
4. prefill의 큰 GEMM은 GPU가 맡고, decode의 작은 연산은 CPU로 넘기는 phase split이 효과적일 수 있다.
5. 일부 row split 또는 layer split은 성능 향상보다도 "GPU 메모리가 부족한 환경에서 실행 가능성 확보"에 가치가 있을 수 있다.

## 우선 실험할 분할 전략

### 1. CPU only baseline

모든 비교의 기준점이다. 먼저 안정적으로 돌아가야 한다.

### 2. Prefill on GPU, Decode on CPU

큰 연산은 GPU, 짧고 자주 반복되는 연산은 CPU에 두는 방식이다. 초기 실험 가치가 가장 높다.

### 3. Row split matmul

하나의 matmul 일부를 CPU가, 일부를 GPU가 맡는다. 절대 성능보다도 실행 가능성과 오버헤드 구조를 파악하는 것이 목적이다.

### 4. Layer split

일부 레이어는 CPU, 일부 레이어는 GPU에서 처리한다. 구현은 단순하지만 장치 간 이동 비용이 병목일 수 있으므로 비교가 필요하다.

## 백엔드 방향

초기 구조는 아래처럼 나누는 것이 좋다.

1. `Model Loader`
2. `Graph Builder`
3. `Planner`
4. `Executor`
5. `Profiler`

백엔드 구현 우선순위는 아래 순서를 권장한다.

1. `CPU backend`를 기준 구현으로 만든다.
2. `NVIDIA backend`를 1차 GPU backend로 붙인다.
3. `Intel GPU backend`를 2차 GPU backend로 붙인다.
4. 세 장치를 공통 planner 아래에서 다룰 수 있게 정리한다.

이때 중요한 것은 backend를 늘리는 것보다 `공통 planner + backend capability query + fallback` 구조를 먼저 고정하는 것이다.

## 성공 기준

아래 다섯 가지를 초기 성공 기준으로 둔다.

1. 모델이 로드된다.
2. 프롬프트를 넣으면 토큰이 실제로 생성된다.
3. 한 backend가 실패해도 CPU fallback으로 이어진다.
4. 최소 한 가지 혼합 실행 모드가 동작한다.
5. CLI 한 번으로 실행부터 결과 확인까지 가능하다.

## 측정 지표

초기에는 복잡한 지표보다 아래만 기록한다.

- first token latency
- decode tokens/sec
- peak memory usage
- backend 선택 및 fallback 로그
- 장치 간 데이터 이동 크기 또는 횟수

이 프로젝트에서는 "더 빠름"만큼 "어떤 조합에서 왜 느린지"를 남기는 것도 중요하다.

## 비목표

초기 단계에서 아래 항목은 하지 않는다.

- 모든 모델 아키텍처 지원
- 학습 기능
- 멀티노드 분산 학습
- 최고 성능 벤치마크 달성
- 모든 GPU backend 동시 완성
- PyTorch 대체 프레임워크 수준의 범용성

## 다음 작업자용 우선순위

### 1단계: 범위 고정

- 대상 모델 1개 선정
- 양자화 포맷 1개 선정
- 지원 시나리오를 `batch=1`, `single session`으로 고정
- CLI 입출력 형태 결정

### 2단계: CPU only 추론 MVP

- 모델 로드
- prefill
- decode
- sampling
- 토큰 출력

### 3단계: 첫 GPU backend 연결

- 큰 연산 하나부터 GPU로 오프로드
- 실패 시 CPU fallback
- backend capability logging 추가

### 4단계: 혼합 실행 모드 추가

- `prefill = GPU / decode = CPU`
- 결과 검증
- latency, memory, fallback 로그 기록

### 5단계: Intel GPU 확장

- 같은 planner 아래에서 Intel GPU backend도 선택 가능하게 한다.
- 최소한 "동작은 한다" 수준까지 맞춘다.

## 첫 4주 제안 일정

### 1주차

- 모델/양자화/실험 범위 확정
- CPU only 추론 경로 작성
- 토큰 생성 CLI 완성

### 2주차

- 첫 GPU backend 연결
- prefill / decode 실행 경로 분리
- 실패 시 CPU fallback 추가

### 3주차

- 혼합 실행 모드 1개 구현
- 기본 profiling 출력
- first token latency, tokens/sec 기록

### 4주차

- Intel GPU까지 같은 구조로 연결 시도
- CPU only 대비 비교 로그 남기기
- README와 실행 예제 업데이트

## 작업하면서 계속 기록해야 할 것

다음 작업자는 구현과 함께 아래 항목을 꾸준히 남겨야 한다.

- 어떤 연산을 어떤 장치에 배치했는지
- 그 판단의 이유
- 실제로 빨랐는지, 느렸는지
- 느리다면 병목이 계산인지 전송인지 초기화 비용인지
- fallback이 발생한 조건
- 특정 장치 조합에서만 깨지는 문제

이 프로젝트는 설계 가설이 많기 때문에, 코드만큼 실험 기록이 중요하다.

## 열린 질문

아직 결정되지 않은 항목들이다.

- 첫 대상 모델은 무엇으로 할지
- 첫 양자화 포맷은 무엇으로 할지
- NVIDIA backend를 어떤 API로 시작할지
- Intel GPU backend를 OpenCL로 시작할지 다른 경로로 갈지
- 세 장치를 동시에 쓰는 모드는 MVP 이후로 미룰지 여부

## 요약

Jakal-Core의 초기 목표는 "모든 환경에서 가장 빠른 LLM runtime"이 아니다. 더 현실적인 목표는, CPU와 GPU를 함께 쓰는 방식으로 저사양 장비에서도 작은 LLM을 끝까지 돌릴 수 있는 실행기를 만드는 것이다. CPU를 다시 주요 계산 자원으로 쓰고, NVIDIA GPU와 Intel GPU를 공통 구조 안에서 다루며, 성능이 조금 부족하더라도 확실히 작동하는 완성품을 먼저 만드는 것이 이 프로젝트의 방향이다.
