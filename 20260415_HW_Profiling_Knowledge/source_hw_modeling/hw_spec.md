# HW spec
- Shared Memory Size : 4MB
- Unified Buffer Size : 256KB
- Queue Size : 128KB
- ISA Record Size : ISA.md 문서에 따른다.
- \# of core per cluster : 2개 (acore - pcore 1쌍을 1개 core라고 생각한다.)
- \# of cluster : 6개

## 전역 원칙 — 내부 tensor datapath

NPU 의 연산 datapath 는 **i8 기반 end-to-end** 이다 (ISA.md §Conv / §Gemm /
§Elementwise 참조). UB / SM / Queue 에 상주하거나 DMA 로 이동하는 모든
**data tensor** (input, weight, bias, activation, residual 등) 는 **1
byte/element (i8)** 로 회계한다. 컴파일러가 tile 크기, UB footprint, SM
peak, DMA data_size 를 계산할 때 반드시 이 규칙을 따른다.

예외 — **scale / zero_point / requant 계수**: layer_start record 의 고정
필드 (128 bit scales + 128 bit zero_points + 3 bit quant_ctrl, ISA.md
§layer_start Conv 참조) 로 전달되며 DMA / UB / SM accounting 대상이
**아니다**. 컴파일러의 tile/footprint 계산 시 완전히 제외한다.

StableHLO / stablehlo_quantized 등 상위 IR 은 rescale chain 을 f32 로
표현할 수 있으나, 이 표현은 dtype lowering 이전 중간 형태일 뿐이다.
tile/footprint 계산은 이 f32 값을 무시하고 항상 i8 기준을 적용한다.

# clock spec

## clock
- 800 MHz (1 clock = 1.25 ns)

## data flow
- AXI bus 기준, 128bit(16B) bus width
- 1 beat = 128bit = 16 Bytes
- burst 전송 시 1 beat/clock 이상적이나, arbitration/contention에 따라 지연 발생

| 경로 | latency (clock) | throughput | 비고 |
|:---|:---|:---|:---|
| SM to UB | 4~8 clock/beat | 2~4 GB/s | NoC 경유. bank 충돌 시 +2~4 clock |
| UB to SM | 4~8 clock/beat | 2~4 GB/s | SM write port 경합 가능 |
| P to A (pair, PC 경유) | 2~4 clock/beat | 4~8 GB/s | 동일 cluster 내 직접 연결 |
| A to P (pair, PC 경유) | 2~4 clock/beat | 4~8 GB/s | 동일 cluster 내 직접 연결 |
| P to P (in cluster, NC ring) | 3~6 clock/beat | 2.7~5.3 GB/s | 1-hop ring |
| A to A (in cluster, NC ring) | 3~6 clock/beat | 2.7~5.3 GB/s | 1-hop ring |
| P to P (out cluster, NC ring) | 6~15 clock/beat | 1~2.7 GB/s | multi-hop. hop 수에 비례 |
| A to A (out cluster, NC ring) | 6~15 clock/beat | 1~2.7 GB/s | multi-hop. hop 수에 비례 |

- 위 값은 시뮬레이터 기본 프로파일용 추정치이며, 실제 하드웨어 구현에 따라 변경된다.
- throughput 계산: 16B / (N clock × 1.25ns) = 16B / (N × 1.25ns)
  - 예: 4 clock/beat → 16B / 5ns = 3.2 GB/s
- DMA burst 전송 시 총 시간: burst_beats × clock_per_beat + initial_latency
  - initial_latency: SM 경로 2~4 clock, NC/PC 경로 1~2 clock (추정)

## calculate

### 기본 전제
- clock : 800 MHz (1 clock = 1.25 ns)
- MAC unit : 1 clock에 1 MAC operation (multiply-accumulate)
- Vector processor : 16 data lane, 1 clock에 16개 element 동시 처리
- Activation : hardwired 또는 PWL. conv 직후 1~2 clock overhead

### opcode별 연산 시간

#### Conv (opcode 0b00001)
- 연산량 : C_out × C_in × Kh × Kw × H_out × W_out MAC operations
- MAC unit가 1 clock에 1 MAC:
  - 총 clock = C_out × C_in × Kh × Kw × H_out × W_out / (병렬도)
  - 병렬도는 하드웨어 MAC array 크기에 따라 다름 (추정: 16~256 MACs 병렬)
- requant : 출력 element당 1 clock (mul + shift + round + saturate)
  - 총 clock = C_out × H_out × W_out / 16 (vector processor 16-lane)
- activation : 출력 element당 1 clock
  - 총 clock = C_out × H_out × W_out / 16

```
예시: Conv 3×3, C_in=64, C_out=64, H_out=80, W_out=80
  MAC ops  = 64 × 64 × 3 × 3 × 80 × 80 = 188,743,680
  MAC array = 256 병렬 가정
  MAC clock = 188,743,680 / 256 = 737,280 clock
  MAC 시간  = 737,280 × 1.25 ns = 921.6 μs

  requant  = 64 × 80 × 80 / 16 = 25,600 clock = 32 μs
  activation = 25,600 clock = 32 μs

  총 연산 시간 = 921.6 + 32 + 32 = 985.6 μs (약 0.99 ms)
```

#### Gemm (opcode 0b00010)
- 연산량 : M × K × N MAC operations
- 총 clock = M × K × N / (MAC 병렬도)
- requant : M × N / 16 clock

```
예시: Gemm M=64, K=64, N=64
  MAC ops  = 64 × 64 × 64 = 262,144
  MAC array = 256 병렬 가정
  MAC clock = 262,144 / 256 = 1,024 clock = 1.28 μs
  requant  = 64 × 64 / 16 = 256 clock = 0.32 μs
  총 = 1.60 μs
```

#### Attention (opcode 0b00011)
- QK^T : B × num_heads × seq_len × head_dim MACs (= B × seq_len × C)
- Softmax : seq_len × num_heads × (max + exp + sum + div) ≈ 4 × seq_len × num_heads clock
- AV : B × num_heads × seq_len × head_dim MACs
- 총 clock ≈ 2 × Gemm(seq_len, head_dim, seq_len) + softmax overhead

#### Elementwise 계열
- 연산량 : C × H × W element operations
- 16-lane vector processor:
  - 총 clock = C × H × W / 16

| sub_opcode | 연산 | element당 clock | 비고 |
|:---|:---|---:|:---|
| Add (0) | add + requant | 2 | add 1 + requant 1 |
| Mul (1) | mul + requant | 2 | mul 1 + requant 1 |
| Sigmoid (2) | LUT lookup | 1 | hardwired/PWL |
| MaxPool (3) | compare × K² | K² | 5×5 pool → 25 compare |
| Resize (4) | index 계산 + copy | 1 | nearest neighbor |
| TopK (5) | partial sort | O(N log K) | N=8400, K=300 기준 |
| Transpose (6) | memory reorder | 1 | address 재배치 |
| Gather (7) | indirect read | 2 | index read + data read |
| Copy (8) | copy | 1 | 단순 복사 |

```
예시: Add, C=64, H=80, W=80
  elements = 64 × 80 × 80 = 409,600
  clock = 409,600 × 2 / 16 = 51,200 clock = 64 μs
```

#### Concat (opcode 0b00100)
- 연산 없음. 순수 DMA 복사.
- 시간 = DMA 전송 시간 (data flow 참조)

### DMA 전송 시간 계산

```
DMA 시간 = initial_latency + ceil(data_size / 16) × clock_per_beat

  - initial_latency : 2~4 clock (경로에 따라)
  - data_size : bytes
  - 16 : bus width (16 Bytes/beat)
  - clock_per_beat : data flow 테이블 참조 (경로별)
```

```
예시: SM → UB, weight 73,728 bytes
  beats = ceil(73728 / 16) = 4608 beats
  clock_per_beat = 4~8 (SM to UB)
  DMA clock = 4 + 4608 × 4 ~ 4 + 4608 × 8 = 18,436 ~ 36,868 clock
  DMA 시간 = 23.0 ~ 46.1 μs
```

### instruction layer 총 시간 계산

하나의 layer (layer_start → layer_run × N → layer_end)의 총 시간:

```
T_layer = T_layer_start + N × T_layer_run + T_layer_end

여기서:
  T_layer_start = max(DMA_weight, DMA_bias)
    - weight와 bias DMA가 동시 진행 가능하면 max, 순차면 sum
    - 현재 DMA bus 6개 공유 → 동시 가능 (다른 slot 사용 시)

  T_layer_run = max(T_compute, T_DMA_input, T_DMA_output)
    - double buffering 사용 시: 연산과 DMA가 겹침 → max
    - double buffering 미사용 시: 순차 → sum
    
    T_compute = MAC_clock + requant_clock + activation_clock
    T_DMA_input = initial_latency + ceil(input_tile_size / 16) × clock_per_beat
    T_DMA_output = initial_latency + ceil(output_tile_size / 16) × clock_per_beat

  T_layer_end = Σ DMA packs (최대 6개, 동시 or 순차)

  N = tile 수 = ceil(H_out / tile_H) × ceil(W_out / tile_W)
```

```
예시: Conv 3×3, C_in=64, C_out=64, H=80, W=80, tile=32×32

  layer_start:
    weight DMA = 73,728 B → 18,436~36,868 clock (23~46 μs)
    bias DMA = 256 B → 4+16×4=68 clock (0.09 μs)
    T_layer_start = 18,436~36,868 clock (weight 지배적)

  layer_run (1 tile):
    input tile = 34×34×64 = 73,984 B → DMA 18,500~36,996 clock
    output tile = 32×32×64 = 65,536 B → DMA 16,388~32,772 clock
    compute = (64×64×9×32×32)/256 + 32×32×64/16×2 = 73,728 + 8,192 = 81,920 clock
    
    double buffering: T_layer_run = max(81,920, 18,500~36,996) = 81,920 clock (compute-bound)
    no double buffering: T_layer_run = 81,920 + 18,500 + 16,388 = 116,808 clock

  N = ceil(80/32) × ceil(80/32) = 3 × 3 = 9 tiles

  layer_end:
    T_layer_end ≈ 0 (추가 DMA 없으면)

  총 시간 (double buffering):
    T_layer = 36,868 + 9 × 81,920 + 0 = 774,148 clock = 967.7 μs ≈ 0.97 ms

  총 시간 (no double buffering):
    T_layer = 36,868 + 9 × 116,808 + 0 = 1,088,140 clock = 1,360 μs ≈ 1.36 ms
```
