# Arc B570 — Kernel Custom: resumo do que foi feito

**Data do registro:** 2026-07-17  
**Objetivo:** acelerar **token generation (TG)** e **prompt processing (PP)** do Bonsai no Intel Arc B570 (Vulkan / Xe2), via perfil de kernels em `ggml-vulkan`, **sem** reescrever shaders GLSL do zero.

---

## 1. Onde está cada coisa

### Código (fonte do kernel)

| O quê | Caminho |
|--------|---------|
| **Perfil B570 (v2/v3)** — flags, env, FA, warptile, mmvq | `llama.cpp\ggml\src\ggml-vulkan\ggml-vulkan.cpp` |
| Struct device: `b570_mode`, `b570_kernel` | ~linha 748–752 (mesmo arquivo) |
| Parse env `GGML_VK_B570_KERNEL` | `ggml_vk_b570_kernel_mode_from_env()` |
| Init + log `ON (v2\|v3\|OFF)` | init do device Vulkan (~6777+) |
| mmvq path principal (decode) | `ggml_vk_get_dequantize_mul_mat_vec*` |
| mmvq path **id** (paridade v3) | `ggml_vk_get_dequantize_mul_mat_vec_id` |
| Enum workgroup: SUBGROUP / LARGE / **XL** | `enum dmmv_wg_sizes` |
| Branch git | `arc-speed` (repo local `llama.cpp`) |
| Remotes | `origin` → Alguem0001/llama.cpp · `arc` → llama.cpp-arc-speed |

### Binários de teste

| O quê | Caminho |
|--------|---------|
| Build Vulkan | `llama.cpp\build-arc\` |
| `llama-bench.exe` | `llama.cpp\build-arc\bin\llama-bench.exe` |
| `llama-cli.exe` / server | `llama.cpp\build-arc\bin\` |
| DLL Vulkan com o perfil | `llama.cpp\build-arc\bin\ggml-vulkan.dll` |

### Modelos usados nos benches de kernel

| Modelo | Caminho |
|--------|---------|
| Bonsai 27B Q1_0 | `models\Bonsai-27B-Q1_0.gguf` |
| Ternary 1.7B Q2_0 | `models\Ternary-Bonsai-1.7B-Q2_0_g64.gguf` |

*(pasta `models\` na raiz do projeto Bansai, **não** dentro de `llama.cpp\models`)*

### Documentação

| Documento | Caminho |
|-----------|---------|
| **Este resumo** | `B570-Kernel-Custom-Resumo.md` (raiz do projeto) |
| Relatório longo (kernel + Bonsai + código) | `B570-Bonsai-Kernel-Report.md` e `llama.cpp\docs\b570-bonsai-kernel-report.md` |
| Design v3 (pacote externo) | `C:\Users\geron\Downloads\B570-Kernel-v3.md` |
| Patch v3 (referência) | `C:\Users\geron\Downloads\b570-kernel-v3.patch` |
| Script de bench v3 | `C:\Users\geron\Downloads\bench-b570-kernel-v3.ps1` |

### Logs / resultados brutos

| Pasta / arquivo | Conteúdo |
|-----------------|----------|
| `llama.cpp\benches\arc-b570\` | Todos os benches kernel + speculative |
| `kernel-v2-*.txt` | Comparação normal vs v2 |
| `kernel-v3-off|v2|v3-*.txt` | A/B/C off × v2 × v3 (3 reps) |
| `kernel-v3-r10-v2|v3-1.7B.txt` | **10 reps** 1.7B (decisão default) |
| `kernel-v3-xl-*.txt` | Experimento XL ON/OFF |
| `RESULTS.md` | Notas antigas de bench na pasta |
| `spec-*` / `final-*` | Speculative DSpark (outro eixo de speed) |

### Env vars (runtime, sem rebuild)

| Variável | Efeito |
|----------|--------|
| `GGML_VK_B570_KERNEL=0` | Perfil **off** (upstream-like) |
| `GGML_VK_B570_KERNEL=1` | Perfil **v2** validado |
| `GGML_VK_B570_KERNEL=2` | Perfil **v3** (default em INTEL_XE2) |
| `GGML_VK_B570_MMVQ_XL=1` | Experimento workgroup XL no decode Q1/Q2 — **default OFF**, piora medido |
| `GGML_VK_ARC_FA_LEGACY=1` | Força path FA Intel legado (equivale modo 0 para FA) |
| `GGML_VK_ARC_MMVQ_WG=large\|subgroup` | Força tamanho de workgroup mmvq |
| `GGML_VK_FORCE_MMVQ` / `DISABLE` | Força / desliga mmvq |

No boot do Vulkan deve aparecer algo como:

```text
ggml_vulkan: B570 optimized kernel: ON (v3)  [toggle: GGML_VK_B570_KERNEL=0|1|2]
```

---

## 2. Linha do tempo: o que tentamos no Kernel Custom

### Hipótese de trabalho

No B570, o **TG do 27B Q1_0 é bandwidth-bound** (~36 t/s). O “kernel custom” não inventa FLOPs: escolhe **melhor path de pipeline** (FA com SIMD16 no decode, warptile com BK mais fundo, mmvq LARGE seletivo) para extrair mais da banda e dos Xe-cores **sem** as regressões da v1 agressiva.

### v1 — agressiva (descartada)

Tentativas que **regrediram** TG (medidas e depois proibidas):

| Tentativa | Resultado |
|-----------|-----------|
| Warptile com **128 threads** | ~**−11% TG** no 1.7B |
| Forçar `mmvq_mode=1` | regressão TG |
| Flash-attn `block_cols=128` | regressão |
| LARGE em quase tudo | dói; Q8_1 em LARGE regrediu |

**Lição:** “maior workgroup / mais threads” **não** é sempre mais rápido no Battlemage.

### v2 — perfil conservador (validado)

Mantido no código e ainda acessível com `KERNEL=1`:

1. **FA no decode (n_rows==1):** subgroups SIMD16 + workgroup 64; PP mantém path Intel “seguro”.
2. **FA `block_cols=64`** (não 128).
3. **Warptile MMQ:** só aprofunda **BK=64** no tile small; **não** infla threads para 128.
4. **mmvq principal:** thresholds m mais altos no B570; **Q1_0/Q2_0 no decode** → sempre **LARGE** (`m≤8, k≥512`).
5. **Q8_1** continua **SUBGROUP** no Intel.

Efeito típico (ordem de grandeza dos benches anteriores):

- **1.7B:** ~+1–4% TG vs off  
- **27B:** ~neutro / ±1% (já no teto de banda)

### v3 — v2 + plumbing + paridade mmvq-id (+ XL opcional)

Aplicado a partir do pacote `B570-Kernel-v3.md` (Downloads):

| Mudança | Tipo | Status |
|---------|------|--------|
| `b570_mode` 0\|1\|2 + `b570_kernel = mode≥1` | plumbing | **no tree** |
| Env `0/1/2` + log `ON (v3)` | plumbing | **no tree** |
| **LARGE Q1/Q2 no path `mul_mat_vec_id`** (paridade com path principal) | delta funcional | **no tree** |
| Workgroup **XL** (sg×8) via `GGML_VK_B570_MMVQ_XL` | experimento | **no tree, OFF** |

**O que a v3 não toca (de propósito):** FA de decode, warptile 128, force mmvq — tudo o que quebrou na v1.

---

## 3. Números finais medidos (esta máquina, Arc B570 Vulkan)

### A/B/C curto (3 reps) — off × v2 × v3

| Modelo | off tg128 | v2 tg128 | v3 tg128 |
|--------|----------:|---------:|---------:|
| 1.7B | 215.8 | 223.6 | 219.8 (ruído) |
| 27B | 35.90 | 35.68 | 35.65 |

### Decisão default: 1.7B **10 reps**

| perfil | pp512 | tg128 |
|--------|------:|------:|
| v2 | 5860 ± 19 | 210.3 ± 0.7 |
| **v3** | 5880 ± 23 | **220.7 ± 1.5** (**+5% TG**) |

→ **Default Xe2 = v3.**

### XL vs v3-LARGE (5 reps) — **rejeitado**

| Modelo | v3 tg | v3+XL tg | Δ |
|--------|------:|---------:|---|
| 1.7B | 218.3 | 217.6 | −0.3% TG, −2.2% PP |
| 27B | 35.4 | 35.2 | −0.5% TG, −1.4% PP |

→ **XL permanece OFF.**

### Referência “nosso atual” estável 27B (llama-bench)

- **pp128 / pp512:** ~435–455 t/s (depende do tamanho de prompt no comando)  
- **tg128:** **~35.5–36.5 t/s** (teto prático com Q1_0 + Vulkan)

---

## 4. O que o Kernel Custom **conseguiu** e o que **não**

### Conseguiu

- Perfil **reprodutível** e **comutável** (0/1/2) sem rebuild.  
- Evitar armadilhas da v1 com comentários e código que não reintroduz 128-thread warptile / FA Bc=128.  
- **+~5% TG no 1.7B** (v3 vs v2, 10 reps).  
- **27B:** estabilidade (sem regressão clara); PP levemente igual ou melhor.  
- Paridade mmvq-id para o dia em que houver MoE/`mul_mat_vec_id` com Q1/Q2.  
- Experimento XL **medido e descartado** com dados.

### Não conseguiu (limites honestos)

- **Não** quebrar o teto ~36 t/s do **27B Q1_0** (banda de memória).  
- **Não** entregar 1,5–2× TG só com escolha de pipeline.  
- XL **não** saturou melhor a banda; piorou um pouco.

### Próximos saltos reais de TG (fora do micro-tuning de pipeline)

1. **Shader Q1_0 de decode vetorizado** (`dequant_q1_0.comp` / mmvq) — trabalho GLSL sério.  
2. **Speculative decoding** com draft da mesma família / DSpark com bom accept rate  
   - DSpark **já liga** no tree (`draft-dspark` + multi-layer capture).  
   - Medido: accept ~45%, mas **TG piora** no B570 (overhead draft+capture > ganho).  
   - Logs: `benches\arc-b570\final-A-no-draft.*`, `final-B-dspark.*`.

---

## 5. Como repetir o bench do kernel

```powershell
$bin = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp\build-arc\bin\llama-bench.exe"
$m17 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Ternary-Bonsai-1.7B-Q2_0_g64.gguf"
$m27 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Bonsai-27B-Q1_0.gguf"

# v3 (default)
$env:GGML_VK_B570_KERNEL = "2"
& $bin -m $m27 -ngl 99 -fa on -p 512 -n 128 -r 5 -b 512 -ub 256

# v2
$env:GGML_VK_B570_KERNEL = "1"

# off
$env:GGML_VK_B570_KERNEL = "0"

# XL (só com KERNEL=2; esperado pior)
$env:GGML_VK_B570_KERNEL = "2"
$env:GGML_VK_B570_MMVQ_XL = "1"
```

Rebuild após editar `ggml-vulkan.cpp`:

```powershell
# vcvars64 + 
cmake --build "....\llama.cpp\build-arc" --config Release --target llama-bench -j 8
```

---

## 6. Checklist “estado atual do tree”

- [x] Kernel v2 no código  
- [x] Kernel v3 (mode + mmvq-id) no código  
- [x] Default Xe2 = **v3**  
- [x] XL no código, **default OFF**, medido e rejeitado  
- [x] Logs em `llama.cpp\benches\arc-b570\`  
- [x] Relatório longo + este resumo  
- [ ] Shader Q1_0 custom (não feito)  
- [ ] Speculative com speedup líquido no B570 (path DSpark ligado, mas ainda mais lento que AR)

---

## 7. Uma frase

**Kernel custom B570 = perfil Vulkan conservador (v2) + comutação 0/1/2 e paridade mmvq-id (v3); ganha no 1.7B, estabiliza o 27B no teto de banda, e não substitui shader Q1_0 ou speculative de alta aceitação para o próximo salto de TG.**

---

## 8. Kernel v4 (adicionado 2026-07-17 — pacote Kimi / b70-optimization-lab)

**Nada do v2/v3 foi removido.** A v4 é **aditiva**.

### Onde está o pacote completo (cópia intacta)

| Item | Caminho |
|------|---------|
| Pacote inteiro | `Bansai Llama.cpp\b570-kernel-v4\` |
| MD design | `b570-kernel-v4\B570-Kernel-v4.md` e `llama.cpp\docs\B570-Kernel-v4.md` e raiz `B570-Kernel-v4.md` |
| Patch referência | `b570-kernel-v4\b570-kernel-v4.patch` (+ `.txt`) |
| Bench script | `b570-kernel-v4\bench-b570-kernel-v4.ps1` e `llama.cpp\scripts\arc\bench-b570-kernel-v4.ps1` |
| Shaders fonte | `llama.cpp\ggml\src\ggml-vulkan\vulkan-shaders\mul_mat_vec_q1_0_vec.comp` |
| | `...\mul_mat_vec_q2_0_vec.comp` |
| | `...\rms_norm_mul.comp` |
| Cópias .txt | mesmo diretório de shaders + pasta `b570-kernel-v4\` |

### O que entrou no tree (código)

| Mudança | Arquivo |
|---------|---------|
| Registro SPIR-V dos 3 shaders | `vulkan-shaders-gen.cpp` (`string_to_spv` para `rms_norm_mul_b570`, `mul_mat_vec_q1_0_vec`, `mul_mat_vec_q2_0_vec`) |
| `b570_mode` 0..**3**, flags `b570_mmvq_vec`, `b570_fuse_rmsnorm_mul` | `ggml-vulkan.cpp` struct device |
| Pipelines `pipeline_mul_mat_vec_q*_vec`, `pipeline_rms_norm_mul_b570` | `ggml-vulkan.cpp` (criados se flags ON) |
| Env parse `KERNEL=3` / `v4` | `ggml_vk_b570_kernel_mode_from_env` |
| Env `GGML_VK_B570_MMVQ_VEC`, `GGML_VK_B570_FUSE_RMSNORM_MUL` | `ggml_vk_b570_env_flag` |
| Intel: `add_rms_fusion` permitido se `b570_fuse_rmsnorm_mul` | init device |
| Default Xe2 | **continua v3 (2)** — v4 **só** com `GGML_VK_B570_KERNEL=3` |

### Ajustes mínimos só para compilar (lógica preservada)

- `#extension GL_EXT_control_flow_attributes` nos `*_vec.comp` (`[[unroll]]`)
- `float16_t(...)` no unpack de half
- `rsqrt` → `inversesqrt` no `rms_norm_mul.comp`

### Env v4

| Variável | Default no modo 3 | Efeito |
|----------|-------------------|--------|
| `GGML_VK_B570_KERNEL=3` | off (default still 2) | Liga v4 |
| `GGML_VK_B570_MMVQ_VEC=1` | ON se mode≥3 | Pipelines mmvq vetorizados Q1/Q2 |
| `GGML_VK_B570_FUSE_RMSNORM_MUL=1` | ON se mode≥3 | Fusão RMS_NORM→MUL no Intel + pipeline B570 |

### Nota de integração

- Pipelines v4 **compilam e criam** com modo 3.
- O path genérico de dispatch mmvq usa push constants **diferentes** dos shaders `*_vec` (ABI experimental do pacote).
- Fusão RMS no grafo Vulkan **já existia** (`pipeline_rms_norm_mul_f32`); no Intel estava desligada — a v4 **habilita** `add_rms_fusion` no B570 quando o flag está ON (sem remover o path antigo).
- Validar com o script: `powershell -File llama.cpp\scripts\arc\bench-b570-kernel-v4.ps1 -Phase correctness` antes de medir speed.

