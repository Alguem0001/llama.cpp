# Arc B570 — Kernel v4: design + o que veio do b70-optimization-lab

**Data:** 2026-07-17
**Base:** tree local `llama.cpp` branch `arc-speed` (kernel v3 já no tree, default Xe2 = v3)
**Fontes externas:** [steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) (lab Intel XPU, 4× Arc Pro B70), estado do upstream `ggml-vulkan` (PR #14001 coopmat Xe2 etc.)

---

## 0. TL;DR

1. **O teto de ~36 t/s do 27B Q1_0 não é o teto de banda real.** Roofline: B570 = 380 GB/s; Q1_0 g64 ≈ 4,2 GB de pesos ⇒ teto teórico ≈ **90 t/s**. Hoje estamos em **~40% da banda**. O mesmo no 1.7B Q2_0: teto ≈ 795 t/s, medido 220 ⇒ **~28%**. Há headroom de kernel de verdade — exatamente o "shader Q1_0 vetorizado" que o resumo já apontava como próximo salto.
2. **O lab do Steve Seguin provou 3 fusões que preservam qualidade** no ggml (SYCL, mas o conceito porta para Vulkan): **RMS_NORM+MUL fundido** (+1–5%), **MMVQ2+SwiGLU fundido** (+0,4–2,5%), e **Q8 activation cache**. A v4 porta a primeira (quant-agnóstica, risco baixo) e deixa a segunda como v5.
3. **Ganho grátis sem rebuild:** a matriz de knobs de runtime que o lab usou no Vulkan — `--poll`, `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, sweep de `-ub` — nunca foi varrida no nosso B570/Windows. No lab, fila gráfica + poll + Mesa ajustado tiraram o Vulkan de 22,2 → 25,2 t/s (**+13%**) sem tocar em shader.
4. **Anti-lições do lab (coisas medidas que regrediram — não repetir):** lane-por-bloco (VDR4) regrediu; forçar MMVQ onde DMMV ganha regrediu; workgroups maiores regrediram (bate com nossa v1/XL); FP16 em variance de norm regrediu.

---

## 1. Roofline do B570 (por que dá para ir além de 36 t/s)

| Modelo | bits/param (g64, scale fp16) | Peso total | Teto @380 GB/s | Medido hoje | % do teto |
|---|---:|---:|---:|---:|---:|
| Bonsai 27B Q1_0 | 1,25 | ~4,22 GB | **~90 t/s** | ~36 t/s | ~40% |
| Ternary 1.7B Q2_0 | 2,25 | ~0,48 GB | **~795 t/s** | ~220 t/s | ~28% |

Premissas: group size 64, 1 escala fp16 por grupo, sem contar KV cache, lm head nem overhead de launch (ou seja, o teto prático é um pouco menor — mas kernels GEMV bons chegam a 70–85% da banda de pico).

**Leitura:** o gargalo hoje é eficiência do kernel de dequant+matvec (mmvq) e custo de launch por camada, não a banda em si. Isso muda a prioridade: vale trabalho GLSL sério no decode, e vale reduzir launches (fusões).

---

## 2. O que o lab tem de aplicável (e o que não tem)

### 2.1 Transferível direto (já no patch v4)

| Lição do lab | Evidência deles | Como aplicamos no B570/Vulkan |
|---|---|---|
| **Fusão RMS_NORM → MUL** (escala 1D F32) | +1,3% single-card, **+5,5%** no TP3; byte-exact | Novo shader `rms_norm_mul.comp` + matcher no `ggml-vulkan.cpp`, gate `GGML_VK_B570_FUSE_RMSNORM_MUL=1` (default ON no modo 3) |
| **Knob matrix Vulkan** (poll / graphics queue / ub) | +13% no Linux (22,2→25,2) | Script `bench-b570-kernel-v4.ps1` varre `--poll 0/25/50/100`, `GGML_VK_ALLOW_GRAPHICS_QUEUE=0/1`, `-ub 64/128/256/512`, `-fa on/off` |
| **Q8 activation cache** | mantido em todas as receitas vencedoras deles | Já temos Q8_1 SUBGROUP no Intel (v2); manter |
| **Disciplina de validação** | greedy byte-compare SHA256 + median de reps | Bench script compara saída greedy `llama-completion` off × v4 antes de aceitar número |

### 2.2 Transferível com trabalho (candidato a v5)

| Lição | Evidência | Por que não entrou na v4 |
|---|---|---|
| **MMVQ2+SwiGLU fundido** (gate/up compartilham ativação; kernel escreve `silu(gate)*up` direto) | +0,4% single, +2,5% TP3 | Precisa de matcher de grafo mais invasivo no Vulkan + shader GLU custom para Q1_0/Q2_0. Escopo de v5. |
| **Fusão allreduce+ADD / single-kernel allreduce** | grandes ganhos em 2–4 GPUs | Temos 1× B570. Não se aplica. |
| **ESIMD block-loaded scales** (carregar metadados de escala em bloco) | ganho standalone no harness deles | Específico de SYCL/ESIMD; o análogo Vulkan é o nosso shader vetorizado com `uint4` loads (na v4). |

### 2.3 Negativos medidos no lab (não reintroduzir)

- **VDR4** (1 lane por bloco de quant no MMVQ reordenado): −5% TG. Hipótese de "menos loads duplicados de escala" não se pagou — escalas baratas, scheduling caro.
- **Forçar MMVQ no Q4_0 Intel**: 10,35 t/s (desastre) — upstream desliga MMVQ p/ Q4_0 Intel de propósito. *(Nosso caso é diferente: Q1_0/Q2_0 custom, onde LARGE mmvq mediu melhor — manter nossos dados, não os deles.)*
- **Workgroup maior ≠ mais rápido**: confirmado por eles e pela nossa v1 (warptile 128 threads, XL).
- **Variance de RMS em FP16** para economizar banda: regrediu qualidade/velocidade. Manter F32.

### 2.4 Upstream ggml-vulkan (estado em 2026-07)

- **PR #14001 (merged): coopmat habilitado para Intel Xe2** — no B580, coopmat2 deu **pp512 488 → 1607 t/s** em Q4_K_M. Se o nosso fork ainda não tem coopmat ativo para o B570 (PCI ID table + `VK_KHR_cooperative_matrix`), **isso é provavelmente o maior ganho de PP disponível**. Verificar com `vulkaninfo` + log `ggml_vulkan` se `cooperative_matrix` está ON. Se não: backport do device-table entry do B570 do upstream.
- Upstream mantém MMVQ de Q4_0 desligado em Intel; nosso fork custom é independente disso.

---

## 3. Kernel v4 — o que muda no tree

A v4 = v3 + **modo 3** (`GGML_VK_B570_KERNEL=3`) com dois deltas, ambos default-ON no modo 3 e comutáveis por env sem rebuild:

### 3.1 Shader mmvq vetorizado para Q1_0/Q2_0 (`GGML_VK_B570_MMVQ_VEC`, default ON no modo 3)

Arquivos novos:
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q1_0_vec.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_0_vec.comp`

Técnicas (mirando os ~60–80% de banda):

1. **Loads de 128 bits**: pesos lidos como `uint4`/`uvec2` (16/8 B por instrução) em vez de byte-a-byte — Q1_0 g64 = 10 B/bloco lido como `uvec2`+`uint16`; Q2_0 g64 = 18 B/bloco lido como `uint4`+`uint16`.
2. **Unpack por bit-twiddling** (`bitfieldExtract`, shifts) em registrador, sem shared memory para pesos.
3. **Acumulação fp32 por lane + `subgroupAdd`** na saída (1 linha por workgroup, lanes varrem blocos com stride = subgroup_size).
4. **2 blocos por iteração por lane** (unroll ×2) para esconder latência de load — análogo ao que o lab viu funcionar; unroll ×4 (estilo VDR4) NÃO fazer (regrediu lá).
5. Ativação Q8_1 lida do cache Q8 existente (dot Q1_0×Q8_1 / Q2_0×Q8_1), preservando o path mmvq atual.

> ⚠️ **Formato de bloco assumido** (confirmar com o `ggml_vec_dot` de referência do fork antes de ligar — detalhes no patch):
> - `block_q1_0 = { half d; uint8_t qs[8]; }`, valor = bit ? +d : −d
> - `block_q2_0 = { half d; uint8_t qs[16]; }`, valor = (q − 1) · d, q ∈ {0,1,2} (ternário)
>
> Se o fork usa offsets/mins diferentes, só a função `dequant_*_block()` do shader muda; o esqueleto de loads/redução fica igual.

### 3.2 Fusão RMS_NORM→MUL (`GGML_VK_B570_FUSE_RMSNORM_MUL`, default ON no modo 3)

Arquivo novo:
- `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm_mul.comp`

+ matcher conservador no `ggml-vulkan.cpp`: só funde `RMS_NORM(F32) → MUL` quando o outro operando do MUL é tensor 1D F32 contíguo no mesmo device (mesmo critério do patch SYCL do lab). Variance em **F32**, escala aplicada na escrita — matematicamente idêntico, validado por byte-compare greedy.

Ganho esperado (espelhando o lab): **+1–2% TG single-card**; corta 1 launch + 1 round-trip de memória por camada (no 27B são dezenas de camadas × 2 norms).

### 3.3 Plumbing

- `enum b570_kernel_mode` ganha `B570_KERNEL_V4=3`; `b570_kernel = mode >= 1` continua válido.
- Log de boot: `ggml_vulkan: B570 optimized kernel: ON (v4) [toggle: GGML_VK_B570_KERNEL=0|1|2|3]`
- Modo 3 liga: tudo da v3 + `mmvq_vec` + `fuse_rmsnorm_mul`. Env individuais permitem isolar cada um no bench.

---

## 4. Plano de validação (mesma disciplina do lab)

1. **Corretude antes de velocidade:** `llama-completion` greedy, 8+ tokens, prompt fixo → SHA256 da saída **idêntico** off × v4 (o lab fez exatamente isso para cada fusão).
2. **Bench:** `bench-b570-kernel-v4.ps1` — A/B/C `off(0) × v3(2) × v4(3)`, `-r 5` (1.7B) e `-r 5` (27B), mediana.
3. **Knob matrix de runtime** (pode ser rodada antes mesmo do rebuild, no binário v3 atual): poll × graphics-queue × ub × fa.
4. **Critério de aceite:** v4 vira default se TG(27B) não regredir e TG(1.7B) ≥ v3 + 2% com correção byte-exata.

Expectativa realista: +5–15% TG no 27B (shader vetorizado) e +1–3% no 1.7B (launch-bound, fusão ajuda pouco). Se o shader vetorizado chegar a ~65% da banda, 27B → **~55–60 t/s**.

---

## 5. Backlog pós-v4 (ordem de valor esperado)

1. **Coopmat p/ PP no B570** (backport PR #14001 se ausente) — potencial de PP em ×2–3.
2. **MMVQ2+SwiGLU fundido** para Q1_0/Q2_0 (port do patch SYCL do lab) — +1–2%.
3. **Fusão same-activation mais ampla** (QKV num launch só) — lab aponta como próximo alvo deles também.
4. **Rebase do fork em cima do upstream atual** — upstream acumulou melhorias FA/coopmat2 que o fork não tem.

---

## 6. Arquivos deste pacote

| Arquivo | Conteúdo |
|---|---|
| `B570-Kernel-v4.md` | Este documento |
| `b570-kernel-v4.patch` | Patch unificado: plumbing modo 3 + matcher RMS_NORM_MUL + dispatch mmvq_vec |
| `vulkan-shaders/mul_mat_vec_q1_0_vec.comp` | Shader novo (vetorizado) |
| `vulkan-shaders/mul_mat_vec_q2_0_vec.comp` | Shader novo (vetorizado) |
| `vulkan-shaders/rms_norm_mul.comp` | Shader novo (fusão) |
| `bench-b570-kernel-v4.ps1` | Bench A/B/C + knob matrix (Windows/PowerShell) |
