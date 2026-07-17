# bench-b570-kernel-v4.ps1 — A/B/C off x v3 x v4 + knob matrix de runtime
# Metodologia espelhada do b70-optimization-lab (scripts/bench-*-vulkan-matrix.sh):
#   - reps fixas, mediana; -fa on/off; --poll 0/25/50/100; -ub 64/128/256/512;
#     GGML_VK_ALLOW_GRAPHICS_QUEUE 0/1  (ganho de +13% no B70 do lab, nunca varrido aqui)
#   - correcao PRIMEIRO: greedy byte-compare off x v4
#
# Uso:
#   powershell -File bench-b570-kernel-v4.ps1 -Phase correctness
#   powershell -File bench-b570-kernel-v4.ps1 -Phase abc
#   powershell -File bench-b570-kernel-v4.ps1 -Phase matrix   # nao precisa do build v4
param(
    [ValidateSet("correctness","abc","matrix","vecwg")]
    [string]$Phase = "abc",
    [string]$Bin     = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp\build-arc\bin",
    [string]$Model17 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Ternary-Bonsai-1.7B-Q2_0_g64.gguf",
    [string]$Model27 = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\models\Bonsai-27B-Q1_0.gguf",
    [string]$OutDir  = "C:\Users\geron\OneDrive\Desktop\AI\Bansai Llama.cpp\llama.cpp\benches\arc-b570",
    [int]$Reps = 5
)

$bench = Join-Path $Bin "llama-bench.exe"
$compl = Join-Path $Bin "llama-completion.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log   = Join-Path $OutDir "kernel-v4-$Phase-$stamp.txt"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Run-Bench($model, $tag, $reps, $extra) {
    Write-Host "== $tag ==" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -fa on -p 512 -n 128 -r $reps -b 512 -ub 256 @extra 2>&1 |
        Tee-Object -FilePath $log -Append
}

switch ($Phase) {

"correctness" {
    $prompt = "The capital of France is"
    foreach ($mode in @("0","3")) {
        $env:GGML_VK_B570_KERNEL = $mode
        $out = & $compl -m $Model17 -p $prompt -n 32 --temp 0 -ngl 99 -fa on 2>$null
        $hash = (Get-FileHash -InputStream ([System.IO.MemoryStream]::new([System.Text.Encoding]::UTF8.GetBytes($out -join "`n")))).Hash
        "{0} mode={1} sha256={2}" -f (Get-Date -Format HH:mm:ss), $mode, $hash |
            Tee-Object -FilePath $log -Append
    }
    Write-Host "Os dois SHA256 devem ser IDENTICOS. Se diferirem, NAO medir velocidade — reportar." -ForegroundColor Yellow
}

"abc" {
    foreach ($mode in @("0","2","3")) {
        $env:GGML_VK_B570_KERNEL = $mode
        Run-Bench $Model17 "1.7B KERNEL=$mode" $Reps @()
        Run-Bench $Model27 "27B  KERNEL=$mode" $Reps @()
    }
    # isolar deltas da v4 (so no modo 3)
    $env:GGML_VK_B570_KERNEL = "3"
    $env:GGML_VK_B570_MMVQ_VEC = "0";  Run-Bench $Model27 "27B v4 sem mmvq_vec" $Reps @()
    $env:GGML_VK_B570_MMVQ_VEC = "1"
    $env:GGML_VK_B570_FUSE_RMSNORM_MUL = "0"; Run-Bench $Model27 "27B v4 sem rmsnorm_mul" $Reps @()
    Remove-Item Env:GGML_VK_B570_FUSE_RMSNORM_MUL
}

"matrix" {
    # Knobs de runtime do lab — roda em QUALQUER build (v3 serve). Procurar:
    # graphics queue > compute? poll 25/50 > 0? ub otimo p/ cada modelo?
    $env:GGML_VK_B570_KERNEL = "2"
    foreach ($gq in @("0","1")) {
        foreach ($poll in @("0","25","50","100")) {
            foreach ($ub in @("64","128","256","512")) {
                if ($gq -eq "1") { $env:GGML_VK_ALLOW_GRAPHICS_QUEUE = "1" }
                else { Remove-Item Env:GGML_VK_ALLOW_GRAPHICS_QUEUE -ErrorAction SilentlyContinue }
                Run-Bench $Model27 "27B gq=$gq poll=$poll ub=$ub" 3 @("--poll",$poll,"-ub",$ub)
            }
        }
    }
}

"vecwg" {
    # varrer workgroup do shader vetorizado: requer rebuild trocando {128,1,1} do
    # create_pipeline -> rodar 1 build por tamanho. Aqui so lembra o procedimento.
    Write-Host "Trocar {128,1,1} por {64,1,1}/{256,1,1} no ggml-vulkan.cpp, rebuild, e rodar -Phase abc." -ForegroundColor Yellow
    Write-Host "Lembrar: XL (sg x8) mediu pior na v3. Nao testar >256." -ForegroundColor Yellow
}
}

Write-Host "`nLog: $log" -ForegroundColor Green
