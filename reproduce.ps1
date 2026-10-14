$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $stepArgs = @()
    if ($env:STACR_STEPS) { $stepArgs = @("--steps", $env:STACR_STEPS) }
    foreach ($method in @("baseline", "checkpoint", "full", "turbo")) {
        python -m stacr reproduce run "gpt2-medium-quality-$method" @stepArgs @args
        if ($LASTEXITCODE -ne 0) { throw "Protocol failed: $method (exit $LASTEXITCODE)" }
    }
} finally {
    Pop-Location
}
