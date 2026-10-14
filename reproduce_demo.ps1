$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($method in @("baseline", "stacr", "turbofast")) {
        python -m stacr train --config synthetic_stacr --device cpu --set "method=$method" --output "outputs/demo/$method.csv" @args
        if ($LASTEXITCODE -ne 0) { throw "Demo failed: $method (exit $LASTEXITCODE)" }
    }
} finally {
    Pop-Location
}
