Unregister-ScheduledTask -TaskName "Isotherm Logger" -Confirm:$false
Write-Host "Removed 'Isotherm Logger'. Data in data\isotherm.db is untouched."
