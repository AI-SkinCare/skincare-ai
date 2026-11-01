param([long]$AfterRecord = 0)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$latest = Get-WinEvent -LogName System -MaxEvents 1
$events = @()
if ($AfterRecord -gt 0) {
    $xpath = "*[System[(EventRecordID > $AfterRecord) and ((Provider[@Name='nvlddmkm'] and (Level=1 or Level=2 or Level=3)) or (Provider[@Name='Display'] and EventID=4101) or (Provider[@Name='Microsoft-Windows-WHEA-Logger'] and (Level=1 or Level=2 or Level=3)))]]"
    $events = @(Get-WinEvent -LogName System -FilterXPath $xpath -ErrorAction SilentlyContinue | ForEach-Object {
        @{record_id=$_.RecordId; provider=$_.ProviderName; event_id=$_.Id; level=$_.Level;
          time=$_.TimeCreated.ToUniversalTime().ToString('o'); xml=$_.ToXml()}
    })
    # SilentlyContinue permits an empty query, but access failures must not look healthy.
    if ($Error.Count -gt 0 -and $Error[0].FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') {
        throw $Error[0]
    }
}
@{latest_record=[long]$latest.RecordId; events=$events} | ConvertTo-Json -Depth 5 -Compress
