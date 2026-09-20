$ErrorActionPreference = 'Stop'
$b = 'http://127.0.0.1:5050'
$out = @()

function Cek($label, $url) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing -MaximumRedirection 0 -ErrorAction Stop $url
    return "$label -> HTTP $($r.StatusCode), bytes=$($r.Content.Length)"
  } catch {
    $code = $_.Exception.Response.StatusCode.value__
    return "$label -> HTTP $code"
  }
}

$out += Cek 'index' "$b/index.php"
$out += Cek 'admin (belum login)' "$b/admin.php"
$out += Cek 'template' "$b/download.php?jenis=template"
$out += Cek 'template+contoh' "$b/download.php?jenis=template&contoh=1"
$out += Cek 'ekspor TANPA login' "$b/download.php?jenis=ekspor"
$out += Cek 'laporan TANPA login' "$b/download.php?jenis=laporan"
$out += Cek 'log TANPA login' "$b/download.php?jenis=log"
$out += Cek 'sampel tidak ada' "$b/download.php?jenis=sampel&nama=tidakada.csv"
$out += Cek 'data id=1' "$b/data.php?id=1"
$out += Cek 'jenis ngawur' "$b/download.php?jenis=ngawur"
$out -join [Environment]::NewLine