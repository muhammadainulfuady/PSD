$b = 'http://127.0.0.1:5050'
$urls = @(
  'index.php',
  'admin.php',
  'download.php?jenis=template',
  'download.php?jenis=template&contoh=1',
  'download.php?jenis=ekspor',
  'download.php?jenis=laporan',
  'download.php?jenis=log',
  'download.php?jenis=sampel&nama=tidakada.csv',
  'data.php?id=1',
  'download.php?jenis=ngawur'
)
$out = @()
foreach ($u in $urls) {
  $line = & curl.exe -s -o NUL -w "%{http_code} %{redirect_url}" "$b/$u"
  $out += ("{0,-45} -> {1}" -f $u, $line)
}
$out -join [Environment]::NewLine