# =====================================================================
# CEK AWAN HIMAWARI TERKINI DI BATU (Versi Otomatis GitHub Actions)
# =====================================================================
import os
import json
import bz2
import html
import builtins
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI wajib untuk server Linux GitHub Actions
import matplotlib.pyplot as plt
import s3fs
from satpy import Scene
import ee

warnings.filterwarnings('ignore')

# =====================================================================
# 0. AUTENTIKASI GOOGLE EARTH ENGINE
# =====================================================================
sa_info = json.loads(os.environ['GCP_SA_KEY'])
ee_credentials = ee.ServiceAccountCredentials(sa_info['client_email'], key_data=json.dumps(sa_info))
ee.Initialize(ee_credentials, project='bsw-um-2403')

# =====================================================================
# 1. PARAMETER & KONFIGURASI WILAYAH BATU
# =====================================================================
JAM_KE_BELAKANG = 3                                    # deret waktu untuk laju pendinginan
REGION   = (112.20, -8.30, 113.00, -7.55)              # potongan peta (Malang Raya)
BATU_BOX = (112.45, -7.96, 112.62, -7.72)              # hulu Batu (sama dengan SEL 9)
TITIK    = {'Batu': (112.525, -7.87), 'Bumiaji': (112.55, -7.81), 'Sumber Brantas': (112.53, -7.76),
            'Singosari': (112.65, -7.89), 'Kota Malang': (112.63, -7.98)}

HIST_TB = {'Banjir bandang 4 Nov 2021 (Bulukerto)': 187.1, 'Banjir bandang 8–10 Nov 2025 (Coban Rais)': 192.0,
           'Luapan 20–22 Feb 2026 (Bumiaji)': 186.7, 'Luapan 3–4 Jan 2026 (Dusun Beru)': 187.8,
           'Luapan 29–30 Mar 2026 (Krecek)': 200.3, 'Genangan drainase 2 Jul 2025 (Selecta)': 219.5}

fs = s3fs.S3FileSystem(anon=True)
TMP = './tmp_hima'
OUT_DIR = './output'
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RUN_WIB = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
LAPORAN = []
def print(*a, **k):
    builtins.print(*a, **k)
    LAPORAN.append(k.get('sep', ' ').join(str(x) for x in a))

# =====================================================================
# 2. AMBIL DATA HIMAWARI-9 TERBARU DARI NOAA AWS S3
# =====================================================================
def latest_folders(n_hours):
    """Daftar folder 10-menit terbaru yang sudah berisi file B13 segmen 6."""
    now = pd.Timestamp.now(tz='UTC').tz_localize(None).floor('10min')
    out = []
    for d in pd.date_range(now - pd.Timedelta(hours=n_hours + 6), now, freq='10min')[::-1]:
        pref = f'noaa-himawari9/AHI-L1b-FLDK/{d:%Y/%m/%d/%H%M}/'
        try:
            f = [x for x in fs.ls(pref, refresh=True) if '_B13_' in x and 'S0610' in x]
        except Exception:
            continue
        if f:
            out.append((d, f[0]))
        if len(out) >= n_hours * 6 + 1:
            break
    return out[::-1]

GRID = {}
def read(path):
    local = os.path.join(TMP, os.path.basename(path)[:-4])
    with fs.open(path, 'rb') as a, open(local, 'wb') as b:
        b.write(bz2.decompress(a.read()))
    scn = Scene([local], reader='ahi_hsd')
    scn.load(['B13'])
    arr = scn['B13']
    if not GRID:
        lo, la = arr.attrs['area'].get_lonlats()
        ok = (lo >= REGION[0]) & (lo <= REGION[2]) & (la >= REGION[1]) & (la <= REGION[3])
        r, c = np.where(ok)
        GRID.update(r0=r.min(), r1=r.max()+1, c0=c.min(), c1=c.max()+1)
        GRID['lon'] = lo[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']]
        GRID['lat'] = la[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']]
    tb = np.asarray(arr[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']].values, dtype=np.float32)
    del arr, scn
    os.remove(local)
    return tb

fl = latest_folders(JAM_KE_BELAKANG)
if not fl:
    raise SystemExit('Tidak ada citra B13 dalam ~9 jam terakhir di noaa-himawari9 (server NOAA terlambat/akses gagal).')

print(f'{len(fl)} citra, terakhir {fl[-1][0] + pd.Timedelta(hours=7):%d %b %Y %H:%M} WIB '
      f'(data NOAA biasanya terlambat ±20–40 menit)')

rec, tb_last = [], None
for t, p in fl:
    tb = read(p)
    m = ((GRID['lon'] >= BATU_BOX[0]) & (GRID['lon'] <= BATU_BOX[2]) &
         (GRID['lat'] >= BATU_BOX[1]) & (GRID['lat'] <= BATU_BOX[3]))
    v = tb[m]
    rec.append(dict(WIB=t + pd.Timedelta(hours=7), Tb_min=v.min(), Tb_median=np.median(v),
                    f235=(v < 235).mean(), f221=(v < 221).mean(), f210=(v < 210).mean()))
    tb_last = tb

TS = pd.DataFrame(rec).set_index('WIB')
TS['Laju_K_per10mnt'] = TS['Tb_min'].diff()
now = TS.iloc[-1]

# =====================================================================
# 3. RINGKASAN & GRAFIK
# =====================================================================
def kelas(t):
    return ('puncak menembus tropopause (sangat dalam)' if t < 210 else
            'Cumulonimbus dalam' if t < 221 else 'awan konvektif' if t < 235 else 'awan sedang/hangat / tidak konvektif')

print(f"\nHULU BATU saat ini: Tb min = {now.Tb_min:.1f} K ({now.Tb_min-273.15:.0f} °C) → {kelas(now.Tb_min)}")
print(f"  luas awan < 235 K = {now.f235*100:.0f}% | < 221 K = {now.f221*100:.0f}% | < 210 K = {now.f210*100:.0f}% wilayah")
print(f"  pendinginan tercepat {JAM_KE_BELAKANG} jam terakhir: {-TS['Laju_K_per10mnt'].min():.1f} K per 10 menit "
      f"(banjir bandang 2021: 15,8 K/10 mnt; median banjir: ±10 K/10 mnt)")

for nama, (lo_, la_) in TITIK.items():
    i = np.unravel_index(np.argmin((GRID['lon'] - lo_) ** 2 + (GRID['lat'] - la_) ** 2), GRID['lon'].shape)
    print(f'  {nama:15s}: {tb_last[i]:.1f} K')

cmp_ = pd.DataFrame({'Tb_min_K': HIST_TB})
cmp_.loc['>>> SAAT INI', 'Tb_min_K'] = now.Tb_min
cmp_['Selisih_vs_sekarang'] = cmp_['Tb_min_K'] - now.Tb_min
cmp_sorted = cmp_.sort_values('Tb_min_K').round(1)

# Buat Grafik
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.1, 1]})
pm = ax[0].pcolormesh(GRID['lon'], GRID['lat'], tb_last, cmap='turbo_r', vmin=185, vmax=300, shading='auto')
ax[0].contour(GRID['lon'], GRID['lat'], tb_last, levels=[210, 221, 235], colors=['k', 'm', 'w'], linewidths=1)
ax[0].add_patch(plt.Rectangle(BATU_BOX[:2], BATU_BOX[2]-BATU_BOX[0], BATU_BOX[3]-BATU_BOX[1], fill=False, ec='r', lw=2))
for nama, (lo_, la_) in TITIK.items():
    ax[0].plot(lo_, la_, 'k^', ms=6)
    ax[0].text(lo_ + .01, la_, nama, fontsize=8)
plt.colorbar(pm, ax=ax[0], label='Tb (K)')
ax[0].set_title(f'Himawari-9 B13 {TS.index[-1]:%d %b %Y %H:%M} WIB\n(kontur: hitam 210 K, ungu 221 K, putih 235 K)')

ax[1].plot(TS.index, TS['Tb_min'], 'b-o', ms=3, label='Tb min hulu Batu (saat ini)')
for nama, v in HIST_TB.items():
    ax[1].axhline(v, ls='--', lw=1, color='red' if 'bandang' in nama else 'grey')
ax[1].axhspan(184, 207, color='red', alpha=.08, label='rentang semua banjir historis (184–207 K)')
ax[1].axhline(221, color='m', lw=1, label='221 K (Cb dalam)')
ax[1].invert_yaxis()
ax[1].set_ylabel('Tb (K) — makin atas makin dingin')
ax[1].grid(ls='--', alpha=.4)
ax[1].set_title(f'{JAM_KE_BELAKANG} jam terakhir vs Tb banjir historis (garis merah = banjir bandang)')
ax[1].legend(fontsize=8)

DATA_WIB = TS.index[-1]
fig.suptitle(f'Citra Himawari-9: {DATA_WIB:%d %b %Y %H:%M} WIB   |   dijalankan: {RUN_WIB:%d %b %Y %H:%M:%S} WIB',
             fontsize=11, y=1.02)
fig.tight_layout()

png_latest = f'{OUT_DIR}/LATEST_peta_tren.png'
fig.savefig(png_latest, dpi=160, bbox_inches='tight')
plt.close(fig)

# =====================================================================
# 4. INTERPRETASI OTOMATIS & HUJAN PENDAHULU GSMaP (GEE)
# =====================================================================
PAKAI_GEE  = True
SIMPAN_LOG = True

ante = None
if PAKAI_GEE:
    try:
        roi = ee.Geometry.Rectangle(list(BATU_BOX))
        t_end = ee.Date(pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%S'))
        gs = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRate')
        def tot(h):
            return gs.filterDate(t_end.advance(-h, 'hour'), t_end).sum() \
                     .reduceRegion(ee.Reducer.mean(), roi, 5000).get('hourlyPrecipRate')
        last = gs.filterDate(t_end.advance(-24, 'hour'), t_end).aggregate_max('system:time_start')
        v = ee.Dictionary({'h3': tot(3), 'h24': tot(24), 'h72': tot(72), 'last': last}).getInfo()
        ante = dict(P3j=v['h3'] or 0, P24j=v['h24'] or 0, P72j=v['h72'] or 0,
                    data_terakhir=pd.to_datetime(v['last'], unit='ms') + pd.Timedelta(hours=7) if v['last'] else None)
    except Exception as e:
        print('GSMaP/GEE tidak dipakai:', str(e)[:120])

win1h = TS[TS.index >= TS.index[-1] - pd.Timedelta(hours=1)]
dt_h_ = (TS.index[1] - TS.index[0]).total_seconds() / 3600 if len(TS) > 1 else 1/6
dur221 = (TS['f221'] >= 0.5).sum() * dt_h_
cool1h = -win1h['Laju_K_per10mnt'].min() if len(win1h) > 1 else 0
trend30 = TS['Tb_min'].iloc[-1] - TS['Tb_min'].asof(TS.index[-1] - pd.Timedelta(minutes=30))
f210_naik = TS['f210'].diff(3).max()
onset = f210_naik >= 0.3

skor, alasan = 0, []
tb = now.Tb_min
if tb <= 192:   skor += 3; alasan.append(f'Tb {tb:.0f} K ≤ 192 K: sedingin banjir bandang 2021/2025')
elif tb <= 207: skor += 2; alasan.append(f'Tb {tb:.0f} K dalam rentang semua banjir historis (184–207 K)')
elif tb <= 221: skor += 1; alasan.append(f'Tb {tb:.0f} K: Cb dalam, tetapi lebih hangat dari semua banjir historis')
else:           alasan.append(f'Tb {tb:.0f} K: lebih hangat dari semua banjir historis (paling hangat 207 K)')

if now.f210 >= 0.5:    skor += 2; alasan.append(f'{now.f210*100:.0f}% hulu Batu tertutup awan < 210 K (luas)')
elif now.f210 >= 0.25: skor += 1; alasan.append(f'{now.f210*100:.0f}% hulu Batu tertutup awan < 210 K (sebagian)')

if dur221 >= 1:        skor += 1; alasan.append(f'awan < 221 K bertahan ±{dur221:.1f} jam dalam {JAM_KE_BELAKANG} jam terakhir')
if cool1h >= 10:       skor += 1; alasan.append(f'pendinginan {cool1h:.1f} K/10 mnt dalam 1 jam terakhir (≥ median banjir)')

if ante is not None:
    if ante['P72j'] >= 25:   skor += 2; alasan.append(f"hujan 3 hari {ante['P72j']:.0f} mm: tanah kemungkinan basah (median banjir ±25 mm)")
    elif ante['P72j'] >= 10: skor += 1; alasan.append(f"hujan 3 hari {ante['P72j']:.0f} mm: tanah agak basah")
    else:                    alasan.append(f"hujan 3 hari {ante['P72j']:.0f} mm: tanah relatif kering → risiko limpasan lebih rendah")

skor_maks = 7 + (2 if ante is not None else 0)
tingkat = ('TINGGI — pola awan menyerupai kejadian banjir historis' if skor >= 6 else
           'SEDANG — sebagian ciri banjir historis terpenuhi' if skor >= 3 else
           'RENDAH — belum menyerupai kejadian banjir historis')

inti = min(TITIK, key=lambda n: tb_last[np.unravel_index(np.argmin((GRID['lon']-TITIK[n][0])**2 +
                                                                   (GRID['lat']-TITIK[n][1])**2), GRID['lon'].shape)])
arah = ('makin dingin (menguat)' if trend30 < -3 else 'menghangat (melemah)' if trend30 > 3 else 'relatif stabil')

print('\n' + '=' * 78)
print(f'INTERPRETASI {TS.index[-1]:%d %b %Y %H:%M} WIB  |  skor kemiripan {skor} / {skor_maks}  →  {tingkat}')
print('=' * 78)
for a in alasan:
    print('  •', a)
print(f'  • tren 30 menit terakhir: {arah} ({trend30:+.1f} K); titik terdingin saat ini: {inti}')
if onset:
    print('  • luas awan < 210 K melonjak ≥ 30% dalam 30 menit → sistem awan BARU MASUK/berkembang cepat; '
          'laju pendinginan sebagian karena awan bergerak masuk, bukan hanya tumbuh di tempat')
if ante is not None:
    print(f"  • GSMaP (data s/d {ante['data_terakhir']:%H:%M} WIB): 3 jam {ante['P3j']:.1f} mm | 24 jam {ante['P24j']:.1f} mm | "
          f"3 hari {ante['P72j']:.1f} mm  (GSMaP cenderung underestimate hujan konvektif lokal)")

print('\nCatatan:')
print('  - Skor ini membandingkan dengan 16 banjir/luapan historis; ambang awan belum diuji terhadap hari hujan')
print('    tanpa banjir (SEL 9b). Awan sangat dingin juga sering terjadi tanpa banjir.')
print('  - Bukan peringatan dini resmi. Untuk keselamatan ikuti BMKG Jawa Timur & BPBD Kota Batu.')

# =====================================================================
# 5. SIMPAN HASIL KE FOLDER OUTPUT REPOSITORI GITHUB
# =====================================================================
TS.assign(waktu_run_WIB=RUN_WIB).to_csv(f'{OUT_DIR}/LATEST_deret_awan.csv', index_label='waktu_citra_WIB')
cmp_sorted.to_csv(f'{OUT_DIR}/LATEST_banding_historis.csv')

teks_laporan = (f'Waktu citra : {DATA_WIB:%d %b %Y %H:%M} WIB\n'
                f'Waktu run   : {RUN_WIB:%d %b %Y %H:%M:%S} WIB\n\n' + '\n'.join(LAPORAN))
with open(f'{OUT_DIR}/LATEST_interpretasi.txt', 'w', encoding='utf-8') as fo:
    fo.write(teks_laporan)

# Buat halaman HTML siap-embed untuk Google Sites
html_out = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pantauan Awan Himawari-9 Hulu Batu</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px; background: #fff; color: #1f2937; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e5e7eb; }}
  pre {{ background: #f8fafc; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
</style>
</head>
<body>
  <img src="LATEST_peta_tren.png?v={RUN_WIB:%Y%m%d%H%M}" alt="Peta Tren Himawari-9 Batu">
  <pre>{html.escape(teks_laporan)}</pre>
</body>
</html>"""
with open(f'{OUT_DIR}/index.html', 'w', encoding='utf-8') as fh:
    fh.write(html_out)

# Perbarui Log Studi Kasus Kumulatif
if SIMPAN_LOG:
    log_path = f'{OUT_DIR}/log_cek_himawari_batu.csv'
    row = dict(
        waktu_citra_WIB=TS.index[-1], waktu_run_WIB=RUN_WIB, Tb_min=tb,
        f235=now.f235, f221=now.f221, f210=now.f210, cool1h=cool1h,
        dur221_j=dur221, tren30=trend30, inti=inti, skor=skor,
        tingkat=tingkat.split(' —')[0],
        **({k: ante[k] for k in ['P3j', 'P24j', 'P72j']} if ante else {}),
        **{f'Tb_{n}': tb_last[np.unravel_index(np.argmin((GRID['lon']-x)**2 + (GRID['lat']-y)**2),
                                               GRID['lon'].shape)] for n, (x, y) in TITIK.items()}
    )
    new_df = pd.DataFrame([row])
    if os.path.exists(log_path):
        new_df = pd.concat([pd.read_csv(log_path), new_df], ignore_index=True)
    new_df.to_csv(log_path, index=False)

builtins.print('\nSelesai! Seluruh hasil berhasil disimpan ke folder output/.')
