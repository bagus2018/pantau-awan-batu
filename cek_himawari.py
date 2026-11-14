# =====================================================================
# CEK AWAN HIMAWARI TERKINI DI BATU — v2 (ambang real-time hasil SEL 9d)
# Jalankan di NOTEBOOK COLAB TERPISAH. Setiap run disimpan ke Drive, dinamai tanggal & waktu akses.
# ---------------------------------------------------------------------
# Indikator utama: DINGIN TERINTEGRASI bergulir di hulu Batu
#     DI = Σ max(0, 221 − Tb_min) · Δt   selama W jam terakhir (citra tiap 30 menit)      [K·jam]
# Ambang (jendela W, Waspada, Siaga) dibaca dari ambang_realtime.json hasil SEL 9d di Drive.
# Hasil SEL 9b–9d: metrik ini paling membedakan banjir vs hujan biasa (lebih baik dari Tb minimum
# saja atau hujan GPM); kondisi tanah saja tidak membedakan.
# Indikator penelitian, BUKAN peringatan dini resmi — ikuti BMKG Jawa Timur & BPBD Kota Batu.
# =====================================================================
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'satpy', 's3fs', 'pyorbital'], check=False)
import os, bz2, json, warnings, builtins
import numpy as np, pandas as pd, matplotlib.pyplot as plt, s3fs
from satpy import Scene
from IPython.display import display
warnings.filterwarnings('ignore')

# ---------------- PENGATURAN ----------------
PAKAI_GEE   = True                    # hujan GSMaP 3 jam/24 jam/3 hari sebagai informasi tambahan
GEE_PROJECT = 'bsw-um-2403'
OUT_BASE    = '/content/drive/MyDrive/HMS_GDG_MERIT_output'
REGION   = (112.20, -8.30, 113.00, -7.55)
BATU_BOX = (112.45, -7.96, 112.62, -7.72)             # hulu Batu (sama dengan SEL 9)
TITIK    = {'Batu': (112.525, -7.87), 'Bumiaji': (112.55, -7.81), 'Sumber Brantas': (112.53, -7.76),
            'Singosari': (112.65, -7.89), 'Kota Malang': (112.63, -7.98)}
HIST_TB  = {'Banjir bandang 4 Nov 2021 (Bulukerto)': 187.1, 'Banjir bandang 8–10 Nov 2025 (Coban Rais)': 192.0,
            'Luapan 20–22 Feb 2026 (Bumiaji)': 186.7, 'Luapan 3–4 Jan 2026 (Dusun Beru)': 187.8,
            'Luapan 29–30 Mar 2026 (Krecek)': 200.3, 'Genangan drainase 2 Jul 2025 (Selecta)': 219.5}

# ---------------- DRIVE, LOG TEKS, AMBANG ----------------
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'): drive.mount('/content/drive')
RUN_WIB = pd.Timestamp.now(tz='Asia/Jakarta').tz_localize(None)
LAPORAN = []
def print(*a, **k):
    builtins.print(*a, **k); LAPORAN.append(k.get('sep', ' ').join(str(x) for x in a))
AMB = dict(jendela_jam=12, ambang_siaga_Kjam=91.0, ambang_waspada_Kjam=64.0, tss_loocv=0.57,
           sumber='bawaan SEL 9b (±12 jam) — jalankan SEL 9d untuk ambang real-time')
fj = f'{OUT_BASE}/ambang_realtime.json'
if os.path.exists(fj):
    AMB.update(json.load(open(fj))); AMB['sumber'] = f"SEL 9d ({AMB.get('dibuat', '?')})"
W = int(AMB['jendela_jam']); SIAGA = float(AMB['ambang_siaga_Kjam']); WASPADA = float(AMB['ambang_waspada_Kjam'])

# ---------------- CITRA ----------------
fs = s3fs.S3FileSystem(anon=True)
TMP = '/content/tmp_hima'; os.makedirs(TMP, exist_ok=True)
def b13(d):
    try:
        f = [x for x in fs.ls(f'noaa-himawari9/AHI-L1b-FLDK/{d:%Y/%m/%d/%H%M}/', refresh=True)
             if '_B13_' in x and 'S0610' in x]
        return f[0] if f else None
    except Exception:
        return None
now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None).floor('10min')
latest = next(((d, p) for d in pd.date_range(now_utc - pd.Timedelta(hours=6), now_utc, freq='10min')[::-1]
               for p in [b13(d)] if p), None)
if latest is None:
    raise SystemExit('Tidak ada citra B13 dalam 6 jam terakhir di noaa-himawari9.')
# citra tiap 30 menit selama W jam ke belakang (sama dengan langkah analisis SEL 9) + citra terbaru
t30 = pd.date_range(latest[0].floor('30min') - pd.Timedelta(hours=W) + pd.Timedelta(minutes=30),
                    latest[0].floor('30min'), freq='30min')
fl = [(d, b13(d)) for d in t30]
fl = [x for x in fl if x[1]]
if latest[0] not in [d for d, _ in fl]: fl.append(latest)

GRID = {}
def read(path):
    local = os.path.join(TMP, os.path.basename(path)[:-4])
    with fs.open(path, 'rb') as a, open(local, 'wb') as b: b.write(bz2.decompress(a.read()))
    scn = Scene([local], reader='ahi_hsd'); scn.load(['B13']); arr = scn['B13']
    if not GRID:
        lo, la = arr.attrs['area'].get_lonlats()
        ok = (lo >= REGION[0]) & (lo <= REGION[2]) & (la >= REGION[1]) & (la <= REGION[3])
        r, c = np.where(ok); GRID.update(r0=r.min(), r1=r.max()+1, c0=c.min(), c1=c.max()+1)
        GRID['lon'] = lo[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']]
        GRID['lat'] = la[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']]
    tb = np.asarray(arr[GRID['r0']:GRID['r1'], GRID['c0']:GRID['c1']].values, dtype=np.float32)
    del arr, scn; os.remove(local); return tb

builtins.print(f'Mengunduh {len(fl)} citra ({W} jam terakhir, tiap 30 menit + terbaru)...')
rec, tb_last, mask = [], None, None
for t, p in fl:
    tb = read(p)
    if mask is None:
        mask = ((GRID['lon'] >= BATU_BOX[0]) & (GRID['lon'] <= BATU_BOX[2]) &
                (GRID['lat'] >= BATU_BOX[1]) & (GRID['lat'] <= BATU_BOX[3]))
    v = tb[mask]
    rec.append(dict(WIB=t + pd.Timedelta(hours=7), Tb_min=float(v.min()), Tb_median=float(np.median(v)),
                    f235=(v < 235).mean(), f221=(v < 221).mean(), f210=(v < 210).mean()))
    tb_last = tb
TS = pd.DataFrame(rec).set_index('WIB').sort_index()
DATA_WIB = TS.index[-1]; now = TS.iloc[-1]

# ---------------- INDIKATOR UTAMA: DINGIN TERINTEGRASI BERGULIR ----------------
T30 = TS[TS.index.minute.isin([0, 30])]                    # hanya citra 30-menitan (konsisten dengan SEL 9d)
x = np.clip(221 - T30['Tb_min'], 0, None) * 0.5
T30 = T30.assign(DI=x.rolling(f'{W}h').sum())               # DI W jam terakhir di setiap waktu
DI = float(T30['DI'].iloc[-1]) if len(T30) else 0.0
DI_1j_lalu = float(T30['DI'].asof(T30.index[-1] - pd.Timedelta(hours=1))) if len(T30) > 2 else np.nan
status = 'SIAGA' if DI >= SIAGA else 'WASPADA' if DI >= WASPADA else 'NORMAL'
n_cit = len(T30); n_harus = W * 2
lengkap = n_cit >= 0.8 * n_harus

# ---------------- HUJAN GSMaP (informasi tambahan) ----------------
ante = None
if PAKAI_GEE:
    try:
        import ee
        try: ee.Initialize(project=GEE_PROJECT)
        except Exception: ee.Authenticate(); ee.Initialize(project=GEE_PROJECT)
        roi = ee.Geometry.Rectangle(list(BATU_BOX))
        gs = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRate')
        now_ee = ee.Date(pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%S'))
        # GSMaP terlambat beberapa jam → hitung mundur dari JAM DATA TERAKHIR yang tersedia, bukan dari sekarang
        t_end = ee.Date(gs.filterDate(now_ee.advance(-2, 'day'), now_ee).aggregate_max('system:time_start')).advance(1, 'hour')
        tot = lambda h: ee.Dictionary(gs.filterDate(t_end.advance(-h, 'hour'), t_end).sum()
                                        .reduceRegion(ee.Reducer.mean(), roi, 5000)).get('hourlyPrecipRate', 0)
        v_ = ee.Dictionary({'h3': tot(3), 'h24': tot(24), 'h72': tot(72), 't': t_end.millis()}).getInfo()
        ante = dict(P3j=v_['h3'] or 0.0, P24j=v_['h24'] or 0.0, P72j=v_['h72'] or 0.0,
                    s_d=pd.to_datetime(v_['t'], unit='ms') + pd.Timedelta(hours=7))
    except Exception as e:
        builtins.print('GSMaP dilewati:', str(e)[:120])

# ---------------- LAPORAN ----------------
kelas = ('puncak menembus tropopause' if now.Tb_min < 210 else 'Cumulonimbus dalam' if now.Tb_min < 221
         else 'awan konvektif' if now.Tb_min < 235 else 'tidak konvektif')
tb_titik = {n: float(tb_last[np.unravel_index(np.argmin((GRID['lon']-a)**2 + (GRID['lat']-b)**2), tb_last.shape)])
            for n, (a, b) in TITIK.items()}
tren = now.Tb_min - TS['Tb_min'].asof(DATA_WIB - pd.Timedelta(minutes=30))
print('=' * 78)
print(f'Waktu akses : {RUN_WIB:%d %b %Y %H:%M:%S} WIB   |   Waktu citra : {DATA_WIB:%d %b %Y %H:%M} WIB')
print('=' * 78)
print(f'STATUS: {status}   —   dingin terintegrasi {W} jam terakhir = {DI:.0f} K·jam '
      f'(Waspada ≥ {WASPADA:.0f} | Siaga ≥ {SIAGA:.0f})')
if np.isfinite(DI_1j_lalu): print(f'  satu jam lalu: {DI_1j_lalu:.0f} K·jam → {"naik" if DI > DI_1j_lalu + 1 else "turun" if DI < DI_1j_lalu - 1 else "stabil"}')
if not lengkap: print(f'  ⚠ hanya {n_cit} dari {n_harus} citra 30-menitan tersedia → nilai DI bisa lebih rendah dari sebenarnya')
print(f'  ambang: {AMB["sumber"]}; kemampuan pada data historis: TSS {AMB.get("tss_loocv", "?")}'
      + (f', POD {AMB["pod_loocv"]}, FAR {AMB["far_loocv"]}' if 'pod_loocv' in AMB else ''))
print(f'\nKondisi awan saat ini: Tb min {now.Tb_min:.1f} K ({now.Tb_min-273.15:.0f} °C) → {kelas}; '
      f'tren 30 menit {tren:+.1f} K')
print(f'  luas hulu Batu tertutup awan <235 / <221 / <210 K = {now.f235*100:.0f} / {now.f221*100:.0f} / {now.f210*100:.0f} %')
print('  ' + ' | '.join(f'{n} {v:.0f} K' for n, v in tb_titik.items()))
if ante:
    print(f"  Hujan GSMaP hulu Batu (s/d {ante['s_d']:%d %b %H:%M} WIB): 3 jam {ante['P3j']:.1f} | 24 jam {ante['P24j']:.1f} | "
          f"3 hari {ante['P72j']:.1f} mm (cenderung underestimate hujan konvektif lokal)")
print('\nArti status:')
print(f'  NORMAL  : awan dingin belum bertahan cukup lama. Hari hujan biasa umumnya di sini.')
print(f'  WASPADA : ≥70% ambang; pantau ulang tiap 30–60 menit.')
print(f'  SIAGA   : setara pola awan saat kejadian banjir historis (16 kejadian 2020–2026).')
far_, pod_ = AMB.get('far_loocv'), AMB.get('pod_loocv')
if far_ is not None and pod_ is not None:
    print(f'Catatan: pada data historis ±{far_*100:.0f}% status SIAGA tidak diikuti banjir, dan ±{(1-pod_)*100:.0f}% banjir')
    print('  (sel sangat lokal/singkat) tidak terdeteksi.')
print('Bukan peringatan resmi — ikuti BMKG Jawa Timur & BPBD Kota Batu.')

cmp_ = pd.DataFrame({'Tb_min_K': HIST_TB}); cmp_.loc['>>> SAAT INI', 'Tb_min_K'] = now.Tb_min
display(cmp_.sort_values('Tb_min_K').round(1))

# ---------------- GRAFIK ----------------
STAMP = f'{RUN_WIB:%Y%m%d_%H%M%S}WIB_citra{DATA_WIB:%H%M}'
OUT_RUN = f'{OUT_BASE}/cek_himawari_batu/{RUN_WIB:%Y%m%d}'; os.makedirs(OUT_RUN, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1.1, 1, 1]})
pm = ax[0].pcolormesh(GRID['lon'], GRID['lat'], tb_last, cmap='turbo_r', vmin=185, vmax=300, shading='auto')
ax[0].contour(GRID['lon'], GRID['lat'], tb_last, levels=[210, 221, 235], colors=['k', 'm', 'w'], linewidths=1)
ax[0].add_patch(plt.Rectangle(BATU_BOX[:2], BATU_BOX[2]-BATU_BOX[0], BATU_BOX[3]-BATU_BOX[1], fill=False, ec='r', lw=2))
for n, (a, b) in TITIK.items(): ax[0].plot(a, b, 'k^', ms=6); ax[0].text(a + .01, b, n, fontsize=8)
plt.colorbar(pm, ax=ax[0], label='Tb (K)'); ax[0].set_title('a. Himawari-9 B13 (kontur 210/221/235 K)')
ax[1].plot(TS.index, TS['Tb_min'], 'b-o', ms=3, label='Tb min hulu Batu')
ax[1].axhspan(184, 207, color='red', alpha=.08, label='rentang banjir historis (184–207 K)')
ax[1].axhline(221, color='m', lw=1, label='221 K (dasar perhitungan DI)')
ax[1].invert_yaxis(); ax[1].set_ylabel('Tb (K) — makin atas makin dingin'); ax[1].legend(fontsize=8)
ax[1].grid(ls='--', alpha=.4); ax[1].set_title(f'b. Tb minimum {W} jam terakhir')
col = {'NORMAL': 'tab:green', 'WASPADA': 'orange', 'SIAGA': 'red'}[status]
ax[2].fill_between(T30.index, T30['DI'], color=col, alpha=.3); ax[2].plot(T30.index, T30['DI'], color=col, lw=2)
ax[2].axhline(SIAGA, color='red', ls='--', label=f'Siaga ≥ {SIAGA:.0f}')
ax[2].axhline(WASPADA, color='orange', ls=':', label=f'Waspada ≥ {WASPADA:.0f}')
ax[2].set_ylim(0, max(SIAGA * 1.3, T30['DI'].max() * 1.1 if len(T30) else 1))
ax[2].set_ylabel(f'Dingin terintegrasi {W} jam (K·jam)'); ax[2].legend(fontsize=8); ax[2].grid(ls='--', alpha=.4)
ax[2].set_title(f'c. STATUS {status}: {DI:.0f} K·jam')
fig.suptitle(f'Citra {DATA_WIB:%d %b %Y %H:%M} WIB   |   akses {RUN_WIB:%d %b %Y %H:%M:%S} WIB', fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(f'{OUT_RUN}/peta_tren_{STAMP}.png', dpi=160, bbox_inches='tight'); plt.show()

# ---------------- SIMPAN (setiap run file sendiri; log kumulatif satu baris per run) ----------------
TS.join(T30[['DI']]).assign(waktu_akses_WIB=RUN_WIB).to_csv(f'{OUT_RUN}/deret_awan_{STAMP}.csv', index_label='waktu_citra_WIB')
with open(f'{OUT_RUN}/interpretasi_{STAMP}.txt', 'w') as fo: fo.write('\n'.join(LAPORAN))
logf = f'{OUT_BASE}/log_cek_himawari_batu.csv'
row = pd.DataFrame([dict(waktu_akses_WIB=RUN_WIB, waktu_citra_WIB=DATA_WIB, status=status, DI=DI, jendela_jam=W,
                         ambang_siaga=SIAGA, Tb_min=now.Tb_min, f235=now.f235, f221=now.f221, f210=now.f210,
                         n_citra=n_cit, **({k: ante[k] for k in ['P3j', 'P24j', 'P72j']} if ante else {}), **{f'Tb_{k}': v for k, v in tb_titik.items()}, ada_banjir='')])
if os.path.exists(logf): row = pd.concat([pd.read_csv(logf), row], ignore_index=True)
row.to_csv(logf, index=False)
builtins.print(f'\nTersimpan di {OUT_RUN}/: peta_tren_{STAMP}.png, deret_awan_{STAMP}.csv, interpretasi_{STAMP}.txt')
builtins.print(f'Log kumulatif: {logf}')
