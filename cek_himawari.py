"""
PANTAU AWAN HIMAWARI-9 HULU BATU — v2 untuk GitHub Actions (repositori pantau-awan-batu)
=========================================================================================
Dijalankan tiap jam oleh workflow (cron '15 * * * *'); hasil disimpan ke folder output/ (workflow: git add output/).

Indikator utama: DINGIN TERINTEGRASI bergulir di hulu Batu (hasil SEL 9b–9d)
    DI = Σ max(0, 221 − Tb_min) · Δt   selama W jam terakhir (citra tiap 30 menit)      [K·jam]
Status: NORMAL < WASPADA (≥ ambang_waspada) < SIAGA (≥ ambang_siaga; setara pola awan saat banjir historis)
Ambang dibaca dari ambang_realtime.json (hasil SEL 9d); bila tidak ada → bawaan 12 jam / 64 / 91 K·jam.

Keluaran setiap run (DITIMPA, untuk tampilan dashboard / Google Sites):
    output/index.html                 dashboard (status, DI, peta, narasi, riwayat 24 run)
    output/LATEST_peta_tren.png       peta awan + Tb + DI
    output/LATEST_interpretasi.txt    narasi
    output/LATEST_deret_awan.csv      deret Tb & DI W jam terakhir
    output/LATEST_banding_historis.csv
    output/log_cek_himawari_batu.csv  log kumulatif (satu baris per run, kolom ada_banjir diisi manual)
Arsip BERLABEL WAKTU (tidak pernah ditimpa) — hanya bila status ≥ SIMPAN_MULAI (bawaan SIAGA):
    output/arsip/YYYY/MM/DD/<waktu akses>_{peta.png, interpretasi.txt, deret_awan.csv}
    output/arsip/grid/YYYY/MM/DD/tb_YYYYMMDD_HHMM.npz   grid Tb mentah tiap citra
Hujan GSMaP (opsional) memakai secret GCP_SA_KEY (JSON service account Earth Engine).
Indikator penelitian, BUKAN peringatan dini resmi — ikuti BMKG Jawa Timur & BPBD Kota Batu.
"""
import os, bz2, json, html, warnings, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s3fs
from satpy import Scene
warnings.filterwarnings('ignore')

OUT = os.environ.get('OUT_DIR', 'output')
SIMPAN_MULAI = os.environ.get('SIMPAN_MULAI', 'SIAGA').upper()
GEE_PROJECT = os.environ.get('GEE_PROJECT', 'bsw-um-2403')
AMB = dict(jendela_jam=12, ambang_siaga_Kjam=91.0, ambang_waspada_Kjam=64.0, sumber='bawaan SEL 9b (±12 jam)')
if os.path.exists('ambang_realtime.json'):
    AMB.update(json.load(open('ambang_realtime.json'))); AMB['sumber'] = f"SEL 9d ({AMB.get('dibuat', '?')})"
W = int(AMB['jendela_jam']); SIAGA = float(AMB['ambang_siaga_Kjam']); WASPADA = float(AMB['ambang_waspada_Kjam'])
REGION   = (112.20, -8.30, 113.00, -7.55)
BATU_BOX = (112.45, -7.96, 112.62, -7.72)
TITIK    = {'Batu': (112.525, -7.87), 'Bumiaji': (112.55, -7.81), 'Sumber Brantas': (112.53, -7.76),
            'Singosari': (112.65, -7.89), 'Kota Malang': (112.63, -7.98)}
HIST_TB  = {'Banjir bandang 4 Nov 2021 (Bulukerto)': 187.1, 'Banjir bandang 8–10 Nov 2025 (Coban Rais)': 192.0,
            'Luapan 20–22 Feb 2026 (Bumiaji)': 186.7, 'Luapan 3–4 Jan 2026 (Dusun Beru)': 187.8,
            'Luapan 29–30 Mar 2026 (Krecek)': 200.3, 'Genangan drainase 2 Jul 2025 (Selecta)': 219.5}
WIB = pd.Timedelta(hours=7)
RUN_WIB = pd.Timestamp.now(tz='UTC').tz_localize(None) + WIB
os.makedirs(OUT, exist_ok=True)
fs = s3fs.S3FileSystem(anon=True); TMP = tempfile.mkdtemp(); LAP = []
def say(*a):
    t = ' '.join(str(x) for x in a); print(t); LAP.append(t)

# ---------------- 1. CITRA: tiap 30 menit selama W jam + citra 10-menit terbaru ----------------
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
if latest is None: raise SystemExit('Tidak ada citra B13 dalam 6 jam terakhir di noaa-himawari9.')
t30 = pd.date_range(latest[0].floor('30min') - pd.Timedelta(hours=W) + pd.Timedelta(minutes=30),
                    latest[0].floor('30min'), freq='30min')
fl = [(d, p) for d in t30 for p in [b13(d)] if p]
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

rec, GRIDS, mask = [], [], None
for t, p in fl:
    tb = read(p); t_wib = t + WIB; GRIDS.append((t_wib, tb))
    if mask is None:
        mask = ((GRID['lon'] >= BATU_BOX[0]) & (GRID['lon'] <= BATU_BOX[2]) &
                (GRID['lat'] >= BATU_BOX[1]) & (GRID['lat'] <= BATU_BOX[3]))
    v = tb[mask]
    rec.append(dict(waktu_citra_WIB=t_wib, Tb_min=float(v.min()), Tb_median=float(np.median(v)),
                    f235=(v < 235).mean(), f221=(v < 221).mean(), f210=(v < 210).mean(),
                    **{f'Tb_{n}': float(tb[np.unravel_index(np.argmin((GRID['lon']-a)**2 + (GRID['lat']-b)**2), tb.shape)])
                       for n, (a, b) in TITIK.items()}))
TS = pd.DataFrame(rec).set_index('waktu_citra_WIB').sort_index()
now = TS.iloc[-1]; DATA_WIB = TS.index[-1]; tb_last = GRIDS[-1][1]

# ---------------- 2. DINGIN TERINTEGRASI (hanya citra 30-menitan, konsisten dengan SEL 9d) ----------------
T30 = TS[TS.index.minute.isin([0, 30])].copy()
T30['DI'] = (np.clip(221 - T30['Tb_min'], 0, None) * 0.5).rolling(f'{W}h').sum()
DI = float(T30['DI'].iloc[-1]) if len(T30) else 0.0
DI_1j = float(T30['DI'].asof(T30.index[-1] - pd.Timedelta(hours=1))) if len(T30) > 2 else np.nan
status = 'SIAGA' if DI >= SIAGA else 'WASPADA' if DI >= WASPADA else 'NORMAL'
lengkap = len(T30) >= 0.8 * W * 2

# ---------------- 3. GSMaP (opsional) ----------------
ante = None
key = os.environ.get('GCP_SA_KEY') or os.environ.get('EE_SERVICE_ACCOUNT_JSON')
if key:
    try:
        import ee
        info = json.loads(key)
        ee.Initialize(ee.ServiceAccountCredentials(info['client_email'], key_data=key),
                      project=GEE_PROJECT or info.get('project_id'))
        roi = ee.Geometry.Rectangle(list(BATU_BOX))
        gs = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational').select('hourlyPrecipRate')
        now_ee = ee.Date(pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%S'))
        t_end = ee.Date(gs.filterDate(now_ee.advance(-2, 'day'), now_ee).aggregate_max('system:time_start')).advance(1, 'hour')
        tot = lambda h: ee.Dictionary(gs.filterDate(t_end.advance(-h, 'hour'), t_end).sum()
                                        .reduceRegion(ee.Reducer.mean(), roi, 5000)).get('hourlyPrecipRate', 0)
        v_ = ee.Dictionary({'h3': tot(3), 'h24': tot(24), 'h72': tot(72), 't': t_end.millis()}).getInfo()
        ante = dict(P3j=v_['h3'] or 0.0, P24j=v_['h24'] or 0.0, P72j=v_['h72'] or 0.0,
                    s_d=pd.to_datetime(v_['t'], unit='ms') + WIB)
    except Exception as e:
        print('GSMaP dilewati:', str(e)[:150])

# ---------------- 4. NARASI ----------------
kelas = ('puncak menembus tropopause' if now.Tb_min < 210 else 'Cumulonimbus dalam' if now.Tb_min < 221
         else 'awan konvektif' if now.Tb_min < 235 else 'tidak ada awan konvektif')
tren = now.Tb_min - TS['Tb_min'].asof(DATA_WIB - pd.Timedelta(minutes=30))
say(f'Waktu citra : {DATA_WIB:%d %b %Y %H:%M} WIB')
say(f'Waktu run   : {RUN_WIB:%d %b %Y %H:%M:%S} WIB')
say('')
say(f'STATUS: {status}  —  dingin terintegrasi {W} jam terakhir = {DI:.0f} K·jam  (Waspada ≥ {WASPADA:.0f} | Siaga ≥ {SIAGA:.0f})')
if np.isfinite(DI_1j):
    say(f'  satu jam lalu {DI_1j:.0f} K·jam → {"naik" if DI > DI_1j + 1 else "turun" if DI < DI_1j - 1 else "stabil"}')
if not lengkap: say(f'  ⚠ hanya {len(T30)} dari {W*2} citra 30-menitan tersedia; DI bisa lebih rendah dari sebenarnya')
say(f'  ambang: {AMB["sumber"]}' + (f'; pada data historis POD {AMB["pod_loocv"]}, FAR {AMB["far_loocv"]}, TSS {AMB["tss_loocv"]}'
                                     if 'pod_loocv' in AMB else ''))
say('')
say(f'Awan saat ini (hulu Batu): Tb min {now.Tb_min:.1f} K ({now.Tb_min-273.15:.0f} °C) → {kelas}; tren 30 menit {tren:+.1f} K')
say(f'  luas tertutup awan <235 / <221 / <210 K = {now.f235*100:.0f} / {now.f221*100:.0f} / {now.f210*100:.0f} %')
say('  ' + ' | '.join(f"{n} {now[f'Tb_{n}']:.0f} K" for n in TITIK))
if ante:
    say(f"  Hujan GSMaP (s/d {ante['s_d']:%d %b %H:%M} WIB): 3 jam {ante['P3j']:.1f} | 24 jam {ante['P24j']:.1f} | "
        f"3 hari {ante['P72j']:.1f} mm")
say('')
say('Arti status: NORMAL = awan dingin belum bertahan lama (hari hujan biasa umumnya di sini);')
say('  WASPADA = pantau ulang; SIAGA = setara pola awan saat 16 kejadian banjir historis 2020–2026.')
say('  Status dicapai rata-rata sekitar saat puncak hujan → berguna sebagai konfirmasi cepat di hulu dan')
say('  tenggang beberapa jam untuk wilayah hilir (Kota Malang).')
say('Bukan peringatan dini resmi. Untuk keselamatan ikuti BMKG Jawa Timur & BPBD Kota Batu.')

# ---------------- 5. GRAFIK ----------------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={'width_ratios': [1.1, 1, 1]})
pm = ax[0].pcolormesh(GRID['lon'], GRID['lat'], tb_last, cmap='turbo_r', vmin=185, vmax=300, shading='auto')
ax[0].contour(GRID['lon'], GRID['lat'], tb_last, levels=[210, 221, 235], colors=['k', 'm', 'w'], linewidths=1)
ax[0].add_patch(plt.Rectangle(BATU_BOX[:2], BATU_BOX[2]-BATU_BOX[0], BATU_BOX[3]-BATU_BOX[1], fill=False, ec='r', lw=2))
for n, (a, b) in TITIK.items(): ax[0].plot(a, b, 'k^', ms=5); ax[0].text(a+.01, b, n, fontsize=8)
plt.colorbar(pm, ax=ax[0], label='Tb (K)'); ax[0].set_title('a. Himawari-9 B13 (kontur 210/221/235 K)')
ax[1].plot(TS.index, TS['Tb_min'], 'b-o', ms=3, label='Tb min hulu Batu')
ax[1].axhspan(184, 207, color='red', alpha=.08, label='rentang banjir historis')
ax[1].axhline(221, color='m', lw=1, label='221 K'); ax[1].invert_yaxis(); ax[1].legend(fontsize=8)
ax[1].set_ylabel('Tb (K) — makin atas makin dingin'); ax[1].grid(ls='--', alpha=.4); ax[1].set_title(f'b. Tb minimum {W} jam terakhir')
col = {'NORMAL': 'tab:green', 'WASPADA': 'orange', 'SIAGA': 'red'}[status]
ax[2].fill_between(T30.index, T30['DI'], color=col, alpha=.3); ax[2].plot(T30.index, T30['DI'], color=col, lw=2)
ax[2].axhline(SIAGA, color='red', ls='--', label=f'Siaga ≥ {SIAGA:.0f}')
ax[2].axhline(WASPADA, color='orange', ls=':', label=f'Waspada ≥ {WASPADA:.0f}')
ax[2].set_ylim(0, max(SIAGA * 1.3, (T30['DI'].max() if len(T30) else 0) * 1.1))
ax[2].set_ylabel(f'Dingin terintegrasi {W} jam (K·jam)'); ax[2].legend(fontsize=8); ax[2].grid(ls='--', alpha=.4)
ax[2].set_title(f'c. STATUS {status}: {DI:.0f} K·jam')
fig.suptitle(f'Citra {DATA_WIB:%d %b %Y %H:%M} WIB  |  akses {RUN_WIB:%d %b %Y %H:%M:%S} WIB', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/LATEST_peta_tren.png', dpi=110, bbox_inches='tight')

# ---------------- 6. FILE TERKINI (ditimpa) + LOG ----------------
open(f'{OUT}/LATEST_interpretasi.txt', 'w').write('\n'.join(LAP))
TS.join(T30[['DI']]).assign(waktu_run_WIB=RUN_WIB).to_csv(f'{OUT}/LATEST_deret_awan.csv', index_label='waktu_citra_WIB')
cmp_ = pd.DataFrame({'Tb_min_K': HIST_TB}); cmp_.loc['>>> SAAT INI', 'Tb_min_K'] = now.Tb_min
cmp_.sort_values('Tb_min_K').to_csv(f'{OUT}/LATEST_banding_historis.csv')
logf = f'{OUT}/log_cek_himawari_batu.csv'
row = pd.DataFrame([dict(waktu_citra_WIB=DATA_WIB, waktu_run_WIB=RUN_WIB, status=status, DI=round(DI, 1), jendela_jam=W,
                         ambang_siaga=SIAGA, Tb_min=now.Tb_min, f235=now.f235, f221=now.f221, f210=now.f210,
                         n_citra=len(T30), **({k: ante[k] for k in ['P3j', 'P24j', 'P72j']} if ante else {}),
                         **{f'Tb_{n}': now[f'Tb_{n}'] for n in TITIK}, ada_banjir='')])
LOG = pd.concat([pd.read_csv(logf), row], ignore_index=True) if os.path.exists(logf) else row
LOG.to_csv(logf, index=False)

# ---------------- 7. ARSIP BERLABEL WAKTU (hanya bila status ≥ SIMPAN_MULAI) ----------------
rank = {'NORMAL': 0, 'WASPADA': 1, 'SIAGA': 2}
arsip_ok = rank[status] >= rank.get(SIMPAN_MULAI, 2)
if arsip_ok:
    STAMP = f'{RUN_WIB:%Y%m%d_%H%M%S}WIB_citra{DATA_WIB:%H%M}'
    d_ar = f'{OUT}/arsip/{RUN_WIB:%Y/%m/%d}'; os.makedirs(d_ar, exist_ok=True)
    fig.savefig(f'{d_ar}/{STAMP}_peta.png', dpi=110, bbox_inches='tight')
    open(f'{d_ar}/{STAMP}_interpretasi.txt', 'w').write('\n'.join(LAP))
    TS.join(T30[['DI']]).to_csv(f'{d_ar}/{STAMP}_deret_awan.csv', index_label='waktu_citra_WIB')
    for t_wib, g in GRIDS:
        gf = f"{OUT}/arsip/grid/{t_wib:%Y/%m/%d}/tb_{t_wib:%Y%m%d_%H%M}.npz"
        if not os.path.exists(gf):
            os.makedirs(os.path.dirname(gf), exist_ok=True)
            np.savez_compressed(gf, tb=g.astype(np.float16), waktu_WIB=str(t_wib))
    geo = f'{OUT}/arsip/grid/lonlat.npz'
    if not os.path.exists(geo): np.savez_compressed(geo, lon=GRID['lon'], lat=GRID['lat'])

# ---------------- 8. DASHBOARD output/index.html (untuk disematkan di Google Sites) ----------------
warna = {'NORMAL': '#16a34a', 'WASPADA': '#f59e0b', 'SIAGA': '#dc2626'}[status]
hist = LOG.tail(24).iloc[::-1]
def _st(r):
    s = r.get('status') if isinstance(r.get('status'), str) else (r.get('tingkat') if isinstance(r.get('tingkat'), str) else '–')
    return s
rows_h = ''.join(
    f"<tr><td>{html.escape(str(r['waktu_citra_WIB'])[:16])}</td><td>{html.escape(_st(r))}</td>"
    f"<td>{'' if pd.isna(r.get('DI', np.nan)) else f'{float(r.get('DI')):.0f}'}</td><td>{float(r['Tb_min']):.0f}</td></tr>"
    for _, r in hist.iterrows())
ver = f'{RUN_WIB:%Y%m%d%H%M}'
page = f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="900">
<title>Pantauan Awan Himawari-9 Hulu Batu</title>
<style>
 body{{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:16px;background:#fff;color:#1f2937}}
 .badge{{display:inline-block;padding:10px 18px;border-radius:10px;color:#fff;font-size:22px;font-weight:700;background:{warna}}}
 .kpi{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:8px 0 14px}}
 .kpi div{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px}}
 img{{max-width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb}}
 pre{{background:#f8fafc;padding:14px;border-radius:8px;border:1px solid #e2e8f0;overflow-x:auto;font-size:13px;line-height:1.5}}
 table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid #e5e7eb;padding:4px 8px;text-align:left}}
 small{{color:#6b7280}}
</style></head><body>
<div class="kpi">
 <span class="badge">{status}</span>
 <div>Dingin terintegrasi {W} jam<br><b>{DI:.0f} K·jam</b></div>
 <div>Ambang<br>Waspada ≥ {WASPADA:.0f} · Siaga ≥ {SIAGA:.0f}</div>
 <div>Tb min hulu Batu<br><b>{now.Tb_min:.0f} K</b> ({now.Tb_min-273.15:.0f} °C)</div>
 <div>Citra<br><b>{DATA_WIB:%d %b %Y %H:%M} WIB</b></div>
</div>
<img src="LATEST_peta_tren.png?v={ver}" alt="Peta dan tren awan Himawari-9 hulu Batu">
<pre>{html.escape(chr(10).join(LAP))}</pre>
<h4>Riwayat 24 pengecekan terakhir</h4>
<table><tr><th>Waktu citra (WIB)</th><th>Status</th><th>DI (K·jam)</th><th>Tb min (K)</th></tr>{rows_h}</table>
<p><small>Diperbarui otomatis tiap jam. Indikator penelitian (Universitas Negeri Malang), bukan peringatan dini resmi —
ikuti BMKG Jawa Timur &amp; BPBD Kota Batu.</small></p>
</body></html>"""
open(f'{OUT}/index.html', 'w').write(page)
print(f'Selesai: status {status}, DI {DI:.0f} K·jam; arsip berlabel waktu ' + ('DISIMPAN' if arsip_ok else 'tidak disimpan'))
