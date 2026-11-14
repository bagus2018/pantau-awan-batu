"""
MONITOR AWAN HIMAWARI-9 HULU BATU — v2 (ambang real-time SEL 9d), dijalankan tiap jam oleh GitHub Actions
==========================================================================================================
Indikator: DINGIN TERINTEGRASI bergulir di hulu Batu
    DI = Σ max(0, 221 − Tb_min) · Δt  selama W jam terakhir (citra tiap 30 menit)   [K·jam]
Status: NORMAL < WASPADA (≥ 70% ambang) < SIAGA (≥ ambang; setara pola awan saat banjir historis).
ARSIP HANYA DISIMPAN BILA STATUS ≥ SIMPAN_MULAI (bawaan SIAGA). Bila tidak, tidak ada file yang disimpan.
Ambang dibaca dari ambang_realtime.json di repositori (salin dari Drive hasil SEL 9d), atau dari variabel
lingkungan DI_JENDELA_JAM / DI_SIAGA / DI_WASPADA, atau nilai bawaan (12 jam, 91, 64 K·jam).
Arsip:
  arsip/grid/YYYY/MM/DD/tb_YYYYMMDD_HHMM.npz   grid Tb Malang Raya (citra 30-menitan W jam terakhir)
  arsip/log_metrik_30menit.csv                 metrik setiap citra (tanpa duplikat)
  arsip/log_run.csv                            satu baris per run tersimpan (status, DI, ...); kolom ada_banjir manual
  arsip/laporan/YYYY/MM/DD/<waktu akses>.txt   narasi
  arsip/peta/YYYY/MM/DD/<waktu akses>.png      peta + Tb + DI
Indikator penelitian, BUKAN peringatan dini resmi (ikuti BMKG Jawa Timur & BPBD Kota Batu).
"""
import os, bz2, json, warnings, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s3fs
from satpy import Scene
warnings.filterwarnings('ignore')

ARSIP = os.environ.get('ARSIP_DIR', 'arsip')
SIMPAN_MULAI = os.environ.get('SIMPAN_MULAI', 'SIAGA').upper()      # 'SIAGA' atau 'WASPADA'
AMB = dict(jendela_jam=12, ambang_siaga_Kjam=91.0, ambang_waspada_Kjam=64.0, sumber='bawaan SEL 9b')
if os.path.exists('ambang_realtime.json'):
    AMB.update(json.load(open('ambang_realtime.json'))); AMB['sumber'] = f"SEL 9d ({AMB.get('dibuat', '?')})"
W = int(os.environ.get('DI_JENDELA_JAM', AMB['jendela_jam']))
SIAGA = float(os.environ.get('DI_SIAGA', AMB['ambang_siaga_Kjam']))
WASPADA = float(os.environ.get('DI_WASPADA', AMB['ambang_waspada_Kjam']))
REGION   = (112.20, -8.30, 113.00, -7.55)
BATU_BOX = (112.45, -7.96, 112.62, -7.72)
TITIK    = {'Batu': (112.525, -7.87), 'Bumiaji': (112.55, -7.81), 'Sumber Brantas': (112.53, -7.76),
            'Singosari': (112.65, -7.89), 'Kota Malang': (112.63, -7.98)}
WIB = pd.Timedelta(hours=7)
RUN_WIB = pd.Timestamp.now(tz='UTC').tz_localize(None) + WIB
fs = s3fs.S3FileSystem(anon=True); TMP = tempfile.mkdtemp(); LAP = []
def say(*a):
    t = ' '.join(str(x) for x in a); print(t); LAP.append(t)

def b13(d):
    try:
        f = [x for x in fs.ls(f'noaa-himawari9/AHI-L1b-FLDK/{d:%Y/%m/%d/%H%M}/', refresh=True)
             if '_B13_' in x and 'S0610' in x]
        return f[0] if f else None
    except Exception:
        return None
now_utc = pd.Timestamp.now(tz='UTC').tz_localize(None).floor('30min')
t30 = pd.date_range(now_utc - pd.Timedelta(hours=W + 3), now_utc, freq='30min')
fl = [(d, p) for d in t30 for p in [b13(d)] if p]
if not fl: raise SystemExit('Tidak ada citra B13 di noaa-himawari9.')
t_last = fl[-1][0]
fl = [(d, p) for d, p in fl if d > t_last - pd.Timedelta(hours=W)]

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
TS['DI'] = (np.clip(221 - TS['Tb_min'], 0, None) * 0.5).rolling(f'{W}h').sum()
now = TS.iloc[-1]; DATA_WIB = TS.index[-1]; DI = float(now.DI)
status = 'SIAGA' if DI >= SIAGA else 'WASPADA' if DI >= WASPADA else 'NORMAL'
rank = {'NORMAL': 0, 'WASPADA': 1, 'SIAGA': 2}

say(f'Waktu akses {RUN_WIB:%d %b %Y %H:%M:%S} WIB | citra {DATA_WIB:%d %b %Y %H:%M} WIB | {len(TS)} citra / {W} jam')
say(f'STATUS {status}: dingin terintegrasi {W} jam = {DI:.0f} K·jam (Waspada ≥ {WASPADA:.0f}, Siaga ≥ {SIAGA:.0f}); '
    f'ambang: {AMB["sumber"]}')
say(f'Tb min hulu Batu {now.Tb_min:.1f} K; luas <235/<221/<210 K = {now.f235*100:.0f}/{now.f221*100:.0f}/{now.f210*100:.0f}%')
say('Titik: ' + ' | '.join(f"{n} {now[f'Tb_{n}']:.0f} K" for n in TITIK))
say('Bukan peringatan resmi — ikuti BMKG Jawa Timur & BPBD Kota Batu.')

if rank[status] < rank.get(SIMPAN_MULAI, 2):
    print(f'Status {status} < {SIMPAN_MULAI} → tidak ada arsip yang disimpan.'); raise SystemExit(0)

# ---------------- ARSIP ----------------
for t_wib, g in GRIDS:
    gf = f"{ARSIP}/grid/{t_wib:%Y/%m/%d}/tb_{t_wib:%Y%m%d_%H%M}.npz"
    if not os.path.exists(gf):
        os.makedirs(os.path.dirname(gf), exist_ok=True)
        np.savez_compressed(gf, tb=g.astype(np.float16), waktu_WIB=str(t_wib))
geo = f'{ARSIP}/grid/lonlat.npz'
if not os.path.exists(geo):
    os.makedirs(os.path.dirname(geo), exist_ok=True); np.savez_compressed(geo, lon=GRID['lon'], lat=GRID['lat'])
logm = f'{ARSIP}/log_metrik_30menit.csv'; Mx = TS.reset_index()
if os.path.exists(logm): Mx = pd.concat([pd.read_csv(logm, parse_dates=['waktu_citra_WIB']), Mx])
Mx.drop_duplicates('waktu_citra_WIB', keep='last').sort_values('waktu_citra_WIB').to_csv(logm, index=False)
STAMP = f'{RUN_WIB:%Y%m%d_%H%M%S}WIB_citra{DATA_WIB:%H%M}'; d_rel = f'{RUN_WIB:%Y/%m/%d}'
os.makedirs(f'{ARSIP}/laporan/{d_rel}', exist_ok=True)
open(f'{ARSIP}/laporan/{d_rel}/{STAMP}.txt', 'w').write('\n'.join(LAP))
logr = f'{ARSIP}/log_run.csv'
row = pd.DataFrame([dict(waktu_akses_WIB=RUN_WIB, waktu_citra_WIB=DATA_WIB, status=status, DI=DI, jendela_jam=W,
                         ambang_siaga=SIAGA, Tb_min=now.Tb_min, f210=now.f210, n_citra=len(TS), ada_banjir='')])
if os.path.exists(logr): row = pd.concat([pd.read_csv(logr), row], ignore_index=True)
row.to_csv(logr, index=False)
os.makedirs(f'{ARSIP}/peta/{d_rel}', exist_ok=True)
tb_last = GRIDS[-1][1]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={'width_ratios': [1.1, 1, 1]})
pm = ax[0].pcolormesh(GRID['lon'], GRID['lat'], tb_last, cmap='turbo_r', vmin=185, vmax=300, shading='auto')
ax[0].contour(GRID['lon'], GRID['lat'], tb_last, levels=[210, 221, 235], colors=['k', 'm', 'w'], linewidths=1)
ax[0].add_patch(plt.Rectangle(BATU_BOX[:2], BATU_BOX[2]-BATU_BOX[0], BATU_BOX[3]-BATU_BOX[1], fill=False, ec='r', lw=2))
for n, (a, b) in TITIK.items(): ax[0].plot(a, b, 'k^', ms=5); ax[0].text(a+.01, b, n, fontsize=8)
plt.colorbar(pm, ax=ax[0], label='Tb (K)'); ax[0].set_title('B13 (kontur 210/221/235 K)')
ax[1].plot(TS.index, TS['Tb_min'], 'b-o', ms=3); ax[1].axhspan(184, 207, color='red', alpha=.08)
ax[1].invert_yaxis(); ax[1].set_ylabel('Tb min hulu Batu (K)'); ax[1].grid(ls='--', alpha=.4)
ax[2].plot(TS.index, TS['DI'], color='red' if status == 'SIAGA' else 'orange', lw=2)
ax[2].axhline(SIAGA, color='red', ls='--', label=f'Siaga ≥ {SIAGA:.0f}')
ax[2].axhline(WASPADA, color='orange', ls=':', label=f'Waspada ≥ {WASPADA:.0f}')
ax[2].set_ylabel(f'DI {W} jam (K·jam)'); ax[2].legend(fontsize=8); ax[2].grid(ls='--', alpha=.4)
ax[2].set_title(f'STATUS {status}: {DI:.0f} K·jam')
fig.suptitle(f'Citra {DATA_WIB:%d %b %Y %H:%M} WIB | akses {RUN_WIB:%d %b %Y %H:%M:%S} WIB', fontsize=11)
fig.tight_layout(); fig.savefig(f'{ARSIP}/peta/{d_rel}/{STAMP}.png', dpi=100, bbox_inches='tight')
print(f'Arsip disimpan: {STAMP}')
