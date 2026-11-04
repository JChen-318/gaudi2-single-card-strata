"""Generate static README SVGs using Python's standard library.

Usage: python docs/assets/make_gaudi2_assets.py
Performance source: docs/benchmarks/gaudi2-performance.csv
"""

import csv
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
BG, FG, MUTED, GRID = '#0d1117', '#f0f6fc', '#a5b4c5', '#303b4a'
BLUE, GREEN = '#58a6ff', '#3fb950'
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def start(height, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
            f'  <title id="title">{escape(title)}</title>',
            f'  <desc id="desc">{escape(desc)}</desc>',
            f'  <rect width="1200" height="{height}" rx="16" fill="{BG}"/>',
            f'  <g font-family="{FONT}">']


def text(x, y, value, size=22, color=FG, weight=400, anchor='start'):
    return f'    <text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=GRID, width=1):
    return f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def save(name, parts):
    (HERE / name).write_text('\n'.join([*parts, '  </g>', '</svg>', '']), encoding='utf-8', newline='\n')


def native_path():
    s = start(340, 'Strata native Gaudi2 backend',
              '512 IQ4 experts resident in HBM, 36 GDN layers and 12 QSA layers compiled as SynapseAI graphs, '
              'with four-row DFlash2 verification. See the README for the full architecture.')
    s += [text(56, 48, 'STRATA / NATIVE SYNAPSEAI BACKEND', 20, BLUE, 600),
          text(1144, 48, '1 × GAUDI2', 20, MUTED, anchor='end'),
          text(56, 116, '512 experts. One Gaudi2.', 60, weight=650),
          text(56, 157, 'Qwen3.8-Flash-Next · IQ4_XS · custom TPC kernels', 25, MUTED)]
    for x, heading, first, second in [
        (56, 'Resident in HBM', '512 IQ4 experts', '+ F32 scales'),
        (448, 'Native layer graphs', '36 GDN + 12 QSA', 'MME + TPC execution'),
        (840, 'DFlash2 verification', 'Four-row windows', 'Device-side argmax'),
    ]:
        s += [line(x, 196, x + 304, 196, BLUE, 2),
              text(x, 235, heading, 25, weight=600),
              text(x, 274, first, 24), text(x, 307, second, 22, MUTED)]
    save('gaudi2-native-path.svg', s)


def performance():
    with (HERE.parent / 'benchmarks/gaudi2-performance.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    s = start(710, 'Recorded Gaudi2 throughput',
              'Decode service measurements use DFlash2 block four and greedy sampling. '
              'Prefill uses 8192-row windows; 16K and 32K are service results, longer prompts are standalone probes. '
              'Decode and prefill have independent axes, both in tokens per second.')
    s += [text(56, 54, 'Measured on one Gaudi2.', 38, weight=650),
          text(56, 92, 'IQ4_XS · single request · service and probe results shown separately', 23, MUTED),
          line(56, 130, 82, 130, BLUE, 8), text(96, 138, 'Production service', 21, MUTED),
          line(400, 130, 426, 130, GREEN, 3), text(440, 138, 'Standalone probe', 21, MUTED)]
    panels = [('decode', 56, 'Decode', 'DFlash2 block 4 · greedy', 180, [0, 60, 120, 180]),
              ('prefill', 652, 'Prefill', '8,192-row windows', 4000, [0, 1000, 2000, 3000, 4000])]
    for phase, left, title, subtitle, maximum, ticks in panels:
        s += [text(left, 193, title, 32, weight=600), text(left, 229, subtitle, 22, MUTED)]
        x0, span = left + 74, 350
        for value in ticks:
            x = x0 + value / maximum * span
            s += [line(x, 270, x, 589), text(round(x, 2), 261, f'{value:,}', 19, MUTED, anchor='middle')]
        for i, row in enumerate(r for r in rows if r['phase'] == phase):
            y = 301 + i * 66
            val = float(row['tokens_per_s'])
            w = val / maximum * span
            probe = row['measurement'] == 'standalone probe'
            s.append(text(left, y + 8, row['context_label'], 24, weight=600))
            style = f'fill="none" stroke="{GREEN}" stroke-width="2"' if probe else f'fill="{BLUE}"'
            s.append(f'    <rect x="{x0}" y="{y-16}" width="{w:.2f}" height="30" rx="3" {style}/>')
            displayed = ('~' if row['approximate'] == 'true' else '') + (f'{val:,.0f}' if phase == 'prefill' else f'{val:.1f}')
            s.append(text(round(x0 + w + 10, 2), y + 7, displayed, 23, weight=600))
        s.append(text(left, 622, 'tokens/s · independent axis', 20, MUTED))
    s += [line(56, 648, 1144, 648),
          text(56, 681, 'Recorded project results · decode dated Oct 26, 2026 · benchmark notes in docs/', 20, MUTED)]
    save('gaudi2-performance.svg', s)


if __name__ == '__main__':
    native_path()
    performance()
    print('Generated two Gaudi2 README assets.')
