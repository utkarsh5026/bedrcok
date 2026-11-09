"""Shared look and feel for every HTML card kbforge renders in a notebook.

The styles are scoped under a ``.kbf`` class so they never leak into the rest of the
notebook. Colours come from JupyterLab's own CSS variables where possible, so the cards
follow the light/dark theme of SageMaker Studio, JupyterLab and VS Code automatically.
"""

from __future__ import annotations

import html

CSS = """
<style>
.kbf {
  --kbf-fg: var(--jp-ui-font-color1, #1f2328);
  --kbf-muted: var(--jp-ui-font-color2, #59636e);
  --kbf-card: var(--jp-layout-color1, #ffffff);
  --kbf-soft: var(--jp-layout-color2, #f6f8fa);
  --kbf-border: var(--jp-border-color2, #d1d9e0);
  --kbf-green: #1a7f37; --kbf-amber: #9a6700; --kbf-red: #cf222e;
  --kbf-blue: #0969da; --kbf-grey: #59636e;
  font-family: var(--jp-ui-font-family, -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif);
  font-size: 13px; line-height: 1.5; color: var(--kbf-fg);
  max-width: 980px; margin: 4px 0 8px;
}
body[data-jp-theme-light="false"] .kbf, body.vscode-dark .kbf, body.vscode-high-contrast .kbf {
  --kbf-green: #3fb950; --kbf-amber: #d29922; --kbf-red: #f85149; --kbf-blue: #4493f8; --kbf-grey: #9198a1;
  --kbf-border: var(--jp-border-color2, #3d444d); --kbf-soft: var(--jp-layout-color2, #151b23);
}
.kbf * { box-sizing: border-box; }
.kbf .card { background: var(--kbf-card); border: 1px solid var(--kbf-border); border-radius: 10px; overflow: hidden; }
.kbf .head { padding: 16px 20px 14px; border-bottom: 1px solid var(--kbf-border); }
.kbf .eyebrow { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--kbf-muted); font-weight: 600; }
.kbf h2 { margin: 2px 0 8px; font-size: 20px; font-weight: 650; line-height: 1.25; color: var(--kbf-fg); border: 0; padding: 0; }
.kbf h3 { margin: 0 0 10px; font-size: 12px; letter-spacing: .05em; text-transform: uppercase; color: var(--kbf-muted); font-weight: 600; border: 0; padding: 0; }
.kbf .section { padding: 16px 20px; border-bottom: 1px solid var(--kbf-border); }
.kbf .section:last-child { border-bottom: 0; }
.kbf .chips { display: flex; flex-wrap: wrap; gap: 6px; }
.kbf .chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--kbf-soft); border: 1px solid var(--kbf-border); color: var(--kbf-muted); }
.kbf .chip b { color: var(--kbf-fg); font-weight: 600; }
.kbf code, .kbf .mono { font-family: var(--jp-code-font-family, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace); font-size: 12px; }
.kbf .muted { color: var(--kbf-muted); }
.kbf .badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  padding: 1px 7px; border-radius: 5px; color: var(--c); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);
  background: color-mix(in srgb, var(--c) 10%, transparent); white-space: nowrap; }
.kbf .c-green { --c: var(--kbf-green); } .kbf .c-amber { --c: var(--kbf-amber); } .kbf .c-red { --c: var(--kbf-red); }
.kbf .c-blue { --c: var(--kbf-blue); } .kbf .c-grey { --c: var(--kbf-grey); }
.kbf .summary { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 13px; }
.kbf .summary span b { font-size: 15px; }
.kbf .banner { margin-top: 12px; padding: 10px 12px; border-radius: 8px; border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
  background: color-mix(in srgb, var(--c) 8%, transparent); }
.kbf .banner b { color: var(--c); }
.kbf .pipeline { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0; }
.kbf .stage { flex: 1 1 150px; min-width: 140px; padding: 10px 12px; background: var(--kbf-soft); border: 1px solid var(--kbf-border); border-radius: 8px; }
.kbf .stage .label { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--kbf-muted); font-weight: 600; }
.kbf .stage .value { font-weight: 600; margin-top: 2px; word-break: break-word; }
.kbf .stage .sub { font-size: 12px; color: var(--kbf-muted); margin-top: 2px; }
.kbf .arrow { display: flex; align-items: center; padding: 0 6px; color: var(--kbf-muted); font-size: 16px; }
.kbf .checks { list-style: none; margin: 0; padding: 0; }
.kbf .checks li { display: grid; grid-template-columns: 22px 1fr; gap: 6px; padding: 5px 0; }
.kbf .icon { width: 18px; height: 18px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 800; color: #fff; background: var(--c); margin-top: 1px; }
.kbf .bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 6px 0 4px; max-width: 420px; background: var(--kbf-soft); }
.kbf .bar i { display: block; height: 100%; }
.kbf .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; color: var(--kbf-muted); }
.kbf .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: 0; }
.kbf .steps { list-style: none; margin: 0; padding: 0; }
.kbf .steps > li { display: grid; grid-template-columns: 28px 1fr; gap: 10px; padding: 10px 0; border-top: 1px dashed var(--kbf-border); }
.kbf .steps > li:first-child { border-top: 0; padding-top: 0; }
.kbf .num { width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--kbf-border); display: flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 600; color: var(--kbf-muted); background: var(--kbf-soft); }
.kbf .row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; }
.kbf .title { font-weight: 600; }
.kbf .res { color: var(--kbf-muted); word-break: break-all; }
.kbf .explain { color: var(--kbf-muted); margin-top: 2px; }
.kbf .note { margin-top: 2px; font-size: 12px; }
.kbf .warn { margin-top: 6px; padding: 8px 10px; border-radius: 6px; border-left: 3px solid var(--c); background: color-mix(in srgb, var(--c) 8%, transparent); white-space: pre-wrap; }
.kbf details { margin-top: 6px; }
.kbf summary { cursor: pointer; color: var(--kbf-blue); font-size: 12px; }
.kbf pre { margin: 6px 0 0; padding: 10px 12px; background: var(--kbf-soft); border: 1px solid var(--kbf-border); border-radius: 6px;
  overflow-x: auto; font-size: 12px; line-height: 1.45; white-space: pre; }
.kbf .next { background: var(--kbf-soft); }
.kbf .grid2 { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
.kbf .grid2 .k { color: var(--kbf-muted); }
.kbf .spin { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--kbf-border); border-top-color: var(--kbf-blue);
  animation: kbf-spin .8s linear infinite; display: inline-block; margin-top: 2px; }
@keyframes kbf-spin { to { transform: rotate(360deg); } }
.kbf .prog > li { grid-template-columns: 24px 1fr auto; align-items: start; }
.kbf .time { color: var(--kbf-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.kbf .log { font-size: 12px; color: var(--kbf-blue); margin-top: 2px; }
.kbf .result { padding: 12px 0; border-top: 1px dashed var(--kbf-border); }
.kbf .result:first-child { border-top: 0; padding-top: 0; }
.kbf .score { display: inline-block; width: 60px; height: 6px; border-radius: 3px; background: var(--kbf-soft); vertical-align: middle; overflow: hidden; }
.kbf .score i { display: block; height: 100%; background: var(--kbf-blue); }
.kbf .snippet { margin-top: 4px; white-space: pre-wrap; }
.kbf a { color: var(--kbf-blue); }
@media (max-width: 640px) { .kbf .arrow { display: none; } .kbf .stage { flex-basis: 100%; margin-bottom: 6px; } }
</style>
"""

#: Colours used for file-type bars in the source check, picked to stay distinct in both themes.
SERIES = ["#2f81f7", "#3fb950", "#d29922", "#a371f7", "#f778ba", "#39c5cf", "#db6d28", "#8b949e"]


def esc(value: object) -> str:
    """HTML-escape anything before putting it in a card."""
    return html.escape(str(value), quote=True)


def card(inner: str) -> str:
    """Wrap card content in the scoped container with the stylesheet."""
    return f'{CSS}<div class="kbf"><div class="card">{inner}</div></div>'
