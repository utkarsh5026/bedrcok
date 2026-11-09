"""The look of ``kbforge.wizard()``: its stylesheet and the small HTML pieces it is built from.

The wizard is made of ipywidgets (text boxes, buttons) mixed with HTML snippets (the header,
section titles, the live preview). ipywidgets have a plain default look, so this module ships
a stylesheet that restyles them — but only inside the wizard's root box, which carries the
``kbf-wiz`` class, so nothing else in the notebook changes.

Everything here is plain functions returning HTML strings, so it can be tested without a
browser or ipywidgets. Colours come from the ``--kbf-*`` variables in ``theme.py`` and so
follow JupyterLab's light/dark theme like every other kbforge card.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .theme import CSS, esc

WIZARD_CSS = """
<style>
.kbf.kbf-wiz { max-width: 1120px; margin: 6px 0 12px; overflow: visible; }
.kbf-wiz .kbf-shell { background: var(--kbf-card); border: 1px solid var(--kbf-border); border-radius: 14px; overflow: clip;
  box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px -12px rgba(0,0,0,.18); }
.kbf-wiz .widget-html, .kbf-wiz .widget-html-content { margin: 0; padding: 0; line-height: inherit; font-size: inherit; width: 100%; height: auto; }
.kbf-wiz .widget-box { overflow: visible; }

/* Header: brand mark, title, reassurance chip and the three-stage stepper */
.kbf-wiz .hero { position: relative; padding: 22px 26px 20px; border-bottom: 1px solid var(--kbf-border);
  background:
    radial-gradient(900px 160px at 0% 0%, color-mix(in srgb, var(--kbf-blue) 14%, transparent), transparent 70%),
    radial-gradient(700px 180px at 100% 0%, color-mix(in srgb, var(--kbf-purple) 12%, transparent), transparent 70%),
    var(--kbf-card); }
.kbf-wiz .hero-top { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.kbf-wiz .mark { flex: 0 0 auto; width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  color: #fff; background: linear-gradient(135deg, var(--kbf-blue), var(--kbf-purple)); box-shadow: 0 6px 16px -6px color-mix(in srgb, var(--kbf-blue) 70%, transparent); }
.kbf-wiz .mark svg { width: 24px; height: 24px; }
.kbf-wiz .hero-text { flex: 1 1 320px; min-width: 0; }
.kbf-wiz .hero h2 { margin: 1px 0 4px; font-size: 22px; letter-spacing: -.01em; }
.kbf-wiz .hero .lede { color: var(--kbf-muted); max-width: 640px; }
.kbf-wiz .safe { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 5px 10px;
  border-radius: 999px; color: var(--kbf-green); background: color-mix(in srgb, var(--kbf-green) 10%, var(--kbf-card));
  border: 1px solid color-mix(in srgb, var(--kbf-green) 30%, transparent); }
.kbf-wiz .safe svg { width: 14px; height: 14px; }
.kbf-wiz .stepper { display: flex; align-items: center; margin-top: 18px; max-width: 760px; }
.kbf-wiz .stp { display: flex; align-items: center; gap: 10px; flex: 0 0 auto; }
.kbf-wiz .stp .dot { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;
  border: 2px solid var(--kbf-border); color: var(--kbf-muted); background: var(--kbf-card); transition: all .2s ease; }
.kbf-wiz .stp .dot svg { width: 15px; height: 15px; }
.kbf-wiz .stp .t { font-weight: 600; line-height: 1.2; }
.kbf-wiz .stp .s { font-size: 11.5px; color: var(--kbf-muted); line-height: 1.2; }
.kbf-wiz .stp.on .dot { border-color: var(--kbf-blue); background: var(--kbf-blue); color: #fff; box-shadow: 0 0 0 5px color-mix(in srgb, var(--kbf-blue) 18%, transparent); }
.kbf-wiz .stp.done .dot { border-color: var(--kbf-green); background: var(--kbf-green); color: #fff; }
.kbf-wiz .stp.todo .t { color: var(--kbf-muted); }
.kbf-wiz .rail { flex: 1 1 40px; height: 2px; margin: 0 14px; border-radius: 2px; background: var(--kbf-border); position: relative; overflow: hidden; }
.kbf-wiz .rail.done { background: var(--kbf-green); }

/* Two-column body: the form on the left, the live preview on the right */
.kbf-wiz .kbf-body { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 0; }
.kbf-wiz .kbf-form { flex: 1 1 520px; min-width: 0; padding: 6px 26px 10px; }
.kbf-wiz .kbf-aside { flex: 0 1 330px; min-width: 280px; padding: 20px 22px 22px; align-self: stretch;
  background: var(--kbf-soft); border-left: 1px solid var(--kbf-border); }
.kbf-wiz .kbf-aside > .widget-html-content { position: sticky; top: 12px; }
.kbf-wiz .kbf-sec { padding: 18px 0 16px; border-bottom: 1px solid var(--kbf-border); }
.kbf-wiz .kbf-sec:last-child { border-bottom: 0; }
.kbf-wiz .sec-head { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
.kbf-wiz .sec-num { flex: 0 0 auto; width: 24px; height: 24px; border-radius: 7px; display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: var(--kbf-blue); background: color-mix(in srgb, var(--kbf-blue) 12%, transparent); }
.kbf-wiz .sec-title { font-size: 14.5px; font-weight: 650; line-height: 24px; }
.kbf-wiz .sec-hint { color: var(--kbf-muted); font-size: 12.5px; margin-top: 1px; }
.kbf-wiz .sec-body { padding-left: 36px; }
.kbf-wiz .lbl { font-size: 12px; font-weight: 600; color: var(--kbf-muted); margin: 16px 0 6px; }

/* Text inputs */
.kbf-wiz .kbf-input { width: 100%; margin: 0; height: auto; }
.kbf-wiz .kbf-input input, .kbf-wiz .kbf-input textarea {
  width: 100%; height: 38px; padding: 8px 12px; font-size: 13.5px; font-family: inherit; color: var(--kbf-fg);
  background: var(--kbf-card); border: 1px solid var(--kbf-border); border-radius: 8px; outline: none; box-shadow: none;
  transition: border-color .15s ease, box-shadow .15s ease; }
.kbf-wiz .kbf-input textarea { height: 86px; resize: vertical; line-height: 1.5; font-family: var(--jp-code-font-family, ui-monospace, monospace); font-size: 12.5px; }
.kbf-wiz .kbf-input input::placeholder, .kbf-wiz .kbf-input textarea::placeholder { color: var(--kbf-muted); opacity: .7; }
.kbf-wiz .kbf-input input:focus, .kbf-wiz .kbf-input textarea:focus {
  border-color: var(--kbf-blue); box-shadow: 0 0 0 3px color-mix(in srgb, var(--kbf-blue) 20%, transparent); }
.kbf-wiz .kbf-input.bad input, .kbf-wiz .kbf-input.bad textarea { border-color: var(--kbf-red); }
.kbf-wiz .kbf-input.bad input:focus, .kbf-wiz .kbf-input.bad textarea:focus { box-shadow: 0 0 0 3px color-mix(in srgb, var(--kbf-red) 18%, transparent); }
.kbf-wiz .kbf-input.mono input { font-family: var(--jp-code-font-family, ui-monospace, monospace); font-size: 12.5px; }
.kbf-wiz .fhint { font-size: 12px; margin-top: 6px; color: var(--kbf-muted); display: flex; gap: 6px; align-items: flex-start; min-height: 18px; }
.kbf-wiz .fhint svg { width: 14px; height: 14px; flex: 0 0 auto; margin-top: 2px; }
.kbf-wiz .fhint.ok { color: var(--kbf-green); }
.kbf-wiz .fhint.bad { color: var(--kbf-red); }
.kbf-wiz .fhint code { background: var(--kbf-soft); border: 1px solid var(--kbf-border); border-radius: 4px; padding: 0 4px; color: var(--kbf-fg); }

/* Option cards: one clickable card per choice. A transparent button covers each card. */
.kbf-wiz .kbf-opts { display: grid; gap: 10px; width: 100%; margin: 0; }
.kbf-wiz .kbf-opt { position: relative; margin: 0; min-width: 0; border-radius: 10px; border: 1px solid var(--kbf-border); background: var(--kbf-card);
  transition: border-color .15s ease, box-shadow .15s ease, background .15s ease, transform .15s ease; }
.kbf-wiz .kbf-opt:hover { border-color: color-mix(in srgb, var(--kbf-blue) 55%, var(--kbf-border)); box-shadow: 0 4px 14px -8px rgba(0,0,0,.35); }
.kbf-wiz .kbf-opt:focus-within { box-shadow: 0 0 0 3px color-mix(in srgb, var(--kbf-blue) 25%, transparent); }
.kbf-wiz .kbf-opt.on { border-color: var(--kbf-blue); background: color-mix(in srgb, var(--kbf-blue) 7%, var(--kbf-card));
  box-shadow: 0 0 0 1px var(--kbf-blue) inset; }
.kbf-wiz .kbf-opt .widget-button.kbf-hit { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; padding: 0; opacity: 0; cursor: pointer; z-index: 2; border: 0; }
.kbf-wiz .opt { display: flex; gap: 11px; padding: 12px 14px; align-items: flex-start; }
.kbf-wiz .opt .ico { flex: 0 0 auto; width: 32px; height: 32px; border-radius: 9px; display: flex; align-items: center; justify-content: center;
  color: var(--kbf-muted); background: var(--kbf-soft); border: 1px solid var(--kbf-border); transition: all .15s ease; }
.kbf-wiz .opt .ico svg { width: 17px; height: 17px; }
.kbf-wiz .kbf-opt.on .opt .ico { color: #fff; background: var(--kbf-blue); border-color: var(--kbf-blue); }
.kbf-wiz .opt .txt { flex: 1 1 auto; min-width: 0; }
.kbf-wiz .opt .ttl { font-weight: 650; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding-right: 22px; }
.kbf-wiz .opt .bdy { color: var(--kbf-muted); font-size: 12.5px; margin-top: 2px; line-height: 1.45; }
.kbf-wiz .opt .meta { font-size: 11.5px; margin-top: 6px; color: var(--kbf-muted); }
.kbf-wiz .opt .tag { font-size: 10.5px; font-weight: 700; letter-spacing: .03em; padding: 1px 7px; border-radius: 999px; color: var(--c);
  background: color-mix(in srgb, var(--c) 12%, transparent); border: 1px solid color-mix(in srgb, var(--c) 30%, transparent); }
.kbf-wiz .opt .tick { position: absolute; top: 10px; right: 10px; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; border: 1.5px solid var(--kbf-border); color: transparent; background: var(--kbf-card); transition: all .15s ease; }
.kbf-wiz .opt .tick svg { width: 11px; height: 11px; }
.kbf-wiz .kbf-opt.on .opt .tick { border-color: var(--kbf-blue); background: var(--kbf-blue); color: #fff; }
.kbf-wiz .kbf-opts.pills .opt { padding: 8px 12px; gap: 8px; align-items: center; }
.kbf-wiz .kbf-opts.pills .opt .ttl { padding-right: 0; }
.kbf-wiz .kbf-opts.pills .opt .bdy { margin-top: 0; font-size: 11.5px; }
.kbf-wiz .kbf-opts.pills .opt .tick { display: none; }

/* Advanced options: a disclosure row and a few compact fields */
.kbf-wiz .widget-button.kbf-disclose { width: 100%; height: auto; margin: 0; padding: 0; text-align: left; background: transparent; border: 0; box-shadow: none;
  color: var(--kbf-fg); font-size: 14.5px; font-weight: 650; display: flex; align-items: center; gap: 10px; cursor: pointer; }
.kbf-wiz .widget-button.kbf-disclose:hover { color: var(--kbf-blue); }
.kbf-wiz .widget-button.kbf-disclose i { width: 24px; height: 24px; border-radius: 7px; display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; color: var(--kbf-muted); background: var(--kbf-soft); margin: 0; }
.kbf-wiz .adv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0 16px; padding-left: 36px; margin-top: 4px; width: 100%; }
.kbf-wiz .adv-grid > .widget-box { min-width: 0; }
.kbf-wiz .kbf-switch { width: auto; margin: 14px 0 0 36px; height: auto; }
.kbf-wiz .kbf-switch label { display: inline-flex; align-items: center; gap: 10px; cursor: pointer; font-size: 13px; color: var(--kbf-fg); }
.kbf-wiz .kbf-switch input[type="checkbox"] { appearance: none; -webkit-appearance: none; width: 34px; height: 20px; margin: 0; border-radius: 999px; position: relative;
  background: var(--kbf-border); cursor: pointer; transition: background .15s ease; flex: 0 0 auto; }
.kbf-wiz .kbf-switch input[type="checkbox"]::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,.3); transition: transform .15s ease; }
.kbf-wiz .kbf-switch input[type="checkbox"]:checked { background: var(--kbf-green); }
.kbf-wiz .kbf-switch input[type="checkbox"]:checked::after { transform: translateX(14px); }

/* Live preview in the side column */
.kbf-wiz .pv-h { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 14px; }
.kbf-wiz .pv-h h3 { margin: 0; }
.kbf-wiz .live { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: var(--kbf-muted); }
.kbf-wiz .live i { width: 7px; height: 7px; border-radius: 50%; background: var(--kbf-green); box-shadow: 0 0 0 3px color-mix(in srgb, var(--kbf-green) 22%, transparent); }
.kbf-wiz .flow { list-style: none; margin: 0; padding: 0; position: relative; }
.kbf-wiz .flow li { position: relative; display: grid; grid-template-columns: 30px 1fr; gap: 10px; padding-bottom: 14px; }
.kbf-wiz .flow li:last-child { padding-bottom: 0; }
.kbf-wiz .flow li:not(:last-child)::before { content: ""; position: absolute; left: 14px; top: 30px; bottom: 2px; width: 2px; border-radius: 2px;
  background: linear-gradient(var(--kbf-border), var(--kbf-border)); }
.kbf-wiz .flow .fi { width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center; color: var(--kbf-blue);
  background: var(--kbf-card); border: 1px solid var(--kbf-border); position: relative; z-index: 1; }
.kbf-wiz .flow .fi svg { width: 16px; height: 16px; }
.kbf-wiz .flow li.unset .fi { color: var(--kbf-muted); border-style: dashed; }
.kbf-wiz .flow .fl { font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--kbf-muted); font-weight: 600; }
.kbf-wiz .flow .fv { font-weight: 600; word-break: break-word; line-height: 1.35; }
.kbf-wiz .flow li.unset .fv { color: var(--kbf-muted); font-weight: 500; font-style: italic; }
.kbf-wiz .flow .fs { font-size: 12px; color: var(--kbf-muted); }
.kbf-wiz .pv-names { margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--kbf-border); }
.kbf-wiz .pv-names .nm { display: grid; grid-template-columns: 1fr; gap: 0; margin-bottom: 7px; }
.kbf-wiz .pv-names .nm span { font-size: 11.5px; color: var(--kbf-muted); }
.kbf-wiz .pv-names .nm code { word-break: break-all; font-size: 11.5px; }
.kbf-wiz .pv-note { margin-top: 16px; font-size: 12px; color: var(--kbf-muted); display: flex; gap: 8px; align-items: flex-start; }
.kbf-wiz .pv-note svg { width: 15px; height: 15px; flex: 0 0 auto; margin-top: 1px; color: var(--kbf-green); }

/* Action bar under the form */
.kbf-wiz .kbf-actions { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 14px 26px; border-top: 1px solid var(--kbf-border);
  background: color-mix(in srgb, var(--kbf-soft) 60%, var(--kbf-card)); }
.kbf-wiz .kbf-actions > .widget-html { flex: 1 1 260px; width: auto; min-width: 0; }
.kbf-wiz .status { display: flex; align-items: center; gap: 9px; font-size: 13px; color: var(--kbf-muted); }
.kbf-wiz .status .sd { width: 22px; height: 22px; border-radius: 50%; flex: 0 0 auto; display: flex; align-items: center; justify-content: center; color: #fff; background: var(--c); }
.kbf-wiz .status .sd svg { width: 12px; height: 12px; }
.kbf-wiz .status b { color: var(--kbf-fg); font-weight: 600; }
.kbf-wiz .status .spin { margin: 0; flex: 0 0 auto; width: 18px; height: 18px; }
.kbf-wiz .widget-button.kbf-btn { flex: 0 0 auto; height: 36px; width: auto; min-width: max-content; margin: 0; padding: 0 16px; border-radius: 8px; font-size: 13px; font-weight: 600; font-family: inherit;
  border: 1px solid var(--kbf-border); background: var(--kbf-card); color: var(--kbf-fg); box-shadow: 0 1px 1px rgba(0,0,0,.04);
  transition: background .15s ease, border-color .15s ease, box-shadow .15s ease, transform .05s ease; }
.kbf-wiz .widget-button.kbf-btn:hover:enabled { border-color: var(--kbf-muted); }
.kbf-wiz .widget-button.kbf-btn:active:enabled { transform: translateY(1px); }
.kbf-wiz .widget-button.kbf-btn:focus-visible { outline: none; box-shadow: 0 0 0 3px color-mix(in srgb, var(--kbf-blue) 30%, transparent); }
.kbf-wiz .widget-button.kbf-btn i { margin-right: 7px; }
.kbf-wiz .widget-button.kbf-btn.primary { background: var(--kbf-blue); border-color: var(--kbf-blue); color: #fff;
  box-shadow: 0 4px 12px -4px color-mix(in srgb, var(--kbf-blue) 60%, transparent); }
.kbf-wiz .widget-button.kbf-btn.primary:hover:enabled { background: color-mix(in srgb, var(--kbf-blue) 88%, #000); border-color: transparent; }
.kbf-wiz .widget-button.kbf-btn.go { background: var(--kbf-green); border-color: var(--kbf-green); color: #fff;
  box-shadow: 0 4px 12px -4px color-mix(in srgb, var(--kbf-green) 60%, transparent); }
.kbf-wiz .widget-button.kbf-btn.go:hover:enabled { background: color-mix(in srgb, var(--kbf-green) 88%, #000); border-color: transparent; }
.kbf-wiz .widget-button.kbf-btn.ghost { background: transparent; box-shadow: none; }
.kbf-wiz .widget-button.kbf-btn:disabled { opacity: .45; cursor: not-allowed; box-shadow: none; }

/* After a plan is made the form folds into this one-line recap */
.kbf-wiz .kbf-recap { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 14px 26px; width: 100%; }
.kbf-wiz .kbf-recap > .widget-html { flex: 1 1 400px; width: auto; min-width: 0; }
.kbf-wiz .recap { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.kbf-wiz .recap .it { display: inline-flex; align-items: center; gap: 7px; padding: 5px 10px 5px 6px; border-radius: 8px; max-width: 100%;
  background: var(--kbf-soft); border: 1px solid var(--kbf-border); font-size: 12.5px; font-weight: 600; }
.kbf-wiz .recap .it span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kbf-wiz .recap .it svg { width: 15px; height: 15px; flex: 0 0 auto; color: var(--kbf-blue); }
.kbf-wiz .recap .sep { color: var(--kbf-muted); }

/* Output area: code, plan, progress, result and the "try it" box */
.kbf-wiz .kbf-out .kbf { max-width: none; margin: 18px 0 0; }
.kbf-wiz .kbf-out .stale { opacity: .45; filter: grayscale(.6); transition: opacity .2s ease; }
.kbf-wiz .code2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 0; }
.kbf-wiz .code2 .section { border-bottom: 0; }
.kbf-wiz .code2 .section + .section { border-left: 1px solid var(--kbf-border); }
.kbf-wiz .loading { display: flex; gap: 14px; align-items: center; padding: 22px 20px; }
.kbf-wiz .loading .spin { width: 22px; height: 22px; border-width: 3px; margin: 0; }
.kbf-wiz .try { margin: 18px 0 0; padding: 18px 20px; border-radius: 12px; border: 1px solid color-mix(in srgb, var(--kbf-green) 35%, var(--kbf-border));
  background: linear-gradient(135deg, color-mix(in srgb, var(--kbf-green) 8%, var(--kbf-card)), color-mix(in srgb, var(--kbf-blue) 6%, var(--kbf-card))); }
.kbf-wiz .try-h { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
.kbf-wiz .try-h .ico { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #fff; background: var(--kbf-green); }
.kbf-wiz .try-h .ico svg { width: 18px; height: 18px; }
.kbf-wiz .try-h .t { font-weight: 650; font-size: 14.5px; }
.kbf-wiz .try-h .s { color: var(--kbf-muted); font-size: 12.5px; }
.kbf-wiz .ask-row { display: flex; gap: 10px; align-items: center; width: 100%; }
.kbf-wiz .ask-row .kbf-input { flex: 1 1 auto; }
.kbf-wiz .ask-row .kbf-input input { height: 40px; padding-left: 14px; font-size: 14px; }
.kbf-wiz .ask-row .widget-button.kbf-btn { height: 40px; }
@media (max-width: 860px) { .kbf-wiz .kbf-aside { border-left: 0; border-top: 1px solid var(--kbf-border); flex-basis: 100%; }
  .kbf-wiz .stp .s { display: none; } .kbf-wiz .sec-body, .kbf-wiz .adv-grid { padding-left: 0; } .kbf-wiz .kbf-switch { margin-left: 0; } }
</style>
"""

#: Small line icons drawn for kbforge (24×24 grid, stroked with the current text colour).
ICONS = {
    "logo": '<path d="M12 3 3 7.5l9 4.5 9-4.5z"/><path d="m3 12 9 4.5 9-4.5"/><path d="m3 16.5 9 4.5 9-4.5"/>',
    "folder": '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v7.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c2.6 2.6 3.9 5.6 3.9 9s-1.3 6.4-3.9 9c-2.6-2.6-3.9-5.6-3.9-9S9.4 5.6 12 3z"/>',
    "sparkle": '<path d="M12 3.5 13.9 9l5.6 1.9-5.6 1.9L12 18.5l-1.9-5.7-5.6-1.9L10.1 9z"/><path d="M19 3v4M17 5h4"/>',
    "database": '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>',
    "scissors": '<circle cx="6" cy="7" r="2.5"/><circle cx="6" cy="17" r="2.5"/><path d="M8 8.5 20 17M8 15.5 20 7"/>',
    "tree": '<rect x="9" y="3" width="6" height="5" rx="1.2"/><rect x="3" y="16" width="6" height="5" rx="1.2"/><rect x="15" y="16" width="6" height="5" rx="1.2"/><path d="M12 8v4M6 16v-2.5h12V16"/>',
    "wave": '<path d="M3 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/><path d="M3 18h18M3 6h18" opacity=".45"/>',
    "file": '<path d="M7 3h7l5 5v11.5A1.5 1.5 0 0 1 17.5 21h-10A1.5 1.5 0 0 1 6 19.5v-15A1.5 1.5 0 0 1 7.5 3z"/><path d="M14 3v5h5"/>',
    "book": '<path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H19v15H7.5A2.5 2.5 0 0 0 5 20.5z"/><path d="M5 20.5A2.5 2.5 0 0 0 7.5 21H19v-3"/>',
    "shield": '<path d="M12 3 4.5 6v5.5c0 4.6 3.2 8.1 7.5 9.5 4.3-1.4 7.5-4.9 7.5-9.5V6z"/><path d="m8.8 12 2.2 2.2 4.3-4.4"/>',
    "check": '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
    "cross": '<path d="M6 6l12 12M18 6 6 18"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.5v.5"/>',
    "alert": '<path d="M12 3.5 21.5 20h-19z"/><path d="M12 10v4.5M12 17.2v.3"/>',
    "chat": '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-4.5 4v-4h0A1.5 1.5 0 0 1 4 14.5z"/>',
    "dot": '<circle cx="12" cy="12" r="1.5"/>',
}


def icon(name: str, width: float = 1.8) -> str:
    """An inline SVG icon from ``ICONS``; it takes its colour and size from the surrounding CSS."""
    return (
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


#: The wizard's three stages: (title, short explanation).
STAGES = [("Describe", "your documents"), ("Review", "the read-only plan"), ("Build", "and try it out")]


def stepper(active: int) -> str:
    """The "Describe → Review → Build" progress line.

    Stages before ``active`` show a tick and ``active`` is highlighted; pass ``len(STAGES) + 1``
    once everything has finished so every stage shows a tick.
    """
    parts = []
    for i, (title, sub) in enumerate(STAGES, 1):
        if i > 1:
            parts.append(f'<div class="rail{" done" if i <= active else ""}"></div>')
        state = "done" if i < active else ("on" if i == active else "todo")
        mark = icon("check", 3) if i < active else str(i)
        parts.append(
            f'<div class="stp {state}"><div class="dot">{mark}</div>'
            f'<div><div class="t">{title}</div><div class="s">{sub}</div></div></div>'
        )
    return f'<div class="stepper">{"".join(parts)}</div>'


def hero(active: int, region: str = "") -> str:
    """The wizard's header: brand mark, title, a "nothing changes yet" chip and the stepper.

    It also carries the stylesheets, so it must stay the first thing inside the wizard.
    """
    where = f" in <b>{esc(region)}</b>" if region else ""
    if active <= 2:
        safe = f'<span class="safe">{icon("shield")}Nothing is created until you click Build</span>'
    elif active == 3:
        safe = (
            f'<span class="safe" style="--kbf-green:var(--kbf-blue)">{icon("sparkle")}Building in your account</span>'
        )
    else:
        safe = f'<span class="safe">{icon("check", 2.4)}Built and ready to use</span>'

    return (
        f"{CSS}{WIZARD_CSS}"
        '<div class="hero"><div class="hero-top">'
        f'<div class="mark">{icon("logo", 2)}</div>'
        '<div class="hero-text"><div class="eyebrow">kbforge · wizard</div>'
        "<h2>Create a Bedrock knowledge base</h2>"
        f'<div class="lede">Answer a few questions and kbforge works out every AWS resource{where} for you. '
        "Review the plan first — it only reads from AWS.</div></div>"
        f"{safe}</div>{stepper(active)}</div>"
    )


def section_head(number: object, title: str, hint: str = "") -> str:
    """A numbered heading for one part of the form, with an optional one-line explanation."""
    hint_html = f'<div class="sec-hint">{esc(hint)}</div>' if hint else ""
    return (
        f'<div class="sec-head"><div class="sec-num">{esc(number)}</div>'
        f'<div><div class="sec-title">{esc(title)}</div>{hint_html}</div></div>'
    )


def label(text: str) -> str:
    """A small field label shown above an input."""
    return f'<div class="lbl">{esc(text)}</div>'


def field_hint(text: str, state: str = "") -> str:
    """The line under an input: neutral advice, or ``state="ok"``/``"bad"`` for validation results.

    ``text`` may contain ``<code>`` markup, so callers must escape any user content in it.
    """
    glyph = {"ok": icon("check", 2.6), "bad": icon("alert", 2), "": icon("info", 2)}[state]
    return f'<div class="fhint {state}">{glyph}<span>{text}</span></div>'


def option(title: str, body: str = "", icon_name: str = "", tag: Tuple[str, str] = ("", ""), meta: str = "") -> str:
    """The inside of one option card: icon, title (with an optional coloured tag), text and a tick."""
    tag_text, tag_colour = tag
    tag_html = f'<span class="tag {tag_colour}">{esc(tag_text)}</span>' if tag_text else ""
    ico = f'<div class="ico">{icon(icon_name)}</div>' if icon_name else ""
    body_html = f'<div class="bdy">{esc(body)}</div>' if body else ""
    meta_html = f'<div class="meta">{esc(meta)}</div>' if meta else ""
    return (
        f'<div class="opt">{ico}<div class="txt"><div class="ttl">{esc(title)}{tag_html}</div>'
        f'{body_html}{meta_html}</div><span class="tick">{icon("check", 3.2)}</span></div>'
    )


#: One row of the live preview: (label, value or "" when not set yet, detail line, icon name).
FlowRow = Tuple[str, str, str, str]


def preview(rows: Sequence[FlowRow], names: Optional[List[Tuple[str, str]]] = None) -> str:
    """The side panel that redraws as the form changes: the data flow, then the AWS names to be used."""
    items = []
    for title, value, sub, icon_name in rows:
        unset = not value
        items.append(
            f'<li class="{"unset" if unset else ""}"><div class="fi">{icon(icon_name)}</div><div>'
            f'<div class="fl">{esc(title)}</div><div class="fv">{esc(value or "not set yet")}</div>'
            + (f'<div class="fs">{esc(sub)}</div>' if sub and not unset else "")
            + "</div></li>"
        )
    names_html = ""
    if names:
        names_html = (
            '<div class="pv-names"><h3>AWS names it will use</h3>'
            + "".join(f'<div class="nm"><span>{esc(k)}</span><code>{esc(v)}</code></div>' for k, v in names)
            + "</div>"
        )
    return (
        '<div class="pv-h"><h3>Your knowledge base</h3><span class="live"><i></i>live preview</span></div>'
        f'<ul class="flow">{"".join(items)}</ul>{names_html}'
        f'<div class="pv-note">{icon("shield")}<span>Names are derived from the knowledge-base name, so running '
        "the wizard again finds the same resources instead of making duplicates.</span></div>"
    )


#: Status-line looks: (CSS colour, icon) for each kind of message.
_STATUS = {
    "todo": ("var(--kbf-grey)", "dot"),
    "ok": ("var(--kbf-green)", "check"),
    "info": ("var(--kbf-blue)", "info"),
    "warn": ("var(--kbf-amber)", "alert"),
    "bad": ("var(--kbf-red)", "cross"),
}


def status(kind: str, text: str) -> str:
    """The message beside the action buttons. ``kind`` is busy/todo/ok/info/warn/bad; ``text`` may hold markup."""
    if kind == "busy":
        return f'<div class="status"><span class="spin"></span><span>{text}</span></div>'
    colour, glyph = _STATUS[kind]
    return f'<div class="status"><span class="sd" style="--c:{colour}">{icon(glyph, 3)}</span><span>{text}</span></div>'


def recap(rows: Sequence[FlowRow]) -> str:
    """The folded-up form: one small chip per answer, in data-flow order (knowledge base first)."""
    ordered = [rows[-1], *rows[:-1]]
    chips = [
        f'<span class="it" title="{esc(t)}">{icon(i)}<span>{esc(v or "not set")}</span></span>'
        for t, v, _, i in ordered
    ]
    return f'<div class="recap">{"".join(chips)}</div>'


def loading(title: str, text: str) -> str:
    """A placeholder card with a spinner, shown while the plan's read-only checks run."""
    return (
        f'{CSS}<div class="kbf"><div class="card"><div class="loading"><span class="spin"></span>'
        f'<div><div class="title">{esc(title)}</div><div class="muted">{esc(text)}</div></div></div></div></div>'
    )


def try_it_header(name: str) -> str:
    """The heading of the "ask your knowledge base" box shown once a build succeeds."""
    return (
        f'<div class="try-h"><div class="ico">{icon("chat", 2)}</div><div>'
        f'<div class="t">Try {esc(name)}</div>'
        '<div class="s">Ask a question your documents answer and see which chunks come back.</div></div></div>'
    )
