/* The "video" tour of kbforge.

   It isn't a video file: it's a small player that animates the real screenshots in
   assets/screens/ (code being typed into a notebook cell, then its output appearing and
   scrolling). That keeps it crisp at any size, light to download, and always in step with
   the screenshots, which scripts/docs_screenshots.py regenerates from real kbforge output.

   Everything on screen is a pure function of the playback time (see `renderAt`), so seeking
   is instant and scripts/docs_video.py can step through it frame by frame to export an MP4.

   Usage: <div class="screen" data-demo></div>, plus optional .controls/.chapters elements
   inside the same .player wrapper. */

(function () {
  "use strict";

  const STAGE_W = 1280;
  const STAGE_H = 720;
  const CHROME_H = 40;
  const OUT_LEFT = 90;
  const OUT_RIGHT = 40;
  const CAPTION_ROOM = 96;
  const SHOTS = (document.currentScript && document.currentScript.dataset.shots) || "assets/screens/";

  /* Light or dark: follows the reader's system setting, like the screenshots on the page.
     `?theme=light` or `?theme=dark` forces one (scripts/docs_video.py uses this). */
  const FORCED_THEME = (location.search.match(/[?&]theme=(light|dark)\b/) || [])[1];
  const DARK_QUERY = window.matchMedia("(prefers-color-scheme: dark)");
  const currentTheme = () => FORCED_THEME || (DARK_QUERY.matches ? "dark" : "light");
  /* Each screenshot has a -dark twin of the same size, so crops and timings work for both. */
  const shotUrl = (file, theme) => SHOTS + (theme === "dark" ? file.replace(/\.png$/, "-dark.png") : file);

  /* The notebook screenshots show the whole cell (code + output). The tour types the code
     itself, so it crops each screenshot to just the output card: [top, left, right] in image
     pixels. The card always spans the same columns; only where it starts differs. */
  const CARD_CROP = (top) => [top, 140, 2128];

  /* The script of the tour. Times are seconds from the start of each scene.
     Outputs appear once the cell has "run"; a later output (`at`) replaces the earlier one. */
  const SCENES = [
    {
      kind: "title", dur: 4.5, chapter: "Intro",
      html: '<div class="mark">kbforge</div><h1>From documents in S3 to a<br>Bedrock knowledge base.</h1>' +
        "<p>Four notebook cells. Under two minutes to watch.</p>",
    },
    {
      kind: "cell", dur: 6, chapter: "Install", label: "Install",
      code: '%pip install -q "kbforge @ git+https://github.com/utkarsh5026/bedrcok.git"',
      outputs: [{ text: '<span class="ok">✓</span> Successfully installed kbforge-0.1.0' }],
      captions: [[0, "<b>Install</b> once, in any SageMaker or Jupyter notebook."]],
    },
    {
      kind: "cell", dur: 9.5, chapter: "1 · Describe", label: "Step 1 of 4 · Describe",
      code: 'from kbforge import KnowledgeBase\n\nkb = KnowledgeBase("hr-policies", source="s3://my-company-docs/hr/")\nkb',
      outputs: [{ img: "describe.png", w: 2524, h: 950, crop: CARD_CROP(180) }],
      captions: [
        [0, "<b>Describe it.</b> Just a name and where your documents live."],
        [5.2, "kbforge shows the pipeline it will set up. <b>Nothing in AWS has changed.</b>"],
      ],
    },
    {
      kind: "cell", dur: 15, chapter: "2 · Plan", label: "Step 2 of 4 · Plan",
      code: "kb.plan()",
      outputs: [{ img: "plan.png", w: 2524, h: 2554, crop: CARD_CROP(72) }],
      captions: [
        [0, "<b>Read the plan.</b> It's read-only, so run it as often as you like."],
        [4, "Preflight checks: your credentials, model access, and what's in the bucket."],
        [9.5, "Then every AWS resource it will create, <b>by name</b>, before it exists."],
      ],
    },
    {
      kind: "cell", dur: 9, chapter: "Mistakes", label: "Caught early",
      code: 'typo = KnowledgeBase("finance-docs", source="s3://finanse-reports/2026/")\ntypo.plan()',
      outputs: [{ img: "blocked.png", w: 2524, h: 2606, crop: CARD_CROP(108), panMax: 110 }],
      captions: [
        [0, "What if something's wrong? Here the bucket name has a typo&hellip;"],
        [4, "&hellip;so the plan is <b>blocked</b> and says why. Nothing gets half-built."],
      ],
    },
    {
      kind: "cell", dur: 14, chapter: "3 · Build", label: "Step 3 of 4 · Build",
      code: "kb.apply()",
      outputs: [
        { img: "progress.png", w: 2360, h: 1000 },
        { img: "apply.png", w: 2524, h: 1972, at: 6.5, crop: CARD_CROP(72) },
      ],
      captions: [
        [0, "<b>Build it.</b> The card updates live, one step at a time."],
        [6.5, "Done: a <b>Ready</b> card with the IDs and code your apps need."],
      ],
    },
    {
      kind: "cell", dur: 9.5, chapter: "4 · Ask", label: "Step 4 of 4 · Ask",
      code: 'kb.ask("How many weeks of parental leave do we get?")',
      outputs: [{ img: "ask.png", w: 2524, h: 870, crop: CARD_CROP(72) }],
      captions: [
        [0, "<b>Try it.</b> Ask a question in plain English&hellip;"],
        [4.6, "&hellip;and see the passages that match best, with their source files."],
      ],
    },
    {
      kind: "cell", dur: 9.5, chapter: "Answers", label: "Step 4 of 4 · Ask",
      code: 'kb.ask("How many weeks of parental leave do we get?", answer=True)',
      outputs: [{ img: "answer.png", w: 2524, h: 932, crop: CARD_CROP(72) }],
      captions: [
        [0, "Want a <b>written answer</b>? Add <code>answer=True</code>&hellip;"],
        [4.6, "&hellip;and a model answers from your documents, citing each passage it used."],
      ],
    },
    {
      kind: "cell", dur: 8, chapter: "Find it", label: "Afterwards",
      code: 'kbforge.knowledge_bases()   # everything kbforge built here',
      outputs: [{ img: "list.png", w: 2524, h: 996, crop: CARD_CROP(72) }],
      captions: [
        [0, "New notebook? <b>List</b> what's already built&hellip;"],
        [4, "&hellip;and pick one up with <code>KnowledgeBase.open(name)</code>."],
      ],
    },
    {
      kind: "cell", dur: 9, chapter: "Later", label: "Afterwards",
      code: "kb.sync()      # documents changed? re-ingest them\nkb.destroy()   # preview what would be deleted",
      outputs: [{ img: "destroy.png", w: 2524, h: 1742, crop: CARD_CROP(72) }],
      captions: [
        [0, "<b>Later:</b> sync when documents change, and clean up when you're done."],
        [5, "destroy() only removes what kbforge created. <b>Your documents are never touched.</b>"],
      ],
    },
    {
      kind: "title", dur: 6, chapter: "Get started",
      html: '<div class="mark">kbforge</div><h1>That\'s the whole thing.</h1>' +
        "<ol><li><span>1</span>describe</li><li><span>2</span>plan</li><li><span>3</span>apply</li><li><span>4</span>ask</li></ol>" +
        '<p class="small">utkarsh5026.github.io/bedrcok</p>',
    },
  ];

  let start = 0;
  for (const s of SCENES) {
    s.start = start;
    start += s.dur;
  }
  const DURATION = start;

  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  const fade = (t, from, len) => clamp((t - from) / len, 0, 1);
  const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  /* Tiny Python highlighter: strings, comments, keywords and calls. Works on partly-typed code. */
  function highlight(code) {
    const pattern = /(#[^\n]*)|("[^"\n]*"?)|\b(from|import|True|False|None)\b|(%pip)|\b([A-Za-z_]\w*)(?=\()/g;
    let out = "";
    let last = 0;
    for (let m; (m = pattern.exec(code));) {
      out += escapeHtml(code.slice(last, m.index));
      const cls = m[1] ? "cm" : m[2] ? "st" : m[3] || m[4] ? "kw" : "fn";
      out += `<span class="${cls}">${escapeHtml(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    return out + escapeHtml(code.slice(last));
  }

  /* When typing starts/ends, when the cell runs, and when its output appears. */
  function timing(scene) {
    const typeStart = 0.5;
    const typeDur = clamp(scene.code.length / 32, 0.5, 2.4);
    const runAt = typeStart + typeDur + 0.35;
    return { typeStart, typeDur, runAt, outAt: runAt + 0.75 };
  }

  class Demo {
    constructor(screen) {
      this.screen = screen;
      this.player = screen.closest(".player") || screen.parentElement;
      this.t = 0;
      this.playing = false;
      this.sceneIndex = -1;
      this.built = null;

      this.stage = document.createElement("div");
      this.stage.className = "stage";
      this.stage.setAttribute("aria-hidden", "true");
      screen.appendChild(this.stage);
      screen.setAttribute("role", "img");
      screen.setAttribute("aria-label", "Animated tour: installing kbforge, describing a knowledge base, reading the plan, building it and asking a question.");

      this.fit();
      new ResizeObserver(() => this.fit()).observe(screen);
      this.buildControls();
      screen.addEventListener("click", () => this.toggle());
      this.setTheme(currentTheme());
      if (!FORCED_THEME) DARK_QUERY.addEventListener("change", () => this.setTheme(currentTheme()));
    }

    /* Switch between the light and dark notebook look, keeping the current playback position. */
    setTheme(theme) {
      this.theme = theme;
      // Point "Download as MP4" at the export that matches what's on screen.
      document.querySelectorAll("a[data-tour-mp4]").forEach((link) => {
        link.href = link.getAttribute("href").replace(/kbforge-tour(-dark)?\.mp4$/, theme === "dark" ? "kbforge-tour-dark.mp4" : "kbforge-tour.mp4");
      });
      this.stage.classList.toggle("dark", theme === "dark");
      this.ready = this.preload();
      this.sceneIndex = -1; // rebuild the current scene with the other screenshots
      this.renderAt(this.t);
    }

    fit() {
      const w = this.screen.clientWidth || STAGE_W;
      this.stage.style.transform = `scale(${w / STAGE_W})`;
    }

    preload() {
      const files = new Set();
      SCENES.forEach((s) => (s.outputs || []).forEach((o) => o.img && files.add(o.img)));
      return Promise.all([...files].map((f) => {
        const img = new Image();
        img.src = shotUrl(f, this.theme);
        return img.decode().catch(() => {});
      }));
    }

    buildControls() {
      const controls = this.player.querySelector(".controls");
      const chapters = this.player.querySelector(".chapters");
      if (controls) {
        controls.innerHTML = '<button class="play" type="button" aria-label="Play">▶</button>' +
          '<div class="timeline" role="slider" tabindex="0" aria-label="Seek" aria-valuemin="0"></div>' +
          '<span class="time"></span>';
        this.playButton = controls.querySelector(".play");
        this.timeLabel = controls.querySelector(".time");
        this.timeline = controls.querySelector(".timeline");
        this.timeline.setAttribute("aria-valuemax", String(Math.round(DURATION)));
        this.segments = SCENES.map((s) => {
          const seg = document.createElement("div");
          seg.className = "seg";
          seg.style.flex = String(s.dur);
          seg.title = s.chapter;
          seg.appendChild(document.createElement("i"));
          this.timeline.appendChild(seg);
          return seg.firstChild;
        });
        this.playButton.addEventListener("click", () => this.toggle());
        this.timeline.addEventListener("click", (e) => {
          const box = this.timeline.getBoundingClientRect();
          this.seek(((e.clientX - box.left) / box.width) * DURATION);
        });
        this.timeline.addEventListener("keydown", (e) => {
          if (e.key === "ArrowRight") this.seek(this.t + 5);
          if (e.key === "ArrowLeft") this.seek(this.t - 5);
        });
      }
      if (chapters) {
        this.chapterButtons = SCENES.map((s) => {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = s.chapter;
          b.addEventListener("click", () => {
            this.seek(s.start);
            this.play();
          });
          chapters.appendChild(b);
          return b;
        });
      }
    }

    toggle() {
      this.playing ? this.pause() : this.play();
    }

    async play() {
      if (this.t >= DURATION - 0.05) this.t = 0;
      this.playing = true;
      this.updateButton();
      await this.ready;
      if (!this.playing) return;
      let last = performance.now();
      const tick = (now) => {
        if (!this.playing) return;
        this.t = Math.min(DURATION, this.t + (now - last) / 1000);
        last = now;
        this.renderAt(this.t);
        if (this.t >= DURATION) {
          this.pause();
          return;
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame((now) => {
        last = now;
        tick(now);
      });
    }

    pause() {
      this.playing = false;
      this.updateButton();
    }

    seek(t) {
      this.t = clamp(t, 0, DURATION);
      this.renderAt(this.t);
    }

    updateButton() {
      if (!this.playButton) return;
      const ended = !this.playing && this.t >= DURATION - 0.05;
      this.playButton.textContent = this.playing ? "❚❚" : ended ? "↻" : "▶";
      this.playButton.setAttribute("aria-label", this.playing ? "Pause" : "Play");
    }

    /* Draw the frame at time t (seconds). Pure: the same t always gives the same picture. */
    renderAt(t) {
      this.t = t;
      let index = SCENES.findIndex((s) => t < s.start + s.dur);
      if (index < 0) index = SCENES.length - 1;
      const scene = SCENES[index];
      if (index !== this.sceneIndex) {
        this.sceneIndex = index;
        this.build(scene, index);
      }
      const local = t - scene.start;
      const isLast = index === SCENES.length - 1;
      const visible = Math.min(index === 0 ? 1 : fade(local, 0, 0.35), isLast ? 1 : 1 - fade(local, scene.dur - 0.3, 0.3));
      this.built.root.style.opacity = visible.toFixed(3);
      if (scene.kind === "cell") this.renderCell(scene, local);
      else this.built.root.style.transform = `scale(${(index === 0 ? 1 : 0.97 + 0.03 * ease(fade(local, 0, 0.8))).toFixed(4)})`;
      this.renderCaption(scene, local);
      this.renderControls(index);
    }

    build(scene, index) {
      this.stage.innerHTML = "";
      const root = document.createElement("div");
      if (scene.kind === "title") {
        root.className = "titlecard";
        root.innerHTML = scene.html;
        this.stage.appendChild(root);
        this.built = { root };
        return;
      }
      root.className = "window";
      const cellNumber = SCENES.slice(0, index).filter((s) => s.kind === "cell").length + 1;
      root.innerHTML =
        '<div class="chrome"><span class="dots"><i></i><i></i><i></i></span>' +
        '<span class="tab">hr-policies.ipynb</span>' +
        `<span class="chip">${scene.label}</span></div>` +
        '<div class="nb"><div class="cellrow"><div class="prompt"></div><div class="code"></div></div></div>' +
        '<div class="out"></div><div class="caption"></div>';
      this.stage.appendChild(root);
      const out = root.querySelector(".out");
      const code = root.querySelector(".code");
      // Size the code box for the finished code so nothing jumps while typing.
      code.innerHTML = highlight(scene.code);
      const cellRow = root.querySelector(".cellrow");
      const outTop = CHROME_H + cellRow.offsetTop + cellRow.offsetHeight + 18;
      code.style.minHeight = code.offsetHeight + "px";
      out.style.top = outTop + "px";
      const outWidth = STAGE_W - OUT_LEFT - OUT_RIGHT;
      const avail = STAGE_H - outTop - CAPTION_ROOM;
      const outputs = scene.outputs.map((o) => {
        const el = document.createElement(o.img ? "img" : "div");
        let height = 0;
        if (o.img) {
          const [top, left, right] = o.crop || [0, 0, o.w];
          const scale = outWidth / (right - left);
          el.src = shotUrl(o.img, this.theme);
          el.alt = "";
          el.style.width = `${o.w * scale}px`;
          el.style.left = `${-left * scale}px`;
          el.style.top = `${-top * scale}px`;
          height = (o.h - top) * scale;
        } else {
          el.className = "textout";
          el.innerHTML = o.text;
        }
        out.appendChild(el);
        const travel = Math.min(o.panMax != null ? o.panMax : Infinity, Math.max(0, height - avail));
        return { el, spec: o, travel };
      });
      this.built = { root, code, out, outputs, cellNumber, prompt: root.querySelector(".prompt"), caption: root.querySelector(".caption") };
    }

    renderCell(scene, local) {
      const b = this.built;
      const { typeStart, typeDur, runAt, outAt } = timing(scene);
      const typed = Math.floor(scene.code.length * clamp((local - typeStart) / typeDur, 0, 1));
      const caretOn = local < runAt && Math.floor(local * 2.2) % 2 === 0;
      const html = highlight(scene.code.slice(0, typed)) + (caretOn ? '<span class="caret"></span>' : "");
      if (b.code.innerHTML !== html) b.code.innerHTML = html;
      b.prompt.textContent = local < runAt ? "[ ]:" : local < outAt ? "[*]:" : `[${b.cellNumber}]:`;
      b.root.classList.toggle("running", local >= runAt && local < outAt);

      b.outputs.forEach((o, i) => {
        const shownAt = o.spec.at != null ? o.spec.at : outAt;
        const next = b.outputs[i + 1];
        const hideAt = next ? next.spec.at : scene.dur + 1;
        const opacity = Math.min(fade(local, shownAt, 0.4), 1 - fade(local, hideAt, 0.4));
        o.el.style.opacity = opacity.toFixed(3);
        o.el.style.visibility = opacity > 0 ? "visible" : "hidden";
        const panStart = shownAt + 1.6;
        const panEnd = Math.min(hideAt, scene.dur) - 1.4;
        const p = panEnd > panStart ? ease(clamp((local - panStart) / (panEnd - panStart), 0, 1)) : 0;
        const rise = 14 * (1 - ease(fade(local, shownAt, 0.5)));
        o.el.style.transform = `translateY(${(rise - p * o.travel).toFixed(1)}px)`;
      });
    }

    renderCaption(scene, local) {
      const el = this.built.caption;
      if (!el || !scene.captions) return;
      let current = scene.captions[0];
      for (const c of scene.captions) if (local >= c[0]) current = c;
      if (el.dataset.at !== String(current[0])) {
        el.dataset.at = String(current[0]);
        el.innerHTML = current[1];
      }
      const since = local - current[0];
      el.style.opacity = fade(since, 0, 0.3).toFixed(3);
    }

    renderControls(index) {
      if (this.segments) {
        SCENES.forEach((s, i) => {
          const p = clamp((this.t - s.start) / s.dur, 0, 1);
          this.segments[i].style.width = (p * 100).toFixed(2) + "%";
        });
        this.timeLabel.textContent = `${fmt(this.t)} / ${fmt(DURATION)}`;
        this.timeline.setAttribute("aria-valuenow", String(Math.round(this.t)));
      }
      if (this.chapterButtons) this.chapterButtons.forEach((b, i) => b.classList.toggle("on", i === index));
      this.updateButton();
    }
  }

  function init() {
    document.querySelectorAll("[data-demo]").forEach((screen) => {
      const demo = new Demo(screen);
      window.kbDemo = { duration: DURATION, renderAt: (t) => demo.renderAt(t), ready: demo.ready, play: () => demo.play(), pause: () => demo.pause() };
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const recording = document.documentElement.classList.contains("record");
      if (screen.dataset.demo === "autoplay" && !reduceMotion && !recording) {
        // Start the tour the first time the player is mostly on screen.
        const seen = new IntersectionObserver((entries) => {
          if (entries.some((e) => e.isIntersecting)) {
            seen.disconnect();
            demo.play();
          }
        }, { threshold: 0.6 });
        seen.observe(screen);
      }
      document.querySelectorAll("[data-play-demo]").forEach((button) => button.addEventListener("click", () => {
        demo.seek(0);
        demo.play();
      }));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
