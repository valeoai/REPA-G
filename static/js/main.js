/* REPA-G project page — interactive figures. Plain JS, no dependencies besides MathJax. */
(function () {
  "use strict";
  const D = window.REPAG_DATA;
  const IMG = "static/img/";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs = {}, parent) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  const sv = (tag, attrs = {}, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === "text") e.textContent = v; else e.setAttribute(k, v); }
    if (parent) parent.appendChild(e);
    return e;
  };
  const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  const fmt = (v, d) => (v == null ? "–" : (d != null ? v.toFixed(d) : (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v < 1 ? v.toFixed(2) : v.toFixed(2))));
  const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };

  /* ---------------- theme ---------------- */
  const themeOrder = ["system", "light", "dark"];
  function applyTheme(t) {
    if (t === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    const b = $("#themeBtn"); if (b) b.textContent = "◐ " + t[0].toUpperCase() + t.slice(1);
    rerenderAll();
  }
  let theme = store.get("repag-theme") || "system";
  $("#themeBtn").addEventListener("click", () => { theme = themeOrder[(themeOrder.indexOf(theme) + 1) % 3]; store.set("repag-theme", theme); applyTheme(theme); });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => rerenderAll());

  /* ---------------- segmented controls ---------------- */
  function seg(id, onChange) {
    const g = document.getElementById(id); if (!g) return;
    g.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b || !g.contains(b)) return;
      $$("button", g).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onChange(b.dataset.v ?? b.dataset.k ?? b.dataset.a, b);
    });
  }

  /* ---------------- nav highlight ---------------- */
  const navLinks = $$(".nav-links a");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => { if (en.isIntersecting) navLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id)); });
    }, { rootMargin: "-45% 0px -50% 0px" });
    $$("section[id]").forEach((s) => io.observe(s));
  }

  /* ---------------- chart infrastructure ---------------- */
  const charts = [];
  function chart(container, render) {
    const c = typeof container === "string" ? $(container) : container;
    if (!c) return null;
    const tip = el("div", { class: "tip", role: "status" }, c);
    const obj = { c, tip, render: () => { $$("svg", c).forEach((s) => s.remove()); render(c, obj); } };
    charts.push(obj);
    obj.render();
    return obj;
  }
  function rerenderAll() { charts.forEach((o) => o.render()); if (window.__toyRedraw) window.__toyRedraw(); if (window.__potRedraw) window.__potRedraw(); }
  let rT; window.addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(rerenderAll, 120); });
  function showTip(o, x, y, html) { o.tip.innerHTML = html; o.tip.style.left = x + "px"; o.tip.style.top = y + "px"; o.tip.classList.add("on"); }
  function hideTip(o) { o.tip.classList.remove("on"); }
  function niceTicks(max, n = 4) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    const step = (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mag;
    const t = []; for (let v = 0; v <= max + 1e-9; v += step) t.push(+v.toFixed(10));
    if (t[t.length - 1] < max) t.push(+(t[t.length - 1] + step).toFixed(10));
    return t;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));

  /* grouped vertical bars: groups [{label, bars:[{key,label,value,color}]}] */
  function groupedBars(c, o, { groups, height = 300, digits, yLabel }) {
    const W = Math.max(300, c.clientWidth), H = height, m = { l: 46, r: 10, t: 22, b: 40 };
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, c);
    const max = Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value)));
    const ticks = niceTicks(max), ymax = ticks[ticks.length - 1];
    const y = (v) => m.t + (H - m.t - m.b) * (1 - v / ymax);
    ticks.forEach((t) => { sv("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === 0 ? "base" : "grid" }, svg); sv("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", class: "ax", text: t }, svg); });
    if (yLabel) sv("text", { x: m.l - 8, y: 12, "text-anchor": "end", class: "axt", text: yLabel }, svg);
    const gw = (W - m.l - m.r) / groups.length;
    groups.forEach((g, gi) => {
      const n = g.bars.length, bw = Math.min(46, (gw * 0.72) / n), gap = 2, tot = n * bw + (n - 1) * gap;
      const x0 = m.l + gi * gw + (gw - tot) / 2;
      sv("text", { x: m.l + gi * gw + gw / 2, y: H - m.b + 22, "text-anchor": "middle", class: "axt", text: g.label }, svg);
      g.bars.forEach((b, bi) => {
        const x = x0 + bi * (bw + gap), yy = y(b.value), h = y(0) - yy, r = Math.min(4, bw / 2, h);
        const p = sv("path", { d: `M${x},${y(0)} V${yy + r} Q${x},${yy} ${x + r},${yy} H${x + bw - r} Q${x + bw},${yy} ${x + bw},${yy + r} V${y(0)} Z`, fill: b.color, class: "mark" }, svg);
        if (bw >= 30) sv("text", { x: x + bw / 2, y: yy - 6, "text-anchor": "middle", class: "vlab", text: fmt(b.value, digits) }, svg);
        const hit = sv("rect", { x: x - gap, y: m.t, width: bw + 2 * gap, height: y(0) - m.t, class: "hit", tabindex: 0 }, svg);
        const on = () => { p.setAttribute("opacity", ".8"); showTip(o, x + bw / 2, yy, `<b>${fmt(b.value, digits)}</b>${esc(g.label)} · ${esc(b.label)}`); };
        const off = () => { p.removeAttribute("opacity"); hideTip(o); };
        hit.addEventListener("pointerenter", on); hit.addEventListener("pointerleave", off); hit.addEventListener("focus", on); hit.addEventListener("blur", off);
      });
    });
  }

  /* horizontal bars: rows [{label, value, color, sub}] */
  function hBars(c, o, { rows, digits, max, unit = "" }) {
    const W = Math.max(280, c.clientWidth), rowH = 34, m = { l: Math.min(Math.max(...rows.map((r) => r.label.length)) * 7.4 + 14, W * 0.56), r: 56, t: 6, b: 6 };
    const H = m.t + m.b + rows.length * rowH;
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, c);
    const mx = max || Math.max(...rows.map((r) => r.value));
    const x = (v) => m.l + (W - m.l - m.r) * (v / mx);
    sv("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "base" }, svg);
    rows.forEach((r, i) => {
      const y0 = m.t + i * rowH + 7, h = rowH - 14, w = Math.max(1, x(r.value) - m.l), rr = Math.min(4, w, h / 2);
      sv("text", { x: m.l - 10, y: y0 + h / 2 + 4, "text-anchor": "end", class: "axt", text: r.label }, svg);
      const p = sv("path", { d: `M${m.l},${y0} H${m.l + w - rr} Q${m.l + w},${y0} ${m.l + w},${y0 + rr} V${y0 + h - rr} Q${m.l + w},${y0 + h} ${m.l + w - rr},${y0 + h} H${m.l} Z`, fill: r.color, class: "mark" }, svg);
      sv("text", { x: m.l + w + 6, y: y0 + h / 2 + 4, class: "vlab", text: fmt(r.value, digits) + unit }, svg);
      const hit = sv("rect", { x: 0, y: y0 - 5, width: W, height: h + 10, class: "hit", tabindex: 0 }, svg);
      const on = () => { p.setAttribute("opacity", ".8"); showTip(o, m.l + w / 2, y0, `<b>${fmt(r.value, digits)}${unit}</b>${esc(r.label)}${r.sub ? " · " + esc(r.sub) : ""}`); };
      const off = () => { p.removeAttribute("opacity"); hideTip(o); };
      hit.addEventListener("pointerenter", on); hit.addEventListener("pointerleave", off); hit.addEventListener("focus", on); hit.addEventListener("blur", off);
    });
  }

  /* line chart with categorical x, optional error bars: series [{label,color,values:[{mean,lo,hi}|number]}] */
  function lineChart(c, o, { xLabels, series, height = 260, yMin, yMax, digits = 2, title, xTitle, compact }) {
    const W = Math.max(compact ? 180 : 300, c.clientWidth), H = height, m = { l: 40, r: 14, t: title ? 26 : 14, b: xTitle ? 46 : 30 };
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, c);
    const vals = series.flatMap((s) => s.values.flatMap((v) => (typeof v === "number" ? [v] : [v.lo ?? v.mean, v.hi ?? v.mean])));
    let lo = yMin ?? Math.min(...vals), hi = yMax ?? Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.1; if (yMin == null) lo -= pad; if (yMax == null) hi += pad;
    const step = (() => { const raw = (hi - lo) / 4, mg = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mg; return (r <= 1 ? 1 : r <= 2 ? 2 : r <= 2.5 ? 2.5 : r <= 5 ? 5 : 10) * mg; })();
    const t0 = Math.ceil(lo / step) * step;
    const x = (i) => m.l + 12 + (W - m.l - m.r - 24) * (xLabels.length === 1 ? 0.5 : i / (xLabels.length - 1));
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
    if (title) sv("text", { x: m.l - 30, y: 14, class: "axt", text: title }, svg);
    for (let t = t0; t <= hi + 1e-9; t += step) { sv("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "grid" }, svg); sv("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "ax", text: +t.toFixed(3) }, svg); }
    sv("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "base" }, svg);
    xLabels.forEach((l, i) => sv("text", { x: x(i), y: H - m.b + 18, "text-anchor": "middle", class: "ax", text: l }, svg));
    if (xTitle) sv("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axt", text: xTitle }, svg);
    const cross = sv("line", { y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 }, svg);
    series.forEach((s) => {
      const pts = s.values.map((v, i) => [x(i), y(typeof v === "number" ? v : v.mean)]);
      s.values.forEach((v, i) => { if (typeof v !== "number" && v.lo != null) { sv("line", { x1: x(i), x2: x(i), y1: y(v.lo), y2: y(v.hi), stroke: s.color, "stroke-width": 1.5, opacity: 0.7 }, svg); sv("line", { x1: x(i) - 4, x2: x(i) + 4, y1: y(v.lo), y2: y(v.lo), stroke: s.color, "stroke-width": 1.5, opacity: 0.7 }, svg); sv("line", { x1: x(i) - 4, x2: x(i) + 4, y1: y(v.hi), y2: y(v.hi), stroke: s.color, "stroke-width": 1.5, opacity: 0.7 }, svg); } });
      sv("path", { d: "M" + pts.map((p) => p.join(",")).join(" L"), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
      pts.forEach((p) => sv("circle", { cx: p[0], cy: p[1], r: 4.5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg));
    });
    const hit = sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit", tabindex: 0 }, svg);
    const at = (i) => {
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 1);
      const rows = series.map((s) => { const v = s.values[i]; const mv = typeof v === "number" ? v : v.mean; return `<span style="display:flex;gap:8px;align-items:center"><i style="width:12px;height:2px;background:${s.color};display:inline-block"></i><span style="font-family:var(--f-mono);font-weight:700">${fmt(mv, digits)}</span> ${esc(s.label)}</span>`; }).join("");
      showTip(o, x(i), m.t + 6, `<span style="opacity:.75">${esc(xLabels[i])}</span>${rows}`);
    };
    const near = (ev) => { const r = svg.getBoundingClientRect(); const px = (ev.clientX - r.left) * (W / r.width); let bi = 0, bd = 1e9; xLabels.forEach((_, i) => { const d = Math.abs(x(i) - px); if (d < bd) { bd = d; bi = i; } }); return bi; };
    hit.addEventListener("pointermove", (ev) => at(near(ev)));
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(o); });
    hit.addEventListener("focus", () => at(xLabels.length - 1));
    hit.addEventListener("blur", () => { cross.setAttribute("opacity", 0); hideTip(o); });
  }

  /* ---------------- Figure 3 · embedding scatter ---------------- */
  chart("#embChart", (c, o) => {
    const W = Math.max(280, c.clientWidth), H = Math.round(Math.min(360, W * 0.72)), m = { l: 52, r: 14, t: 12, b: 44 };
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Scatter of KL divergence versus squared feature distance" }, c);
    const lx0 = Math.log10(0.55), lx1 = Math.log10(2.1), ly0 = Math.log10(1.2), ly1 = Math.log10(16);
    const x = (v) => m.l + (W - m.l - m.r) * (Math.log10(v) - lx0) / (lx1 - lx0);
    const y = (v) => m.t + (H - m.t - m.b) * (1 - (Math.log10(v) - ly0) / (ly1 - ly0));
    [0.6, 0.8, 1, 1.5, 2].forEach((t) => { sv("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "grid" }, svg); sv("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle", class: "ax", text: t }, svg); });
    [2, 5, 10].forEach((t) => { sv("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "grid" }, svg); sv("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end", class: "ax", text: t === 10 ? "10⁷" : t + "×10⁶" }, svg); });
    sv("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axt", text: "squared feature distance ‖φ₁ − φ₂‖²" }, svg);
    sv("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", class: "axt", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, text: "symmetrised KL" }, svg);
    const B = 6.776, A = 2.605;
    const cid = "clip" + Math.random().toString(36).slice(2, 8);
    sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, sv("clipPath", { id: cid }, sv("defs", {}, svg)));
    const lg = sv("g", { "clip-path": `url(#${cid})` }, svg);
    sv("line", { x1: x(0.55), y1: y(B * 0.55), x2: x(16 / B), y2: y(16), stroke: css("--accent"), "stroke-width": 2, "stroke-dasharray": "7 4" }, lg);
    sv("line", { x1: x(0.55), y1: y(A * 0.55), x2: x(2.1), y2: y(A * 2.1), stroke: css("--patch"), "stroke-width": 2, "stroke-dasharray": "2 4", "stroke-linecap": "round" }, lg);
    const pts = D.fig3.map(([a, b]) => ({ a, b, px: x(a), py: y(b) }));
    const g = sv("g", {}, svg);
    pts.forEach((p, i) => { const ci = sv("circle", { cx: p.px, cy: p.py, r: 0, fill: css("--ink-2"), "fill-opacity": 0.55, stroke: css("--surface"), "stroke-width": 0.8 }, g); p.el = ci; if (reduceMotion) ci.setAttribute("r", 2.8); else setTimeout(() => ci.setAttribute("r", 2.8), 4 + i * 3); });
    sv("text", { x: W - m.r - 6, y: H - m.b - 10, "text-anchor": "end", class: "axt", text: "B/A = 2.60" }, svg);
    const ring = sv("circle", { r: 6, fill: "none", stroke: css("--accent"), "stroke-width": 2, opacity: 0 }, svg);
    const hit = sv("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width), py = (ev.clientY - r.top) * (H / r.height);
      let best = null, bd = 1e9; pts.forEach((p) => { const d = (p.px - px) ** 2 + (p.py - py) ** 2; if (d < bd) { bd = d; best = p; } });
      if (!best || bd > 900) { ring.setAttribute("opacity", 0); hideTip(o); return; }
      ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("opacity", 1);
      showTip(o, best.px * (r.width / W), best.py * (r.height / H), `<b>KL ≈ ${best.b.toFixed(2)}×10⁶</b>‖Δφ‖² ≈ ${best.a.toFixed(3)} · ratio ${(best.b / best.a).toFixed(2)}×10⁶`);
    });
    hit.addEventListener("pointerleave", () => { ring.setAttribute("opacity", 0); hideTip(o); });
  });

  /* ---------------- Figure 4 · interpolation pad ---------------- */
  (function interp() {
    const pad = $("#pad"), knob = $("#knob"), big = $("#interpBig"), grid = $("#igrid"), wt = $("#padW");
    if (!pad) return;
    const spritePos = (r, c) => `${(c / 7) * 100}% ${(r / 3) * 100}%`;
    const btns = [];
    for (let r = 0; r < 4; r++) for (let cc = 0; cc < 8; cc++) {
      const b = el("button", { type: "button", "aria-label": `Interpolation row ${r + 1}, column ${cc + 1}` }, grid);
      const sp = el("span", { class: "isp" }, b); sp.style.backgroundPosition = spritePos(r, cc);
      b.addEventListener("click", () => { stopAuto(); set(cc / 7, r / 3); });
      btns.push(b);
    }
    let u = 3 / 7, v = 1 / 3, cur = -1;
    function set(nu, nv) {
      u = Math.min(1, Math.max(0, nu)); v = Math.min(1, Math.max(0, nv));
      knob.style.left = u * 100 + "%"; knob.style.top = v * 100 + "%";
      const cc = Math.round(u * 7), r = Math.round(v * 3), idx = r * 8 + cc;
      const w = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v].map((x) => Math.round(x * 100));
      wt.textContent = `${w[0]}% · ${w[1]}%\n${w[2]}% · ${w[3]}%`; wt.style.whiteSpace = "pre";
      pad.setAttribute("aria-valuetext", `weights ${w.join(", ")} percent`);
      if (idx !== cur) { cur = idx; big.style.backgroundPosition = spritePos(r, cc); btns.forEach((b, i) => b.classList.toggle("on", i === idx)); }
    }
    const fromEv = (ev) => { const r = pad.getBoundingClientRect(); set((ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height); };
    let drag = false;
    pad.addEventListener("pointerdown", (ev) => { stopAuto(); drag = true; pad.setPointerCapture(ev.pointerId); fromEv(ev); });
    pad.addEventListener("pointermove", (ev) => { if (drag) fromEv(ev); });
    pad.addEventListener("pointerup", () => (drag = false));
    pad.addEventListener("keydown", (ev) => {
      const k = ev.key; let du = 0, dv = 0;
      if (k === "ArrowLeft") du = -1 / 7; else if (k === "ArrowRight") du = 1 / 7; else if (k === "ArrowUp") dv = -1 / 3; else if (k === "ArrowDown") dv = 1 / 3; else return;
      ev.preventDefault(); stopAuto(); set(u + du, v + dv);
    });
    // ambient tour until the reader takes over
    let raf = null, t0 = null, visible = false;
    function tick(ts) { if (t0 == null) t0 = ts; const t = (ts - t0) / 1000; set(0.5 + 0.5 * Math.sin(t * 0.45), 0.5 + 0.5 * Math.sin(t * 0.7 + 1.2)); raf = requestAnimationFrame(tick); }
    function stopAuto() { if (raf) cancelAnimationFrame(raf); raf = null; auto = false; }
    let auto = !reduceMotion;
    set(u, v);
    if (auto && "IntersectionObserver" in window) {
      new IntersectionObserver((e) => { visible = e[0].isIntersecting; if (visible && auto && !raf) raf = requestAnimationFrame(tick); if (!visible && raf) { cancelAnimationFrame(raf); raf = null; } }).observe(pad);
    }
  })();

  /* ---------------- Figures 2 & 7 · clusters ---------------- */
  (function clusters() {
    const refs = $("#refs"), spaces = $("#spaces"); if (!refs) return;
    const EX = [{ k: "car", n: 8, label: "Car" }, { k: "flamingo", n: 16, label: "Flamingo" }, { k: "banjo", n: 16, label: "Banjo player" }];
    const SP = [
      { k: "dinov2", t: "DINOv2", tag: "teacher", v: "Concept-level clusters: same object category, varied scenes." },
      { k: "sit", t: "SiT", tag: "no alignment", v: "Members share colour or layout, not meaning.", bad: true },
      { k: "align", t: "SiT REPA", tag: "before proj.", v: "REPA features reproduce the teacher's grouping." },
      { k: "alignproj", t: "SiT REPA + Proj", tag: "after proj.", v: "Equally semantic after the projection head." }
    ];
    EX.forEach((e, i) => {
      const b = el("button", { class: "ref", type: "button", "aria-pressed": String(i === 0), "aria-label": e.label }, refs);
      const ri = el("span", { class: "ctile" }, b); ri.style.cssText = `display:block;width:100%;height:100%;animation:none;opacity:1;transform:none;background-image:url(${IMG}clusters/${e.k}_dinov2.webp);background-size:${e.n * 100}% 100%;background-position:0 0`;
      b.addEventListener("click", () => { $$(".ref", refs).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(e); });
    });
    function draw(e) {
      spaces.innerHTML = "";
      SP.forEach((s, si) => {
        const d = el("div", { class: "space" + (s.bad ? " bad" : "") }, spaces);
        const h = el("h4", {}, d); h.append(s.t); el("span", { text: s.tag }, h);
        el("p", { class: "verdict", text: s.v }, d);
        const g = el("div", { class: "cgrid" }, d);
        for (let j = 0; j < e.n; j++) {
          const im = el("div", { class: "ctile", role: "img", "aria-label": j === 0 ? `Reference image (${e.label})` : `Cluster member ${j + 1}` }, g);
          im.style.backgroundImage = `url(${IMG}clusters/${e.k}_${s.k}.webp)`; im.style.backgroundSize = `${e.n * 100}% 100%`; im.style.backgroundPosition = `${(j / (e.n - 1)) * 100}% 0`;
          if (j === 0) im.classList.add("refimg");
          im.style.animationDelay = (reduceMotion ? 0 : si * 70 + j * 25) + "ms";
        }
      });
    }
    draw(EX[0]);
  })();

  /* ---------------- Potential explorer ---------------- */
  (function potentials() {
    const cond = $("#potCond"), out = $("#potOut"), heat = $("#potHeat"); if (!cond) return;
    const ctx = cond.getContext("2d"), hctx = heat.getContext("2d");
    const N = 16; let mode = "full", anchor = 2, target = [12, 14];
    const MODES = {
      full: { t: "Full feature map", eq: "\\[\\mathcal{V} = \\tfrac{1}{N}\\textstyle\\sum_{n} \\langle [h]_n, [h^\\star]_n \\rangle\\]", d: "Every generated patch is pulled toward the anchor patch at the same position. This is the training objective itself; with a large λ it gives a stochastic reconstruction of the anchor." },
      masked: { t: "Masked feature map", eq: "\\[\\mathcal{V} = \\tfrac{1}{|S|}\\textstyle\\sum_{n \\in S} \\langle [h]_n, [h^\\star]_n \\rangle\\]", d: "Only the subset S of patches inside a mask is constrained. The mask comes for free: threshold the first principal component of the DINOv2 patches. The object keeps its shape and pose; the background is resampled." },
      average: { t: "Average concept", eq: "\\[\\mathcal{V} = \\Big\\langle \\tfrac{\\bar h}{\\lVert \\bar h \\rVert}, \\tfrac{\\bar h^\\star}{\\lVert \\bar h^\\star \\rVert} \\Big\\rangle\\]", d: "Patches are averaged into one global token before comparison. All spatial constraints disappear; only the semantic concept of the anchor remains." },
      spa: { t: "Selective Patch Alignment", eq: "\\[\\mathcal{V}^{\\mathrm{SPA}} = T \\log \\textstyle\\sum_{n} \\exp\\!\\big(\\langle [h]_n, [h^\\star]_i \\rangle / T\\big)\\]", d: "A single target token is matched by whichever generated patches already resemble it, through a soft-max with temperature T. As T → 0 only the best patch is pulled; as T → ∞ it falls back to averaging. In compositions, this lets the background token claim the right region." }
    };
    // anchor picker
    const picker = el("div", { class: "thumbs", role: "group", "aria-label": "Anchor image" });
    $("#potDesc").after(picker);
    [2, 1, 0, 3].forEach((i, k) => {
      const b = el("button", { type: "button", "aria-pressed": String(k === 0), "aria-label": `Anchor ${k + 1}` }, picker);
      el("img", { src: `${IMG}single/${i}_anchor.webp`, alt: "" }, b);
      b.addEventListener("click", () => { anchor = i; $$("button", picker).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); draw(); });
    });
    const imgs = {};
    const get = (src) => (imgs[src] ||= loadImg(src));
    const maskCells = async (i) => {
      const m = await get(`${IMG}single/${i}_mask.webp`); const c = document.createElement("canvas"); c.width = c.height = N;
      const x = c.getContext("2d"); x.imageSmoothingEnabled = false; x.drawImage(m, 0, 0, N, N);
      const d = x.getImageData(0, 0, N, N).data, on = [];
      for (let k = 0; k < N * N; k++) on.push(d[k * 4] > 150 && d[k * 4 + 1] > 150);
      return on;
    };
    const patchMeans = async (src) => {
      const im = await get(src); const c = document.createElement("canvas"); c.width = c.height = 64;
      const x = c.getContext("2d"); x.drawImage(im, 0, 0, 64, 64); const d = x.getImageData(0, 0, 64, 64).data; const out = [];
      for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) { let s = [0, 0, 0]; for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) { const k = ((r * 4 + a) * 64 + q * 4 + b) * 4; s[0] += d[k]; s[1] += d[k + 1]; s[2] += d[k + 2]; } out.push(s.map((z) => z / 16)); }
      return out;
    };
    const T = () => Math.pow(10, -2 + 3 * (+$("#tempR").value / 100));
    async function draw() {
      const M = MODES[mode];
      $("#potTitle").textContent = M.t; $("#potDesc").textContent = M.d;
      const eq = $("#potEq");
      if (eq) { eq.textContent = M.eq; if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([eq]).catch(() => {}); }
      $("#spaCtl").hidden = mode !== "spa"; picker.hidden = mode === "spa";
      const W = cond.width, cs = W / N, accent = css("--accent"), patch = css("--patch");
      ctx.clearRect(0, 0, W, W);
      if (mode === "spa") {
        const lava = await get(`${IMG}teaser/lava.webp`); ctx.drawImage(lava, 0, 0, W, W);
        ctx.fillStyle = "rgba(0,0,0,.35)"; ctx.fillRect(0, 0, W, W);
        ctx.drawImage(lava, target[0] * cs, target[1] * cs, cs, cs, target[0] * cs, target[1] * cs, cs, cs);
        ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 4; ctx.strokeRect(target[0] * cs + 2, target[1] * cs + 2, cs - 4, cs - 4);
        $("#potCondCap").textContent = "Target token";
        out.hidden = true; heat.hidden = false;
        const gen = await get(`${IMG}teaser/repag_compose.webp`);
        const [gm, tm] = await Promise.all([patchMeans(`${IMG}teaser/repag_compose.webp`), patchMeans(`${IMG}teaser/lava.webp`)]);
        const tv = tm[target[1] * N + target[0]], t = T();
        const sim = gm.map((g) => 1 - Math.hypot(g[0] - tv[0], g[1] - tv[1], g[2] - tv[2]) / 441.7);
        const mx = Math.max(...sim), w = sim.map((s) => Math.exp((s - mx) / t)), wm = Math.max(...w), ws = w.reduce((a, b) => a + b, 0);
        hctx.clearRect(0, 0, W, W); hctx.drawImage(gen, 0, 0, W, W); hctx.fillStyle = "rgba(10,12,18,.55)"; hctx.fillRect(0, 0, W, W);
        for (let k = 0; k < N * N; k++) { const a = w[k] / wm; if (a < 0.02) continue; hctx.fillStyle = `rgba(63,215,207,${(0.15 + 0.75 * a).toFixed(3)})`; hctx.fillRect((k % N) * cs, Math.floor(k / N) * cs, cs, cs); }
        const neff = (ws * ws) / w.reduce((a, b) => a + b * b, 0);
        $("#potOutCap").textContent = `Soft-max weight per patch · effective patches ≈ ${neff.toFixed(1)} / 256`;
        $("#tempV").textContent = t < 1 ? t.toFixed(2) : t.toFixed(1);
        return;
      }
      out.hidden = false; heat.hidden = true;
      const a = await get(`${IMG}single/${anchor}_anchor.webp`); ctx.drawImage(a, 0, 0, W, W);
      const on = mode === "masked" ? await maskCells(anchor) : null;
      if (mode === "average") {
        ctx.fillStyle = "rgba(10,12,18,.5)"; ctx.fillRect(0, 0, W, W);
        const cx = W / 2, cy = W / 2;
        ctx.strokeStyle = "rgba(240,123,25,.35)"; ctx.lineWidth = 1;
        for (let r = 0; r < N; r += 2) for (let q = 0; q < N; q += 2) { ctx.beginPath(); ctx.moveTo(q * cs + cs / 2, r * cs + cs / 2); ctx.lineTo(cx, cy); ctx.stroke(); }
        ctx.fillStyle = "#f07b19"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.fillRect(cx - 34, cy - 34, 68, 68); ctx.strokeRect(cx - 34, cy - 34, 68, 68);
        $("#potCondCap").textContent = "256 patch tokens → 1 mean token";
      } else {
        for (let k = 0; k < N * N; k++) {
          const q = k % N, r = Math.floor(k / N), use = mode === "full" || on[k];
          if (!use) { ctx.fillStyle = "rgba(10,12,18,.62)"; ctx.fillRect(q * cs, r * cs, cs, cs); }
          else { ctx.strokeStyle = "rgba(63,215,207,.55)"; ctx.lineWidth = 1.2; ctx.strokeRect(q * cs + 0.5, r * cs + 0.5, cs - 1, cs - 1); }
        }
        const n = mode === "full" ? 256 : on.filter(Boolean).length;
        $("#potCondCap").textContent = `${n} of 256 patch tokens used`;
      }
      out.src = `${IMG}single/${anchor}_${mode}.webp`;
      $("#potOutCap").textContent = "REPA-G sample · REPA-E, DINOv2 features";
    }
    window.__potRedraw = () => draw();
    seg("potTabs", (k) => { mode = k; draw(); });
    $("#tempR").addEventListener("input", () => draw());
    cond.style.cursor = "default";
    // MathJax may finish after first draw
    window.addEventListener("load", () => setTimeout(draw, 300));
    draw();
  })();

  /* ---------------- Figure 6 · toy ---------------- */
  (function toy() {
    const dc = $("#toyData"), fc = $("#toyFeat"); if (!dc) return;
    const dx = dc.getContext("2d"), fx = fc.getContext("2d");
    let theta = Math.PI, lam = 2, colorMode = true, samples = [], anim = null;
    const active = (x, y) => ((Math.floor(x) + Math.floor(y)) % 2 + 2) % 2 === 1;
    function feat(x, y) {
      const i = Math.floor(x), j = Math.floor(y), u = x - (i + 0.5), v = y - (j + 0.5), rmax = Math.SQRT2 / 2;
      const t = Math.min(1, Math.max(0, Math.hypot(u, v) / rmax)) * 2 - 1, h = (0.5 - Math.abs(u)) * 2, w = (0.5 - Math.abs(v)) * 2;
      const nrm = Math.hypot(t, h - w) || 1e-9; return Math.acos(t / nrm) * (Math.sign(h - w) || 1);
    }
    const hue = (a) => `hsl(${((a * 180) / Math.PI + 360) % 360} 70% 52%)`;
    function sample() {
      const out = []; let tries = 0;
      while (out.length < 2000 && tries < 3e6) {
        tries++; const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1; if (!active(x, y)) continue;
        const a = feat(x, y); if (Math.random() < Math.exp(lam * (Math.cos(a - theta) - 1))) out.push([x, y, a]);
      }
      return out;
    }
    let field = null;
    function buildField(S) {
      const c = document.createElement("canvas"); c.width = c.height = 200; const x = c.getContext("2d"), im = x.createImageData(200, 200);
      for (let r = 0; r < 200; r++) for (let q = 0; q < 200; q++) {
        const X = q / 100 - 1 + 0.005, Y = 1 - r / 100 - 0.005, k = (r * 200 + q) * 4; if (!active(X, Y)) { im.data[k + 3] = 0; continue; }
        const h = ((feat(X, Y) * 180) / Math.PI + 360) % 360; const [R, G, B] = hsl2rgb(h / 360, 0.7, 0.55); im.data[k] = R; im.data[k + 1] = G; im.data[k + 2] = B; im.data[k + 3] = 46;
      }
      x.putImageData(im, 0, 0); return c;
    }
    function hsl2rgb(h, s, l) { const f = (n) => { const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l); return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))); }; return [f(0), f(8), f(4)]; }
    function drawData(count) {
      const W = dc.width, pad = 40, s = (W - 2 * pad) / 2; const X = (x) => pad + (x + 1) * s, Y = (y) => pad + (1 - y) * s;
      dx.clearRect(0, 0, W, W);
      dx.fillStyle = css("--surface-2"); dx.fillRect(X(-1), Y(1), s, s); dx.fillRect(X(0), Y(0), s, s);
      if (colorMode) { field ||= buildField(); dx.imageSmoothingEnabled = true; dx.drawImage(field, pad, pad, 2 * s, 2 * s); }
      dx.strokeStyle = css("--line"); dx.lineWidth = 1; dx.strokeRect(pad, pad, 2 * s, 2 * s);
      const base = css("--ink-2");
      for (let k = 0; k < count && k < samples.length; k++) { const [x, y, a] = samples[k]; dx.fillStyle = colorMode ? hue(a) : base; dx.globalAlpha = colorMode ? 0.9 : 0.55; dx.beginPath(); dx.arc(X(x), Y(y), 2.6, 0, 7); dx.fill(); }
      dx.globalAlpha = 1;
      dx.fillStyle = css("--muted"); dx.font = "22px IBM Plex Mono, monospace"; dx.fillText(`${Math.min(count, samples.length)} samples · λ = ${lam.toFixed(1)}`, pad, W - 12);
    }
    function drawFeat() {
      const W = fc.width, cx = W / 2, cy = W / 2, R = W * 0.34;
      fx.clearRect(0, 0, W, W);
      fx.strokeStyle = css("--line"); fx.lineWidth = 1; fx.beginPath(); fx.moveTo(cx - R - 30, cy); fx.lineTo(cx + R + 30, cy); fx.moveTo(cx, cy - R - 30); fx.lineTo(cx, cy + R + 30); fx.stroke();
      for (let k = 0; k < 360; k++) { const a0 = (k * Math.PI) / 180; fx.strokeStyle = hue(a0); fx.lineWidth = 10; fx.beginPath(); fx.arc(cx, cy, R, -a0 - 0.01, -a0 - Math.PI / 180 - 0.012, true); fx.stroke(); }
      // histogram of sampled feature angles
      const bins = new Array(72).fill(0); samples.forEach(([, , a]) => { bins[Math.floor((((a + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI)) * 72) % 72]++; });
      const mx = Math.max(1, ...bins); fx.strokeStyle = css("--ink-2"); fx.lineWidth = 5; fx.lineCap = "round";
      bins.forEach((b, i) => { if (!b) return; const a = ((i + 0.5) / 72) * 2 * Math.PI, l = 8 + (b / mx) * R * 0.55; fx.beginPath(); fx.moveTo(cx + Math.cos(a) * (R + 10), cy - Math.sin(a) * (R + 10)); fx.lineTo(cx + Math.cos(a) * (R + 10 + l), cy - Math.sin(a) * (R + 10 + l)); fx.stroke(); });
      // target arrow
      const tx = cx + Math.cos(theta) * R, ty = cy - Math.sin(theta) * R, acc = css("--accent");
      fx.strokeStyle = acc; fx.lineWidth = 4; fx.beginPath(); fx.moveTo(cx, cy); fx.lineTo(tx, ty); fx.stroke();
      fx.fillStyle = acc; fx.beginPath(); fx.arc(tx, ty, 13, 0, 7); fx.fill(); fx.strokeStyle = css("--surface"); fx.lineWidth = 4; fx.stroke();
      fx.fillStyle = css("--ink"); fx.font = "600 24px IBM Plex Mono, monospace"; fx.textAlign = "center";
      fx.fillText(`c = [${Math.cos(theta).toFixed(2)}, ${Math.sin(theta).toFixed(2)}]`, cx, W - 16); fx.textAlign = "left";
    }
    function run() {
      samples = sample(); drawFeat();
      if (anim) cancelAnimationFrame(anim);
      if (reduceMotion) { drawData(samples.length); return; }
      let n = 0; const step = () => { n += 160; drawData(n); if (n < samples.length) anim = requestAnimationFrame(step); }; step();
    }
    window.__toyRedraw = () => { drawData(samples.length); drawFeat(); };
    const lamR = $("#lamR"); lamR.addEventListener("input", () => { lam = +lamR.value / 10; $("#lamV").textContent = lam.toFixed(1); run(); });
    $("#colorMode").addEventListener("change", (e) => { colorMode = e.target.checked; drawData(samples.length); });
    seg("toyPresets", (a) => { theta = (+a * Math.PI) / 180; run(); });
    let drag = false;
    const setFromEv = (ev) => { const r = fc.getBoundingClientRect(); const x = ev.clientX - r.left - r.width / 2, y = -(ev.clientY - r.top - r.height / 2); theta = Math.atan2(y, x); $$("#toyPresets button").forEach((b) => b.setAttribute("aria-pressed", "false")); drawFeat(); };
    fc.addEventListener("pointerdown", (ev) => { drag = true; fc.setPointerCapture(ev.pointerId); setFromEv(ev); });
    fc.addEventListener("pointermove", (ev) => { if (drag) setFromEv(ev); });
    fc.addEventListener("pointerup", () => { if (drag) { drag = false; run(); } });
    run();
  })();

  /* ---------------- Gallery (Fig 14/15) ---------------- */
  (function gallery() {
    const g = $("#gal"); if (!g) return;
    const st = { ds: "imagenet", model: "repae", mode: "full", all: false };
    const rows = { imagenet: 14, coco: 15 };
    function draw() {
      g.innerHTML = "";
      const n = st.all ? rows[st.ds] : 8;
      for (let r = 0; r < n; r++) {
        const card = el("div", { class: "gcard" }, g);
        const p = el("div", { class: "gpair", tabindex: "0" }, card);
        el("img", { src: `${IMG}gallery/${st.ds}/${r}_${st.model}_${st.mode}.webp`, alt: `Sample ${r + 1}, ${st.model === "repae" ? "REPA-E" : "SiT"}, ${st.mode} features`, loading: "lazy" }, p);
        el("img", { class: "anc", src: `${IMG}gallery/${st.ds}/${r}_anchor.webp`, alt: "", loading: "lazy" }, p);
        const mini = el("div", { class: "mini" }, p); el("img", { src: `${IMG}gallery/${st.ds}/${r}_anchor.webp`, alt: "", loading: "lazy" }, mini);
        if (st.mode === "masked") { const mm = el("div", { class: "mini mask" }, p); el("img", { src: `${IMG}gallery/${st.ds}/${r}_mask.webp`, alt: "", loading: "lazy" }, mm); }
        const cap = el("div", { class: "gcap" }, card); el("span", { text: `#${String(r + 1).padStart(2, "0")}` }, cap); el("span", { text: `${st.model === "repae" ? "REPA-E" : "SiT"} · ${st.mode}` }, cap);
      }
      $("#galMore").textContent = st.all ? "Show fewer" : `Show all ${rows[st.ds]} anchors`;
    }
    seg("galDs", (v) => { st.ds = v; draw(); });
    seg("galModel", (v) => { st.model = v; draw(); });
    seg("galMode", (v) => { st.mode = v; draw(); });
    $("#galMore").addEventListener("click", () => { st.all = !st.all; draw(); });
    draw();
  })();

  /* ---------------- Tables 1 & 3 ---------------- */
  (function dist() {
    if (!$("#distChart")) return;
    const st = { ds: "imagenet", g: 0, m: 0 };
    const names = ["FID ↓", "sFID ↓", "IS ↑", "Prec. ↑", "Rec. ↑"];
    const col = { uncond: "--s-base", FDINO: "--s-dino", FSiT: "--s-sit" }, lab = { uncond: "Unconditional", FDINO: "REPA-G · DINOv2", FSiT: "REPA-G · SiT" };
    const o = chart("#distChart", (c, o) => {
      const T = D.dist[st.ds];
      const groups = Object.entries(T).map(([model, conds]) => ({ label: model, bars: Object.entries(conds).map(([k, v]) => ({ key: k, label: lab[k], value: v[st.g][st.m], color: css(col[k]) })) }));
      groupedBars(c, o, { groups, height: 300, digits: st.m >= 3 ? 2 : st.m === 2 ? 1 : 2, yLabel: names[st.m] });
    });
    function table() {
      const T = D.dist[st.ds], wrap = $("#distTable"); wrap.innerHTML = "";
      const t = el("table", { class: "t" }, wrap), th = el("thead", {}, t), r1 = el("tr", {}, th), r2 = el("tr", {}, th);
      el("th", { text: "Model", rowspan: 2, class: "l" }, r1); el("th", { text: "Cond.", rowspan: 2, class: "l" }, r1);
      ["Full feature map", "Masked feature map", "Average feature map"].forEach((g) => el("th", { text: g, colspan: 5, style: "text-align:center" }, r1));
      for (let g = 0; g < 3; g++) names.forEach((n) => el("th", { text: n }, r2));
      // best among guided rows per column
      const best = []; for (let g = 0; g < 3; g++) for (let m = 0; m < 5; m++) { const vals = []; Object.values(T).forEach((cs) => Object.entries(cs).forEach(([k, v]) => { if (k !== "uncond") vals.push(v[g][m]); })); best.push(m < 2 ? Math.min(...vals) : Math.max(...vals)); }
      const tb = el("tbody", {}, t);
      Object.entries(T).forEach(([model, conds]) => Object.entries(conds).forEach(([k, v], i) => {
        const tr = el("tr", { class: i === 0 ? "grp" : "" }, tb); el("td", { text: i === 0 ? model : "" }, tr);
        el("td", { text: k === "uncond" ? "–" : k === "FDINO" ? "F_DINO" : "F_SiT", class: k === "uncond" ? "l" : "l ours" }, tr);
        for (let g = 0; g < 3; g++) for (let m = 0; m < 5; m++) { const val = v[g][m]; el("td", { text: val.toFixed(2), class: "num" + (k !== "uncond" && val === best[g * 5 + m] ? " best" : "") }, tr); }
      }));
    }
    seg("dDs", (v) => { st.ds = v; o.render(); table(); });
    seg("dGran", (v) => { st.g = +v; o.render(); });
    seg("dMet", (v) => { st.m = +v; o.render(); });
    table();
  })();

  /* ---------------- Table 2 ---------------- */
  (function inst() {
    const wrap = $("#instTable"); if (!wrap) return;
    const t = el("table", { class: "t" }, wrap), hr = el("tr", {}, el("thead", {}, t));
    ["Model", "Cond.", "DINOv2 ↑", "JEPA ↑", "CLIP ↑", "PSNR ↑"].forEach((h, i) => el("th", { text: h, class: i < 2 ? "l" : "" }, hr));
    const tb = el("tbody", {}, t), cells = [];
    D.inst.forEach(([model, cond], i) => {
      const tr = el("tr", { class: i === 0 || D.inst[i - 1][0] !== model ? "grp" : "" }, tb);
      el("td", { text: i === 0 || D.inst[i - 1][0] !== model ? model : "" }, tr); el("td", { text: cond === "FDINO" ? "F_DINO" : "F_SiT", class: "l ours" }, tr);
      const row = []; for (let k = 0; k < 4; k++) { const td = el("td", { class: "num bar-cell", style: "min-width:96px" }, tr); const f = el("div", { class: "fill" }, td); const s = el("span", {}, td); row.push([f, s]); }
      cells.push(row);
    });
    function upd(g) { D.inst.forEach(([, , v], i) => v[g].forEach((x, k) => { const [f, s] = cells[i][k]; f.style.width = (k === 3 ? x / 25 : x) * 100 + "%"; s.textContent = k === 3 ? x.toFixed(2) : x.toFixed(2); })); }
    seg("iGran", (v) => upd(+v)); upd(0);
  })();


  /* ---------------- Alignment requirement (Tables 1 & 2 extract) ---------------- */
  (function alignReq() {
    const wrap = $("#alignTable"); if (!wrap) return;
    const rows = [["SiT", "no"], ["REPA", "yes"], ["REPA-E", "yes"], ["Self-Flow", "no"]];
    function draw(g) {
      wrap.innerHTML = "";
      const t = el("table", { class: "t" }, wrap), hr = el("tr", {}, el("thead", {}, t));
      [["Model", "l"], ["REPA loss", "l"], ["FID, no guidance", ""], ["FID, REPA-G", ""], ["DINOv2 sim. to anchor", ""]].forEach(([h, c]) => el("th", { text: h, class: c }, hr));
      const tb = el("tbody", {}, t);
      rows.forEach(([m, rl]) => {
        let before, after, sim;
        if (m === "Self-Flow") { if (g !== 0) { const tr = el("tr", { class: "grp" }, tb); el("td", { text: m, class: "l ours" }, tr); el("td", { text: "✗ no (self-sup.)", class: "l", style: "color:var(--muted)" }, tr); el("td", { text: "full map only", class: "l muted", colspan: 3 }, tr); return; }
          const sf = D.selfflow; before = sf[2][2][0]; after = sf[3][2][0]; sim = sf[3][2][5]; }
        else { const d = D.dist.imagenet[m]; before = d.uncond[g][0]; after = d.FSiT[g][0]; sim = D.inst.find((r) => r[0] === m && r[1] === "FSiT")[2][g][0]; }
        const tr = el("tr", { class: m === "Self-Flow" ? "grp" : "" }, tb);
        el("td", { text: m, class: m === "Self-Flow" ? "l ours" : "l" }, tr);
        el("td", { text: rl === "yes" ? "✓ yes" : m === "Self-Flow" ? "✗ no (self-sup.)" : "✗ no", class: "l", style: rl === "yes" ? "color:var(--patch);font-weight:600" : "color:var(--muted)" }, tr);
        el("td", { text: before.toFixed(2), class: "num" }, tr);
        const td = el("td", { class: "num" }, tr);
        el("span", { text: after.toFixed(2) + " ", style: "font-weight:700" }, td);
        const tg = el("span", { text: after > before ? "▲" : "▼", style: `font-family:var(--f-ui);font-size:12px;color:${after > before ? "var(--accent)" : "var(--patch)"}` }, td);
        el("span", { class: "wd", text: after > before ? " worse" : " better" }, tg);
        const sc = el("td", { class: "num bar-cell sim" }, tr);
        const f = el("div", { class: "fill" }, sc); f.style.width = sim * 100 + "%"; el("span", { text: sim.toFixed(2) }, sc);
      });
    }
    const sfg = $("#sfGrid");
    if (sfg) for (let r = 0; r < 3; r++) { const p = el("div", { class: "sf-pair" }, sfg); [["anchor", "Anchor"], ["selfflow", "REPA-G · Self-Flow"]].forEach(([k, l]) => { const f = el("figure", {}, p); el("img", { src: `${IMG}baselines/${r}_${k}.webp`, alt: l, loading: "lazy" }, f); el("span", { text: l }, f); }); }
    seg("aGran", (v) => draw(+v)); draw(0);
  })();
  /* ---------------- Table 4 ---------------- */
  chart("#textChart", (c, o) => hBars(c, o, { rows: D.text.map(([m, l, v]) => ({ label: l, sub: m, value: v[0], color: css(m === "CAD-I" ? "--s-base" : "--s-sit") })), digits: 2 }));

  /* ---------------- Table 6 ---------------- */
  (function enc() {
    const ser = [["Uncond.", "--s-base"], ["Full", "--s-sit"], ["Average", "--s-dino"], ["Masked", "--s-3"]];
    const lg = $("#encLegend"); ser.forEach(([n, c]) => { const s = el("span", {}, lg); el("i", { style: `background:var(${c})` }, s); s.append(n); });
    chart("#encChart", (c, o) => groupedBars(c, o, { groups: D.encoders.enc.map((e, i) => ({ label: e, bars: ser.map(([n, cc]) => ({ label: n, value: D.encoders.fid[n][i], color: css(cc) })) })), height: 260, digits: 1, yLabel: "FID ↓" }));
  })();

  /* ---------------- Composition ---------------- */
  (function comp() {
    const g = $("#comp"); if (!g) return;
    for (let k = 0; k < 12; k++) {
      const c = el("div", { class: "ccard" }, g), s = el("div", { class: "srcs" }, c);
      const a = el("div", {}, s); el("img", { src: `${IMG}compose/${k}_anchor.webp`, alt: "Anchor object", loading: "lazy" }, a); el("span", { text: "anchor" }, a);
      const t = el("div", {}, s); el("img", { src: `${IMG}compose/${k}_target.webp`, alt: "Target background", loading: "lazy" }, t); el("span", { text: "target" }, t);
      const ar = el("div", { class: "flow hot arrow", "aria-hidden": "true" }, c); ar.style.width = "26px";
      const r = el("div", { class: "res" }, c); el("img", { src: `${IMG}compose/${k}_out.webp`, alt: "Composed sample", loading: "lazy" }, r);
    }
    let m = 1; const cm = $("#cMet"); if (!cm) return;
    D.compose.cols.forEach((n, i) => el("button", { type: "button", "data-v": i, "aria-pressed": String(i === m), text: n }, cm));
    const o = chart("#compChart", (c, o) => hBars(c, o, { rows: D.compose.rows.map(([l, v]) => ({ label: l, value: v[m], color: css(l.startsWith("IPA") ? "--s-sit" : "--s-base"), sub: l.startsWith("IPA") ? "REPA-G" : "baseline" })), digits: 3 }));
    seg("cMet", (v) => { m = +v; o.render(); });
  })();

  /* ---------------- Baselines (Fig 11, Tables 7/14/15) ---------------- */
  (function base() {
    const g = $("#cmpTable"); if (!g) return;
    [["Anchor", ""], ["TFG", ""], ["IP-Adapter", ""], ["REPA-G · REPA-E", "ours"], ["REPA-G · Self-Flow", "ours"]].forEach(([h, c]) => el("div", { class: "hd " + c, text: h }, g));
    for (let r = 0; r < 3; r++) ["anchor", "tfg", "ipadapter", "repae", "selfflow"].forEach((k) => el("img", { src: `${IMG}baselines/${r}_${k}.webp`, alt: `${k} sample ${r + 1}`, loading: "lazy" }, g));
    const mk = (wrap, head, rows, firstCols) => {
      if (!wrap) return;
      wrap.innerHTML = ""; const t = el("table", { class: "t" }, wrap), hr = el("tr", {}, el("thead", {}, t));
      head.forEach((h, i) => el("th", { text: h, class: i < firstCols ? "l" : "" }, hr));
      const tb = el("tbody", {}, t);
      rows.forEach((r) => { const tr = el("tr", {}, tb); r.forEach((v, i) => el("td", { text: v == null ? "–" : typeof v === "number" ? (v >= 100 ? v.toFixed(1) : v.toFixed(2)) : v, class: i < firstCols ? (String(v).startsWith("REPA-G") ? "l ours" : "l") : "num" }, tr)); });
    };
    mk($("#tfgTable"), ["Method", "Features", ...D.tfg.cols], D.tfg.rows.map(([a, b, v]) => [a, b, ...v]), 2);
    const ad = (ds) => mk($("#adTable"), ["Condition (" + (ds === "imagenet" ? "ImageNet" : "COCO") + ")", ...D.adapter.cols], D.adapter[ds].map(([a, v]) => [a, ...v]), 1);
    ad("imagenet"); seg("adDs", ad);
    mk($("#sfTable"), ["Model", "Cond.", "FID↓", "sFID↓", "IS↑", "Prec.↑", "Rec.↑", "DINOv2↑"], D.selfflow.map(([a, b, v]) => [a, b, ...v]), 2);
  })();

  /* ---------------- Table 8 · λ ---------------- */
  (function lam() {
    let met = "FID";
    const o = chart("#lamChart", (c, o) => lineChart(c, o, { xLabels: ["5k", "10k", "20k", "50k", "100k"], xTitle: "guidance scale λ", series: [["Full", "--s-sit"], ["Masked", "--s-dino"], ["Average", "--s-base"]].map(([n, cc]) => ({ label: n, color: css(cc), values: D.lambda[n][met] })), digits: met === "FID" ? 1 : 2, height: 280 }));
    seg("lMet", (v) => { met = v; o.render(); });
  })();

  /* ---------------- Figures 9 & 10 · diversity ---------------- */
  (function div() {
    const box = $("#divCharts"); if (!box) return;
    let ds = "imagenet"; const objs = [];
    [["lpips", "LPIPS distance"], ["dino_cos", "DINOv2 cosine dist."], ["vendi", "DINOv2 Vendi"]].forEach(([k, t]) => {
      const d = el("div", { class: "chart" }, box);
      objs.push(chart(d, (c, o) => lineChart(c, o, { xLabels: ["2k", "10k", "50k"], title: t, xTitle: "guidance scale λ", compact: true, height: 216, digits: 2, series: [["avg", "Average", "--s-dino"], ["full", "Full map", "--s-sit"]].map(([kk, n, cc]) => ({ label: n, color: css(cc), values: D.fig9[ds][k][kk] })) })));
    });
    const strip = $("#divStrip");
    const drawStrip = () => { strip.innerHTML = ""; ["anchor", "avg0", "avg1", "avg2", "full0", "full1", "full2"].forEach((k, i) => el("img", { src: `${IMG}diversity/${ds}_${k}.webp`, alt: i === 0 ? "Anchor" : `Sample (${k.startsWith("avg") ? "average" : "full"} features)`, class: i === 0 ? "first" : "", loading: "lazy" }, strip)); };
    seg("vDs", (v) => { ds = v; objs.forEach((o) => o.render()); drawStrip(); });
    drawStrip();
  })();

  /* ---------------- Figure 12 · wipe ---------------- */
  (function wipe() {
    const w = $("#wipe"); if (!w) return; const top = $("#wipeB"), bar = $("#wipeBar");
    const set = (p) => { p = Math.max(0, Math.min(100, p)); top.style.clipPath = `inset(0 0 0 ${p}%)`; bar.style.left = p + "%"; w.setAttribute("aria-valuenow", Math.round(p)); w._p = p; };
    let drag = false; const fromEv = (ev) => { const r = w.getBoundingClientRect(); set(((ev.clientX - r.left) / r.width) * 100); };
    w.addEventListener("pointerdown", (ev) => { drag = true; w.setPointerCapture(ev.pointerId); fromEv(ev); });
    w.addEventListener("pointermove", (ev) => { if (drag) fromEv(ev); });
    w.addEventListener("pointerup", () => (drag = false));
    w.addEventListener("keydown", (ev) => { if (ev.key === "ArrowLeft") { set((w._p ?? 50) - 5); ev.preventDefault(); } if (ev.key === "ArrowRight") { set((w._p ?? 50) + 5); ev.preventDefault(); } });
    set(50);
    const th = $("#hiThumbs");
    for (let i = 0; i < 4; i++) { const b = el("button", { type: "button", "aria-pressed": String(i === 1), "aria-label": `Example ${i + 1}` }, th); el("img", { src: `${IMG}hires/${i}_anchor.webp`, alt: "" }, b); b.addEventListener("click", () => { $$("button", th).forEach((x) => x.setAttribute("aria-pressed", String(x === b))); $("#wipeA").src = `${IMG}hires/${i}_anchor.webp`; top.src = `${IMG}hires/${i}_out.webp`; }); }
  })();

  /* ---------------- Table 17 ---------------- */
  chart("#rtChart", (c, o) => hBars(c, o, { rows: D.runtime.filter((r) => r[0] === "256").map(([, n, v, mem]) => ({ label: n, value: v, sub: `${mem} GB peak`, color: css(n.startsWith("REPA-G") ? "--s-sit" : "--s-base") })), digits: 2 }));

  /* ---------------- Figure 13 · failures ---------------- */
  (function fails() {
    const f = $("#fails"); if (!f) return;
    ["Inconsistent aspect ratio", "Failed blending", "Inconsistent viewpoint"].forEach((t, r) => {
      const d = el("div", { class: "fail" }, f);
      ["anchor", "target", "out"].forEach((k) => el("img", { src: `${IMG}failures/${r}_${k}.webp`, alt: `${t}: ${k}`, loading: "lazy" }, d));
      const p = el("p", {}, d); el("b", { text: t }, p);
    });
  })();

  /* ---------------- BibTeX copy ---------------- */
  $("#copyBib").addEventListener("click", async () => {
    const b = $("#copyBib"), txt = $("#bibText").textContent;
    try { await navigator.clipboard.writeText(txt); b.textContent = "Copied"; }
    catch (e) { const r = document.createRange(); r.selectNodeContents($("#bibText")); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "Selected, press Ctrl+C"; }
    setTimeout(() => (b.textContent = "Copy"), 1800);
  });

  if (theme !== "system") applyTheme(theme); else { const b = $("#themeBtn"); if (b) b.textContent = "◐ System"; }
})();
