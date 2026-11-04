// Real-time viewer for the trained Gaussians of my globe capture (3D Gaussian splatting, rendered in the browser).
// WebGL2, no libraries. Same image formation as training with gsplat: each Gaussian is projected to a 2D ellipse
// (covariance J W Σ Wᵀ Jᵀ plus 0.3 px), coloured by spherical harmonics up to degree 3 for the viewing direction,
// and the ellipses are blended front to back over a white background. A worker re-sorts them by depth as the
// camera moves.
//
// The data file is written by nerf-globe/tools/export_web_splat.py: 20 uint32 words per Gaussian (see there).
//
//   import { createSplatViewer } from './splat-viewer.js';
//   const v = await createSplatViewer(canvas, { url: 'globe-gaussians', onProgress, onFps });
//   v.start(); v.stop(); v.reset();

const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D u_tex;
uniform mat4 u_view;
uniform vec3 u_campos;
uniform vec2 u_focal;
uniform vec2 u_size;
uniform vec3 u_shs;
uniform float u_covs;
in vec2 a_corner;
in uint a_idx;
out vec4 v_col;
out vec2 v_pos;
uint W[12];
float s8(int j){ return float(int(W[j >> 2] << uint(24 - 8 * (j & 3))) >> 24); }
vec3 cf(int k){ return vec3(s8(3 * k), s8(3 * k + 1), s8(3 * k + 2)); }
void main(){
  int x0 = int(a_idx % 400u) * 5, y0 = int(a_idx / 400u);
  uvec4 t0 = texelFetch(u_tex, ivec2(x0, y0), 0);
  vec3 p = uintBitsToFloat(t0.xyz);
  vec4 c = u_view * vec4(p, 1.0);
  vec2 half_ = 0.5 * u_size;
  vec2 ctr = u_focal * c.xy / c.z;
  if (c.z < 0.02 || any(greaterThan(abs(ctr), 1.3 * half_))){ gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  uvec4 t1 = texelFetch(u_tex, ivec2(x0 + 1, y0), 0);
  vec2 h0 = unpackHalf2x16(t0.w), h1 = unpackHalf2x16(t1.x);
  vec2 a = unpackHalf2x16(t1.y), b = unpackHalf2x16(t1.z), d = unpackHalf2x16(t1.w);
  mat3 V = mat3(a.x, a.y, b.x,  a.y, b.y, d.x,  b.x, d.x, d.y) * u_covs;
  float iz = 1.0 / c.z;
  mat3 J = mat3(u_focal.x * iz, 0.0, 0.0,  0.0, u_focal.y * iz, 0.0,  -u_focal.x * c.x * iz * iz, -u_focal.y * c.y * iz * iz, 0.0);
  mat3 T = J * mat3(u_view);
  mat3 S = T * V * transpose(T);
  float sa = S[0][0] + 0.3, sb = S[0][1], sc = S[1][1] + 0.3;
  float mid = 0.5 * (sa + sc), rad = length(vec2(0.5 * (sa - sc), sb));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 dv = abs(sb) < 1e-9 ? (sa >= sc ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(sb, l1 - sa));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dv;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dv.y, -dv.x);

  uvec4 t2 = texelFetch(u_tex, ivec2(x0 + 2, y0), 0), t3 = texelFetch(u_tex, ivec2(x0 + 3, y0), 0), t4 = texelFetch(u_tex, ivec2(x0 + 4, y0), 0);
  W[0] = t2.x; W[1] = t2.y; W[2] = t2.z; W[3] = t2.w; W[4] = t3.x; W[5] = t3.y; W[6] = t3.z; W[7] = t3.w;
  W[8] = t4.x; W[9] = t4.y; W[10] = t4.z; W[11] = t4.w;
  vec3 r = normalize(p - u_campos);
  float x = r.x, y = r.y, z = r.z, xx = x * x, yy = y * y, zz = z * z;
  vec3 col = 0.28209479177387814 * vec3(h0.y, h1.x, h1.y);
  col += u_shs.x * 0.4886025119029199 * (-y * cf(0) + z * cf(1) - x * cf(2));
  col += u_shs.y * (1.0925484305920792 * x * y * cf(3) - 1.0925484305920792 * y * z * cf(4)
       + 0.31539156525252005 * (2.0 * zz - xx - yy) * cf(5) - 1.0925484305920792 * x * z * cf(6)
       + 0.5462742152960396 * (xx - yy) * cf(7));
  col += u_shs.z * (-0.5900435899266435 * y * (3.0 * xx - yy) * cf(8) + 2.890611442640554 * x * y * z * cf(9)
       - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * cf(10) + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * cf(11)
       - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * cf(12) + 1.445305721320277 * z * (xx - yy) * cf(13)
       - 0.5900435899266435 * x * (xx - 3.0 * yy) * cf(14));
  v_col = vec4(max(col + 0.5, 0.0), h0.x);          // like gsplat: no upper clip here, only on the finished image
  v_pos = a_corner;
  vec2 px = ctr + a_corner.x * major + a_corner.y * minor;
  gl_Position = vec4(px.x / half_.x, -px.y / half_.y, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec4 v_col;
in vec2 v_pos;
out vec4 o;
void main(){
  float A = dot(v_pos, v_pos);
  if (A > 4.0) discard;
  float al = min(0.99, v_col.a * exp(-A));
  if (al < 1.0 / 255.0) discard;
  o = vec4(v_col.rgb * al, al);
}`;

// second pass: the accumulated colour and opacity (16-bit float) over the background, clipped to 0..1 like gsplat
const VS2 = `#version 300 es
in vec2 a; void main(){ gl_Position = vec4(a, 0.0, 1.0); }`;
const FS2 = `#version 300 es
precision highp float;
uniform sampler2D u_acc;
uniform vec3 u_bg;
out vec4 o;
void main(){
  vec4 c = texelFetch(u_acc, ivec2(gl_FragCoord.xy), 0);
  o = vec4(clamp(c.rgb + (1.0 - c.a) * u_bg, 0.0, 1.0), 1.0);
}`;

// depth sort in a worker: 16-bit counting sort of the camera-space depth, nearest first
const WORKER = `
let P = null, n = 0, counts = new Uint32Array(65536), dep = null;
onmessage = e => {
  if (e.data.pos){ P = e.data.pos; n = P.length / 3; dep = new Float32Array(n); return; }
  const v = e.data.row, a = v[0], b = v[1], c = v[2], d = v[3];
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++){ const z = a * P[3 * i] + b * P[3 * i + 1] + c * P[3 * i + 2] + d; dep[i] = z; if (z < mn) mn = z; if (z > mx) mx = z; }
  const s = 65535 / Math.max(mx - mn, 1e-9);
  counts.fill(0);
  const key = new Uint16Array(n);
  for (let i = 0; i < n; i++){ const k = ((dep[i] - mn) * s) | 0; key[i] = k; counts[k]++; }
  for (let k = 0, acc = 0; k < 65536; k++){ const t = counts[k]; counts[k] = acc; acc += t; }
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[counts[key[i]]++] = i;
  postMessage({ order: out, id: e.data.id }, [out.buffer]);
};`;

async function fetchWithProgress(url, onProgress){
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const buf = new Uint8Array(total), reader = res.body.getReader();
  let got = 0;
  for (;;){
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got); got += value.length;
    onProgress && onProgress(got / total);
  }
  return buf;
}

export async function createSplatViewer(canvas, opts = {}){
  const base = opts.url || 'globe-gaussians';
  const meta = opts.meta || await (await fetch(base + '.json')).json();
  const bytes = await fetchWithProgress(base + '.bin', opts.onProgress);
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const n = meta.count;

  const gl = canvas.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: true, preserveDrawingBuffer: !!opts.preserve });
  if (!gl) throw new Error('WebGL2 is not available in this browser');

  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const link = (vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; };
  const prog = link(VS, FS), prog2 = link(VS2, FS2);
  const bg = opts.background || [1, 1, 1];
  // accumulate in a 16-bit float buffer when the GPU allows it (almost all do), else straight into the canvas
  const floatOK = !!gl.getExtension('EXT_color_buffer_float');
  let acc = null, accTex = null, accW = 0, accH = 0;
  const vao2 = gl.createVertexArray(); gl.bindVertexArray(vao2);
  const tri = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, tri); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const aT = gl.getAttribLocation(prog2, 'a'); gl.enableVertexAttribArray(aT); gl.vertexAttribPointer(aT, 2, gl.FLOAT, false, 0, 0);
  gl.useProgram(prog2); gl.uniform1i(gl.getUniformLocation(prog2, 'u_acc'), 1); gl.uniform3fv(gl.getUniformLocation(prog2, 'u_bg'), bg);
  function ensureAcc(w, h){
    if (!floatOK || (acc && accW === w && accH === h)) return;
    if (accTex) gl.deleteTexture(accTex);
    if (!acc) acc = gl.createFramebuffer();
    accTex = gl.createTexture(); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, accTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, acc); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, accTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.activeTexture(gl.TEXTURE0);
    accW = w; accH = h;
  }
  gl.useProgram(prog);
  const U = name => gl.getUniformLocation(prog, name);

  // data texture: 400 Gaussians per row, 5 RGBA32UI texels each
  const rows = Math.ceil(n / 400), tex = gl.createTexture(), padded = new Uint32Array(rows * 400 * 20);
  padded.set(words.subarray(0, n * 20));
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 2000, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, padded);
  gl.uniform1i(U('u_tex'), 0);
  gl.uniform3fv(U('u_shs'), meta.shScale);
  gl.uniform1f(U('u_covs'), 1 / meta.covScale);

  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
  const aC = gl.getAttribLocation(prog, 'a_corner'); gl.enableVertexAttribArray(aC); gl.vertexAttribPointer(aC, 2, gl.FLOAT, false, 0, 0);
  const idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ARRAY_BUFFER, n * 4, gl.DYNAMIC_DRAW);
  const aI = gl.getAttribLocation(prog, 'a_idx'); gl.enableVertexAttribArray(aI); gl.vertexAttribIPointer(aI, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(aI, 1);

  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);   // front to back
  gl.clearColor(0, 0, 0, 0);

  // sorting worker
  const pos = new Float32Array(n * 3);
  const f32 = new Float32Array(words.buffer, words.byteOffset, words.length);
  for (let i = 0; i < n; i++){ pos[3 * i] = f32[20 * i]; pos[3 * i + 1] = f32[20 * i + 1]; pos[3 * i + 2] = f32[20 * i + 2]; }
  const worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: 'text/javascript' })));
  worker.postMessage({ pos }, [pos.buffer]);
  let sorted = 0, sortBusy = false, sortRow = null, lastRow = null;
  worker.onmessage = e => {
    gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, e.data.order);
    sorted = n; sortBusy = false; dirty = true;
    if (sortRow) requestSort(sortRow);
  };
  function requestSort(row){
    if (sortBusy){ sortRow = row; return; }
    sortRow = null; sortBusy = true; lastRow = row.slice();
    worker.postMessage({ row });
  }

  // camera: orbit around the globe, z up, OpenCV camera (x right, y down, z forward)
  const home = {
    az: opts.azimuth ?? -0.35,
    el: opts.elevation ?? Math.asin(Math.min(0.9, meta.cameraHeight / meta.cameraDistance)),
    dist: opts.distance ?? meta.cameraDistance,
    target: opts.target || [0, 0, 0.02],
  };
  const cam = { ...home, target: [...home.target] }, goal = { dist: home.dist };
  let vAz = 0, vEl = 0, lastInput = -1e9, auto = opts.autoRotate !== false;
  const autoSpeed = opts.autoSpeed ?? 0.22;          // radians per second

  function viewMatrix(){
    const ce = Math.cos(cam.el), t = cam.target;
    const eye = [t[0] + cam.dist * ce * Math.cos(cam.az), t[1] + cam.dist * ce * Math.sin(cam.az), t[2] + cam.dist * Math.sin(cam.el)];
    let f = [t[0] - eye[0], t[1] - eye[1], t[2] - eye[2]]; const fl = Math.hypot(...f); f = f.map(v => v / fl);
    let r = [f[1] * 1 - f[2] * 0, f[2] * 0 - f[0] * 1, 0]; const rl = Math.hypot(...r) || 1; r = r.map(v => v / rl);   // f x up(0,0,1)
    const dn = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];                    // f x r
    const tx = -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]), ty = -(dn[0] * eye[0] + dn[1] * eye[1] + dn[2] * eye[2]), tz = -(f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
    return { eye, m: new Float32Array([r[0], dn[0], f[0], 0, r[1], dn[1], f[1], 0, r[2], dn[2], f[2], 0, tx, ty, tz, 1]), row: [f[0], f[1], f[2], tz] };
  }

  // input: drag to turn (with a little inertia), wheel or pinch to zoom, double-click to go back
  const pointers = new Map();
  let pinch0 = null;
  canvas.style.touchAction = 'none';
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, t: performance.now() }); lastInput = performance.now(); vAz = vEl = 0; if (pointers.size === 2){ const [p, q] = [...pointers.values()]; pinch0 = { d: Math.hypot(p.x - q.x, p.y - q.y), dist: goal.dist }; } });
  canvas.addEventListener('pointermove', e => {
    const p = pointers.get(e.pointerId); if (!p) return;
    const now = performance.now(), rect = canvas.getBoundingClientRect(), k = 3.2 / Math.max(rect.width, 1);
    if (pointers.size === 1){
      const dAz = -(e.clientX - p.x) * k, dEl = (e.clientY - p.y) * k * 0.8, dt = Math.max(now - p.t, 1) / 1000;
      cam.az += dAz; cam.el = Math.max(-1.1, Math.min(1.45, cam.el + dEl));
      vAz = 0.7 * vAz + 0.3 * dAz / dt; vEl = 0.7 * vEl + 0.3 * dEl / dt;
    }
    p.x = e.clientX; p.y = e.clientY; p.t = now; lastInput = now; dirty = true;
    if (pointers.size === 2 && pinch0){ const [a, b] = [...pointers.values()]; goal.dist = clampDist(pinch0.dist * pinch0.d / Math.max(Math.hypot(a.x - b.x, a.y - b.y), 1)); }
  });
  const up = e => { const p = pointers.get(e.pointerId); pointers.delete(e.pointerId); if (pointers.size < 2) pinch0 = null; if (p && performance.now() - p.t > 60){ vAz = vEl = 0; } };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('wheel', e => { e.preventDefault(); goal.dist = clampDist(goal.dist * Math.exp(e.deltaY * 0.0012)); lastInput = performance.now(); }, { passive: false });
  canvas.addEventListener('dblclick', () => reset());
  const clampDist = d => Math.max(0.6, Math.min(4, d));
  function reset(){ Object.assign(cam, { az: home.az, el: home.el, dist: home.dist }); goal.dist = home.dist; vAz = vEl = 0; lastInput = -1e9; dirty = true; }

  // render loop
  let raf = 0, last = 0, dirty = true, fpsT = 0, fpsN = 0, fps = 0;
  function frame(now){
    raf = requestAnimationFrame(frame);
    const dt = Math.min((now - (last || now)) / 1000, 0.05); last = now;
    const idle = now - lastInput > 3500 && pointers.size === 0;
    if (pointers.size === 0){
      if (Math.abs(vAz) + Math.abs(vEl) > 1e-3){ cam.az += vAz * dt; cam.el = Math.max(-1.1, Math.min(1.45, cam.el + vEl * dt)); const k = Math.exp(-dt * 3.5); vAz *= k; vEl *= k; dirty = true; }
      else if (auto && idle){ cam.az += autoSpeed * dt; dirty = true; }
    }
    if (Math.abs(goal.dist - cam.dist) > 1e-4){ cam.dist += (goal.dist - cam.dist) * (1 - Math.exp(-dt * 10)); dirty = true; }

    // frames per second of the page (every animation frame counts, drawn or not: a slow draw lowers it)
    fpsN++;
    if (now - fpsT > 1000){ fps = fpsN * 1000 / (now - fpsT); fpsT = now; fpsN = 0; opts.onFps && opts.onFps(fps); }
    const dpr = Math.min(window.devicePixelRatio || 1, opts.maxDpr || 2), rect = canvas.getBoundingClientRect();
    const Wp = Math.max(1, Math.round(rect.width * dpr)), Hp = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== Wp || canvas.height !== Hp){ canvas.width = Wp; canvas.height = Hp; dirty = true; }
    if (!dirty && sorted && !api.redrawAlways) return;
    dirty = false;
    draw(Wp, Hp);
  }
  function draw(Wp, Hp){
    const v = api.fixedView || viewMatrix();
    if (!lastRow || Math.abs(v.row[0] - lastRow[0]) + Math.abs(v.row[1] - lastRow[1]) + Math.abs(v.row[2] - lastRow[2]) + Math.abs(v.row[3] - lastRow[3]) > 0.002) requestSort(v.row);
    // same framing as the iPhone frames at the same distance: their focal length relative to width (or height)
    const f = v.f || meta.focalRatio * Math.min(Wp, Hp * 1080 / 607) * (opts.zoom || 1);
    ensureAcc(Wp, Hp);
    gl.viewport(0, 0, Wp, Hp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, floatOK ? acc : null);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (sorted){
      gl.enable(gl.BLEND);
      gl.useProgram(prog); gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniformMatrix4fv(U('u_view'), false, v.m);
      gl.uniform3fv(U('u_campos'), v.eye);
      gl.uniform2f(U('u_focal'), f, f);
      gl.uniform2f(U('u_size'), Wp, Hp);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, sorted);
    }
    if (floatOK){
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.disable(gl.BLEND);
      gl.useProgram(prog2); gl.bindVertexArray(vao2);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, accTex);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.activeTexture(gl.TEXTURE0);
    }
  }
  // milliseconds the GPU needs for one frame at the canvas's current size: draw n frames, then wait for them
  function benchmark(n = 40){
    if (!sorted || !canvas.width) return null;
    const px = new Uint8Array(4);
    draw(canvas.width, canvas.height); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const t = performance.now();
    for (let i = 0; i < n; i++) draw(canvas.width, canvas.height);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return (performance.now() - t) / n;
  }
  // redrawAlways: draw every frame even when nothing moves (for measuring the frame rate)
  // fixedView: { m: world-to-camera 4x4 (column-major, OpenCV axes), eye, row: depth row, f } replaces the orbit
  // camera, used to check the renderer against gsplat on the test photos
  const api = {
    fixedView: null,
    count: n, meta,
    start(){ if (!raf){ last = 0; fpsT = performance.now(); fpsN = 0; dirty = true; raf = requestAnimationFrame(frame); } },
    stop(){ cancelAnimationFrame(raf); raf = 0; },
    reset, benchmark,
    get ready(){ return sorted > 0; },
    setAutoRotate(on){ auto = on; },
    get fps(){ return fps; },
    redrawAlways: false,
  };
  return api;
}
