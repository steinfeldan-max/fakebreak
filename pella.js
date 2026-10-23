// JavaScript port of pellav2's audio preprocessing (pellav2_infer.py: load_clip + Detector.features),
// so the voice check can run fully in the browser.

const PELLA_SR = 16000;
const PELLA_CROP = 4 * PELLA_SR;

// 2nd-order Butterworth high-pass at 20 Hz (same as scipy butter(2, 20, 'highpass', fs=16000)),
// applied forwards and backwards like scipy's sosfiltfilt (odd padding of 9 samples, steady-state start).
function pellaHighpass(x) {
  const w0 = 2 * Math.PI * 20 / PELLA_SR, q = Math.SQRT1_2, alpha = Math.sin(w0) / (2 * q), cos = Math.cos(w0);
  const a0 = 1 + alpha;
  const b0 = (1 + cos) / 2 / a0, b1 = -(1 + cos) / a0, b2 = b0, a1 = -2 * cos / a0, a2 = (1 - alpha) / a0;
  const run = sig => {
    const out = new Float64Array(sig.length);
    let z1 = (b1 + b2) * sig[0], z2 = b2 * sig[0]; // steady state for a constant input (DC gain is 0)
    for (let i = 0; i < sig.length; i++) {
      const v = sig[i], y = b0 * v + z1;
      z1 = b1 * v - a1 * y + z2;
      z2 = b2 * v - a2 * y;
      out[i] = y;
    }
    return out;
  };
  const pad = 9, n = x.length;
  const ext = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) {
    ext[i] = 2 * x[0] - x[pad - i];
    ext[n + pad + i] = 2 * x[n - 1] - x[n - 2 - i];
  }
  ext.set(x, pad);
  const fwd = run(ext).reverse();
  const back = run(fwd).reverse();
  return Float32Array.from(back.subarray(pad, pad + n));
}

// Same steps as load_clip: mono 16 kHz in, remove DC, high-pass, center-crop to 4 s, normalize loudness, pad.
// Loudness uses the RMS estimate (pellav2's own fallback) instead of full BS.1770; the features are
// normalized per clip afterwards, so this only changes how much a very loud clip clips.
function pellaPrep(input) {
  let x = Float32Array.from(input);
  const mean = x.reduce((a, b) => a + b, 0) / Math.max(1, x.length);
  for (let i = 0; i < x.length; i++) x[i] -= mean;
  if (x.length > 3 * PELLA_SR / 100) x = pellaHighpass(x);
  if (x.length > PELLA_CROP) {
    const off = Math.floor((x.length - PELLA_CROP) / 2);
    x = x.slice(off, off + PELLA_CROP);
  }
  if (x.length) {
    const rms = Math.sqrt(x.reduce((a, v) => a + v * v, 0) / x.length);
    const lufs = 20 * Math.log10(rms + 1e-9) - 0.691;
    const gain = Math.pow(10, Math.min(40, Math.max(-40, -23 - lufs)) / 20);
    for (let i = 0; i < x.length; i++) x[i] = Math.max(-1, Math.min(1, x[i] * gain));
  }
  const out = new Float32Array(PELLA_CROP);
  out.set(x.subarray(0, PELLA_CROP));
  return out;
}

// In-place radix-2 FFT on (re, im) arrays of length n (power of two).
function pellaFft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const ai = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k + len / 2] = re[i + k] - ar; im[i + k + len / 2] = im[i + k] - ai;
        re[i + k] += ar; im[i + k] += ai;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

let pellaMelBank = null;
function pellaMel() {
  if (pellaMelBank) return pellaMelBank;
  const mel = f => 1127 * Math.log(1 + f / 700);
  const nb = 80, nfft = 512, lo = mel(20), hi = mel(PELLA_SR / 2), delta = (hi - lo) / (nb + 1);
  pellaMelBank = [];
  for (let m = 0; m < nb; m++) {
    const left = lo + m * delta, center = left + delta, right = center + delta;
    const w = new Float64Array(nfft / 2 + 1);
    for (let i = 0; i < nfft / 2; i++) {
      const v = mel(PELLA_SR / nfft * i);
      w[i] = Math.max(0, Math.min((v - left) / (center - left), (right - v) / (right - center)));
    }
    pellaMelBank.push(w);
  }
  return pellaMelBank;
}

// Kaldi-style 80-bin log-mel filterbank (torchaudio.compliance.kaldi.fbank defaults: 25 ms povey window,
// 10 ms shift, pre-emphasis 0.97, DC removal, 512-point FFT), normalized per bin, then 2 frames stacked
// into 160-dim vectors. Returns { data: Float32Array(T*160), frames: T }.
function pellaFeatures(clip) {
  let end = clip.length;
  while (end > 0 && clip[end - 1] === 0) end--;
  const len = Math.max(end || clip.length, 400);
  const sig = new Float64Array(len);
  for (let i = 0; i < len; i++) sig[i] = (clip[i] || 0) * 32768;

  const N = 400, SHIFT = 160, NFFT = 512, NB = 80, EPS = 1.1920928955078125e-07;
  const frames = 1 + Math.floor((len - N) / SHIFT);
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)), 0.85);
  const bank = pellaMel();
  const feats = new Float64Array(frames * NB);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
  for (let f = 0; f < frames; f++) {
    re.fill(0); im.fill(0);
    let mean = 0;
    for (let i = 0; i < N; i++) mean += sig[f * SHIFT + i];
    mean /= N;
    for (let i = 0; i < N; i++) re[i] = sig[f * SHIFT + i] - mean;
    for (let i = N - 1; i > 0; i--) re[i] -= 0.97 * re[i - 1];
    re[0] -= 0.97 * re[0];
    for (let i = 0; i < N; i++) re[i] *= win[i];
    pellaFft(re, im);
    for (let m = 0; m < NB; m++) {
      const w = bank[m];
      let e = 0;
      for (let i = 0; i < NFFT / 2; i++) if (w[i]) e += w[i] * (re[i] * re[i] + im[i] * im[i]);
      feats[f * NB + m] = Math.log(Math.max(e, EPS));
    }
  }
  // Per-bin mean / unbiased variance normalization over time (torch .var() default).
  for (let m = 0; m < NB; m++) {
    let mean = 0;
    for (let f = 0; f < frames; f++) mean += feats[f * NB + m];
    mean /= frames;
    let v = 0;
    for (let f = 0; f < frames; f++) v += (feats[f * NB + m] - mean) ** 2;
    v /= Math.max(1, frames - 1);
    const sd = Math.sqrt(v + 1e-7);
    for (let f = 0; f < frames; f++) feats[f * NB + m] = (feats[f * NB + m] - mean) / sd;
  }
  const T = Math.floor(frames / 2);
  return { data: Float32Array.from(feats.subarray(0, T * 2 * NB)), frames: T };
}
