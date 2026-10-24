// JavaScript port of pellav2's audio preprocessing (pellav2_infer.py: load_clip + Detector.features),
// so the voice check can run fully in the browser.

const PELLA_SR = 16000;
const PELLA_CROP = 4 * PELLA_SR;

// Exact port of torchaudio.functional.resample (sinc_interp_hann, lowpass_filter_width 6, rolloff 0.99).
// The detector looks at high frequencies, so the browser's own resampler (slightly different filter)
// turned a real 44.1 kHz clip into "99.6% fake"; with this port it matches Python.
function pellaResample(x, orig, target) {
  if (orig === target) return Float32Array.from(x);
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const g = gcd(orig, target), o = orig / g, n = target / g;
  const LPW = 6, base = Math.min(o, n) * 0.99;
  const width = Math.ceil(LPW * o / base);
  const klen = 2 * width + o;
  const kernels = [];
  for (let j = 0; j < n; j++) {
    const k = new Float64Array(klen);
    for (let i = 0; i < klen; i++) {
      let t = (-j / n + (i - width) / o) * base;
      t = Math.max(-LPW, Math.min(LPW, t));
      const win = Math.cos(t * Math.PI / LPW / 2) ** 2;
      const tp = t * Math.PI;
      k[i] = (tp === 0 ? 1 : Math.sin(tp) / tp) * win * base / o;
    }
    kernels.push(k);
  }
  const padded = new Float64Array(x.length + klen);
  padded.set(x, width);
  const frames = Math.floor((padded.length - klen) / o) + 1;
  const outLen = Math.ceil(n * x.length / o);
  const out = new Float32Array(outLen);
  for (let f = 0; f < frames; f++) {
    for (let j = 0; j < n; j++) {
      const idx = f * n + j;
      if (idx >= outLen) return out;
      const k = kernels[j], off = f * o;
      let s = 0;
      for (let i = 0; i < klen; i++) s += padded[off + i] * k[i];
      out[idx] = s;
    }
  }
  return out;
}

// Read a WAV file directly (PCM 8/16/24/32-bit or float), so it's not resampled by the browser on decode.
// Returns { rate, mono } or null if it isn't a WAV we understand.
function pellaParseWav(buf) {
  const v = new DataView(buf);
  const str = (o, len) => String.fromCharCode(...new Uint8Array(buf, o, len));
  if (buf.byteLength < 44 || str(0, 4) !== 'RIFF' || str(8, 4) !== 'WAVE') return null;
  let o = 12, fmt = null;
  while (o + 8 <= buf.byteLength) {
    const id = str(o, 4), size = v.getUint32(o + 4, true);
    if (id === 'fmt ') {
      let format = v.getUint16(o + 8, true);
      if (format === 0xFFFE && size >= 26) format = v.getUint16(o + 32, true); // WAVE_FORMAT_EXTENSIBLE
      fmt = { format, channels: v.getUint16(o + 10, true), rate: v.getUint32(o + 12, true), bits: v.getUint16(o + 22, true) };
    } else if (id === 'data' && fmt) {
      const { format, channels, rate, bits } = fmt, bytes = bits / 8;
      if (!(format === 1 || format === 3) || ![8, 16, 24, 32, 64].includes(bits)) return null;
      const frames = Math.floor(Math.min(size, buf.byteLength - o - 8) / (bytes * channels));
      const mono = new Float32Array(frames);
      let p = o + 8;
      for (let i = 0; i < frames; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++, p += bytes) {
          let s;
          if (format === 3) s = bits === 32 ? v.getFloat32(p, true) : v.getFloat64(p, true);
          else if (bits === 8) s = (v.getUint8(p) - 128) / 128;
          else if (bits === 16) s = v.getInt16(p, true) / 32768;
          else if (bits === 24) s = ((v.getUint8(p) | (v.getUint8(p + 1) << 8) | (v.getInt8(p + 2) << 16))) / 8388608;
          else s = v.getInt32(p, true) / 2147483648;
          sum += s;
        }
        mono[i] = sum / channels;
      }
      return { rate, mono };
    }
    o += 8 + size + (size & 1);
  }
  return null;
}

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
