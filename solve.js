// Faster drop-in for makefun-diff 1.0.0 (make.fun "Diff two texts faster").
//
// Original implementation, written from the competition spec (RULES.txt):
// lines are cut after every \n; the common line prefix is kept, then the
// common suffix of what is left; between them Myers' greedy search runs
// from the first lines, keeping on each diagonal the path reaching
// furthest into the old text, preferring a deletion when a deletion and an
// insertion reach the same point, never moving past either text's end,
// pruning no edge diagonal, and following every run of equal lines as far
// as it goes; the first path to reach both ends is the script.
//
// Performance notes:
// - Lines are interned to integer ids (one Map): the snake test runs once
//   per diagonal cell (~D^2/2 of them), so integer comparison beats
//   string comparison even after paying for the interning.
// - The forward pass records, for every (D, k) cell it visits, one packed
//   word: the snake length and which predecessor (deletion/insertion) the
//   greedy choice took. Backtracking follows those recorded choices, so it
//   reproduces the forward greedy path exactly without keeping a V
//   snapshot per D and without re-running the search. One word per cell,
//   not one per (D, k) pair of the full k-range, so half the memory of
//   V snapshots and no per-level copy.
// - A bidirectional / divide-and-conquer search was evaluated and NOT
//   used: its total diagonal work is the same geometric sum as the
//   forward pass (D^2/2), and with many equal-cost shortest paths the
//   middle snake it picks need not lie on the forward greedy path, so it
//   cannot guarantee the identical script (see builder-B-report.md).

const EQ = 0, DEL = 1, INS = 2;

// Reusable scratch buffer for the packed Myers records (see diffMiddle).
// Holds no answer data; only the current case's search records.
let packBuf = new Uint32Array(1 << 20);
let packBufCap = 1 << 20;

function splitLines(s) {
  const out = [];
  let start = 0;
  for (;;) {
    const nl = s.indexOf('\n', start);
    if (nl < 0) break;
    out.push(s.slice(start, nl + 1));
    start = nl + 1;
  }
  if (start < s.length) out.push(s.slice(start));
  return out;
}

export function run(input) {
  const a = splitLines(input.old);
  const b = splitLines(input.new);
  const N = a.length, M = b.length;

  // Intern lines to integer ids (shared map, so equal lines share an id).
  const map = new Map();
  let next = 1;
  const ai = new Int32Array(N), bi = new Int32Array(M);
  for (let i = 0; i < N; i++) {
    let id = map.get(a[i]);
    if (id === undefined) { id = next++; map.set(a[i], id); }
    ai[i] = id;
  }
  for (let i = 0; i < M; i++) {
    let id = map.get(b[i]);
    if (id === undefined) { id = next++; map.set(b[i], id); }
    bi[i] = id;
  }

  // Common prefix, then common suffix of the remainder.
  let p = 0;
  while (p < N && p < M && ai[p] === bi[p]) p++;
  let s = 0;
  while (s < N - p && s < M - p && ai[N - 1 - s] === bi[M - 1 - s]) s++;
  const n = N - p - s, m = M - p - s;

  // Edit ops for the middle region, as codes, in order.
  const mid = diffMiddle(ai, bi, p, n, m);

  // Full op array: prefix equals + middle ops + suffix equals.
  const all = new Int8Array(p + mid.length + s);
  all.set(mid, p);

  return formatPatch(a, b, all, N, M);
}

// Myers forward greedy on ai[p .. p+n) vs bi[p .. p+m); returns Int8Array ops.
function diffMiddle(a, b, p, n, m) {
  const max = n + m;
  if (max === 0) return new Int8Array(0);
  const v = new Int32Array(2 * max + 1);
  const off = max;
  const km = n - m;

  // Packed per-cell records, in visit order (D ascending, k ascending):
  // pack = (snakeLength << 1) | isDeletion. Cell (d, k) is at index
  // d*(d+1)/2 + (k+d)/2 (level d holds d+1 cells).
  // The buffer is a reusable scratch allocation, reset per case (it holds
  // no answer data, only the current search's records).
  let packCap = packBufCap;
  let packs = packBuf;
  let pidx = 0;
  // Capacity is checked once per D level (level d holds d+1 cells), never
  // per cell: a per-cell check would cost millions of calls in cold code.
  const ensureLevel = (cells) => {
    if (pidx + cells > packCap) {
      do { packCap *= 2; } while (pidx + cells > packCap);
      const np = new Uint32Array(packCap);
      np.set(packs);
      packs = np;
    }
  };

  let D = -1;
  outer:
  for (let d = 0; d <= max; d++) {
    ensureLevel(d + 1);
    if (d === 0) {
      let x = 0, y = 0;
      while (x < n && y < m && a[p + x] === b[p + y]) { x++; y++; }
      v[off] = x;
      packs[pidx++] = x << 1; // no edit, snake of length x
      if (x >= n && y >= m) { D = 0; break outer; }
      continue;
    }
    // k = -d edge: insertion only.
    let k = -d;
    let xe = v[off + k + 1];
    let x = xe, y = x - k;
    while (x < n && y < m && a[p + x] === b[p + y]) { x++; y++; }
    v[off + k] = x;
    packs[pidx++] = ((x - xe) << 1) | 0;
    if (k === km && x >= n) { D = d; break outer; }
    // Interior diagonals: deletion wins ties (del >= ins).
    // vi tracks off+k directly, so each cell costs one indexed read pair.
    let vi;
    for (k = -d + 2, vi = off - d + 2; k <= d - 2; k += 2, vi += 2) {
      const del = v[vi - 1] + 1;
      const ins = v[vi + 1];
      const isDel = del >= ins ? 1 : 0;
      xe = isDel ? del : ins;
      x = xe; y = x - k;
      while (x < n && y < m && a[p + x] === b[p + y]) { x++; y++; }
      v[vi] = x;
      packs[pidx++] = ((x - xe) << 1) | isDel;
      if (k === km && x >= n) { D = d; break outer; }
    }
    // k = d edge: deletion only.
    k = d;
    xe = v[off + k - 1] + 1;
    x = xe; y = x - k;
    while (x < n && y < m && a[p + x] === b[p + y]) { x++; y++; }
    v[off + k] = x;
    packs[pidx++] = ((x - xe) << 1) | 1;
    if (k === km && x >= n) { D = d; break outer; }
  }
  if (D < 0) D = max; // unreachable in theory; D <= n + m always

  // Backtrack the exact greedy path from (n, m) using the recorded cells.
  const ops = new Int8Array(n + m);
  let oi = n + m;
  let x = n, y = m;
  for (let d = D; d > 0; d--) {
    const k = x - y;
    const pack = packs[(d * (d + 1) >> 1) + ((k + d) >> 1)];
    const s = pack >>> 1;
    for (let t = 0; t < s; t++) { ops[--oi] = EQ; x--; y--; }
    if (pack & 1) { ops[--oi] = DEL; x--; }
    else { ops[--oi] = INS; y--; }
  }
  while (x > 0 && y > 0) { ops[--oi] = EQ; x--; y--; }
  packBuf = packs; packBufCap = packCap; // keep the buffer for the next case
  return ops.subarray(oi);
}

// Build the makefun-diff/1.0.0 patch string from the op array.
// all[i]: EQ/DEL/INS consuming a[i] and/or b[j] in order.
function formatPatch(a, b, all, N, M) {
  const total = all.length;
  let deleted = 0, inserted = 0;
  for (let i = 0; i < total; i++) {
    if (all[i] === DEL) deleted++;
    else if (all[i] === INS) inserted++;
  }

  // Change runs: maximal runs of non-equal ops, with old/new line counts
  // at each run boundary, so hunk headers can be computed in one pass.
  // runs: [startOp, endOp, oldBefore, newBefore, oldAfter, newAfter]
  const runs = [];
  let oi = 0, oj = 0;
  let i = 0;
  while (i < total) {
    if (all[i] === EQ) { i++; oi++; oj++; continue; }
    const start = i, oldBefore = oi, newBefore = oj;
    while (i < total && all[i] !== EQ) {
      if (all[i] === DEL) oi++; else oj++;
      i++;
    }
    runs.push([start, i, oldBefore, newBefore, oi, oj]);
  }

  // Group runs into hunks: up to 3 kept lines before/after each run;
  // runs at most 6 kept lines apart share a hunk.
  const hunks = []; // [startOp, endOp, oldStart, newStart, oldEnd, newEnd]
  for (const r of runs) {
    // context start: at most 3 equals before the run
    let hs = r[0];
    let c = 0;
    while (hs > 0 && c < 3 && all[hs - 1] === EQ) { hs--; c++; }
    let he = r[1];
    c = 0;
    while (he < total && c < 3 && all[he] === EQ) { he++; c++; }
    // Context ops are all equals, each consuming one old and one new line.
    const before = r[0] - hs, after = he - r[1];
    const os = r[2] - before, ns = r[3] - before;
    const oe = r[4] + after, ne = r[5] + after;
    if (hunks.length > 0) {
      const last = hunks[hunks.length - 1];
      // merge when the previous hunk's end reaches this hunk's start:
      // that is exactly gap <= 6 kept lines between the runs.
      if (hs <= last[1]) {
        last[1] = he; last[4] = oe; last[5] = ne;
        continue;
      }
    }
    hunks.push([hs, he, os, ns, oe, ne]);
  }

  const out = [];
  out.push(`makefun-diff/1 old=${N} new=${M} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`);
  for (const h of hunks) {
    out.push(`@ ${h[2] + 1} ${h[4] - h[2]} ${h[3] + 1} ${h[5] - h[3]}\n`);
    let li = h[2], lj = h[3];
    const emit = (mark, line) => {
      out.push(mark, line);
      if (!line.endsWith('\n')) out.push('\n%\n');
    };
    let q = h[0];
    while (q < h[1]) {
      if (all[q] === EQ) {
        emit('=', a[li++]);
        lj++;
        q++;
        continue;
      }
      // One run of changes: list its deletions before its insertions.
      let end = q;
      while (end < h[1] && all[end] !== EQ) end++;
      for (let r = q; r < end; r++) if (all[r] === DEL) emit('-', a[li++]);
      for (let r = q; r < end; r++) if (all[r] === INS) emit('+', b[lj++]);
      q = end;
    }
  }
  return out.join('');
}
