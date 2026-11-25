// Independent checks on the exact JavaScript kernel shipped inside the offline page.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const kernel = html.match(/<script id="ising-kernel">([\s\S]*?)<\/script>/)[1];
const SpinLattice = vm.runInNewContext(kernel + '\nSpinLattice');

function energy(spins, length) {
  // Count each directed nearest neighbour, then remove the factor of two.
  let sum = 0;
  for (let x = 0; x < length; ++x) for (let y = 0; y < length; ++y) {
    for (const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const neighbour = ((y + dy + length) % length)*length + (x + dx + length) % length;
      sum -= spins[y*length+x]*spins[neighbour];
    }
  }
  return sum/2;
}

test('Every flip on the 2×2 torus gives the full Hamiltonian difference', () => {
  const run = new SpinLattice(2, 0, 'up', 1.3);
  for (let mask = 0; mask < 16; ++mask) {
    const spins = Array.from({length:4}, (_, i) => mask & (1 << i) ? 1 : -1);
    run.spins.set(spins); run.measure();
    assert.equal(run.energy, energy(spins, 2));
    assert.equal(run.magnetization, spins.reduce((a,b) => a+b));
    for (let i = 0; i < 4; ++i) {
      const flipped = spins.slice(); flipped[i] *= -1;
      assert.ok(run.energyChange(i) === energy(flipped, 2) - run.energy);
    }
  }
});

test('A zero-temperature ground state is stationary; seeded runs reproduce both moments', () => {
  const cold = new SpinLattice(5, 42, 'down', 0);
  for (let i = 0; i < 8; ++i) cold.sweep();
  assert.equal(cold.energy, -50); assert.equal(cold.magnetization, -25);
  const a = new SpinLattice(5, 0, 'random', 2.269), b = new SpinLattice(5, 0, 'random', 2.269);
  for (let k = 0; k < 200; ++k) {
    a.sweep(); b.sweep();
    assert.deepEqual(a.spins, b.spins);
    assert.equal(a.energy, energy(a.spins, 5));
    assert.equal(a.magnetization, a.spins.reduce((x,y) => x+y));
  }
});

test('Seeded Metropolis sampling recovers the exact 2×2 canonical means', () => {
  const T = 3;
  let z = 0, e = 0, absM = 0;
  for (let i = 0; i < 16; ++i) {
    const spins = Array.from({length: 4}, (_, j) => (i >> j) & 1 ? 1 : -1);
    const E = energy(spins, 2), w = Math.exp(-E/T);
    z += w; e += w*E/4; absM += w*Math.abs(spins.reduce((a,b) => a+b))/4;
  }
  const run = new SpinLattice(2, 2134, 'random', T);
  for (let i=0; i<64; ++i) run.sweep();
  const n = 45000;
  let sampledE=0, sampledAbsM=0;
  for (let i=0; i<n; ++i) { run.sweep(); sampledE += run.energy/4; sampledAbsM += Math.abs(run.magnetization)/4; }
  assert.ok(Math.abs(sampledE/n - e/z) < 0.025);
  assert.ok(Math.abs(sampledAbsM/n - absM/z) < 0.015);
});
