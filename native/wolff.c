/* Single-cluster Wolff updates on the periodic square lattice (J = k_B = 1, h = 0). */
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef struct {
    uint64_t word[4];
} Random;

static uint64_t splitmix64(uint64_t *seed) {
    uint64_t z = (*seed += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static inline uint64_t rotl(uint64_t x, int shift) {
    return (x << shift) | (x >> (64 - shift));
}

/* xoshiro256++ 1.0, Blackman & Vigna. Each replica owns its generator. */
static inline uint64_t next_u64(Random *random) {
    uint64_t *s = random->word;
    uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

static int random_site(Random *random, uint32_t n) {
    /* Multiply-high sampling, with rejection of the short interval at the origin. */
    uint64_t product = (next_u64(random) >> 32) * n;
    if ((uint32_t)product < n) {
        uint32_t threshold = (uint32_t)(0u - n) % n;
        while ((uint32_t)product < threshold) {
            product = (next_u64(random) >> 32) * n;
        }
    }
    return (int)(product >> 32);
}

typedef struct {
    int32_t n;
    int8_t *spin;
    int32_t (*neighbor)[4];
    int32_t *queue;
    int32_t energy, magnetization;
    uint64_t cutoff;
    Random random;
} Chain;

static inline void flip(Chain *chain, int site, int old) {
    const int32_t *nn = chain->neighbor[site];
    const int8_t *s = chain->spin;
    chain->energy += 2 * old * (s[nn[0]] + s[nn[1]] + s[nn[2]] + s[nn[3]]);
    chain->spin[site] = (int8_t)-old;
}

static int cluster_update(Chain *chain) {
    int first = random_site(&chain->random, (uint32_t)chain->n);
    int old = chain->spin[first];
    int size = 1;
    chain->queue[0] = first;
    flip(chain, first, old);
    for (int head = 0; head < size; ++head) {
        const int32_t *nn = chain->neighbor[chain->queue[head]];
        for (int edge = 0; edge < 4; ++edge) {
            int site = nn[edge];
            if (chain->spin[site] == old && (next_u64(&chain->random) >> 11) < chain->cutoff) {
                /* Flipping on admission marks visited sites. For L=2 a rejected
                   parallel bond still gives the second bond an independent trial. */
                flip(chain, site, old);
                chain->queue[size++] = site;
            }
        }
    }
    chain->magnetization -= 2 * old * size;
    return size;
}

/* initial = 0 (random), +1 (up) or -1 (down). One update is one cluster. */
EXPORT int wolff_sample(int32_t length, double temperature, uint64_t seed, int32_t initial,
                        int32_t warmup, int32_t samples, int32_t stride,
                        int32_t *energy, int32_t *magnetization, double *mean_size) {
    if (length < 2 || (int64_t)length * length > INT32_MAX / 4 ||
        !isfinite(temperature) || temperature <= 0 || initial < -1 || initial > 1 ||
        warmup < 0 || samples < 1 || stride < 1 || !energy || !magnetization || !mean_size) {
        return 1;
    }
    Chain chain = {0};
    chain.n = length * length;
    size_t n = (size_t)chain.n;
    if (n > SIZE_MAX / sizeof *chain.neighbor) return 2;
    chain.spin = malloc(n * sizeof *chain.spin);
    chain.neighbor = malloc(n * sizeof *chain.neighbor);
    chain.queue = malloc(n * sizeof *chain.queue);
    if (!chain.spin || !chain.neighbor || !chain.queue) {
        free(chain.spin);
        free(chain.neighbor);
        free(chain.queue);
        return 2;
    }
    for (int j = 0; j < 4; ++j) chain.random.word[j] = splitmix64(&seed);
    chain.cutoff = (uint64_t)ldexp(-expm1(-2.0 / temperature), 53);
    for (int y = 0; y < length; ++y) {
        for (int x = 0; x < length; ++x) {
            int i = y * length + x;
            chain.neighbor[i][0] = y * length + (x + 1) % length;
            chain.neighbor[i][1] = ((y + 1) % length) * length + x;
            chain.neighbor[i][2] = y * length + (x + length - 1) % length;
            chain.neighbor[i][3] = ((y + length - 1) % length) * length + x;
            chain.spin[i] = initial ? (int8_t)initial : (next_u64(&chain.random) >> 63 ? 1 : -1);
            chain.magnetization += chain.spin[i];
        }
    }
    for (int i = 0; i < chain.n; ++i) {
        chain.energy -= chain.spin[i] * (chain.spin[chain.neighbor[i][0]] +
                                         chain.spin[chain.neighbor[i][1]]);
    }
    for (int step = 0; step < warmup; ++step) cluster_update(&chain);
    double total_size = 0;
    for (int i = 0; i < samples; ++i) {
        for (int step = 0; step < stride; ++step) total_size += cluster_update(&chain);
        energy[i] = chain.energy;
        magnetization[i] = chain.magnetization;
    }
    *mean_size = total_size / ((double)samples * stride);
    free(chain.spin);
    free(chain.neighbor);
    free(chain.queue);
    return 0;
}
