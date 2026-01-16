use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

/// Computes the integer nth-root of `value` using binary search.
/// Returns the largest integer `r` such that `r^n <= value`.
fn integer_root(value: u64, n: u32) -> u64 {
    if value == 0 {
        return 0;
    }
    if n == 0 {
        panic!("0th root is undefined");
    }
    if n == 1 {
        return value;
    }

    // Binary search for the integer nth-root
    let mut low: u64 = 1;
    let mut high: u64 = value;

    // Narrow down the upper bound to avoid overflow
    // The nth root of u64::MAX is at most 2^(64/n)
    let max_root = 1u64 << (64 / n as u64 + 1);
    if high > max_root {
        high = max_root;
    }

    while low < high {
        let mid = low + (high - low + 1) / 2;

        // Check if mid^n <= value, handling potential overflow
        match checked_pow(mid, n) {
            Some(pow) if pow <= value => low = mid,
            _ => high = mid - 1,
        }
    }

    low
}

/// Computes base^exp with overflow checking
fn checked_pow(base: u64, exp: u32) -> Option<u64> {
    let mut result: u64 = 1;
    let mut b = base;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result.checked_mul(b)?;
        }
        e >>= 1;
        if e > 0 {
            b = b.checked_mul(b)?;
        }
    }

    Some(result)
}

/// Benchmark: compute the sum of integer roots for many values
fn sum_integer_roots(values: &[u64], root_degrees: &[u32]) -> u64 {
    values
        .iter()
        .flat_map(|&v| root_degrees.iter().map(move |&n| integer_root(v, n)))
        .sum()
}

/// Parallel version using rayon
fn sum_integer_roots_parallel(values: &[u64], root_degrees: &[u32]) -> u64 {
    values
        .par_iter()
        .flat_map(|&v| root_degrees.par_iter().map(move |&n| integer_root(v, n)))
        .sum()
}

fn benchmark_integer_roots(c: &mut Criterion) {
    // Limit thread count to avoid ThreadPoolBuilder bug
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .ok(); // Ignore error if pool already initialized

    // Generate test data: a range of values to compute roots for
    let values: Vec<u64> = (1..=10000).collect();
    let root_degrees: Vec<u32> = vec![2, 3, 4, 5, 6, 7, 8]; // square, cube, 4th, 5th, 6th, 7th, 8th roots

    c.bench_function("sum_integer_roots_sequential", |b| {
        b.iter(|| sum_integer_roots(black_box(&values), black_box(&root_degrees)))
    });

    c.bench_function("sum_integer_roots_parallel", |b| {
        b.iter(|| sum_integer_roots_parallel(black_box(&values), black_box(&root_degrees)))
    });

    // Benchmark individual root operations
    c.bench_function("integer_sqrt_10000_values", |b| {
        b.iter(|| {
            values
                .iter()
                .map(|&v| integer_root(black_box(v), 2))
                .sum::<u64>()
        })
    });

    c.bench_function("integer_cbrt_10000_values", |b| {
        b.iter(|| {
            values
                .iter()
                .map(|&v| integer_root(black_box(v), 3))
                .sum::<u64>()
        })
    });

    // Benchmark with larger values
    let large_values: Vec<u64> = (1..=1000).map(|x| x * 1_000_000).collect();

    c.bench_function("sum_roots_large_values", |b| {
        b.iter(|| sum_integer_roots(black_box(&large_values), black_box(&root_degrees)))
    });
}

criterion_group!(benches, benchmark_integer_roots);
criterion_main!(benches);
