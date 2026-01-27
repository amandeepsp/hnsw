const std = @import("std");
const hnsw = @import("hnsw.zig");
const index = @import("index.zig");
const distance = @import("distance.zig");

const VectorStore = index.VectorStore;

pub fn bruteForceTopK(
    store: *const VectorStore,
    distance_fn: distance.DistanceFn,
    allocator: std.mem.Allocator,
    query_idx: usize,
    k: usize,
) ![]usize {
    const Entry = struct {
        idx: usize,
        dist: f32,
    };

    var results = try std.ArrayList(Entry).initCapacity(allocator, store.count);
    defer results.deinit(allocator);

    const query_vec = store.vec(query_idx);
    for (0..store.count) |i| {
        if (i == query_idx) continue;
        const d = distance_fn(query_vec, store.vec(i));
        try results.append(allocator, .{ .idx = i, .dist = d });
    }

    std.mem.sort(Entry, results.items, {}, struct {
        fn lessThan(_: void, a: Entry, b: Entry) bool {
            return a.dist < b.dist;
        }
    }.lessThan);

    const count = @min(k, results.items.len);
    const top_k = try allocator.alloc(usize, count);
    for (0..count) |i| {
        top_k[i] = results.items[i].idx;
    }

    return top_k;
}

pub fn computeRecall(hnsw_results: std.ArrayListUnmanaged(usize), bf_results: []const usize) f32 {
    var hits: usize = 0;
    for (hnsw_results.items) |h| {
        for (bf_results) |b| {
            if (h == b) {
                hits += 1;
                break;
            }
        }
    }
    return @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(bf_results.len));
}

pub fn runBenchmark(
    allocator: std.mem.Allocator,
    store: *const VectorStore,
    hnsw_index: *hnsw.HnswIndex,
    distance_fn: distance.DistanceFn,
    num_queries: usize,
    k: usize,
) !void {
    var total_recall: f32 = 0.0;
    var hnsw_time: u64 = 0;
    var bf_time: u64 = 0;

    const step = store.count / num_queries;

    for (0..num_queries) |q| {
        const query_idx = q * step;

        var timer = try std.time.Timer.start();
        const hnsw_results = try hnsw_index.topK(query_idx, k);
        hnsw_time += timer.read();

        timer.reset();
        const bf_results = try bruteForceTopK(store, distance_fn, allocator, query_idx, k);
        bf_time += timer.read();

        const recall = computeRecall(hnsw_results, bf_results);
        total_recall += recall;
    }

    const avg_recall = total_recall / @as(f32, @floatFromInt(num_queries));
    const avg_hnsw_us = @as(f64, @floatFromInt(hnsw_time)) / @as(f64, @floatFromInt(num_queries)) / 1000.0;
    const avg_bf_us = @as(f64, @floatFromInt(bf_time)) / @as(f64, @floatFromInt(num_queries)) / 1000.0;

    std.log.info("Benchmark results ({d} queries, k={d}):", .{ num_queries, k });
    std.log.info("  Average recall: {d:.2}%", .{avg_recall * 100});
    std.log.info("  HNSW avg time: {d:.1}µs", .{avg_hnsw_us});
    std.log.info("  Brute-force avg time: {d:.1}µs", .{avg_bf_us});
    std.log.info("  Speedup: {d:.1}x", .{avg_bf_us / avg_hnsw_us});
}
