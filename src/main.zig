const std = @import("std");
const hnsw = @import("hnsw.zig");
const index = @import("index.zig");
const distance = @import("distance.zig");
const benchmark = @import("benchmark.zig");

const GLOVE_NUM_WORDS = 1291147;
const GLOVE_DIM = 50;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();
    var timer = try std.time.Timer.start();

    // Read vectors from stdin, text, vecs...
    var stdin_buf: [4096]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buf);
    const stdin: *std.io.Reader = &stdin_reader.interface;

    var store_builder = index.VectorStoreBuilder.initWithOptions(
        allocator,
        GLOVE_DIM,
        .{ .normalize = true },
    );
    try store_builder.ensureCapacity(GLOVE_NUM_WORDS);
    const store = try store_builder.buildFromReader(stdin);

    std.log.info(
        "loaded vector store - words: {d}, took: {d:.2}s",
        .{ store.count, @as(f64, @floatFromInt(timer.read())) / 1e9 },
    );
    timer.reset();

    // const word_index = try index.buildWordIndex(allocator, &store);

    var hnsw_index = try hnsw.HnswIndex.init(allocator, &store, &distance.normCosine, .{
        .max_nodes_per_layer = 16,
        .ef_construction = 50,
        .ef_search = 50,
        .num_words = store.count,
    });

    var insert_count: usize = 0;
    var last_report: u64 = 0;
    for (0..GLOVE_NUM_WORDS) |idx| {
        try hnsw_index.insert(idx);
        insert_count += 1;

        if (insert_count % 10000 == 0) {
            const now = timer.read();
            const delta_ms = @as(f64, @floatFromInt(now - last_report)) / 1e6;
            std.log.info("inserted {d} nodes, last 10k took: {d:.0}ms", .{ insert_count, delta_ms });
            last_report = now;
        }
    }

    const elapsed = timer.read();
    std.log.info(
        "hnsw_index built - size: {d}, took: {d:.2}s",
        .{ hnsw_index.nodes.items.len, @as(f64, @floatFromInt(elapsed)) / 1e9 },
    );

    try benchmark.runBenchmark(allocator, &store, &hnsw_index, &distance.normCosine, 100, 10);
}
