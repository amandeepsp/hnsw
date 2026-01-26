const std = @import("std");
const hnsw = @import("hnsw.zig");
const index = @import("index.zig");
const distance = @import("distance.zig");

const GLOVE_NUM_WORDS = 100;
const GLOVE_DIM = 50;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();
    var timer = try std.time.Timer.start();

    // Read vectors from stdin, text, vecs...
    var stdin_buf: [2048]u8 = undefined;
    var stdin_reader = std.fs.File.stdin().reader(&stdin_buf);
    const stdin: *std.io.Reader = &stdin_reader.interface;

    var store_builder = index.VectorStoreBuilder.init(allocator, GLOVE_DIM);
    try store_builder.ensureCapacity(GLOVE_NUM_WORDS);
    const store = try store_builder.buildFromReader(stdin);

    std.log.info("loaded vector store - words: {d}, took: {d}", .{ store.count, timer.read() });
    timer.reset();

    // const word_index = try index.buildWordIndex(allocator, &store);

    var hnsw_index = try hnsw.HnswIndex.init(allocator, &store, &distance.l2, .{
        .max_nodes_per_layer = 10,
        .ef_construction = 5,
        .ef_search = 5,
        .num_words = store.count,
    });

    for (0..GLOVE_NUM_WORDS) |idx| {
        try hnsw_index.insert(idx);
    }

    const elapsed = timer.read();
    std.log.info("hnsw_index built - size: {any}, time to fill: {any}\n", .{ hnsw_index.nodes.items.len, elapsed });

    const query_idx: usize = 89;
    const top3 = try hnsw_index.top_k(query_idx, 3);

    std.log.info("top 3 NNs for '{s}':", .{store.words[query_idx]});
    for (top3) |neighbor_idx| {
        std.log.info("  - {s}", .{store.words[neighbor_idx]});
    }
}
