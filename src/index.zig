const std = @import("std");

pub const VectorStore = struct {
    count: usize,
    dim: usize,
    words: [][]const u8,
    vectors: []f32, // flattened vectors

    pub fn vec(self: *const VectorStore, idx: usize) []const f32 {
        const start = idx * self.dim;
        return self.vectors[start .. start + self.dim];
    }
};

pub const WordIndex = std.StringHashMapUnmanaged(usize);

pub fn buildWordIndex(alloc: std.mem.Allocator, store: *const VectorStore) !WordIndex {
    var map: WordIndex = .empty;
    const num_words: u32 = @intCast(store.count);
    try map.ensureTotalCapacity(alloc, num_words);

    for (store.words, 0..) |word, i| {
        try map.put(alloc, word, i);
    }

    return map;
}

pub const VectorStoreBuilder = struct {
    pub const Options = struct {
        normalize: bool,
    };
    allocator: std.mem.Allocator,
    dim: usize,
    options: Options,

    vectors: std.ArrayListUnmanaged(f32),
    words: std.ArrayListUnmanaged([]const u8),

    pub fn init(alloc: std.mem.Allocator, dim: usize) VectorStoreBuilder {
        return initWithOptions(alloc, dim, .{ .normalize = false });
    }

    pub fn initWithOptions(alloc: std.mem.Allocator, dim: usize, options: Options) VectorStoreBuilder {
        return .{
            .allocator = alloc,
            .dim = dim,
            .options = options,
            .vectors = .empty,
            .words = .empty,
        };
    }

    pub fn ensureCapacity(self: *VectorStoreBuilder, num_words: usize) !void {
        try self.vectors.ensureTotalCapacityPrecise(self.allocator, num_words * self.dim);
        try self.words.ensureTotalCapacityPrecise(self.allocator, num_words);
    }

    pub fn buildFromReader(self: *VectorStoreBuilder, reader: *std.io.Reader) !VectorStore {
        while (try reader.takeDelimiter('\n')) |line| {
            var it = std.mem.tokenizeScalar(u8, line, ' ');

            if (it.next()) |word| {
                try self.words.append(self.allocator, try self.allocator.dupe(u8, word));
            }

            const vec_start = self.vectors.items.len;
            while (it.next()) |vec_str| {
                const vec_val = try std.fmt.parseFloat(f32, vec_str);
                try self.vectors.append(self.allocator, vec_val);
            }

            if (self.vectors.items.len % self.dim != 0) return error.WrongVectorLength;

            if (self.options.normalize) {
                const vec = self.vectors.items[vec_start..];
                var norm: f32 = 0.0;
                for (vec) |v| norm += v * v;
                norm = @sqrt(norm);
                if (norm > 0.0) {
                    for (vec) |*v| v.* /= norm;
                }
            }
        }

        return .{
            .dim = self.dim,
            .count = self.words.items.len,
            .vectors = self.vectors.items,
            .words = self.words.items,
        };
    }
};
