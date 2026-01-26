const std = @import("std");
const distance = @import("distance.zig");
const index = @import("index.zig");
const VectorStore = index.VectorStore;

const Node = struct {
    neighbors: []std.ArrayListUnmanaged(usize),

    pub fn initEmpty(
        alloc: std.mem.Allocator,
        layer: usize,
        layer_size: usize,
        layer_size0: usize,
    ) !Node {
        const neighbors = try alloc.alloc(std.ArrayListUnmanaged(usize), layer + 1);
        for (neighbors, 0..layer + 1) |*nbrs, l| {
            nbrs.* = .empty;
            const node_cap = if (l == 0) layer_size0 else layer_size;
            try nbrs.ensureTotalCapacityPrecise(alloc, node_cap);
        }

        return .{ .neighbors = neighbors };
    }
};

const SortCtx = struct {
    store: *const VectorStore,
    query_idx: usize,
    dist_fn: distance.DistanceFn,
};

fn compareByDistance(ctx: SortCtx, a: usize, b: usize) bool {
    const dist_a = ctx.dist_fn(ctx.store.vec(ctx.query_idx), ctx.store.vec(a));
    const dist_b = ctx.dist_fn(ctx.store.vec(ctx.query_idx), ctx.store.vec(b));
    return dist_a < dist_b;
}

const SearchEntry = struct {
    idx: usize,
    dist: f32,
};

// Min-heap for candidates (explore closest first)
fn minCompareSearch(_: void, a: SearchEntry, b: SearchEntry) std.math.Order {
    return std.math.order(a.dist, b.dist);
}

// Max-heap for results (track furthest)
fn maxCompareSearch(_: void, a: SearchEntry, b: SearchEntry) std.math.Order {
    return std.math.order(b.dist, a.dist); // reversed
}

pub const HnswIndex = struct {
    pub const Params = struct {
        max_nodes_per_layer: usize,
        ef_construction: usize,
        ef_search: usize,
        num_words: usize,
        seed: u64 = 2026,
    };

    params: Params,
    layer_mult: f64,
    max_nodes_layer0: usize,
    prng: std.Random.DefaultPrng,
    allocator: std.mem.Allocator,
    store: *const VectorStore,
    distance_fn: distance.DistanceFn,

    // TODO: we should use multiple entry points
    entry_point: usize,
    layers: usize = 0,
    nodes: std.ArrayListUnmanaged(Node),

    pub fn init(
        alloc: std.mem.Allocator,
        store: *const VectorStore,
        distance_fn: distance.DistanceFn,
        params: Params,
    ) !HnswIndex {
        return .{
            .allocator = alloc,
            .params = params,
            .layer_mult = 1.0 / std.math.log(
                f64,
                std.math.e,
                @as(f64, @floatFromInt(params.max_nodes_per_layer)),
            ),
            .max_nodes_layer0 = 2 * params.max_nodes_per_layer,
            .entry_point = 0,
            .prng = std.Random.DefaultPrng.init(params.seed),
            .store = store,
            .distance_fn = distance_fn,
            .nodes = try std.ArrayListUnmanaged(Node).initCapacity(alloc, params.num_words),
        };
    }

    fn dist(self: *HnswIndex, a: usize, b: usize) f32 {
        return self.distance_fn(self.store.vec(a), self.store.vec(b));
    }

    fn search_layer(
        self: *HnswIndex,
        layer: usize,
        idx: usize,
        entry_point: usize,
        entry_factor: usize,
    ) ![]usize {
        var visited = try std.DynamicBitSetUnmanaged.initEmpty(self.allocator, self.params.num_words);
        defer visited.deinit(self.allocator);
        var candidates = std.PriorityQueue(SearchEntry, void, minCompareSearch).init(self.allocator, undefined);
        defer candidates.deinit();
        var farthest = std.PriorityQueue(SearchEntry, void, maxCompareSearch).init(self.allocator, undefined);
        defer farthest.deinit();

        visited.set(entry_point);
        try candidates.add(.{ .idx = entry_point, .dist = self.dist(idx, entry_point) });
        try farthest.add(.{ .idx = entry_point, .dist = self.dist(idx, entry_point) });

        while (candidates.count() > 0) {
            const curr_candidate = candidates.remove();
            const curr_idx = curr_candidate.idx;
            const curr_distq = curr_candidate.dist;
            const curr_farthest: ?SearchEntry = farthest.peek();

            if (curr_distq > curr_farthest.?.dist) {
                break;
            }

            for (self.nodes.items[curr_idx].neighbors[layer].items) |nbr| {
                if (!visited.isSet(nbr)) {
                    visited.set(nbr);

                    const curr_farthest_ = farthest.peek();
                    if (self.dist(nbr, idx) < curr_farthest_.?.dist) {
                        try candidates.add(.{ .idx = nbr, .dist = self.dist(nbr, idx) });
                        try farthest.add(.{ .idx = nbr, .dist = self.dist(nbr, idx) });

                        if (farthest.count() > entry_factor) {
                            _ = farthest.remove();
                        }
                    }
                }
            }
        }

        const results = try self.allocator.alloc(usize, farthest.count());
        var i: usize = 0;
        while (farthest.count() > 0) {
            const entry = farthest.remove();
            results[i] = entry.idx;
            i += 1;
        }

        return results;
    }

    fn select_neighbors(
        self: *HnswIndex,
        layer: usize,
        idx: usize,
        candidates: []usize,
    ) ![]usize {
        const max_nodes = if (layer == 0) self.max_nodes_layer0 else self.params.max_nodes_per_layer;

        std.mem.sort(
            usize,
            candidates,
            SortCtx{
                .store = self.store,
                .query_idx = idx,
                .dist_fn = self.distance_fn,
            },
            compareByDistance,
        );
        return candidates[0..@min(candidates.len, max_nodes)];
    }

    pub fn insert(
        self: *HnswIndex,
        idx: usize,
    ) !void {
        const random = self.prng.random();

        const urandom = random.float(f64);
        const assigned_layer: usize = @intFromFloat(std.math.floor(
            -std.math.log(f64, std.math.e, urandom) * self.layer_mult,
        ));

        const node = try Node.initEmpty(
            self.allocator,
            assigned_layer,
            self.params.max_nodes_per_layer,
            self.max_nodes_layer0,
        );

        try self.nodes.append(self.allocator, node);

        // First node
        if (self.nodes.items.len == 1) {
            self.entry_point = idx;
            self.layers = assigned_layer;
            return;
        }

        var curr_entry_point = self.entry_point;
        var current_layer = self.layers;

        var current_nearest: []usize = undefined;

        while (current_layer > assigned_layer) {
            current_nearest = try self.search_layer(current_layer, idx, curr_entry_point, 1);
            curr_entry_point = current_nearest[0];
            current_layer -= 1;
        }

        current_layer = @min(assigned_layer, self.layers) + 1;
        while (current_layer > 0) {
            current_layer -= 1;

            current_nearest = try self.search_layer(
                current_layer,
                idx,
                curr_entry_point,
                self.params.ef_construction,
            );

            const neighbors = try self.select_neighbors(current_layer, idx, current_nearest);

            for (neighbors) |nbr_idx| {
                try self.nodes.items[idx].neighbors[current_layer].append(self.allocator, nbr_idx);

                const nbr_neighbors = &self.nodes.items[nbr_idx].neighbors[current_layer];
                try nbr_neighbors.append(self.allocator, idx);

                const max_neighbors = if (current_layer == 0) self.max_nodes_layer0 else self.params.max_nodes_per_layer;
                if (nbr_neighbors.items.len > max_neighbors) {
                    self.prune_neighbors(nbr_neighbors, idx, max_neighbors);
                }
            }

            curr_entry_point = current_nearest[0];
        }

        if (assigned_layer > self.layers) {
            self.layers = assigned_layer;
            self.entry_point = idx;
        }
    }

    fn prune_neighbors(
        self: *HnswIndex,
        neighbors: *std.ArrayListUnmanaged(usize),
        idx: usize,
        max_size: usize,
    ) void {
        std.mem.sort(
            usize,
            neighbors.items,
            SortCtx{
                .store = self.store,
                .query_idx = idx,
                .dist_fn = self.distance_fn,
            },
            compareByDistance,
        );
        neighbors.shrinkRetainingCapacity(max_size);
    }

    pub fn top_k(self: *HnswIndex, idx: usize, k: usize) ![]usize {
        var candidates: []usize = undefined;
        var curr_entry_point = self.entry_point;
        var curr_layer = self.layers;

        while (curr_layer > 0) {
            candidates = try self.search_layer(curr_layer, idx, curr_entry_point, 1);
            curr_entry_point = candidates[0];
            curr_layer -= 1;
        }

        candidates = try self.search_layer(0, idx, curr_entry_point, self.params.ef_search);

        return candidates[0..@min(k, candidates.len)];
    }
};
