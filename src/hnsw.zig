const std = @import("std");
const distance = @import("distance.zig");
const index = @import("index.zig");
const VectorStore = index.VectorStore;

pub const NodeIdx = u32;

const Node = struct {
    neighbors: []std.ArrayListUnmanaged(NodeIdx),

    pub fn initEmpty(
        alloc: std.mem.Allocator,
        layer: usize,
        layer_size: usize,
        layer_size0: usize,
    ) !Node {
        const neighbors = try alloc.alloc(std.ArrayListUnmanaged(NodeIdx), layer + 1);
        for (neighbors, 0..layer + 1) |*nbrs, l| {
            nbrs.* = .empty;
            const node_cap = if (l == 0) layer_size0 else layer_size;
            try nbrs.ensureTotalCapacityPrecise(alloc, node_cap);
        }

        return .{ .neighbors = neighbors };
    }
};

const SearchEntry = struct {
    idx: NodeIdx,
    dist: f32,
};

fn minCompare(_: void, a: SearchEntry, b: SearchEntry) bool {
    return a.dist < b.dist;
}

fn minCompareSearch(_: void, a: SearchEntry, b: SearchEntry) std.math.Order {
    return std.math.order(a.dist, b.dist);
}

fn maxCompareSearch(_: void, a: SearchEntry, b: SearchEntry) std.math.Order {
    return std.math.order(b.dist, a.dist);
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

    entry_points: std.ArrayListUnmanaged(NodeIdx),
    layers: usize = 0,
    nodes: std.ArrayListUnmanaged(Node),
    visited: std.DynamicBitSetUnmanaged,

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
            .entry_points = .empty,
            .prng = std.Random.DefaultPrng.init(params.seed),
            .store = store,
            .distance_fn = distance_fn,
            .nodes = try std.ArrayListUnmanaged(Node).initCapacity(alloc, params.num_words),
            .visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, params.num_words),
        };
    }

    fn dist(self: *HnswIndex, a: NodeIdx, b: NodeIdx) f32 {
        return self.distance_fn(self.store.vec(@intCast(a)), self.store.vec(@intCast(b)));
    }

    fn searchLayer(
        self: *HnswIndex,
        layer: usize,
        idx: NodeIdx,
        entry_points: std.ArrayListUnmanaged(NodeIdx),
        entry_factor: usize,
    ) !std.ArrayListUnmanaged(NodeIdx) {
        self.visited.unsetAll();

        var candidates = std.PriorityQueue(SearchEntry, void, minCompareSearch).init(self.allocator, {});
        defer candidates.deinit();
        var farthest = std.PriorityQueue(SearchEntry, void, maxCompareSearch).init(self.allocator, {});
        defer farthest.deinit();

        for (entry_points.items) |entry_point| {
            self.visited.set(entry_point);
            try candidates.add(.{ .idx = entry_point, .dist = self.dist(idx, entry_point) });
            try farthest.add(.{ .idx = entry_point, .dist = self.dist(idx, entry_point) });
        }

        while (candidates.count() > 0) {
            const curr_candidate = candidates.remove();
            const curr_idx = curr_candidate.idx;
            const curr_distq = curr_candidate.dist;
            const curr_farthest = farthest.peek();

            if (curr_distq > curr_farthest.?.dist) {
                break;
            }

            for (self.nodes.items[curr_idx].neighbors[layer].items) |nbr| {
                if (!self.visited.isSet(nbr)) {
                    self.visited.set(nbr);

                    const nbr_dist = self.dist(nbr, idx);
                    const curr_farthest_ = farthest.peek();
                    if (farthest.count() < entry_factor or nbr_dist < curr_farthest_.?.dist) {
                        try candidates.add(.{ .idx = nbr, .dist = nbr_dist });
                        try farthest.add(.{ .idx = nbr, .dist = nbr_dist });

                        if (farthest.count() > entry_factor) {
                            _ = farthest.remove();
                        }
                    }
                }
            }
        }

        const count = farthest.count();
        var results: std.ArrayListUnmanaged(NodeIdx) = .empty;
        try results.resize(self.allocator, count);
        var i: usize = count;
        while (farthest.count() > 0) {
            i -= 1;
            const entry = farthest.remove();
            results.items[i] = entry.idx;
        }

        return results;
    }

    fn selectNeighbors(
        self: *HnswIndex,
        layer: usize,
        idx: NodeIdx,
        candidates: *std.ArrayListUnmanaged(NodeIdx),
    ) !void {
        const max_nodes = if (layer == 0) self.max_nodes_layer0 else self.params.max_nodes_per_layer;

        var visited: std.AutoHashMapUnmanaged(NodeIdx, void) = .empty;
        defer visited.deinit(self.allocator);
        var candidate_entries: std.ArrayListUnmanaged(SearchEntry) = .empty;
        defer candidate_entries.deinit(self.allocator);

        for (candidates.items) |c| {
            try candidate_entries.append(self.allocator, .{ .idx = c, .dist = self.dist(c, idx) });
            try visited.put(self.allocator, c, {});
        }

        candidates.clearRetainingCapacity();

        std.mem.sort(SearchEntry, candidate_entries.items, {}, minCompare);

        for (candidate_entries.items) |c| {
            var good = true;
            for (candidates.items) |s| {
                if (self.dist(c.idx, s) < c.dist) {
                    good = false;
                    break;
                }
            }
            if (good) {
                try candidates.append(self.allocator, c.idx);
            }
            if (candidates.items.len >= max_nodes) {
                break;
            }
        }
    }

    pub fn insert(self: *HnswIndex, idx: NodeIdx) !void {
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

        if (self.nodes.items.len == 1) {
            try self.entry_points.append(self.allocator, idx);
            self.layers = assigned_layer;
            return;
        }

        if (assigned_layer == self.layers) {
            try self.entry_points.append(self.allocator, idx);
        }

        var curr_entry_points: std.ArrayListUnmanaged(NodeIdx) = self.entry_points;
        var current_layer = self.layers;

        var current_nearest: std.ArrayListUnmanaged(NodeIdx) = undefined;

        while (current_layer > assigned_layer) {
            const new_entry_points = try self.searchLayer(current_layer, idx, curr_entry_points, 1);
            curr_entry_points = new_entry_points;
            current_layer -= 1;
        }

        current_layer = @min(assigned_layer, self.layers) + 1;
        while (current_layer > 0) {
            current_layer -= 1;

            current_nearest = try self.searchLayer(
                current_layer,
                idx,
                curr_entry_points,
                self.params.ef_construction,
            );

            try self.selectNeighbors(current_layer, idx, &current_nearest);

            for (current_nearest.items) |nbr_idx| {
                try self.nodes.items[idx].neighbors[current_layer].append(self.allocator, nbr_idx);

                const nbr_neighbors = &self.nodes.items[nbr_idx].neighbors[current_layer];
                try nbr_neighbors.append(self.allocator, idx);

                const max_neighbors = if (current_layer == 0) self.max_nodes_layer0 else self.params.max_nodes_per_layer;
                if (nbr_neighbors.items.len > max_neighbors) {
                    try self.selectNeighbors(current_layer, nbr_idx, nbr_neighbors);
                }
            }

            curr_entry_points = current_nearest;
        }

        if (assigned_layer > self.layers) {
            self.layers = assigned_layer;
            self.entry_points.clearRetainingCapacity();
            try self.entry_points.append(self.allocator, idx);
        }
    }

    pub fn topK(self: *HnswIndex, idx: NodeIdx, k: usize) !std.ArrayListUnmanaged(NodeIdx) {
        var candidates: std.ArrayListUnmanaged(NodeIdx) = undefined;
        var curr_entry_points = self.entry_points;
        var curr_layer = self.layers;

        while (curr_layer > 0) {
            const new_entry_points = try self.searchLayer(curr_layer, idx, curr_entry_points, 1);
            curr_entry_points = new_entry_points;
            curr_layer -= 1;
        }

        candidates = try self.searchLayer(0, idx, curr_entry_points, self.params.ef_search);
        candidates.shrinkRetainingCapacity(k);
        return candidates;
    }
};
