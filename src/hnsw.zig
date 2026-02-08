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
    ) !Node {
        const neighbors = try alloc.alloc(std.ArrayListUnmanaged(NodeIdx), layer);
        for (neighbors) |*nbrs| {
            nbrs.* = .empty;
            try nbrs.ensureTotalCapacityPrecise(alloc, layer_size);
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
        max_neighbors_per_layer: usize,
        ef_construction: usize,
        ef_search: usize,
        num_words: usize,
        seed: u64 = 2026,
    };

    params: Params,
    layer_mult: f64,
    max_neighbors_layer0: usize,
    layer0_stride: usize,
    prng: std.Random.DefaultPrng,
    allocator: std.mem.Allocator,
    store: *const VectorStore,
    distance_fn: distance.DistanceFn,

    entry_points: std.ArrayListUnmanaged(NodeIdx),
    layers: usize = 0,
    nodes: std.ArrayListUnmanaged(Node),
    visited: std.DynamicBitSetUnmanaged,

    // Layer 0 has all the nodes, so we can optimize searches by keeping layer0 flat
    // to improve cache locality.
    // length = max_neighbors_layer0 * num_words, this will be allocated once and reused.
    layer0_neighbors: []NodeIdx,
    // Number of neighbors for each node in layer 0, keeps track of the end offset in layer0_neighbors.
    // length = num_words, preallocated once and reused.
    layer0_neighbor_counts: []usize,

    pub fn init(
        alloc: std.mem.Allocator,
        store: *const VectorStore,
        distance_fn: distance.DistanceFn,
        params: Params,
    ) !HnswIndex {
        const neighbor_counts = try alloc.alloc(usize, params.num_words);
        @memset(neighbor_counts, 0);
        const max_neighbors_layer0 = 2 * params.max_neighbors_per_layer;
        const layer0_stride = max_neighbors_layer0 + 1;

        return .{
            .allocator = alloc,
            .params = params,
            .layer_mult = 1.0 / std.math.log(
                f64,
                std.math.e,
                @as(f64, @floatFromInt(params.max_neighbors_per_layer)),
            ),
            .max_neighbors_layer0 = max_neighbors_layer0,
            .layer0_stride = layer0_stride,
            .entry_points = .empty,
            .prng = std.Random.DefaultPrng.init(params.seed),
            .store = store,
            .distance_fn = distance_fn,
            .nodes = try std.ArrayListUnmanaged(Node).initCapacity(alloc, params.num_words),
            .visited = try std.DynamicBitSetUnmanaged.initEmpty(alloc, params.num_words),
            .layer0_neighbors = try alloc.alloc(NodeIdx, params.num_words * layer0_stride),
            .layer0_neighbor_counts = neighbor_counts,
        };
    }

    fn neighborsCapAtLayer(self: *HnswIndex, layer: usize) usize {
        if (layer == 0) {
            return self.max_neighbors_layer0;
        }

        return self.params.max_neighbors_per_layer;
    }

    fn neighborsAtLayer(self: *HnswIndex, idx: NodeIdx, layer: usize) []NodeIdx {
        if (layer == 0) {
            const neighbor_count = self.layer0_neighbor_counts[idx];
            const neighbors_start = idx * self.layer0_stride;
            return self.layer0_neighbors[neighbors_start .. neighbors_start + neighbor_count];
        }

        return self.nodes.items[idx].neighbors[layer - 1].items;
    }

    fn appendNeighbor(self: *HnswIndex, idx: NodeIdx, nbr_idx: NodeIdx, layer: usize) !void {
        if (layer == 0) {
            const neighbor_count_idx = self.layer0_neighbor_counts[idx];
            const neighbor_count_nbr = self.layer0_neighbor_counts[nbr_idx];

            self.layer0_neighbors[idx * self.layer0_stride + neighbor_count_idx] = nbr_idx;
            self.layer0_neighbors[nbr_idx * self.layer0_stride + neighbor_count_nbr] = idx;

            self.layer0_neighbor_counts[idx] += 1;
            self.layer0_neighbor_counts[nbr_idx] += 1;
        } else {
            try self.nodes.items[idx].neighbors[layer - 1].append(self.allocator, nbr_idx);

            const nbr_neighbors = &self.nodes.items[nbr_idx].neighbors[layer - 1];
            try nbr_neighbors.append(self.allocator, idx);
        }
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

            for (self.neighborsAtLayer(curr_idx, layer)) |nbr| {
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
        const max_nodes = self.neighborsCapAtLayer(layer);

        var candidate_entries: std.ArrayListUnmanaged(SearchEntry) = .empty;
        defer candidate_entries.deinit(self.allocator);

        for (candidates.items) |c| {
            try candidate_entries.append(self.allocator, .{ .idx = c, .dist = self.dist(c, idx) });
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

    fn pruneNeighbors0(self: *HnswIndex, idx: NodeIdx) !void {
        const start = idx * self.layer0_stride;
        const count = self.layer0_neighbor_counts[idx];
        var nbr_list = std.ArrayListUnmanaged(NodeIdx){
            .items = self.layer0_neighbors[start .. start + count],
            .capacity = self.layer0_stride,
        };
        try self.selectNeighbors(0, idx, &nbr_list);
        self.layer0_neighbor_counts[idx] = nbr_list.items.len;
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
            self.params.max_neighbors_per_layer,
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
                const max_neighbors = self.neighborsCapAtLayer(current_layer);
                try self.appendNeighbor(idx, nbr_idx, current_layer);

                if (current_layer == 0) {
                    const count = self.layer0_neighbor_counts[nbr_idx];
                    if (count > max_neighbors) {
                        try self.pruneNeighbors0(nbr_idx);
                    }
                } else {
                    const nbr_neighbors = &self.nodes.items[nbr_idx].neighbors[current_layer - 1];
                    if (nbr_neighbors.items.len > max_neighbors) {
                        try self.selectNeighbors(current_layer, nbr_idx, nbr_neighbors);
                    }
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
