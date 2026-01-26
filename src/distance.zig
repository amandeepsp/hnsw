pub const DistanceFn = *const fn (
    a: []const f32,
    b: []const f32,
) f32;

pub fn dot(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

pub fn l2(a: []const f32, b: []const f32) f32 {
    var dist: f32 = 0.0;
    for (a, b) |x, y| {
        dist += (x - y) * (x - y);
    }

    return @sqrt(dist);
}
