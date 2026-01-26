pub const DistanceFn = *const fn (
    a: []const f32,
    b: []const f32,
) f32;

pub fn dot(a: []const f32, b: []const f32) f32 {
    return dotSimd(a, b);
}

fn dotScalar(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn dotSimd(a: []const f32, b: []const f32) f32 {
    const Vec8 = @Vector(8, f32);
    var sum: Vec8 = @splat(0.0);

    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const va: Vec8 = a[i..][0..8].*;
        const vb: Vec8 = b[i..][0..8].*;
        sum += va * vb;
    }

    var result = @reduce(.Add, sum);
    while (i < a.len) : (i += 1) {
        result += a[i] * b[i];
    }
    return result;
}

pub fn l2(a: []const f32, b: []const f32) f32 {
    var dist: f32 = 0.0;
    for (a, b) |x, y| {
        dist += (x - y) * (x - y);
    }

    return @sqrt(dist);
}

// Assumes that the input vectors are already normalized.
pub fn normCosine(a: []const f32, b: []const f32) f32 {
    return 1.0 - dot(a, b);
}
