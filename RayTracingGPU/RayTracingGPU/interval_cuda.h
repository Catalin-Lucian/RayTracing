#ifndef INTERVAL_H
#define INTERVAL_H

const float infinity = std::numeric_limits<float>::infinity();
class Interval {
public:
    float min, max;

    Interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    Interval(float _min, float _max) : min(_min), max(_max) {}

    bool contains(float x) const {
        return min <= x && x <= max;
    }

    bool surrounds(float x) const {
        return min < x && x < max;
    }

    float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const Interval empty, universe;
};

const static Interval empty(+infinity, -infinity);
const static Interval universe(-infinity, +infinity);

inline float random_float() {
    // Returns a random real in [0,1).
    return  (float) (rand() / (RAND_MAX + 1.0));
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return (float)(min + (max - min) * random_float());
}

#endif