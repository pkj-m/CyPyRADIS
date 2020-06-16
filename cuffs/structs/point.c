typedef struct Point {
    int x;
    int y;
} Point;

struct Point make_and_send_point(int x, int y);

struct Point make_and_send_point(int x, int y) {
    struct Point p = {x, y};
    return p;
}