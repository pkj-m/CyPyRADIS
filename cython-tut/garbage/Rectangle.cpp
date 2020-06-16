#include <iostream>
#include "Rectangle.h"

namespace shapes {

    // default constructor
    Rectangle::Rectangle() {}

    // overloaded constructor
    Rectangle::Rectangle (int x0, int y0, int x1, int y1) {
        this->x0 = x0;
        this->y0 = y0;
        this->x1 = x1;
        this->y1 = y1;
    }

    // destructor
    Rectangle::~Rectangle() {}

    // return the area of rectangle
    int Rectangle::getArea() {
        return (this->x1 - this->x0) * (this->y1 - this->y0);
    }

    //get the size of rectangle
    void Rectangle::getSize (int *width, int *height) {
        (*width) = x1 - x0;
        (*height) = y1- y0;
    }

    // move by dx dy
    void Rectangle::move(int dx, int dy) {
        this->x0 += dx;
        this->x1 += dx;
        this->y0 += dy;
        this->y1 += dy;
    }
}