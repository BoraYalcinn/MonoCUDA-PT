#pragma once
#include <vector>

// PNG / JPG / BMP / TGA / PPM through stb_image, 8 bit RGBA, top row first
bool loadImageRgba8(const char* filePath, int& imageWidth, int& imageHeight, std::vector<unsigned char>& rgbaPixels);

// EXR through tinyexr, HDR (and LDR formats, gamma decoded) through stb_image, float RGBA, top row first
bool loadImageRgbaFloat(const char* filePath, int& imageWidth, int& imageHeight, std::vector<float>& rgbaPixels);
