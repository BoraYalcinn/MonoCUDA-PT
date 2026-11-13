#include "image_io.h"
#include <cstring>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_GIF
#include "stb_image.h"

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_USE_STB_ZLIB 0
#include "tinyexr.h"

static bool hasExtension(const char* filePath, const char* extension){
    size_t pathLength = std::strlen(filePath);
    size_t extensionLength = std::strlen(extension);
    if (pathLength < extensionLength) return false;
    for (size_t i = 0; i < extensionLength; i++){
        char a = filePath[pathLength - extensionLength + i];
        char b = extension[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (a != b) return false;
    }
    return true;
}

bool loadImageRgba8(const char* filePath, int& imageWidth, int& imageHeight, std::vector<unsigned char>& rgbaPixels){
    int channelsInFile = 0;
    unsigned char* pixels = stbi_load(filePath, &imageWidth, &imageHeight, &channelsInFile, 4);
    if (!pixels) return false;
    rgbaPixels.assign(pixels, pixels + (size_t)imageWidth * imageHeight * 4);
    stbi_image_free(pixels);
    return true;
}

bool loadImageRgbaFloat(const char* filePath, int& imageWidth, int& imageHeight, std::vector<float>& rgbaPixels){
    if (hasExtension(filePath, ".exr")){
        float* pixels = nullptr;
        const char* errorMessage = nullptr;
        int result = LoadEXR(&pixels, &imageWidth, &imageHeight, filePath, &errorMessage);
        if (result != TINYEXR_SUCCESS){
            if (errorMessage) FreeEXRErrorMessage(errorMessage);
            return false;
        }
        rgbaPixels.assign(pixels, pixels + (size_t)imageWidth * imageHeight * 4);
        std::free(pixels);
        return true;
    }
    int channelsInFile = 0;
    float* pixels = stbi_loadf(filePath, &imageWidth, &imageHeight, &channelsInFile, 4);
    if (!pixels) return false;
    rgbaPixels.assign(pixels, pixels + (size_t)imageWidth * imageHeight * 4);
    stbi_image_free(pixels);
    return true;
}
