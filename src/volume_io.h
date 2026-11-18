#pragma once
#include <vector>

// A NanoVDB density grid in the form the renderer needs it: the flat,
// self-contained buffer that gets copied to the device as it is, plus the few
// numbers that have to be known on the host before anything can be traced.
struct VolumeGrid {
    std::vector<char> gridBytes;      // a whole NanoVDB grid, ready for cudaMemcpy
    float maximumDensity = 0.f;       // the majorant that delta tracking needs
    float worldMinimum[3] = {0.f, 0.f, 0.f};
    float worldMaximum[3] = {0.f, 0.f, 0.f};
};

// Reads the first float grid of an .nvdb file. Compressed files need NanoVDB to
// be built with NANOVDB_USE_ZIP / NANOVDB_USE_BLOSC, which we do not, so write
// them uncompressed (vdb_tool's default).
bool loadNanoVdbGrid(const char* filePath, VolumeGrid& grid);

// Builds a fog sphere in memory. Nothing in the renderer needs this, it exists
// so the volume code can be exercised without a data set on disk.
bool createFogSphereGrid(float radiusInWorldUnits, float voxelSize, VolumeGrid& grid);
