#include "volume_io.h"

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include <cstdio>

// Pulls the grid bytes and the metadata out of a handle, whether the grid came
// from a file or was built here.
static bool takeFloatGrid(const nanovdb::GridHandle<nanovdb::HostBuffer>& handle, const char* label, VolumeGrid& grid){
    const nanovdb::FloatGrid* floatGrid = handle.grid<float>();
    if (!floatGrid){
        std::fprintf(stderr, "volume '%s' holds no float grid (only float density grids are supported)\n", label);
        return false;
    }

    const char* bytes = static_cast<const char*>(handle.data());
    grid.gridBytes.assign(bytes, bytes + handle.bufferSize());

    grid.maximumDensity = floatGrid->tree().root().maximum();
    if (!(grid.maximumDensity > 0.f)){
        std::fprintf(stderr, "volume '%s' reports a maximum density of %g, so it carries no statistics; "
                             "delta tracking needs them as its majorant\n", label, (double)grid.maximumDensity);
        return false;
    }

    const nanovdb::Vec3dBBox& worldBounds = floatGrid->worldBBox();
    for (int axis = 0; axis < 3; axis++){
        grid.worldMinimum[axis] = (float)worldBounds.min()[axis];
        grid.worldMaximum[axis] = (float)worldBounds.max()[axis];
    }

    std::printf("volume '%s': %llu voxels, maximum density %.3f, bounds (%.2f %.2f %.2f) to (%.2f %.2f %.2f)\n",
                label,
                (unsigned long long)floatGrid->activeVoxelCount(),
                (double)grid.maximumDensity,
                (double)grid.worldMinimum[0], (double)grid.worldMinimum[1], (double)grid.worldMinimum[2],
                (double)grid.worldMaximum[0], (double)grid.worldMaximum[1], (double)grid.worldMaximum[2]);
    return true;
}

bool loadNanoVdbGrid(const char* filePath, VolumeGrid& grid){
    try {
        nanovdb::GridHandle<nanovdb::HostBuffer> handle = nanovdb::io::readGrid(filePath);
        return takeFloatGrid(handle, filePath, grid);
    } catch (const std::exception& error){
        std::fprintf(stderr, "could not read '%s': %s\n", filePath, error.what());
        return false;
    }
}

bool createFogSphereGrid(float radiusInWorldUnits, float voxelSize, VolumeGrid& grid){
    try {
        nanovdb::GridHandle<nanovdb::HostBuffer> handle =
            nanovdb::tools::createFogVolumeSphere<float>((double)radiusInWorldUnits,
                                                         nanovdb::Vec3d(0.0),
                                                         (double)voxelSize);
        return takeFloatGrid(handle, "fog sphere", grid);
    } catch (const std::exception& error){
        std::fprintf(stderr, "could not build the test fog sphere: %s\n", error.what());
        return false;
    }
}
