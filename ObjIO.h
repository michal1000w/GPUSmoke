#pragma once

#include <iostream>

#define VOXELIZER_IMPLEMENTATION
#include "Voxelizer.h"
#include "third_party/cuda_voxelizer/src/voxelize.cuh"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "third_party/tinyobjloader/tiny_obj_loader.h"



#include <cuda_runtime.h>
// Standard libs
#include <string>
#include <cstdio>
// GLM for maths
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "third_party/cuda_voxelizer/src/util.h"
#include "third_party/cuda_voxelizer/src/util_io.h"
#include "third_party/cuda_voxelizer/src/util_cuda.h"
#include "third_party/cuda_voxelizer/src/timer.h"
// CPU voxelizer fallback
#include "third_party/cuda_voxelizer/src/cpu_voxelizer.h"



unsigned int* vx_voxelize_snap_3dgrid_correct_ratio(vx_mesh_t const* m,
    unsigned int width,
    unsigned int height,
    unsigned int depth)
{
    vx_aabb_t* aabb = NULL;
    vx_aabb_t* meshaabb = NULL;
    float ax, ay, az;

    VX_ASSERT(m->colors);

    for (size_t i = 0; i < m->nindices; i += 3) {
        vx_triangle_t triangle;
        unsigned int i1, i2, i3;

        VX_ASSERT(m->indices[i + 0] < m->nvertices);
        VX_ASSERT(m->indices[i + 1] < m->nvertices);
        VX_ASSERT(m->indices[i + 2] < m->nvertices);

        i1 = m->indices[i + 0];
        i2 = m->indices[i + 1];
        i3 = m->indices[i + 2];

        triangle.p1 = m->vertices[i1];
        triangle.p2 = m->vertices[i2];
        triangle.p3 = m->vertices[i3];

        if (!meshaabb) {
            meshaabb = VX_MALLOC(vx_aabb_t, 1);
            *meshaabb = vx__triangle_aabb(&triangle);
        }
        else {
            vx_aabb_t naabb = vx__triangle_aabb(&triangle);
            *meshaabb = vx__aabb_merge(meshaabb, &naabb);
        }
    }

    float resx = (meshaabb->max.x - meshaabb->min.x) / min(min(width,height),depth);
    float resy = (meshaabb->max.y - meshaabb->min.y) / min(min(width, height), depth);
    float resz = (meshaabb->max.z - meshaabb->min.z) / min(min(width, height), depth);

    vx_point_cloud_t* pc = vx_voxelize_pc(m, resx, resy, resz, 0.0);

    aabb = VX_MALLOC(vx_aabb_t, 1);

    vx__aabb_init(aabb);

    for (size_t i = 0; i < pc->nvertices; i++) {
        for (size_t j = 0; j < 3; j++) {
            aabb->max.v[j] = VX_MAX(aabb->max.v[j], pc->vertices[i].v[j]);
            aabb->min.v[j] = VX_MIN(aabb->min.v[j], pc->vertices[i].v[j]);
        }
    }

    ax = aabb->max.x - aabb->min.x;
    ay = aabb->max.y - aabb->min.y;
    az = aabb->max.z - aabb->min.z;

    unsigned int* data = VX_CALLOC(unsigned int, width * height * depth);

    for (size_t i = 0; i < pc->nvertices; ++i) {
        float rgba[4] = { pc->colors[i].r, pc->colors[i].g, pc->colors[i].b, 1.0 };
        unsigned int color;
        float ox, oy, oz;
        int ix, iy, iz;
        unsigned int index;

        ox = pc->vertices[i].x + fabs(aabb->min.x);
        oy = pc->vertices[i].y + fabs(aabb->min.y);
        oz = pc->vertices[i].z + fabs(aabb->min.z);

        VX_ASSERT(ox >= 0.f);
        VX_ASSERT(oy >= 0.f);
        VX_ASSERT(oz >= 0.f);

        ix = (ax == 0.0) ? 0 : (ox / ax) * (min(min(width, height), depth) - 1);
        iy = (ay == 0.0) ? 0 : (oy / ay) * (min(min(width, height), depth) - 1);
        iz = (az == 0.0) ? 0 : (oz / az) * (min(min(width, height), depth) - 1);


        VX_ASSERT(ix >= 0);
        VX_ASSERT(iy >= 0);
        VX_ASSERT(iz >= 0);

        VX_ASSERT(ix + iy * width + iz * (width * height) < width * height * depth);

        color = vx__rgbaf32_to_abgr8888(rgba);
        index = ix + iy * width + iz * (width * height);

        if (data[index] != 0) {
            data[index] = vx__mix(color, data[index]);
        }
        else {
            data[index] = color;
        }
    }

    VX_FREE(aabb);
    VX_FREE(meshaabb);
    vx_point_cloud_free(pc);

    return data;
}


void LoadTest() {
    std::string inputfile = "input/obj/suzanne.obj";
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            index_offset += fv;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
}

vx_mesh_t* Voxelize(const char* filename, float voxelsizex, float voxelsizey, float voxelsizez, float precision)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

    if (!err.empty()) {
        printf("err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("failed to load : %s\n", filename);
        return nullptr;
    }

    if (shapes.size() == 0) {
        printf("err: # of shapes are zero.\n");
        return nullptr;
    }

    // Only use first shape.
    {
        vx_mesh_t* mesh;
        vx_mesh_t* result;

        mesh = vx_mesh_alloc(attrib.vertices.size(), shapes[0].mesh.indices.size());

        for (size_t f = 0; f < shapes[0].mesh.indices.size(); f++) {
            mesh->indices[f] = shapes[0].mesh.indices[f].vertex_index;
        }

        for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
            mesh->vertices[v].x = attrib.vertices[3 * v + 0];
            mesh->vertices[v].y = attrib.vertices[3 * v + 1];
            mesh->vertices[v].z = attrib.vertices[3 * v + 2];
        }

        result = vx_voxelize(mesh, voxelsizex, voxelsizey, voxelsizez, precision);

        printf("Number of vertices: %ld\n", result->nvertices);
        printf("Number of indices: %ld\n", result->nindices);

        ///////////////////////////////odtad moje
        return result;
        ///////////////
    }
    return nullptr;
}

float* Voxelize2(const char* filename, int x, int y, int z, float density = 0.7f)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

    if (!err.empty()) {
        printf("err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("failed to load : %s\n", filename);
        return nullptr;
    }

    if (shapes.size() == 0) {
        printf("err: # of shapes are zero.\n");
        return nullptr;
    }

    // Only use first shape.
    {
        vx_mesh_t* mesh;
        unsigned int* result;

        mesh = vx_mesh_alloc(attrib.vertices.size(), shapes[0].mesh.indices.size());

        for (size_t f = 0; f < shapes[0].mesh.indices.size(); f++) {
            mesh->indices[f] = shapes[0].mesh.indices[f].vertex_index;
        }

        for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
            mesh->vertices[v].x = attrib.vertices[3 * v + 0];
            mesh->vertices[v].y = attrib.vertices[3 * v + 1];
            mesh->vertices[v].z = attrib.vertices[3 * v + 2];
        }

        //result = vx_voxelize_snap_3dgrid(mesh, x, y, z);
        result = vx_voxelize_snap_3dgrid_correct_ratio(mesh, x, y, z);

        float* resultf = new float[x * y * z];

        unsigned int sum = 0;
        for (int i = 0; i < x * y * z; i++) {
            resultf[i] = result[i];
            if (resultf[i] != 0) {
                resultf[i] = density;
                sum++;
            }
        }
        std::cout << "All count: " << x * y * z << std::endl;
        std::cout << "Sum: " << sum << std::endl;

        ///////////////////////////////odtad moje
        return resultf;
        ///////////////
    }
    return nullptr;
}




















float* meshToGPU_managed(const trimesh::TriMesh* mesh) {
    Timer t; t.start();
    size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
    float* device_triangles;
    fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
    checkCudaErrors(cudaMallocManaged((void**)&device_triangles, n_floats)); // managed memory
    fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
    for (size_t i = 0; i < mesh->faces.size(); i++) {
        glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
        glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
        glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
        size_t j = i * 9;
        memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
        memcpy((device_triangles)+j + 3, glm::value_ptr(v1), sizeof(glm::vec3));
        memcpy((device_triangles)+j + 6, glm::value_ptr(v2), sizeof(glm::vec3));
    }
    t.stop(); fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

    //for (size_t i = 0; i < mesh->faces.size(); i++) {
    //	size_t t = i * 9;
    //	std::cout << "Tri: " << device_triangles[t] << " " << device_triangles[t + 1] << " " << device_triangles[t + 2] << std::endl;
    //	std::cout << "Tri: " << device_triangles[t + 3] << " " << device_triangles[t + 4] << " " << device_triangles[t + 5] << std::endl;
    //	std::cout << "Tri: " << device_triangles[t + 6] << " " << device_triangles[t + 7] << " " << device_triangles[t + 8] << std::endl;
    //}

    return device_triangles;
}

enum class OutputFormat { output_binvox = 0, output_morton = 1, output_obj_points = 2, output_obj_cubes = 3 };
const char* OutputFormats[] = { "binvox file", "morton encoded blob", "obj file (pointcloud)", "obj file (cubes)" };
OutputFormat outputformat = OutputFormat::output_binvox;
void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);
void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code, size_t sizee);
void voxelize_solid(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);






bool forceCPU = false;
bool useThrustPath = false;



float* LoadAndVoxelize(int3 grid_size, std::string filename = "",float density = 0.7 , int deviceIndex = 0, 
                    bool solidVoxelization = true, int objresolution = NULL) {
    fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
    trimesh::TriMesh* themesh = trimesh::TriMesh::read(filename.c_str());
    themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
    fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
    fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
    fprintf(stdout, "[Mesh] Computing bbox \n");
    themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

    int gsize = min(min(grid_size.x, grid_size.y), grid_size.z);
    if (objresolution != NULL)
        gsize = objresolution;

    // SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
    fprintf(stdout, "\n## VOXELISATION SETUP \n");
    // Initialize our own AABox
    AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));
    // Transform that AABox to a cubical box (by padding directions if needed)
    // Create voxinfo struct, which handles all the rest
    voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gsize,gsize,gsize), themesh->faces.size());
    voxelization_info.print();
    // Compute space needed to hold voxel table (1 voxel / bit)
    size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
    //size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)));
    unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this

    bool cuda_ok = false;
    if (!forceCPU)
    {
        // SECTION: Try to figure out if we have a CUDA-enabled GPU
        fprintf(stdout, "\n## CUDA INIT \n");
        cuda_ok = true;// initCuda();
        cuda_ok ? fprintf(stdout, "[Info] CUDA GPU found\n") : fprintf(stdout, "[Info] CUDA GPU not found\n");
    }

    // SECTION: The actual voxelization
    if (cuda_ok && !forceCPU) {
        // GPU voxelization
        fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

        float* device_triangles;
        // Transfer triangles to GPU using either thrust or managed cuda memory
        if (useThrustPath) { 
            //device_triangles = meshToGPU_thrust(themesh);
        }
        else { device_triangles = meshToGPU_managed(themesh); }

        if (!useThrustPath) {
            fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
            checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
        }
        else {
            // ALLOCATE MEMORY ON HOST
            fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
            checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
        }
        fprintf(stdout, "\n## GPU VOXELISATION \n");
        if (solidVoxelization) {
            voxelize_solid(voxelization_info, device_triangles, vtable, useThrustPath, false);// (outputformat == OutputFormat::output_obj_cubes));
        }
        else {
            voxelize(voxelization_info, device_triangles, vtable, useThrustPath, false);// (outputformat == OutputFormat::output_obj_cubes));
        }
    }
    else {
        // CPU VOXELIZATION FALLBACK
        fprintf(stdout, "\n## CPU VOXELISATION \n");
        if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
        else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
        // allocate zero-filled array
        vtable = (unsigned int*)calloc(1, vtable_size);
        /*
        if (!solidVoxelization) {
            cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
        }
        else {
            cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
        }
        */
    }

    //// DEBUG: print vtable
    //for (int i = 0; i < vtable_size; i++) {
    //	char* vtable_p = (char*)vtable;
    //	cout << (int) vtable_p[i] << endl;
    //}
    std::cout << "Allocating memory"<< grid_size.x * grid_size.y * grid_size.z << std::endl;
    float* resultf = new float[grid_size.x * grid_size.y * grid_size.z];

    unsigned int sum = 0;
    unsigned int licznik = 0;
    for (size_t x = 0; x < grid_size.x; x++) { //na tej osi b³¹d
        for (size_t y = 0; y < grid_size.y; y++) {
            for (size_t z = 0; z < grid_size.z; z++) {
                if ((x < gsize && y < gsize && z < gsize)
                    && checkVoxel(x, y, z, voxelization_info.gridsize, vtable)) {
                    sum++;
                    resultf[z * grid_size.y * grid_size.x + y * grid_size.x + x] = density;
                }
                else {
                    resultf[z * grid_size.y * grid_size.x + y * grid_size.x + x] = 0.0f;
                }
            }
        }
    }
    /*
    for (size_t x = 0; x < voxelization_info.gridsize.x; x++) {
        for (size_t y = 0; y < voxelization_info.gridsize.y; y++) {
            for (size_t z = 0; z < voxelization_info.gridsize.z; z++) {
                if (checkVoxel(x, y, z, voxelization_info.gridsize, vtable)) {
                    sum++;
                    resultf[z * grid_size.y * grid_size.x + y * grid_size.x + x] = density;
                }
            }
        }
    }
    */
    std::cout << std::endl << "All count: " << grid_size.x * grid_size.y * grid_size.z << std::endl;
    std::cout << "Sum: " << sum << std::endl;

    return resultf;
}


unsigned int* LoadAndVoxelizeCompressed(int3 grid_size, std::string filename, float density, int deviceIndex, 
    bool solidVoxelization, int &table_size, int objresolution = NULL) {
    fprintf(stdout, "[I/O] Reading mesh from %s \n", filename.c_str());
    trimesh::TriMesh* themesh = trimesh::TriMesh::read(filename.c_str());
    themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
    fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
    fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
    fprintf(stdout, "[Mesh] Computing bbox \n");
    themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

    int gsize = min(min(grid_size.x, grid_size.y), grid_size.z);
    if (objresolution != NULL)
        gsize = objresolution;

    // SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
    fprintf(stdout, "\n## VOXELISATION SETUP \n");
    // Initialize our own AABox
    AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));
    // Transform that AABox to a cubical box (by padding directions if needed)
    // Create voxinfo struct, which handles all the rest
    voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gsize, gsize, gsize), themesh->faces.size());
    voxelization_info.print();
    // Compute space needed to hold voxel table (1 voxel / bit)
    size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
    //size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x) * static_cast<size_t>(voxelization_info.gridsize.y) * static_cast<size_t>(voxelization_info.gridsize.z)));
    unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this

    bool cuda_ok = false;
    if (!forceCPU)
    {
        // SECTION: Try to figure out if we have a CUDA-enabled GPU
        fprintf(stdout, "\n## CUDA INIT \n");
        cuda_ok = true;// initCuda();
        cuda_ok ? fprintf(stdout, "[Info] CUDA GPU found\n") : fprintf(stdout, "[Info] CUDA GPU not found\n");
    }

    // SECTION: The actual voxelization
    if (cuda_ok && !forceCPU) {
        // GPU voxelization
        fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

        float* device_triangles;
        // Transfer triangles to GPU using either thrust or managed cuda memory
        if (useThrustPath) {
            //device_triangles = meshToGPU_thrust(themesh);
        }
        else { device_triangles = meshToGPU_managed(themesh); }

        if (!useThrustPath) {
            fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
            checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
        }
        else {
            // ALLOCATE MEMORY ON HOST
            fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
            checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
        }
        fprintf(stdout, "\n## GPU VOXELISATION \n");
        if (solidVoxelization) {
            voxelize_solid(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_obj_cubes));
        }
        else {
            voxelize(voxelization_info, device_triangles, vtable, useThrustPath, false);// (outputformat == OutputFormat::output_obj_cubes));
        }
    }
    else {
        // CPU VOXELIZATION FALLBACK
        fprintf(stdout, "\n## CPU VOXELISATION \n");
        if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
        else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
        // allocate zero-filled array
        vtable = (unsigned int*)calloc(1, vtable_size);
        /*
        if (!solidVoxelization) {
            cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
        }
        else {
            cpu_voxelizer::cpu_voxelize_mesh_solid(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
        }
        */
    }
    
    table_size = vtable_size;


    return vtable;
}