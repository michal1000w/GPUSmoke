#pragma once

#include <iostream>

#define VOXELIZER_IMPLEMENTATION
#include "Voxelizer.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "third_party/tinyobjloader/tiny_obj_loader.h"




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

        result = vx_voxelize_snap_3dgrid(mesh, x, y, z);

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