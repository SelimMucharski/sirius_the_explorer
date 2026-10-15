// Convert a mesh file (FBX, ...) to OBJ with node transforms baked; print node tree + bbox.
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cstdio>
#include <string>

static void dump(const aiNode* n, int depth) {
  const aiMatrix4x4& m = n->mTransformation;
  std::printf("%*s%s meshes=%u  T=[%.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f] t=(%.3f %.3f %.3f)\n",
              depth * 2, "", n->mName.C_Str(), n->mNumMeshes, m.a1, m.a2, m.a3, m.b1, m.b2, m.b3,
              m.c1, m.c2, m.c3, m.a4, m.b4, m.c4);
  for (unsigned i = 0; i < n->mNumChildren; ++i) dump(n->mChildren[i], depth + 1);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s in.fbx [out.obj]\n", argv[0]); return 1; }
  Assimp::Importer imp;
  // Keep FBX pivots out of the node graph so baking is clean.
  imp.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
  const aiScene* sc = imp.ReadFile(argv[1], aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
  if (!sc) { std::fprintf(stderr, "load failed: %s\n", imp.GetErrorString()); return 1; }
  std::printf("meshes=%u materials=%u textures=%u\n", sc->mNumMeshes, sc->mNumMaterials, sc->mNumTextures);
  for (unsigned i = 0; i < sc->mNumMaterials; ++i) {
    aiString name, tex; sc->mMaterials[i]->Get(AI_MATKEY_NAME, name);
    sc->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, 0, &tex);
    std::printf("  material %u: %s diffuse=%s\n", i, name.C_Str(), tex.C_Str());
  }
  dump(sc->mRootNode, 0);
  if (argc < 3) return 0;
  // Bake transforms into vertices for export.
  imp.ApplyPostProcessing(aiProcess_PreTransformVertices);
  sc = imp.GetScene();
  aiVector3D lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
  size_t nv = 0, nf = 0;
  for (unsigned i = 0; i < sc->mNumMeshes; ++i) {
    const aiMesh* m = sc->mMeshes[i]; nv += m->mNumVertices; nf += m->mNumFaces;
    for (unsigned j = 0; j < m->mNumVertices; ++j) {
      const aiVector3D& v = m->mVertices[j];
      lo.x = std::min(lo.x, v.x); lo.y = std::min(lo.y, v.y); lo.z = std::min(lo.z, v.z);
      hi.x = std::max(hi.x, v.x); hi.y = std::max(hi.y, v.y); hi.z = std::max(hi.z, v.z);
    }
  }
  std::printf("baked: verts=%zu faces=%zu bbox min(%.3f %.3f %.3f) max(%.3f %.3f %.3f)\n",
              nv, nf, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
  Assimp::Exporter ex;
  std::string fmt = std::string(argv[2]).rfind(".glb") != std::string::npos ? "glb2" : "obj";
  if (ex.Export(sc, fmt, argv[2]) != AI_SUCCESS) {
    std::fprintf(stderr, "export failed: %s\n", ex.GetErrorString()); return 1;
  }
  std::printf("wrote %s\n", argv[2]);
  return 0;
}
