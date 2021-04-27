#pragma once
#include <GL/glew.h>
#include "Common.h"
#include "VertexArray.h"
#include "IndexBuffer.h"
#include "Shader.h"
#include <iostream>


class Renderer {
public:
	void Clear() const;
	void SetColor(GLclampf R, GLclampf G, GLclampf B, GLclampf A) const;
	void Draw(const VertexArray& va, const IndexBuffer& ib,
		const Shader& shader) const;
};