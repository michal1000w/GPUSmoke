#pragma once
#include "Common.h"
#include <iostream>



class Texture {
private:
	unsigned int m_RendererID;
	std::string m_FilePath;
	unsigned char* m_LocalBuffer;
	unsigned int m_Width, m_Height, m_BPP;

public:
	Texture(const std::string& path);
	~Texture();

	void UpdateTexture(const std::string& path);
	void UpdateTexture(unsigned char* image, unsigned int width, unsigned int height);


	void Bind(unsigned int slot = 0) const;
	void Unbind() const;

	inline int GetWidth() const { return m_Width; }
	inline int GetHeight() const { return m_Height; }
};