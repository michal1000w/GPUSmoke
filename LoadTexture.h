#pragma once
#include "Common.h"
#include <iostream>
GLuint loadBMP(const char* imagepath) {
	unsigned char header[54];
	unsigned int dataPos;
	unsigned int width, height;
	unsigned int imageSize;
	unsigned char* data;

	FILE* file = fopen(imagepath, "rb");
	if (!file) {
		printf("Error loading texture\n");
		return 0;
	}

	if (fread(header, 1, 54, file) != 54) {
		printf("Error: BMP file not correct!!\n");
		return 0;
	}

	if (header[0] != 'B' || header[1] != 'M') {
		printf("Error: BMP file not correct!!\n");
		return 0;
	}



	// Read ints from the byte array
	dataPos = *(int*)&(header[0x0A]);
	imageSize = *(int*)&(header[0x22]);
	width = *(int*)&(header[0x12]);
	height = *(int*)&(header[0x16]);

	if (imageSize == 0)    imageSize = width * height * 3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)      dataPos = 54; // The BMP header is done that way

	// Create a buffer
	data = new unsigned char[imageSize];

	// Read the actual data from the file into the buffer
	fread(data, 1, imageSize, file);

	//Everything is in memory now, the file can be closed
	fclose(file);

	// Create one OpenGL texture
	GLuint textureID;
	GLCall(glGenTextures(1, &textureID));

	// "Bind" the newly created texture : all future texture functions will modify this texture
	GLCall(glBindTexture(GL_TEXTURE_2D, textureID));

	// Give the image to OpenGL
	GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data));

	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

	std::cout << "Texture loaded successfully\n";

	return textureID;
}

unsigned char* loadTexture(const char* imagepath, unsigned int& Width,
	unsigned int& Height, unsigned int& BPP, unsigned int CHANNELS) {

	unsigned char header[54];
	unsigned int dataPos;
	unsigned int width, height;
	unsigned int imageSize;
	unsigned char* data;

	FILE* file = fopen(imagepath, "rb");
	if (!file) {
		printf("Error loading texture\n");
		return 0;
	}

	if (fread(header, 1, 54, file) != 54) {
		printf("Error: BMP file not correct!!\n");
		return 0;
	}

	if (header[0] != 'B' || header[1] != 'M') {
		printf("Error: BMP file not correct!!\n");
		return 0;
	}



	// Read ints from the byte array
	dataPos = *(int*)&(header[0x0A]);
	imageSize = *(int*)&(header[0x22]);
	width = *(int*)&(header[0x12]);
	height = *(int*)&(header[0x16]);

	if (imageSize == 0)    imageSize = width * height * CHANNELS; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)      dataPos = 54; // The BMP header is done that way

	// Create a buffer
	data = new unsigned char[imageSize];

	// Read the actual data from the file into the buffer
	fread(data, 1, imageSize, file);

	//Everything is in memory now, the file can be closed
	fclose(file);

	std::cout << "Texture loaded successfully\n";
	Width = width;
	Height = height;
	BPP = imageSize;
	return data;

}