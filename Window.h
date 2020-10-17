#include "Common.h"
#include "IndexBuffer.h"
#include "VertexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"






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
	glGenTextures(1, &textureID);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, textureID);

	// Give the image to OpenGL
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return textureID;
}




int Window(float* Img_res) {
	GLFWwindow* window;

	//initialize
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	//create a windowed mode window
	window = glfwCreateWindow(Img_res[0], Img_res[1], "JFlow Alpha 0.0.1", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK)
		std::cout << "Error: glewInit\n";

	std::cout << glGetString(GL_VERSION) << std::endl;


	///////////////////////////////////////////////
	{
		float positions[12] = {
			-0.5f, -0.5f,
			 0.5f, -0.5f,
			 0.5f,  0.5f,
			-0.5f,  0.5f,
		};
		unsigned int indices[] = {
			0,1,2,
			2,3,0
		};
		///////////////////////////////////////////////
		unsigned int vao;
		GLCall(glGenVertexArrays(1, &vao));//1
		GLCall(glBindVertexArray(vao));

		///////////////////////////////////////////////
		VertexArray va;
		VertexBuffer vb(positions, 4 * 2 * sizeof(float));
		VertexBufferLayout layout;
		layout.Push<float>(2);
		va.AddBuffer(vb, layout);

		/////////////////////////////////////////////
		IndexBuffer ib(indices, 6);

		/////////////////SHADERS/////////////////////
		Shader shader("Shaders.shader");
		shader.Bind();

		//////////////////UNIFORM SHADER////////////////////
		shader.SetUniform4f("u_Color", 0.2f, 0.3f, 0.9f, 1.0f);

		float r = 0.0f;
		float increment = 0.01f;
		/////////////////////////////////////////////////
		va.UnBind();
		shader.UnBind();
		vb.Unbind();
		ib.Unbind();


		/////////////////////////////////////////////////
		while (!glfwWindowShouldClose(window)) {
			//////////////////
			GLCall(glClear(GL_COLOR_BUFFER_BIT));
			//loadBMP("file.bmp");
			shader.Bind();
			shader.SetUniform4f("u_Color", r, 0.3f, 0.9f, 1.0f);

			GLCall(glBindVertexArray(vao));
			va.Bind();
			ib.Bind();

			GLCall(glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));

			if (r > 1.0f)
				increment = -0.01f;
			else if (r < 0.0f)
				increment = 0.01f;
			r += increment;

			glfwSwapBuffers(window);
			glfwPollEvents();

		}
	}
	glfwTerminate();
	return 0;
}