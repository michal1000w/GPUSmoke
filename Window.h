#include "Common.h"
#include "IndexBuffer.h"
#include "VertexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Renderer.h"
#include "Texture.h"

#include "Solver.cuh"
Solver solver;


static void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *window, int entered);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);



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

	//////////KURSOR
	glfwSetCursorPosCallback(window, cursorPositionCallback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); //GLFW_CURSOR_HIDDEN
	glfwSetCursorEnterCallback(window, cursorEnterCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);
	glfwSetScrollCallback(window, scrollCallback);
	/////////////KLAWIATURA
	glfwSetKeyCallback(window, keyCallback);
	////////////

	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK)
		std::cout << "Error: glewInit\n";

	std::cout << glGetString(GL_VERSION) << std::endl;


	///////////////////////////////////////////////
	{
		float positions[] = {
			-1.0f, -1.0f, 0.0f, 0.0f,
			 1.0f, -1.0f, 1.0f, 0.0f,
			 1.0f,  1.0f, 1.0f, 1.0f,
			-1.0f,  1.0f, 0.0f, 1.0f
		};
		unsigned int indices[] = {
			0,1,2,
			2,3,0
		};

		GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
		GLCall(glEnable(GL_BLEND));
		///////////////////////////////////////////////
		unsigned int vao;
		GLCall(glGenVertexArrays(1, &vao));//1
		GLCall(glBindVertexArray(vao));

		///////////////////////////////////////////////
		VertexArray va;
		VertexBuffer vb(positions, 4 * 4 * sizeof(float));
		VertexBufferLayout layout;
		layout.Push<float>(2);
		layout.Push<float>(2);
		va.AddBuffer(vb, layout);

		/////////////////////////////////////////////
		IndexBuffer ib(indices, 6);

		/////////////////SHADERS/////////////////////
		Shader shader("Shaders.shader");
		shader.Bind();

		//////////////////UNIFORM SHADER////////////////////
		//shader.SetUniform4f("u_Color", 0.2f, 0.3f, 0.9f, 1.0f);

		float r = 0.0f;
		float increment = 0.01f;
		/////////////////////////////////////////////////

		//Texture texture("file.bmp");
		//texture.Bind(/*slot*/0);
		//shader.SetUniform1i("u_Texture", /*slot*/0);
		Texture texture("mf.bmp");
		///////////////////////////////////////////
		va.UnBind();
		shader.UnBind();
		vb.Unbind();
		ib.Unbind();

		Renderer renderer;
		unsigned int frame = 0;
		/////////////////////////////////////////////////
		while (!glfwWindowShouldClose(window)) {
			//////////////////
			solver.Simulation_Frame(frame);
			frame++;
			//////////////////
			//Texture texture("output/R" + pad_number(frame) + ".bmp");
			texture.UpdateTexture("output/R" + pad_number(frame) + ".bmp");
			texture.Bind(/*slot*/0);
			//shader.SetUniform1i("u_Texture", /*slot*/0);
			//////////////////
			renderer.Clear();
			shader.Bind();

			renderer.Draw(va, ib, shader);


			GLCall(glfwSwapBuffers(window));
			GLCall(glfwPollEvents());

		}
	}
	glfwTerminate();
	return 0;
}

static void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos) {
	
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		/*Rotate*/
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			solver.setRotation(solver.getRotation() - 0.1f);
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			solver.setRotation(solver.getRotation() + 0.1f);
		}
	}
	else{ /*Left-Right*/
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			solver.setCamera(solver.getCamera().x - 2.5f, solver.getCamera().y, solver.getCamera().z);
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			solver.setCamera(solver.getCamera().x + 2.5f, solver.getCamera().y, solver.getCamera().z);
		}
	} /*Forward - Backward*/
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		solver.setCamera(solver.getCamera().x, solver.getCamera().y, solver.getCamera().z + 2.5f);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		solver.setCamera(solver.getCamera().x, solver.getCamera().y, solver.getCamera().z - 2.5f);
	}
		
}

void cursorEnterCallback(GLFWwindow* window, int entered) {
	if (entered) {

	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	double xPos, yPos;
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		//left button released
	}
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
	solver.setCamera(solver.getCamera().x, solver.getCamera().y,
		solver.getCamera().z + 2.5f * yOffset);
	//std::cout <<"   "<< yOffset;
}