#include "Common.h"
#include "IndexBuffer.h"
#include "VertexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Renderer.h"
#include "Texture.h"

#include "Solver.cuh"
extern Solver solver;


#include "third_party/imgui/imgui.h"
#include "third_party/imgui/imgui_impl_opengl3.h"
#include "third_party/imgui/imgui_impl_glfw.h"

#include <thread>


static void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *window, int entered);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

void UpdateSolver() {
	std::cout << "\nRestarting\n";
	//clearing
	//solver.ClearCache();
	solver.frame = 0;
	solver.Clear_Simulation_Data();
	//preparation
	//solver.Initialize();
	solver.UpdateDomainResolution();
	//solver.UpdateTimeStep();
	if (solver.SAMPLE_SCENE == 0)
		solver.ExampleScene();
	else if (solver.SAMPLE_SCENE == 1 || solver.SAMPLE_SCENE == 2)
		solver.ExportVDBScene();
	solver.Initialize_Simulation();
}









void RenderGUI(bool& SAVE_FILE_TAB, bool& OPEN_FILE_TAB, float& fps,
	float& progress, bool& save_panel) {

	/////////////////////////////
			/////    CREATE WINDOW    ///
	if (SAVE_FILE_TAB) {
		ImGui::Begin("Save Panel");
		{
			ImGui::Text("Enter filename");
			ImGui::InputText("Filename", solver.SAVE_FOLDER, IM_ARRAYSIZE(solver.SAVE_FOLDER));
			if (ImGui::Button("Save")) {
				std::string filename = solver.SAVE_FOLDER;
				filename = trim(filename);
				solver.SaveSceneToFile(filename);
				SAVE_FILE_TAB = false;
			}
			if (ImGui::Button("Close")) {
				SAVE_FILE_TAB = false;
			}
			ImGui::Text("Saved projects:");
			ImGui::BeginChild("Scrolling");
			std::string directory = "scenes\\";
			std::vector <std::string> list = solver.getFilesList(directory);
			for (int object = 0; object < list.size(); object++) {
				std::string name = (" -> " + list[object]);
				ImGui::Text(name.c_str());
			}
			ImGui::EndChild();
		}
		ImGui::End();
	}
	if (OPEN_FILE_TAB) {
		ImGui::Begin("Open Panel");
		{
			ImGui::Text("Enter filename");
			ImGui::InputText("Filename", solver.OPEN_FOLDER, IM_ARRAYSIZE(solver.OPEN_FOLDER));
			if (ImGui::Button("Open")) {
				std::string filename = solver.OPEN_FOLDER;
				filename = trim(filename);
				solver.LoadSceneFromFile(filename);
				OPEN_FILE_TAB = false;
			}
			if (ImGui::Button("Close")) {
				OPEN_FILE_TAB = false;
			}
			///////////////////////////////////////
			ImGui::Text("Saved projects:");
			ImGui::BeginChild("Scrolling");
			std::string directory = "scenes\\";
			std::vector <std::string> list = solver.getFilesList(directory);
			for (int object = 0; object < list.size(); object++) {
				std::string name = (" -> " + list[object]);
				ImGui::Text(name.c_str());
			}
			ImGui::EndChild();
			///////////////////////////////////////
		}
		ImGui::End();
	}

	ImGui::Begin("IO Panel", &save_panel, ImGuiWindowFlags_MenuBar);
	{
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open..", "Ctrl+O")) {
					SAVE_FILE_TAB = false;
					OPEN_FILE_TAB = true;
					//solver.LoadSceneFromFile("scene2");
				}
				if (ImGui::MenuItem("Save", "Ctrl+S")) {
					OPEN_FILE_TAB = false;
					SAVE_FILE_TAB = true;
				}
				if (ImGui::MenuItem("Close", "Ctrl+W")) {
					save_panel = false;
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}




		ImGui::Text("Example scenes");
		const char* items[] = { "VDB","VDBFire", "Objects" };// , "vdb", "vdbs" };
		static const char* current_item = "Objects";

		if (ImGui::BeginCombo("##combo", current_item)) // The second parameter is the label previewed before opening the combo.
		{
			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
			{
				bool is_selected = (current_item == items[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(items[n], is_selected)) {
					current_item = items[n];
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
			ImGui::EndCombo();
		}
		if (ImGui::Button("Load Scene")) {
			if (current_item == "VDB")
				solver.SAMPLE_SCENE = 1;
			else if (current_item == "Objects")
				solver.SAMPLE_SCENE = 0;
			else if (current_item == "VDBFire")
				solver.SAMPLE_SCENE = 2;
			solver.preserve_object_list = false;
			UpdateSolver();
			solver.preserve_object_list = true;
		}
		ImGui::Text("Exporting settings");
		ImGui::InputInt("Start frame", &solver.EXPORT_START_FRAME);
		ImGui::InputInt("End frame", &solver.EXPORT_END_FRAME);
		ImGui::InputText("Cache Folder", solver.EXPORT_FOLDER, IM_ARRAYSIZE(solver.EXPORT_FOLDER));
		if (ImGui::Button("Export VDB")) {
			solver.ClearCache();
			solver.EXPORT_VDB = true;
			UpdateSolver();
			progress = 0.0;
		}
		if (solver.EXPORT_VDB) {
			ImGui::ProgressBar(progress, ImVec2(-1, 0));
			progress += 1.0 / ((float)solver.EXPORT_END_FRAME);

		}
		if (solver.EXPORT_VDB)
			if (ImGui::Button("Stop")) {
				progress = 0.0;
				solver.EXPORT_VDB = false;
			}
	}
	ImGui::End();
	ImGui::Begin("Properties Panel");
	{
		ImGui::Text("Domain Resolution");
		ImGui::SliderInt("x", &solver.New_DOMAIN_RESOLUTION.x, 2, 490);
		ImGui::SliderInt("y", &solver.New_DOMAIN_RESOLUTION.y, 2, 490);
		ImGui::SliderInt("z", &solver.New_DOMAIN_RESOLUTION.z, 2, 490);



		ImGui::Text("Simulation Settings");
		ImGui::SliderFloat("Ambient Temp", &solver.Ambient_Temperature, -10.0f, 100.0f);
		ImGui::SliderFloat("Smoke Dissolve", &solver.Smoke_Dissolve, 0.93f, 1.0f);
		ImGui::SliderInt("Simulation accuracy", &solver.ACCURACY_STEPS, 1, 64);
		if (ImGui::Button("Simulate")) {
			if (solver.SIMULATE == false)
				solver.SIMULATE = true;
			else
				solver.SIMULATE = false;
		}
		//ImGui::SliderFloat("Simulation speed", &solver.speed, 0.0f, 3.0f);//bugs
		//ImGui::ColorEdit3("clear color", (float*)&clear_color);


		ImGui::Text("Render Settings:");
		ImGui::Checkbox("Fire&Smoke render", &solver.Smoke_And_Fire);
		ImGui::SliderFloat("Fire Emission Rate", &solver.Fire_Max_Temperature, 1, 100);
		ImGui::SliderInt("Render samples", &solver.STEPS, 1, 512);

		if (ImGui::Button("Reset")) {
			UpdateSolver();
		}
		ImGui::SameLine();
		ImGui::Text(("FPS: " + std::to_string(fps)).c_str());
		ImGui::Checkbox("Preserve object list", &solver.preserve_object_list);
	}
	ImGui::End();
	/////////////////////////////

	ImGui::Begin("Objects Panel");
	{
		ImGui::Text("Emitter type");
		const char* items[] = { "emitter", "force" };// , "vdb", "vdbs" };
		static const char* current_item = "emitter";

		if (ImGui::BeginCombo("##combo", current_item)) // The second parameter is the label previewed before opening the combo.
		{
			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
			{
				bool is_selected = (current_item == items[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(items[n], is_selected)) {
					current_item = items[n];
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
			ImGui::EndCombo();
		}
		if (ImGui::Button("Delete selected")) {
		REPEAT:
			for (int object = 0; object < solver.object_list.size(); object++) {
				if (solver.object_list[object].selected) {
					if (solver.object_list[object].get_type() == "vdb" ||
						solver.object_list[object].get_type() == "vdbs")
						solver.object_list[object].cudaFree();
					solver.object_list[object].free();
					solver.object_list.erase(solver.object_list.begin() + object);
					goto REPEAT;
				}
			}
		}
		if (ImGui::Button("Add Emitter")) {
			solver.object_list.push_back(OBJECT(current_item, 18.0f, 50, 0.9, 5, 0.9, make_float3(solver.getDomainResolution().x * 0.25, 0.0, 0.0), solver.object_list.size()));
		}

		ImGui::Text("Object list:");
		ImGui::BeginChild("Scrolling");

		for (int object = 0; object < solver.object_list.size(); object++) {
			std::string name = ("  -> " + solver.object_list[object].get_name());
			ImGui::Text(name.c_str());
			ImGui::SameLine();
			ImGui::Checkbox(std::to_string(object).c_str(), &solver.object_list[object].selected);
			ImGui::SliderFloat3(("position-" + std::to_string(object)).c_str(), solver.object_list[object].Location, 0, 600);
			ImGui::SliderFloat(("size-" + std::to_string(object)).c_str(), &solver.object_list[object].size, 0.0, 100.0);
			if (solver.object_list[object].type >= 5)
				ImGui::SliderFloat(("force strength-" + std::to_string(object)).c_str(), &solver.object_list[object].force_strength, -100.0, 100.0);
			solver.object_list[object].UpdateLocation();
		}
		ImGui::EndChild();

	}
	ImGui::End();
	/////////////////////////////
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
	window = glfwCreateWindow(Img_res[0], Img_res[1], FULL_NAME.c_str(), NULL, NULL);
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
	//calculate sizes
	int2 Image = make_int2(solver.getImageResoltion().x, solver.getImageResoltion().y);
	int2 Window = make_int2(Img_res[0], Img_res[1]);

	
	float range_x = ((float)Image.x / (float)Window.x);
	range_x = 2.0 * range_x - 1.0;

	float range_y = ((float)Image.y / (float)Window.y);
	range_y = 2.0 * range_y - 1.0;
	if (Image.x == Window.x) range_x = 1.0f;
	if (Image.y == Window.y) range_y = 1.0f;
	{
		float positions[] = {
			-1.0f,	    -1.0f,		0.0f, 0.0f, //lewy dó³
			 range_x,   -1.0f,		1.0f, 0.0f,
			 range_x,	 range_y,	1.0f, 1.0f,
			-1.0f,		 range_y,	0.0f, 1.0f
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
		solver.frame = 0;
		/////////////////////////////////////////////////
		//////////////IMGUI/////////////////////////////
		//Setup IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui::StyleColorsDark();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init((char*)glGetString(GL_NUM_SHADING_LANGUAGE_VERSIONS));


		//ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

		float fps = 0;
		bool save_panel = true;
		bool SAVE_FILE_TAB = false;
		bool OPEN_FILE_TAB = false;
		float progress = 0.0f;
		//std::thread* sim;
		solver.DONE_FRAME = true;
		/////////////////////////////////////////////////
		while (!glfwWindowShouldClose(window)) {
			clock_t startTime = clock();


			//////////////////
			solver.Simulation_Frame();
			//////////////////
			//Texture texture("output/R" + pad_number(frame) + ".bmp");
			texture.UpdateTexture("output/R" + pad_number(solver.frame) + ".bmp");
			texture.Bind(/*slot*/0);
			//shader.SetUniform1i("u_Texture", /*slot*/0);
			//////////////////
			renderer.Clear();
			shader.Bind();
			/////////////////////////////
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			//std::thread GUI_THR( RenderGUI ,std::ref(SAVE_FILE_TAB), std::ref(OPEN_FILE_TAB), std::ref(fps), std::ref(progress), std::ref(save_panel));
			RenderGUI(SAVE_FILE_TAB, OPEN_FILE_TAB, fps, progress, save_panel);
			//New Frame//////////////////
			

			renderer.Draw(va, ib, shader);

			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			

			GLCall(glfwSwapBuffers(window));
			GLCall(glfwPollEvents());

			fps = 1.0 / ((double(clock() - startTime) / (double)CLOCKS_PER_SEC));
		}
	}
	//Shutdown
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
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
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
		UpdateSolver();
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		solver.setCamera(solver.getCamera().x, solver.getCamera().y + 2.5f, solver.getCamera().z);
	}
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
		solver.setCamera(solver.getCamera().x, solver.getCamera().y - 2.5f, solver.getCamera().z);
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