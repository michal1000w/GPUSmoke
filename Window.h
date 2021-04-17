#include "Common.h"
#include "IndexBuffer.h"
#include "VertexBuffer.h"
#include "VertexArray.h"
#include "Shader.h"
#include "Renderer.h"
#include "Texture.h"
#include "Timeline.h"

#include "Solver.cuh"
extern Solver solver;


#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_opengl3.h"
#include "imgui/backends/imgui_impl_glfw.h"


#include <thread>

#include <imgui/imgui_internal.h>
//#include <imgui_tables.cpp>
#include <imgui/misc/cpp/imgui_stdlib.h>
#include <imgui/misc/single_file/imgui_single_file.h>


//#define WINDOWS7_BUILD


static void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *window, int entered);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

void UpdateSolver() {
	std::cout << "\nRestarting\n";
	solver.ThreadsJoin();
	//clearing
	//solver.ClearCache();
	solver.frame = 0;
	solver.Clear_Simulation_Data();
	solver.ResetObjects();
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



bool TimelineInitialized = false;
MySequence Timeline;

void UpdateTimeline() {
	Timeline.myItems.clear();
	//dodawanie elementow
	for (int j = 0; j < solver.object_list.size(); j++) {
		if (solver.object_list[j].get_type() == "explosion") {
			Timeline.myItems.push_back(MySequence::MySequenceItem{ 1, &solver.object_list[j].frame_range_min,
																	&solver.object_list[j].frame_range_max, false });
			Timeline.rampEdit.push_back(RampEdit());
		}
		//Timeline.myItems.push_back(MySequence::MySequenceItem{ 1, 20, 30, true });
	}
}


void UpdateAnimation() {
	int frame = solver.frame;
	for (int j = 0; j < solver.object_list.size(); j++) {
		if (solver.object_list[j].get_type() == "explosion") {
			solver.object_list[j].set_size(Timeline.rampEdit.at(j).GetPointXAtTime(0,frame));
		}
	}
}








static const ImGuiDataTypeInfo GDataTypeInfo[] =
{
	{ sizeof(char),             "S8",   "%d",   "%d"    },  // ImGuiDataType_S8
	{ sizeof(unsigned char),    "U8",   "%u",   "%u"    },
	{ sizeof(short),            "S16",  "%d",   "%d"    },  // ImGuiDataType_S16
	{ sizeof(unsigned short),   "U16",  "%u",   "%u"    },
	{ sizeof(int),              "S32",  "%d",   "%d"    },  // ImGuiDataType_S32
	{ sizeof(unsigned int),     "U32",  "%u",   "%u"    },
#ifdef _MSC_VER
	{ sizeof(ImS64),            "S64",  "%I64d","%I64d" },  // ImGuiDataType_S64
	{ sizeof(ImU64),            "U64",  "%I64u","%I64u" },
#else
	{ sizeof(ImS64),            "S64",  "%lld", "%lld"  },  // ImGuiDataType_S64
	{ sizeof(ImU64),            "U64",  "%llu", "%llu"  },
#endif
	{ sizeof(float),            "float", "%.3f","%f"    },  // ImGuiDataType_Float (float are promoted to double in va_arg)
	{ sizeof(double),           "double","%f",  "%lf"   },  // ImGuiDataType_Double
};





bool SliderPos(const char* label, ImGuiDataType data_type, void* v, int components, const void* v_min, const void* v_max, const char* format = "%.3f", float power = 1)
{
	using namespace ImGui;
	ImGuiWindow* window = GetCurrentWindow();
	if (window->SkipItems)
		return false;

	ImGuiContext& g = *GImGui;
	bool value_changed = false;
	BeginGroup();
	PushID(label);
	PushMultiItemsWidths(components, CalcItemWidth());
	size_t type_size = GDataTypeInfo[data_type].Size;
	for (int i = 0; i < components; i++)
	{
		PushID(i);
		if (i > 0)
			SameLine(0, g.Style.ItemInnerSpacing.x);
		value_changed |= SliderScalar("", data_type, v, v_min, v_max, format, power);
		PopID();
		PopItemWidth();
		v = (void*)((char*)v + type_size);
		v_min = (void*)((char*)v_min + type_size);
		v_max = (void*)((char*)v_max + type_size);
	}
	PopID();

	const char* label_end = FindRenderedTextEnd(label);
	if (label != label_end)
	{
		SameLine(0, g.Style.ItemInnerSpacing.x);
		TextEx(label, label_end);
	}

	EndGroup();
    return value_changed;
}


















float InterfaceScale = 1.0f;
float ImageScale = 1.0f;




void RenderGUI(bool& SAVE_FILE_TAB, bool& OPEN_FILE_TAB, float& fps,
	float& progress, bool& save_panel, bool& helper_window, bool& confirm_button, int2 Image) {

	Timeline.mFrameMin = solver.START_FRAME;
	Timeline.mFrameMax = solver.END_FRAME;

	if (helper_window) {
		ImGui::Begin("Helper Panel");
		{
			ImGui::SetWindowFontScale(InterfaceScale);

			ImGui::Text("Useful shortcuts:");

			ImGui::Text("W/A/S/D - Camera movement");
			ImGui::Text("Q/Z - Camera up/down");
			ImGui::Text("Left mouse + A/D - Camera rotation");
			ImGui::Text("R - reset simulation");
			ImGui::Text("F - stop exporting");

			ImGui::SliderFloat("Interface scale", &InterfaceScale, 0.9, 2.0f);

					
			if (ImGui::Button("Close")) {
				helper_window = false;
			}
		}
		ImGui::End();
	}




	if (SAVE_FILE_TAB) {
		ImGui::Begin("Save Panel");
		{
			ImGui::SetWindowFontScale(InterfaceScale);
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
			ImGui::SetWindowFontScale(InterfaceScale);
			ImGui::Text("Enter filename");
			ImGui::InputText("Filename", solver.OPEN_FOLDER, IM_ARRAYSIZE(solver.OPEN_FOLDER));
			if (ImGui::Button("Open")) {
				solver.ThreadsJoin();
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
		ImGui::SetWindowFontScale(InterfaceScale);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open..", "")) {
					SAVE_FILE_TAB = false;
					OPEN_FILE_TAB = true;
					//solver.LoadSceneFromFile("scene2");
				}
				if (ImGui::MenuItem("Save", "")) {
					OPEN_FILE_TAB = false;
					SAVE_FILE_TAB = true;
				}
				if (ImGui::MenuItem("Close", "")) {
					save_panel = false;
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Help")) {
				if (ImGui::MenuItem("Help", "")) {
					helper_window = true;
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
		ImGui::SetWindowFontScale(InterfaceScale);
		ImGui::Text("Domain Resolution");
		ImGui::SliderInt("x", &solver.New_DOMAIN_RESOLUTION.x, 2, 490);
		ImGui::SliderInt("y", &solver.New_DOMAIN_RESOLUTION.y, 2, 490);
		ImGui::SliderInt("z", &solver.New_DOMAIN_RESOLUTION.z, 2, 490);



		ImGui::Text("Simulation Settings");
		ImGui::SliderFloat("Ambient Temp", &solver.Ambient_Temperature, -10.0f, 100.0f);
		ImGui::SliderFloat("Smoke Dissolve", &solver.Smoke_Dissolve, 0.93f, 1.0f);
		ImGui::SliderFloat("Flame Dissolve", &solver.Flame_Dissolve, 0.9f, 1.0f);
		ImGui::SliderFloat("Diverge rate", &solver.DIVERGE_RATE, 0.1f, 0.8f);
		ImGui::SliderFloat("Buoyancy", &solver.Smoke_Buoyancy, 0.0f, 10.0f);
		ImGui::SliderFloat("Pressure", &solver.Pressure, -1.5f, 0.0f);
		ImGui::SliderFloat("Max Velocity", &solver.max_velocity, 0.0f, 2.0f);
		ImGui::SliderFloat("Influence on Velocity", &solver.influence_on_velocity, 0.0f, 1.1f);
		ImGui::SliderInt("Simulation accuracy", &solver.ACCURACY_STEPS, 1, 96);
		if (ImGui::Button("Simulate")) {
			if (solver.SIMULATE == false)
				solver.SIMULATE = true;
			else
				solver.SIMULATE = false;
		}
		ImGui::SliderFloat("Simulation speed", &solver.speed, 0.1f, 1.5f);


		ImGui::Checkbox("Wavelet Upresing", &solver.Upsampling);
		if (solver.Upsampling || true) {
			ImGui::SliderFloat("Offset", &solver.OFFSET, 0.0001f, 0.3f);
			ImGui::SliderFloat("Scale", &solver.SCALE, 0.01f, 4.0f);
			//ImGui::Checkbox("Simulation Influence", &solver.INFLUENCE_SIM);
			ImGui::SliderFloat("Strength", &solver.noise_intensity, 0.01f, 3.0f);
			ImGui::SliderFloat("Time", &solver.time_anim, 0.0f, 2.0f);
			if (solver.INFLUENCE_SIM) {
				ImGui::Checkbox("Velocity", &solver.UpsamplingVelocity);
				ImGui::SameLine();
			}
			ImGui::Checkbox("Density&Temp", &solver.UpsamplingDensity);
			//ImGui::SameLine();
			//ImGui::Checkbox("Temperature", &solver.UpsamplingTemperature);
			if (solver.UpsamplingDensity)
				ImGui::SliderFloat("Density cutoff", &solver.density_cutoff, 0.0001f, 0.01f);
		}
		//ImGui::ColorEdit3("clear color", (float*)&clear_color);


		ImGui::Text("Render Settings:");
		ImGui::Checkbox("Fire&Smoke render", &solver.Smoke_And_Fire);
		ImGui::Checkbox("Shadows render", &solver.render_shadows);
		ImGui::SliderFloat("Fire Emission Rate", &solver.Fire_Max_Temperature, 0.9, 2);
		ImGui::SliderFloat("Fire Multiply", &solver.fire_multiply, 0, 1);
		ImGui::SliderInt("Render samples", &solver.STEPS, 128, 2048);
		ImGui::SliderFloat("Render Step Size", &solver.render_step_size, 0.5, 2);
		ImGui::SliderFloat("Density", &solver.density_influence, 0, 2);
		if (!solver.render_shadows)
			ImGui::SliderFloat("Transparency Compensation", &solver.transparency_compensation, 0.01, 1);
		else
			ImGui::SliderFloat("Shadow Quality", &solver.shadow_quality, 0.1, 2);
		ImGui::Checkbox("Legacy render", &solver.legacy_renderer);

		if (ImGui::Button("Reset")) {
			UpdateSolver();
		}
		ImGui::SameLine();
		ImGui::Text(("FPS: " + std::to_string(fps)).c_str());
		ImGui::Checkbox("Preserve object list", &solver.preserve_object_list);
		ImGui::Text(("Frame: " + std::to_string(solver.frame)).c_str());
	}
	ImGui::End();
	/////////////////////////////

	ImGui::Begin("Objects Panel");
	{
		ImGui::SetWindowFontScale(InterfaceScale);
		ImGui::Text("Emitter type");
		const char* items[] = { "emitter", "explosion" , "force", "power", 
			"turbulance", "wind", "sphere" };// , "vdb", "vdbs" };
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
		if (ImGui::Button("Delete All")) {
			confirm_button = true;
		}
		if (confirm_button) {
			ImGui::SameLine();
			if (ImGui::Button("Confirm")) {
				for (int object = 0; object < solver.object_list.size(); object++) {
					solver.object_list[object].free();
				}
				solver.object_list.clear();
				confirm_button = false;
				TimelineInitialized = false;
			}
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
			TimelineInitialized = false;
		}
		if (ImGui::Button("Add Emitter")) {
			solver.object_list.push_back(OBJECT(current_item, 18.0f, 50, 5.2, 5, 0.9, make_float3(float(solver.getDomainResolution().x) * 0.5f, 5.0f, float(solver.getDomainResolution().z) * 0.5f), solver.object_list.size()));
			TimelineInitialized = false;
		}

		ImGui::Text("Object list:");
		ImGui::BeginChild("Scrolling");

		for (int object = 0; object < solver.object_list.size(); object++) {
			std::string name = ("  -> " + solver.object_list[object].get_name());
			ImGui::Text(name.c_str());
			ImGui::SameLine();
			ImGui::Checkbox(std::to_string(object).c_str(), &solver.object_list[object].selected);
			
			
			float maxs[] = { solver.New_DOMAIN_RESOLUTION.x,solver.New_DOMAIN_RESOLUTION.y,solver.New_DOMAIN_RESOLUTION.z };
			float minns[] = { 0.0f,0.0f,0.0f };
			
			
			SliderPos(("position-" + std::to_string(object)).c_str(), ImGuiDataType_Float, solver.object_list[object].Location, 3, minns, maxs);
			ImGui::SliderFloat(("size-" + std::to_string(object)).c_str(), &solver.object_list[object].size, 0.0, 200.0);
			if (ImGui::Button("Apply current size as initial")) {
				solver.object_list[object].initial_size = solver.object_list[object].size;
			}
			if (solver.object_list[object].type >= 5 && solver.object_list[object].type < 9) {
				ImGui::SliderFloat(("force strength-" + std::to_string(object)).c_str(), &solver.object_list[object].force_strength, -100.0, 100.0);
				ImGui::SameLine();
				ImGui::Checkbox(("Square-" + std::to_string(object)).c_str(), &solver.object_list[object].square);
				if (solver.object_list[object].type == FORCE_FIELD_TURBULANCE)
					ImGui::SliderFloat(("turbulance frequence-" + std::to_string(object)).c_str(), &solver.object_list[object].velocity_frequence, 0.0, 20);
				if (solver.object_list[object].type == FORCE_FIELD_WIND)
					ImGui::SliderFloat3(("wind direction-" + std::to_string(object)).c_str(), solver.object_list[object].force_direction, -1, 1);
			}
			else if (solver.object_list[object].type == EMITTER) {
				ImGui::SliderFloat(("initial temperature-" + std::to_string(object)).c_str(), &solver.object_list[object].impulseTemp, -50.0, 50.0);
				ImGui::SliderFloat(("velocity frequence-" + std::to_string(object)).c_str(), &solver.object_list[object].velocity_frequence, 0.0, solver.object_list[object].max_vel_freq);
				//solver.object_list[object].set_vel_freq = solver.object_list[object].velocity_frequence;
			}
			else if (solver.object_list[object].type == EXPLOSION) {
				ImGui::SliderFloat(("initial temperature-" + std::to_string(object)).c_str(), &solver.object_list[object].impulseTemp, -50.0, 50.0);
				ImGui::SliderFloat(("velocity frequence-" + std::to_string(object)).c_str(), &solver.object_list[object].velocity_frequence, 0.0, solver.object_list[object].max_vel_freq);
				ImGui::InputInt(("start frame-" + std::to_string(object)).c_str(), &solver.object_list[object].frame_range_min);
				ImGui::InputInt(("end frame-" + std::to_string(object)).c_str(), &solver.object_list[object].frame_range_max);
			}
			solver.object_list[object].UpdateLocation();
		}
		ImGui::EndChild();

	}
	ImGui::End();
	/////////////////////////////



	ImGui::Begin("Timeline && Animation");
	{
		ImGui::SetWindowFontScale(InterfaceScale);

		if (!TimelineInitialized) {
			UpdateTimeline();
			TimelineInitialized = true;
		}

		/*
		unsigned int small = 1;
		unsigned int huge = 100;

		ImGui::InputScalar("Frame Start",ImGuiDataType_U32, &solver.START_FRAME, &small,&huge, "%d");
		ImGui::InputScalar("Frame End", ImGuiDataType_U32, &solver.END_FRAME, &small, &huge, "%d");

		ImGui::SliderScalar("timeline", ImGuiDataType_U32, &solver.frame, &solver.START_FRAME, &solver.END_FRAME, "%d", 1);
		*/


		// let's create the sequencer
		static int selectedEntry = -1;
		static bool expanded = true;

		ImGui::PushItemWidth(130);
		ImGui::InputInt("Frame Min", &solver.START_FRAME);
		//ImGui::SameLine();
		//ImGui::InputInt("Frame ", &currentFrame);
		ImGui::SameLine();
		ImGui::InputInt("Frame Max", &solver.END_FRAME);
		ImGui::PopItemWidth();
		
		
		if (solver.START_FRAME < 0) solver.START_FRAME = 0;
		if (solver.END_FRAME < 0) solver.END_FRAME = 0;

		Timeline.mFrameMin = solver.START_FRAME;
		Timeline.mFrameMax = solver.END_FRAME;

		Sequencer(&Timeline, &solver.frame, &expanded, &selectedEntry, &solver.START_FRAME, ImSequencer::SEQUENCER_EDIT_STARTEND | ImSequencer::SEQUENCER_ADD | ImSequencer::SEQUENCER_DEL | ImSequencer::SEQUENCER_COPYPASTE | ImSequencer::SEQUENCER_CHANGE_FRAME);
		// add a UI to edit that particular item
		if (selectedEntry != -1)
		{
			const MySequence::MySequenceItem& item = Timeline.myItems[selectedEntry];
			ImGui::Text("I am a %s, please edit me", SequencerItemTypeNames[item.mType]);
			// switch (type) ....
		}





		if (solver.frame > solver.END_FRAME)
			UpdateSolver();
	
		
	}
	ImGui::End();

}













#include <Windows.h>
#include <WinUser.h>

int Window(float* Img_res, float dpi) {
#ifndef WINDOWS7_BUILD
	SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE); //skalowanie globalne //bez _V2 chyba lepsze
#endif // WINDOWS7_BUILD
	
	InterfaceScale = dpi;
	
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

	
	{
		float range_x = ((float)Image.x / (float)Window.x);
		range_x = 2.0 * range_x - 1.0;

		float range_y = ((float)Image.y / (float)Window.y);
		range_y = 2.0 * range_y - 1.0;


		if (Image.x == Window.x) range_x = 1.0f;
		if (Image.y == Window.y) range_y = 1.0f;
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
		bool helper_window = true;
		bool confirm_button = false;
		//std::thread* sim;
		solver.DONE_FRAME = true;
		/////////////////////////////////////////////////
		//TImeline

		//Timeline.myItems.push_back(MySequence::MySequenceItem{ 0, 10, 30, false });
		//Timeline.myItems.push_back(MySequence::MySequenceItem{ 1, 20, 30, true });


		///////////////////////////////////////////////////
		while (!glfwWindowShouldClose(window)) {
			clock_t startTime = clock();


			//////////////////
			if (solver.SIMULATE) {
				solver.Simulation_Frame();
			}
			//////////////////
			//Texture texture("output/R" + pad_number(frame) + ".bmp");
			//texture.UpdateTexture("output/R" + pad_number(solver.frame) + ".bmp");
			texture.UpdateTexture("output/temp.bmp");
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
			RenderGUI(SAVE_FILE_TAB, OPEN_FILE_TAB, fps, progress, save_panel, helper_window, confirm_button, Image);
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
	if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
		solver.EXPORT_VDB = false;
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