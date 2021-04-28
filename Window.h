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

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <experimental/filesystem>
#include <thread>

#include <imgui/imgui_internal.h>
//#include <imgui_tables.cpp>
#include <imgui/misc/cpp/imgui_stdlib.h>
#include <imgui/misc/single_file/imgui_single_file.h>
#include <ImGuiFileBrowser.h>


//LEGACY BUILD SETTINGS/////////////////////
//#define WINDOWS7_BUILD



///////////////////////////////////




void AddObject2(int,int,int ps = 0);


static void cursorPositionCallback(GLFWwindow* window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *window, int entered);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

// TODO: Save theme on close & load it on startup

enum Themes
{
	DARK,
	DARKMIDNIGHT,
	GRAY,
	LIGHT
};

int selected_theme = DARK;

const char* theme_strings[] = { "Dark", "Midnight", "Gray", "Light" };


const char* itemse[] = { "emitter", "explosion" , "force", "power", "turbulance", "wind", "sphere", "particle", "object" };
bool TimelineInitialized = false;
bool AddParticleSystem = false;
bool AddObjectSystem = false;
char temp_object_path[512] = { "./input/obj/suzanne/" };
char temp_particle_path[512] = { "./input/exp2/" };
static int selectedEntry = -1;
MySequence Timeline;


//////////////////FUNCTIO
int sinresolution = 8;
float rot = solver.getRotation();
float sinmid = 20;
float sinspeed = 0.1;
float sinsize = 8;
float sinoffset = 0.0;

#define CURVE_SIZE 0
#define CURVE_X 1
#define CURVE_Y 2
#define CURVE_Z 3

imgui_addons::ImGuiFileBrowser file_dialog;



void UpdateTimelinePartially() {

	for (int i = 0; i < Timeline.myItems.size(); i++) {
		Timeline.myItems[i].mFrameStart = &solver.object_list[i].frame_range_min;
		Timeline.myItems[i].mFrameEnd = &solver.object_list[i].frame_range_max;
	}

}

void ClearTimeline(int minn = 0) {
	for (int object = solver.object_list.size()-1; object >= minn; object++) { //>=
		//int t = solver.object_list[object].type;
		solver.object_list[object].free();
		solver.object_list[object].collisions.clear();
		solver.object_list.erase(solver.object_list.begin() + object);
		//Timeline.Del(object);
		Timeline.myItems.erase(Timeline.myItems.begin() + object);
		Timeline.rampEdit.erase(Timeline.rampEdit.begin() + object);
	}
	selectedEntry = -1;
}




static bool Knob(const char* label, float* p_value, float v_min, float v_max)
{
	ImGuiIO& io = ImGui::GetIO();
	ImGuiStyle& style = ImGui::GetStyle();

	float radius_outer = 20.0f;
	ImVec2 pos = ImGui::GetCursorScreenPos();
	ImVec2 center = ImVec2(pos.x + radius_outer, pos.y + radius_outer);
	float line_height = ImGui::GetTextLineHeight();
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	float ANGLE_MIN = 3.141592f * 0.75f;
	float ANGLE_MAX = 3.141592f * 2.25f;

	ImGui::InvisibleButton(label, ImVec2(radius_outer * 2, radius_outer * 2 + line_height + style.ItemInnerSpacing.y));
	bool value_changed = false;
	bool is_active = ImGui::IsItemActive();
	bool is_hovered = ImGui::IsItemActive();
	if (is_active && io.MouseDelta.x != 0.0f)
	{
		float step = (v_max - v_min) / 200.0f;
		*p_value += io.MouseDelta.x * step;
		if (*p_value < v_min) *p_value = v_min;
		if (*p_value > v_max) *p_value = v_max;
		value_changed = true;
	}

	float t = (*p_value - v_min) / (v_max - v_min);
	float angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * t;
	float angle_cos = cosf(angle), angle_sin = sinf(angle);
	float radius_inner = radius_outer * 0.40f;
	draw_list->AddText(ImVec2(pos.x, pos.y + radius_outer * 2 + style.ItemInnerSpacing.y), ImGui::GetColorU32(ImGuiCol_Text), label);
	draw_list->AddCircleFilled(center, radius_outer, ImGui::GetColorU32(ImGuiCol_FrameBg), 16);
	draw_list->AddLine(ImVec2(center.x + angle_cos * radius_inner, center.y + angle_sin * radius_inner), ImVec2(center.x + angle_cos * (radius_outer - 2), center.y + angle_sin * (radius_outer - 2)), ImGui::GetColorU32(ImGuiCol_SliderGrabActive), 2.0f);
	draw_list->AddCircleFilled(center, radius_inner, ImGui::GetColorU32(is_active ? ImGuiCol_FrameBgActive : is_hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), 16);

	if (is_active || is_hovered)
	{
		ImGui::SetNextWindowPos(ImVec2(pos.x - style.WindowPadding.x, pos.y - line_height - style.ItemInnerSpacing.y - style.WindowPadding.y));
		ImGui::BeginTooltip();
		ImGui::Text("%.3f", *p_value);
		ImGui::EndTooltip();
	}

	return value_changed;
}

static bool Knob(const char* label, int* p_value, int v_min, int v_max)
{
	ImGuiIO& io = ImGui::GetIO();
	ImGuiStyle& style = ImGui::GetStyle();

	float radius_outer = 20.0f;
	ImVec2 pos = ImGui::GetCursorScreenPos();
	ImVec2 center = ImVec2(pos.x + radius_outer, pos.y + radius_outer);
	float line_height = ImGui::GetTextLineHeight();
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	float ANGLE_MIN = 3.141592f * 0.75f;
	float ANGLE_MAX = 3.141592f * 2.25f;

	ImGui::InvisibleButton(label, ImVec2(radius_outer * 2, radius_outer * 2 + line_height + style.ItemInnerSpacing.y));
	bool value_changed = false;
	bool is_active = ImGui::IsItemActive();
	bool is_hovered = ImGui::IsItemActive();
	if (is_active && io.MouseDelta.x != 0.0f)
	{
		float step = (v_max - v_min) / 200;
		*p_value += io.MouseDelta.x * step;
		if (*p_value < v_min) *p_value = v_min;
		if (*p_value > v_max) *p_value = v_max;
		value_changed = true;
	}

	float t = (*p_value - v_min) / (v_max - v_min);
	float angle = ANGLE_MIN + (ANGLE_MAX - ANGLE_MIN) * t;
	float angle_cos = cosf(angle), angle_sin = sinf(angle);
	float radius_inner = radius_outer * 0.40f;
	draw_list->AddCircleFilled(center, radius_outer, ImGui::GetColorU32(ImGuiCol_FrameBg), 16);
	draw_list->AddLine(ImVec2(center.x + angle_cos * radius_inner, center.y + angle_sin * radius_inner), ImVec2(center.x + angle_cos * (radius_outer - 2), center.y + angle_sin * (radius_outer - 2)), ImGui::GetColorU32(ImGuiCol_SliderGrabActive), 2.0f);
	draw_list->AddCircleFilled(center, radius_inner, ImGui::GetColorU32(is_active ? ImGuiCol_FrameBgActive : is_hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), 16);
	draw_list->AddText(ImVec2(pos.x, pos.y + radius_outer * 2 + style.ItemInnerSpacing.y), ImGui::GetColorU32(ImGuiCol_Text), label);

	if (is_active || is_hovered)
	{
		ImGui::SetNextWindowPos(ImVec2(pos.x - style.WindowPadding.x, pos.y - line_height - style.ItemInnerSpacing.y - style.WindowPadding.y));
		ImGui::BeginTooltip();
		ImGui::Text("%i", *p_value);
		ImGui::EndTooltip();
	}

	return value_changed;
}















void UpdateTimeline() {
	
	Timeline.myItems.clear();
	
	for (int i = 0; i < Timeline.rampEdit.size(); i++)
		for (int j = 0; j < Timeline.rampEdit[i].mMax.size(); j++) {
			Timeline.rampEdit[i].RemovePoints(j);
		}
	Timeline.focused = false;
	Timeline.rampEdit.clear();

	//dodawanie elementow
	for (int j = 0; j < solver.object_list.size(); j++) {
		int current_item_id = 0;
		for (int i = 0; i < EmitterCount; i++)
			if (solver.object_list[j].get_type2() == itemse[i]) {
				current_item_id = i;
				break;
			}
		if (solver.object_list[j].get_type2() == "particle")
			AddObject2(solver.object_list[j].type, j, 1);
		else if (solver.object_list[j].type == VDBOBJECT)
			AddObject2(solver.object_list[j].type, j, 3);
		else
			AddObject2(solver.object_list[j].type, j);
	}
}


void UpdateAnimation() {
	int frame = solver.frame;
//#pragma omp parallel for num_threads(4)
	for (int j = 0; j < solver.object_list.size(); j++) {
		if (solver.object_list[j].get_type2() == "explosion" && frame >= solver.object_list[j].frame_range_min &&
			frame <= solver.object_list[j].frame_range_max) {
			solver.object_list[j].set_size(Timeline.rampEdit[j].GetPointYAtTime(0,frame));

			float x = Timeline.rampEdit[j].GetPointYAtTime(CURVE_X, frame);
			float y = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Y, frame);
			float z = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Z, frame);
			solver.object_list[j].set_location(make_float3(x,y,z));
		}
		//if (solver.object_list[j].get_type2() == "emitter") {
		else if (solver.object_list[j].type != PARTICLE && solver.object_list[j].type != VDBOBJECT){
			solver.object_list[j].set_size(Timeline.rampEdit[j].GetPointYAtTime(0, frame));
			float x = Timeline.rampEdit[j].GetPointYAtTime(CURVE_X, frame);
			float y = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Y, frame);
			float z = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Z, frame);
			solver.object_list[j].set_location(make_float3(x, y, z));
		}
		else if (solver.object_list[j].type == PARTICLE){
			solver.object_list[j].set_size(Timeline.rampEdit[j].GetPointYAtTime(0, frame));
			float x = Timeline.rampEdit[j].GetPointYAtTime(CURVE_X, frame);
			float y = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Y, frame);
			float z = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Z, frame);
			solver.object_list[j].set_location(make_float3(x, y, z));
		}
		else if (solver.object_list[j].type == VDBOBJECT) {

			/*
			float sizeee = solver.object_list[j].size;
			solver.object_list[j].set_size(Timeline.rampEdit[j).GetPointYAtTime(0, frame));

			if (solver.object_list[j].size != sizeee) {
				int frame_start = solver.object_list[j].frame_range_min;
				int frame_end = solver.object_list[j].frame_range_max;
				solver.object_list[j].collisions.clear();
				int SIZEEEE = min(min(solver.getDomainResolution().x, solver.getDomainResolution().y), solver.getDomainResolution().z);
				solver.object_list[j].LoadObjects(solver.getDomainResolution(),
					solver.devicesCount, solver.deviceIndex, SIZEEEE * solver.object_list[j].size);
				solver.object_list[j].frame_range_min = frame_start;
				solver.object_list[j].frame_range_max = frame_end;
			}
			*/

			float x = Timeline.rampEdit[j].GetPointYAtTime(CURVE_X, frame);
			float y = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Y, frame);
			float z = Timeline.rampEdit[j].GetPointYAtTime(CURVE_Z, frame);
			solver.object_list[j].set_location(make_float3(x, y, z));
		}
	}
}

void AddObject2(int type, int j, int particle_system) {
	type = max(0, type)%EmitterCount;
	int current_item_id = 0;
	for (int i = 0; i < EmitterCount; i++)
		if (solver.object_list[j].get_type2() == itemse[i]) {
			current_item_id = i;
			break;
		}
	Timeline.myItems.push_back(MySequence::MySequenceItem{ current_item_id, &solver.object_list[j].frame_range_min,
																&solver.object_list[j].frame_range_max, false });

	if (type == EXPLOSION || type == PARTICLE) {
		Timeline.rampEdit.push_back(RampEdit(solver.object_list[j].frame_range_min, solver.object_list[j].frame_range_max,
			(float)solver.getDomainResolution().x / 2.f, 5.f, (float)solver.getDomainResolution().z / 2.f, particle_system));
	}
	else if (type == VDBOBJECT) {
		Timeline.rampEdit.push_back(RampEdit(solver.object_list[j].frame_range_min, solver.object_list[j].frame_range_max,
			0, 5.f, 0, particle_system));
	}
	else {
		Timeline.rampEdit.push_back(RampEdit(solver.object_list[j].frame_range_min, solver.object_list[j].frame_range_max,
			(float)solver.getDomainResolution().x / 2.f, 5.f, (float)solver.getDomainResolution().z / 2.f, 2));
	}

	UpdateTimelinePartially();
}

void AddObject(int type) {
	//std::cout << "Adding object of index " << type << std::endl;
	type = max(0,type) % EmitterCount;
	std::string name = itemse[type];
	if (name == "particle") {
		
	}
	else {
		solver.object_list.push_back(OBJECT(name, 18.0f, 50, 5.2, 5, 0.9, make_float3(float(solver.getDomainResolution().x) * 0.5f, 5.0f, float(solver.getDomainResolution().z) * 0.5f), solver.object_list.size(), solver.devicesCount));
		int j = solver.object_list.size() - 1;
		AddObject2(solver.object_list[j].type, j);
	}
}


void DuplicateObject(int index) {
	OBJECT obj = OBJECT(solver.object_list[index], solver.object_list.size(),solver.devicesCount);
	std::string name = obj.get_type2();
	obj.set_name(name + std::to_string(index+1));

	solver.object_list.push_back(OBJECT(obj, solver.object_list.size(), solver.devicesCount));
	int j = solver.object_list.size() - 1;
	if (solver.object_list[j].type == VDBOBJECT) {
		solver.object_list[j].load_density_grid(solver.object_list[index].get_density_grid(),
			solver.object_list[index].get_initial_temp(), solver.deviceIndex);
	}
	//solver.object_list[j).vdb_object = obj.vdb_object;

	int current_item_id = 0;
#pragma unroll
	for (int i = 0; i < EmitterCount; i++)
		if (solver.object_list[j].get_type2() == itemse[i]) {
			current_item_id = i;
			break;
		}


	Timeline.myItems.push_back(MySequence::MySequenceItem{ current_item_id, &solver.object_list[j].frame_range_min, 
														&solver.object_list[j].frame_range_max, false });
	//Timeline.rampEdit.push_back(RampEdit(solver.object_list[j].frame_range_min, solver.object_list[j].frame_range_max));
	RampEdit ramp = RampEdit();
	ramp.Copy(Timeline.rampEdit[index]);
	Timeline.rampEdit.push_back(&ramp); //tu problem?

	UpdateTimeline();
}

void DeleteObject(const int object) {
	/*
	if (solver.object_list[object].get_type() == "object" ||
		solver.object_list[object].get_type() == "vdbs")
		solver.object_list[object].cudaFree();
	*/
	solver.object_list[object].free();
	solver.object_list[object].collisions.clear();
	solver.object_list.erase(solver.object_list.begin() + object);
	//Timeline.Del(object);
	Timeline.myItems.erase(Timeline.myItems.begin() + object);
	Timeline.rampEdit.erase(Timeline.rampEdit.begin() + object);

	UpdateTimeline();
	//std::cout << "Lists: " << solver.object_list.size() << ":" << Timeline.myItems.size() << ":" << Timeline.rampEdit.size() << std::endl;
}



int frame = solver.frame;


std::vector<std::thread> threads;


void UpdateSolver(bool full = false, std::string filename = "") {
	solver.LOCK = true;
	solver.writing = true;
	solver.THIS_IS_THE_END = true;
	for (auto& thread : threads)
		thread.join();


	std::cout << "\nRestarting\n";
	solver.ThreadsJoin();
	//clearing
	//solver.ClearCache();
	solver.frame = 0;
	frame = solver.frame;

	solver.THIS_IS_THE_END = false;
	//solver.SIMULATE = true;

	if ((solver.New_DOMAIN_RESOLUTION.x == solver.getDomainResolution().x) &&
		(solver.New_DOMAIN_RESOLUTION.y == solver.getDomainResolution().y) &&
		(solver.New_DOMAIN_RESOLUTION.z == solver.getDomainResolution().z) &&
		!full && solver.preserve_object_list ) {
		solver.Clear_Simulation_Data2();
		solver.InitGPUNoise(solver.NOISE_SC);
		solver.ResetObjects1(); //loc rot scale
		UpdateTimelinePartially();
	}
	else if (full) {
		//ClearTimeline(1);
		solver.Clear_Simulation_Data();
		solver.LoadSceneFromFile(filename);
		//solver.Initialize_Simulation();
		UpdateTimelinePartially();
	}
	else {// if (solver.preserve_object_list) {
		solver.Clear_Simulation_Data();
		solver.UpdateDomainResolution();
		solver.ResetObjects();
		solver.Initialize_Simulation();
		UpdateTimelinePartially();
	}
	/*
	else {
		solver.preserve_object_list = true;
		ClearTimeline(1);
		solver.Clear_Simulation_Data();
		solver.UpdateDomainResolution();
		solver.Initialize_Simulation();
		UpdateTimeline();
		UpdateAnimation();
		ClearTimeline(0);
		UpdateTimeline();

		solver.LOCK = false;
		solver.writing = false;
		threads.clear();

		return;
	}
	*/

	if (!full) {
		if (solver.SAMPLE_SCENE == 0)
			solver.ExampleScene();
		else if (solver.SAMPLE_SCENE == 1 || solver.SAMPLE_SCENE == 2)
			solver.ExportVDBScene();
	}
	


	//preparation
	//solver.Initialize();
	//solver.UpdateTimeStep();
	// 
	//solver.Initialize_Simulation(); //tutaj
	
	
	solver.frame = 0;
	frame = solver.frame;
	solver.DONE_FRAME = true;
	solver.LOCK = false;
	solver.writing = false;
	threads.clear();

}





void DrawCombo(float dpi, const char* name, int& variable, const char* labels[], int count)
{
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6 * dpi);
	ImGui::Text(name);
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5 * dpi);
	ImGui::Combo(std::string("##COMBO__" + std::string(name)).c_str(), &variable, labels, count);
}

void DrawCombo(float dpi, const char* name, int& variable, bool (*items_getter)(void*, int, const char**), void* data, int count)
{
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 6 * dpi);
	ImGui::Text(name);
	ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5 * dpi);
	ImGui::Combo(std::string("##COMBO__" + std::string(name)).c_str(), &variable, items_getter, data, count);
}
































bool ALLOW_SCROLL = 0;


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
int begin_frame = 0;



void RenderGUI(float DPI, bool& SAVE_FILE_TAB, bool& OPEN_FILE_TAB, float& fps,
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
			ImGui::Text("LM double click on curve - new point");
			ImGui::Text("LCtrl+Scroll on animation panel\n - zoom in/out");
			ImGui::Text("Space - pause simulation");
			ImGui::Text("LCtrl+LM on slider - writing mode");

			ImGui::BeginChild("##Misc settings", ImVec2(0, 0), true);
			{
				ImGui::SliderFloat("Interface scale", &InterfaceScale, 0.9, 2.0f);
				ImGui::Text("\n\n");
				DrawCombo(InterfaceScale, "Theme", selected_theme, theme_strings, ARRAYSIZE(theme_strings));


				if (Knob("Rotation", &rot, -5, 5))
					solver.setRotation(rot);
			}
			ImGui::EndChild();


			ImGuiStyle* style = &ImGui::GetStyle();
			switch (selected_theme)
			{
			case DARK:
				ImGui::StyleColorsDark();
				break;
			case DARKMIDNIGHT:
				style->WindowPadding = ImVec2(15, 15);
				style->WindowRounding = 5.0f;
				style->FramePadding = ImVec2(5, 5);
				style->FrameRounding = 4.0f;
				style->ItemSpacing = ImVec2(12, 8);
				style->ItemInnerSpacing = ImVec2(8, 6);
				style->IndentSpacing = 25.0f;
				style->ScrollbarSize = 15.0f;
				style->ScrollbarRounding = 9.0f;
				style->GrabMinSize = 5.0f;
				style->GrabRounding = 3.0f;

				style->Colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
				style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
				style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
				style->Colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
				style->Colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.88f);
				style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
				style->Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
				style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
				style->Colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 0.98f, 0.95f, 0.75f);
				style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
				style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
				style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
				style->Colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
				style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
				style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
				style->Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
				style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
				style->Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
				style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
				style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
				style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
				style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
				style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
				style->Colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
				style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
				style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
				style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
				style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
				break;
			case GRAY:
				style->Colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
				style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
				style->Colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
				style->Colors[ImGuiCol_ChildBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
				style->Colors[ImGuiCol_PopupBg] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
				style->Colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
				style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
				style->Colors[ImGuiCol_FrameBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
				style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
				style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
				style->Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
				style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
				style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
				style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
				style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
				style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
				style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
				style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
				style->Colors[ImGuiCol_CheckMark] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
				style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
				style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
				style->Colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
				style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
				style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
				style->Colors[ImGuiCol_Header] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
				style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
				style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
				style->Colors[ImGuiCol_Separator] = style->Colors[ImGuiCol_Border];
				style->Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
				style->Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
				style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
				style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
				style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
				style->Colors[ImGuiCol_Tab] = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
				style->Colors[ImGuiCol_TabHovered] = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
				style->Colors[ImGuiCol_TabActive] = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
				style->Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
				style->Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
				style->Colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
				style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
				style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
				style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
				style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
				style->Colors[ImGuiCol_DragDropTarget] = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
				style->Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
				style->Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
				style->Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
				style->Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
				style->GrabRounding = style->FrameRounding = 2.3f;
				break;
			case LIGHT:
				ImGui::StyleColorsLight();
				break;
			default:
				ImGui::StyleColorsDark();
			}

				
					
			if (ImGui::Button("Close")) {
				helper_window = false;
			}
		}
		ImGui::End();
	}




			/*
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
			*/
	if (OPEN_FILE_TAB) {
		ImGui::Begin("Open Panel");
		{
			ImGui::SetWindowFontScale(InterfaceScale);
			ImGui::Text("Enter filename");
			//ImGui::InputText("Filename", solver.OPEN_FOLDER, IM_ARRAYSIZE(solver.OPEN_FOLDER));
			if (ImGui::Button("Open")) {
				//solver.ThreadsJoin();
				std::string filename = solver.OPEN_FOLDER;
				filename = trim(filename);

				//////////////////////////////
				bool temporary = solver.preserve_object_list;
				solver.preserve_object_list = false;
				UpdateSolver(true);
				solver.preserve_object_list = temporary;
				//////////////////////////////

				//solver.LoadSceneFromFile(filename);
				//UpdateTimeline();
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
				//ImGui::Text(name.c_str());
				if (ImGui::Button(name.c_str())) {
					solver.ThreadsJoin();
					std::string filename = list[object];
					filename = trim(filename);


					solver.preserve_object_list = false;
					UpdateSolver(true,filename);
					UpdateTimeline();
					solver.preserve_object_list = true;

					//solver.LoadSceneFromFile(filename);
					//UpdateTimeline();
					OPEN_FILE_TAB = false;
				}
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

		if (SAVE_FILE_TAB) {
			ImGui::OpenPopup("Save File");
			SAVE_FILE_TAB = false;
		}
		file_dialog.interface_scale = InterfaceScale;
		
		if (file_dialog.showFileDialog("Save File", imgui_addons::ImGuiFileBrowser::DialogMode::SAVE, ImVec2(700 * InterfaceScale, 310 * InterfaceScale), ".txt")) //".txt,.jpg,.dll"
		{
			
			std::cout << file_dialog.selected_fn << std::endl;      // The name of the selected file or directory in case of Select Directory dialog mode
			std::cout << file_dialog.selected_path << std::endl;    // The absolute path to the selected file
			std::cout << file_dialog.ext << std::endl;              // Access ext separately (For SAVE mode)
			//Do writing of files based on extension here
			std::string filename = file_dialog.selected_path;
			filename = trim(filename);
			solver.SaveSceneToFile(filename + file_dialog.ext);
			SAVE_FILE_TAB = false;
		}


		ImGui::Text("Example scenes");
		const char* items[] = { "VDB","VDBFire", "Objects" };// , "vdb", "vdbs" };
		static const char* current_item3 = "Objects";

		if (ImGui::BeginCombo("##combo", current_item3)) // The second parameter is the label previewed before opening the combo.
		{
			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
			{
				bool is_selected = (current_item3 == items[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(items[n], is_selected)) {
					current_item3 = items[n];
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
			ImGui::EndCombo();
		}
		if (ImGui::Button("Load Scene")) {
			if (current_item3 == "VDB")
				solver.SAMPLE_SCENE = 1;
			else if (current_item3 == "Objects")
				solver.SAMPLE_SCENE = 0;
			else if (current_item3 == "VDBFire")
				solver.SAMPLE_SCENE = 2;
			solver.preserve_object_list = false;
			UpdateSolver();
			UpdateTimeline();
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
			//progress += 1.0 / ((float)solver.EXPORT_END_FRAME);
			progress = (float)solver.frame / ((float)solver.EXPORT_END_FRAME);

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




		int maxx = round(MAX_BOK * ratioo);
		int maxy = round(MAX_BOK * ratioo);
		int maxz = round(MAX_BOK * ratioo);

		ImGui::SliderInt("x", &solver.New_DOMAIN_RESOLUTION.x, 2, maxx);
		ImGui::SliderInt("y", &solver.New_DOMAIN_RESOLUTION.y, 2, maxy);
		ImGui::SliderInt("z", &solver.New_DOMAIN_RESOLUTION.z, 2, maxz);

		solver.New_DOMAIN_RESOLUTION.x = min(solver.New_DOMAIN_RESOLUTION.x, maxx);
		solver.New_DOMAIN_RESOLUTION.y = min(solver.New_DOMAIN_RESOLUTION.y, maxy);
		solver.New_DOMAIN_RESOLUTION.z = min(solver.New_DOMAIN_RESOLUTION.z, maxz);

		ratioo = (MAX_BOK * 3.f) / float(solver.New_DOMAIN_RESOLUTION.x + solver.New_DOMAIN_RESOLUTION.y + solver.New_DOMAIN_RESOLUTION.z);



		ImGui::Text("Simulation Settings");
		if (ImGui::CollapsingHeader("Simulation Settings", 0, true))
		{
			ImGui::SliderFloat("Ambient Temp", &solver.Ambient_Temperature, -10.0f, 100.0f);
			ImGui::SliderFloat("Smoke Dissolve", &solver.Smoke_Dissolve, 0.93f, 1.0f);
			ImGui::SliderFloat("Flame Dissolve", &solver.Flame_Dissolve, 0.9f, 1.0f);
			ImGui::SliderFloat("Diverge rate", &solver.DIVERGE_RATE, 0.1f, 0.8f);
			ImGui::SliderFloat("Buoyancy", &solver.Smoke_Buoyancy, 0.0f, 10.0f);
			ImGui::SliderFloat("Pressure", &solver.Pressure, -1.5f, 0.0f);
			ImGui::SliderFloat("Max Velocity", &solver.max_velocity, 0.0f, 20.0f);
			ImGui::SliderFloat("Influence on Velocity", &solver.influence_on_velocity, 0.0f, 5.1f);
			ImGui::Text("\n\n");
		}
		
		ImGui::SliderInt("Simulation accuracy", &solver.ACCURACY_STEPS, 1, 150);
		if (ImGui::Button("Simulate")) {
			solver.SIMULATE = !solver.SIMULATE;
		}
		ImGui::SliderFloat("Simulation speed", &solver.speed, 0.1f, 1.5f);


		ImGui::Text("\n\n");
		ImGui::Checkbox("Wavelet Upresing", &solver.Upsampling);
		if (solver.Upsampling) {
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


		ImGui::Text("\n\nRender Settings:");
		if (ImGui::CollapsingHeader("Render Settings", 0, true))
		{
			ImGui::Checkbox("Fire&Smoke render", &solver.Smoke_And_Fire);
			ImGui::Checkbox("Shadows render", &solver.render_shadows);
			ImGui::Checkbox("Collision obj render", &solver.render_collision_objects);
			ImGui::SliderFloat("Fire Emission Rate", &solver.Fire_Max_Temperature, 0.9, 2);
			ImGui::SliderFloat("Fire Multiply", &solver.fire_multiply, 0, 1);
			ImGui::SliderFloat("Render Step Size", &solver.render_step_size, 0.5, 2);
			ImGui::SliderFloat("Density", &solver.density_influence, 0, 2);
			if (!solver.render_shadows)
				ImGui::SliderFloat("Transparency Compensation", &solver.transparency_compensation, 0.01, 1);
			else
				ImGui::SliderFloat("Shadow Quality", &solver.shadow_quality, 0.1, 2);
			ImGui::Checkbox("Legacy render", &solver.legacy_renderer);
		}
		ImGui::SliderInt("Render samples", &solver.STEPS, 128, 2048);
		if (ImGui::Button("Reset")) {
			UpdateSolver();
		}
		ImGui::SameLine();
		ImGui::Text(("FPS: " + std::to_string(fps)).c_str());
		//ImGui::Checkbox("Preserve object list", &solver.preserve_object_list);
		ImGui::Text(("Frame: " + std::to_string(solver.frame)).c_str());
	}
	ImGui::End();
	/////////////////////////////

	ImGui::Begin("Objects Panel");
	{
		ImGui::SetWindowFontScale(InterfaceScale);
		ImGui::Text("Emitter type");
		
		static const char* current_object_emitter = "emitter";
		int current_item_id = 0;

		if (ImGui::BeginCombo("##combo", current_object_emitter)) // The second parameter is the label previewed before opening the combo.
		{
			for (int n = 0; n < IM_ARRAYSIZE(itemse); n++)
			{
				bool is_selected = (current_object_emitter == itemse[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(itemse[n], is_selected)) {
					current_object_emitter = itemse[n];
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
				selectedEntry = -1;
			}
		}
		if (ImGui::Button("Delete selected")) {
		REPEAT:
			for (int object = 0; object < solver.object_list.size(); object++) {
				if (solver.object_list[object].selected) {
					/*
					if (solver.object_list[object].get_type() == "vdb" ||
						solver.object_list[object].get_type() == "vdbs")
						solver.object_list[object].cudaFree();
					solver.object_list[object].free();
					solver.object_list.erase(solver.object_list.begin() + object);
					Timeline.Del(object);
					*/
					DeleteObject(object);
					goto REPEAT;
					//TODO
				}
			}
			selectedEntry = min((int)selectedEntry, int(solver.object_list.size()-1));
			//TimelineInitialized = false;
		}
		if (ImGui::Button("Add Emitter")) {
			/*
			solver.object_list.push_back(OBJECT(current_item, 18.0f, 50, 5.2, 5, 0.9, make_float3(float(solver.getDomainResolution().x) * 0.5f, 5.0f, float(solver.getDomainResolution().z) * 0.5f), solver.object_list.size()));
			//if (current_item == "explosion") {
			for (int i = 0; i < EmitterCount; i++)
				if (current_item == itemse[i]) {
					current_item_id = i;
					break;
				}
			int j = solver.object_list.size() - 1;

			std::cout << current_item_id << " : " << j << std::endl;
			AddObject2(current_item_id,j);
			*/
			for (int i = 0; i < EmitterCount; i++)
				if (current_object_emitter == itemse[i]) {
					current_item_id = i;
					break;
				}
			//std::cout << current_item_id << std::endl;
			//std::cout << current_object_emitter << std::endl;
			if (std::string(current_object_emitter) == "particle") {
				AddParticleSystem = true;
			}
			else if (std::string(current_object_emitter) == "object") {
				AddObjectSystem = true;
			}
			else {
				AddObject(current_item_id);
			}
		}

		if (AddParticleSystem && std::string(current_object_emitter) == "particle") {
			ImGui::InputText("filepath", temp_particle_path, IM_ARRAYSIZE(temp_particle_path));
			if (std::experimental::filesystem::is_directory(temp_particle_path)) {
				if (ImGui::Button("Confirm2")) {
					solver.SIMULATE = false;
					OBJECT prt("particle", 1.0f, make_float3(0, 0, 0), 5.0, 0.8, solver.object_list.size(), solver.devicesCount);
					prt.particle_filepath = temp_particle_path;
					prt.LoadParticles();
					if (prt.velocities.size() != 0) {
						if (prt.velocities[0].size() != 0) {
							solver.object_list.push_back(prt);
							int j = solver.object_list.size() - 1;
							AddObject2(PARTICLE, j, 1);
						}
					}
					solver.SIMULATE = true;
					AddParticleSystem = false;
				}
			}
		}
		if (AddObjectSystem && std::string(current_object_emitter) == "object") {
			ImGui::InputText("filepath2", temp_object_path, IM_ARRAYSIZE(temp_object_path));
			if (std::experimental::filesystem::is_directory(temp_object_path)) {
				if (ImGui::Button("Confirm22")) {
					solver.SIMULATE = false;
					OBJECT prt("object", 1.0f, make_float3(0, 0, 0), 5.0, 0.8, solver.object_list.size(), solver.devicesCount);
					prt.particle_filepath = temp_object_path;
					prt.LoadObjects(solver.getDomainResolution(),solver.devicesCount,solver.deviceIndex);
					if (prt.collisions.size() != 0) {
							solver.object_list.push_back(prt);
							int j = solver.object_list.size() - 1;
							AddObject2(VDBOBJECT, j, 1);
					}
					solver.SIMULATE = true;
					AddObjectSystem = false;
				}
			}
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
			

			if (solver.object_list[object].type != PARTICLE) {
				if (solver.SIMULATE) {
					solver.object_list[object].Location[0] = solver.object_list[object].get_location().x;
					solver.object_list[object].Location[1] = solver.object_list[object].get_location().y;
					solver.object_list[object].Location[2] = solver.object_list[object].get_location().z;


					SliderPos(("position-" + std::to_string(object)).c_str(), ImGuiDataType_Float, solver.object_list[object].Location, 3, minns, maxs);
				}
				else {
					SliderPos(("position-" + std::to_string(object)).c_str(), ImGuiDataType_Float, solver.object_list[object].Location, 3, minns, maxs);


					int position = -1;
					if (Timeline.rampEdit[object].IsOnPoint(CURVE_X, solver.frame, position) &&
						Timeline.rampEdit[object].IsOnPoint(CURVE_Y, solver.frame, position) &&
						Timeline.rampEdit[object].IsOnPoint(CURVE_Z, solver.frame, position)) {
						ImGui::Checkbox(std::string("Edit Keyframe XYZ-" + std::to_string(object)).c_str(), &solver.object_list[object].edit_frame_translation);
						if (solver.object_list[object].edit_frame_translation) {
							Timeline.rampEdit[object].EditPoint(CURVE_X, position, ImVec2(solver.frame, solver.object_list[object].Location[0]));
							Timeline.rampEdit[object].EditPoint(CURVE_Y, position, ImVec2(solver.frame, solver.object_list[object].Location[1]));
							Timeline.rampEdit[object].EditPoint(CURVE_Z, position, ImVec2(solver.frame, solver.object_list[object].Location[2]));
						}
					}
					else {
						if (ImGui::Button(std::string("Add Keyframe XYZ" + std::to_string(object)).c_str())) {
							if (!Timeline.rampEdit[object].IsOnPoint(CURVE_X, solver.frame, position))
								Timeline.rampEdit[object].AddPoint(CURVE_X, ImVec2(solver.frame, solver.object_list[object].Location[0]));
							if (!Timeline.rampEdit[object].IsOnPoint(CURVE_Y, solver.frame, position))
								Timeline.rampEdit[object].AddPoint(CURVE_Y, ImVec2(solver.frame, solver.object_list[object].Location[1]));
							if (!Timeline.rampEdit[object].IsOnPoint(CURVE_Z, solver.frame, position))
								Timeline.rampEdit[object].AddPoint(CURVE_Z, ImVec2(solver.frame, solver.object_list[object].Location[2]));
						}
					}
				}
			}
			else { /////PARTICLE
				if (solver.SIMULATE) {
					solver.object_list[object].Location[0] = solver.object_list[object].get_location().x;
					solver.object_list[object].Location[1] = solver.object_list[object].get_location().y;
					solver.object_list[object].Location[2] = solver.object_list[object].get_location().z;
					SliderPos(("position-" + std::to_string(object)).c_str(), ImGuiDataType_Float, solver.object_list[object].Location, 3, minns, maxs);
				}
				else {
					SliderPos(("position-" + std::to_string(object)).c_str(), ImGuiDataType_Float, solver.object_list[object].Location, 3, minns, maxs);
					
					ImGui::Checkbox(std::string("Edit Keyframe XYZ-" + std::to_string(object)).c_str(), &solver.object_list[object].edit_frame_translation);
					if (solver.object_list[object].edit_frame_translation) {
						for (int position = 0; position < Timeline.rampEdit[object].mPointCount[CURVE_X]; position++)
							Timeline.rampEdit[object].EditPoint(CURVE_X, position, ImVec2(Timeline.rampEdit[object].mPts[CURVE_X][position].x, solver.object_list[object].Location[0]));
						for (int position = 0; position < Timeline.rampEdit[object].mPointCount[CURVE_Y]; position++)
							Timeline.rampEdit[object].EditPoint(CURVE_Y, position, ImVec2(Timeline.rampEdit[object].mPts[CURVE_Y][position].x, solver.object_list[object].Location[1]));
						for (int position = 0; position < Timeline.rampEdit[object].mPointCount[CURVE_Z]; position++)
							Timeline.rampEdit[object].EditPoint(CURVE_Z, position, ImVec2(Timeline.rampEdit[object].mPts[CURVE_Z][position].x, solver.object_list[object].Location[2]));
					}
				}


				ImGui::SliderFloat(("scale-" + std::to_string(object)).c_str(), &solver.object_list[object].scale, 0.01f, 10.f);
			}

			if (solver.object_list[object].type == VDBOBJECT) {
				ImGui::Checkbox(("emitter-" + std::to_string(object)).c_str(), &solver.object_list[object].is_emitter);
			}



			if (solver.SIMULATE) {
				ImGui::SliderFloat(("size-" + std::to_string(object)).c_str(), &solver.object_list[object].size, 0.0, 200.0);
				solver.object_list[object].initial_size = solver.object_list[object].size;
			}
			else {
				ImGui::SliderFloat(("size-" + std::to_string(object)).c_str(), &solver.object_list[object].initial_size, 0.0, 200.0);
				
				//solver.object_list[object].initial_size = solver.object_list[object].size;
				int position = -1;
				if (Timeline.rampEdit[object].IsOnPoint(CURVE_SIZE, solver.frame, position)) {
					ImGui::Checkbox(std::string("Edit Keyframe S" + std::to_string(object)).c_str(), &solver.object_list[object].edit_frame);
					if (solver.object_list[object].edit_frame) {
						Timeline.rampEdit[object].EditPoint(CURVE_SIZE, position, ImVec2(solver.frame, solver.object_list[object].initial_size));
					}
				}
				else {
					if (ImGui::Button(std::string("Add Keyframe S" + std::to_string(object)).c_str())) {
						Timeline.rampEdit[object].AddPoint(CURVE_SIZE, ImVec2(solver.frame, solver.object_list[object].initial_size));
					}
				}
				
			}


			if (solver.object_list[object].type >= 5 && solver.object_list[object].type < 9) {
				ImGui::SliderFloat(("force strength-" + std::to_string(object)).c_str(), &solver.object_list[object].force_strength, -100.0, 100.0);
				ImGui::SameLine();
				ImGui::Checkbox(("Square-" + std::to_string(object)).c_str(), &solver.object_list[object].square);
				if (solver.object_list[object].type == FORCE_FIELD_TURBULANCE) {
					ImGui::SliderFloat(("turbulance frequence-" + std::to_string(object)).c_str(), &solver.object_list[object].velocity_frequence, 0.0, 20);
					ImGui::SliderFloat(("turbulance scale-" + std::to_string(object)).c_str(), &solver.object_list[object].scale, 0.005, 1.3);
				}
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

		// let's create the sequencer
		static bool expanded = true;//true

		ImGui::PushItemWidth(130);
		ImGui::InputInt("Frame Min", &solver.START_FRAME);
		//ImGui::SameLine();
		//ImGui::InputInt("Frame ", &currentFrame);
		ImGui::SameLine();
		ImGui::InputInt("Frame Max", &solver.END_FRAME);
		ImGui::PopItemWidth();
		
		
		if (solver.START_FRAME < 0) solver.START_FRAME = 0;
		if (solver.END_FRAME < 0) solver.END_FRAME = 0;

		//Timeline.mFrameMin = solver.START_FRAME;
		//Timeline.mFrameMax = solver.END_FRAME;

		//Sequencer(&Timeline, &solver.frame, &expanded, &selectedEntry, &solver.START_FRAME, ImSequencer::SEQUENCER_EDIT_STARTEND | ImSequencer::SEQUENCER_ADD | ImSequencer::SEQUENCER_DEL | ImSequencer::SEQUENCER_COPYPASTE | ImSequencer::SEQUENCER_CHANGE_FRAME);
		Sequencer(&Timeline, &solver.frame, &expanded, &selectedEntry, &begin_frame, ImSequencer::SEQUENCER_EDIT_STARTEND | ImSequencer::SEQUENCER_ADD | ImSequencer::SEQUENCER_DEL | ImSequencer::SEQUENCER_COPYPASTE | ImSequencer::SEQUENCER_CHANGE_FRAME);
		// add a UI to edit that particular item
		if (selectedEntry != -1 && Timeline.rampEdit.size() > 0)
		{
			
			const MySequence::MySequenceItem& item = Timeline.myItems[selectedEntry];
			ImGui::SliderFloat("Minimum", &Timeline.rampEdit[selectedEntry].mMin[0].y, 0.0, 500);
			ImGui::SliderFloat("Maksimum", &Timeline.rampEdit[selectedEntry].mMax[0].y, 0.0, 500);




			const char* items2[] = { "sine", "random_noise"};
			static const char* current_item2 = "sine";
			int current_item_id = 0;

			if (ImGui::BeginCombo("##combo1", current_item2)) // The second parameter is the label previewed before opening the combo.
			{
				for (int n = 0; n < IM_ARRAYSIZE(items2); n++)
				{
					bool is_selected = (current_item2 == items2[n]); // You can store your selection however you want, outside or inside your objects
					if (ImGui::Selectable(items2[n], is_selected)) {
						current_item2 = items2[n];
					}
					if (is_selected)
						ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
				}
				ImGui::EndCombo();
			}

			
			const char* axes[] = { "size", "x" , "y", "z" };
			static const char* axis = "size";
			int ax = 0;


			if (ImGui::BeginCombo("##combo2", axis)) {
				for (int n = 0; n < IM_ARRAYSIZE(axes); n++)
				{
					bool is_selected = (axis == axes[n]); // You can store your selection however you want, outside or inside your objects
					if (ImGui::Selectable(axes[n], is_selected)) {
						axis = axes[n];
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
					}
				}
				ImGui::EndCombo();
			}
			
			if (current_item2 == "sine") {

				ImGui::SliderInt("step_size", &sinresolution, 1, 16);
				ImGui::SliderFloat("speed", &sinspeed, 0.01f, 5);
				ImGui::SliderFloat("size", &sinsize, 0.01f, 500);
				ImGui::SliderFloat("mid", &sinmid, 0, 500);
				ImGui::SliderFloat("offset", &sinoffset, 0, 1);

				if (ImGui::Button("Add")) {
					for (int z = 0; z < IM_ARRAYSIZE(axes); z++) {
						if (axes[z] == axis) {
							ax = z;
							break;
						}
					}
					Timeline.rampEdit[selectedEntry].RemovePoints(ax);
					for (int i = 0; i < solver.END_FRAME; i += sinresolution) {
						Timeline.rampEdit[selectedEntry].AddPoint(ax, ImVec2(i, sinsize * std::sinf(sinoffset + (float)i * sinspeed) + sinmid));
					}

				}
			}
			else if (current_item2 == "random_noise") {

				ImGui::SliderInt("step_size", &sinresolution, 1, 16);
				//ImGui::SliderFloat("speed", &sinspeed, 0.01f, 5);
				ImGui::SliderFloat("size", &sinsize, 0.01f, 500);
				ImGui::SliderFloat("mid", &sinmid, 0, 500);
				ImGui::SliderFloat("SEED", &sinoffset, 0, 1);

				srand(unsigned int(sinoffset * 1000));

				if (ImGui::Button("Add")) {
					for (int z = 0; z < IM_ARRAYSIZE(axes); z++) {
						if (axes[z] == axis) {
							ax = z;
							break;
						}
					}
					Timeline.rampEdit[selectedEntry].RemovePoints(ax);
					for (int i = 0; i < solver.END_FRAME; i += sinresolution) {
						Timeline.rampEdit[selectedEntry].AddPoint(ax, ImVec2(i, sinsize * (float(rand() % 1000) / 1000.0f) + sinmid));
					}

				}
			}
			
			

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

	bool FirstUseEver = true;
	
	GLFWwindow* window;

	//initialize
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_MAXIMIZED,1); //maximize the window at start


	//create a windowed mode window
	window = glfwCreateWindow(Img_res[0], Img_res[1], FULL_NAME.c_str(), NULL, NULL);
	if (!window) {
		std::cout << "Cannot create window";
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

		/*
		float positions[] = {
			-1.0f,	    -1.0f,		0.0f, 0.0f, //lewy dó³
			 range_x,   -1.0f,		1.0f, 0.0f,
			 range_x,	 range_y,	1.0f, 1.0f,
			-1.0f,		 range_y,	0.0f, 1.0f
		};
		*/
		float positions[] = {
			-1.0f,	    range_y,		0.0f, 0.0f, //lewy dó³
			 range_x,   range_y,		1.0f, 0.0f,
			 range_x,	  -1.0f,		1.0f, 1.0f,
			-1.0f,		  -1.0f,		0.0f, 1.0f
		};

		unsigned int indices[] = {
			0,1,2,
			2,3,0
		};

		GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA))
		GLCall(glEnable(GL_BLEND))
		///////////////////////////////////////////////
		unsigned int vao;
		GLCall(glGenVertexArrays(1, &vao))//1
		GLCall(glBindVertexArray(vao))

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
		Shader shader("./VertexShader.shader");
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


		renderer.SetColor(0.1, 0.1, 0.1, 1);
		/////////////////////////////////////////////////
		//////////////IMGUI/////////////////////////////
		//Setup IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init((char*)glGetString(GL_NUM_SHADING_LANGUAGE_VERSIONS));



		/////Load theme	

		//ImVec4 clear_color = ImVec4(0->45f, 0.55f, 0.60f, 1.00f);

		float fps = 0;
		bool save_panel = true;
		bool SAVE_FILE_TAB = false;
		bool OPEN_FILE_TAB = false;
		float progress = 0.0f;
		bool helper_window = true;
		bool confirm_button = false;
		//std::thread* sim;
		/////////////////////////////////////////////////
		//TImeline

		//Timeline.myItems.push_back(MySequence::MySequenceItem{ 0, 10, 30, false });
		//Timeline.myItems.push_back(MySequence::MySequenceItem{ 1, 20, 30, true });

		solver.frame = 0;
		solver.state->time_step = solver.speed * 0.1;
		solver.DONE_FRAME = true;
		frame = solver.frame;

		//UpdateSolver();
		UpdateTimeline();
		///////////////////////////////////////////////////
		while (!glfwWindowShouldClose(window)) {
			//clock_t startTime = clock();
			if (solver.LOCK) continue;

			frame = solver.frame;

			if (frame == solver.frame && solver.DONE_FRAME) {
				UpdateAnimation();
				frame++;
			}
			//////////////////
			//solver.Simulation_Frame();

			if (threads.size() == 0) { //0
				threads.push_back(std::thread([&]() {
					while (true) {

						if (solver.frame == frame - 1) {
							clock_t startTime = clock();
							solver.DONE_FRAME = false;

							solver.Simulation_Frame();

							solver.DONE_FRAME = true;
							fps = 1.0 / ((double(clock() - startTime) / (double)CLOCKS_PER_SEC));
						}


						if (solver.THIS_IS_THE_END) {
							std::cout << "I'm breaking out" << std::endl;
							break;
						}
					}
				}));
			}
			/*
			*/
			 
			 
			//////////////////
			//Texture texture("output/R" + pad_number(frame) + ".bmp");
			//texture.UpdateTexture("output/R" + pad_number(solver.frame) + ".bmp");
			if (!solver.writing) {
				//texture.UpdateTexture("./output/temp.bmp");
				texture.UpdateTexture(solver.img,solver.img_d.x,solver.img_d.y);
				texture.Bind(/*slot*/0);
			}
			//shader.SetUniform1i("u_Texture", /*slot*/0);
			//////////////////
			renderer.Clear();
			shader.Bind();
			/////////////////////////////
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();


			//std::thread GUI_THR( RenderGUI ,std::ref(SAVE_FILE_TAB), std::ref(OPEN_FILE_TAB), std::ref(fps), std::ref(progress), std::ref(save_panel));
			RenderGUI(dpi, SAVE_FILE_TAB, OPEN_FILE_TAB, fps, progress, save_panel, helper_window, confirm_button, Image);

			/////
			//Viewer window


			ImGui::SetWindowFontScale(InterfaceScale);
			ImGui::SetNextWindowSize(ImVec2(solver.img_d.x+10, solver.img_d.y+10), FirstUseEver);
			ImGui::Begin("Viewport");
			{
				if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows))
					ALLOW_SCROLL = true;
				else
					ALLOW_SCROLL = false;
				// Using a Child allow to fill all the space of the window.
				// It also alows customization
				ImGui::BeginChild("View");
				// Get the size of the child (i.e. the whole draw size of the windows).
				//ImVec2 wsize(solver.img_d.x,solver.img_d.y);
				ImVec2 wsize = ImGui::GetWindowSize();
				ImGui::Image((ImTextureID)1/*texture id*/, wsize, ImVec2(0, 0), ImVec2(1, 1)); //noflip


				// Because I use the texture from OpenGL, I need to invert the V from the UV.
				//ImGui::Image((ImTextureID)1/*texture id*/, wsize, ImVec2(0, 1), ImVec2(1, 0));
				ImGui::EndChild();
			}
			ImGui::End();

			/////








			//renderer.Draw(va, ib, shader);
			//New Frame//////////////////


			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			

			GLCall(glfwSwapBuffers(window))
			GLCall(glfwPollEvents())

				//fps = 1.0 / ((double(clock() - startTime) / (double)CLOCKS_PER_SEC));
			if (FirstUseEver)
				FirstUseEver = false;
		}
		solver.THIS_IS_THE_END = true;
		std::cout << "Threads join together" << std::endl;
		for (auto& thread : threads)
			thread.join();
		threads.clear();
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
			rot -= 0.1f;
		}
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			solver.setRotation(solver.getRotation() + 0.1f);
			rot += 0.1f;
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

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		solver.SIMULATE = solver.SIMULATE ? false : true;
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
	if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || ALLOW_SCROLL) {
		solver.setCamera(solver.getCamera().x, solver.getCamera().y,
			solver.getCamera().z + 2.5f * yOffset);
	}
}