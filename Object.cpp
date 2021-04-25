#include "Object.h"


#include "BPPointCloud.h"
#include "GetFileList.h"
#include "ObjIO.h"

////////////////////////CONSTRUCTORS//////////////////////////////////
/*
OBJECT::OBJECT(std::string type, float size, float initial_velocity, float velocity_frequence, float3 location, int number) {
	this->set_type(type);
	this->size = size;
	this->initial_velocity = initial_velocity;
	this->set_velocity_frequence(velocity_frequence);
	//this->velocity_frequence = velocity_frequence;
	this->location = location;
	this->set_impulseDensity();
	this->set_impulseTemp();
	this->name = get_object();
	if (number != -1)
		this->name += std::to_string(number);
	this->selected = false;
	this->Location[0] = location.x; this->Location[1] = location.y; this->Location[2] = location.z;
}
*/
void OBJECT::LoadObjects(int3 resolution, int deviceCount, int deviceIndex) {
	auto filelist = get_file_list(this->particle_filepath);

	//this->vdb_object = new GRID3D(resolution, deviceCount, deviceIndex);

	for (int i = 0; i < filelist.size(); i++) {
		std::vector<std::string> elements;
		split(filelist[i], elements, '.');
		if (elements.at(elements.size() - 1) != "obj") continue;

		std::cout << i << "/" << filelist.size() << "\n";
		int table_size = 0;
		unsigned int* current = LoadAndVoxelizeCompressed(resolution, filelist[i], 0.67, deviceIndex, true, table_size); //false
		collisions.push_back(current); //wyciek pamieci
		std::cout << "Table size: " << table_size << std::endl;
	}


	frame_range_min = 0;
	if (filelist.size() > 2)
		frame_range_max = filelist.size();
	else
		frame_range_max = 800;
}

void OBJECT::LoadParticles() {
	auto filelist = get_file_list(this->particle_filepath);
	for (int i = 0; i < filelist.size(); i++) {
		std::cout << i << "/" << filelist.size() << "\n";
		BPReader bpr(filelist[i]);
		bpr.ReadData1();
		std::vector<float3> poss;
		std::vector<float3> vell;
		for (int j = 0; j < bpr.particles.size(); j++) {
			poss.push_back(bpr.particles[j].position);
			vell.push_back(bpr.particles[j].velocity);
		}
		positions.push_back(poss);
		velocities.push_back(vell);
	}
	frame_range_min = 0;
	frame_range_max = filelist.size();
}

OBJECT::OBJECT(std::string type, float size, float initial_velocity, float velocity_frequence, float Temp, float Density, float3 location, int number, int deviceCount) {
	this->set_type(type);
	this->size = size;
	this->initial_size = size;
	this->initial_velocity = initial_velocity;
	this->set_velocity_frequence(velocity_frequence);
	this->location = location;
	this->set_impulseDensity(Density);
	this->set_impulseTemp(Temp);
	this->name = get_object();
	if (number != -1)
		this->name += std::to_string(number);
	this->selected = false;
	this->Location[0] = location.x; this->Location[1] = location.y; this->Location[2] = location.z;
	this->force_strength = 0.0f;
	this->vdb_object = GRID3D(deviceCount);

	this->previous_location = location;
	this->previous_size = size;
}



OBJECT::OBJECT(std::string type, float size, std::vector<std::vector<float3>> velocities, std::vector<std::vector<float3>> positions, float3 location, float Temp, float Density, int number, int deviceCount) {
	this->set_type(type);
	this->size = size;
	this->initial_size = size;
	this->initial_velocity = 1;
	this->set_velocity_frequence(1);
	this->location = location;
	this->set_impulseDensity(Density);
	this->set_impulseTemp(Temp);
	this->name = get_object();
	if (number != -1)
		this->name += std::to_string(number);
	this->selected = false;
	this->Location[0] = location.x; this->Location[1] = location.y; this->Location[2] = location.z;
	this->force_strength = 0.0f;
	this->vdb_object = GRID3D(deviceCount);

	this->previous_location = location;
	this->previous_size = size;

	this->velocities = velocities;
	this->positions = positions;
}
OBJECT::OBJECT(std::string type, float size, float3 location, float Temp, float Density, int number, int deviceCount) {
	this->set_type(type);
	this->size = size;
	this->initial_size = size;
	this->initial_velocity = 1;
	this->set_velocity_frequence(1);
	this->location = location;
	this->set_impulseDensity(Density);
	this->set_impulseTemp(Temp);
	this->name = get_object();
	if (number != -1)
		this->name += std::to_string(number);
	this->selected = false;
	this->Location[0] = location.x; this->Location[1] = location.y; this->Location[2] = location.z;
	this->force_strength = 0.0f;
	this->vdb_object = GRID3D(deviceCount);

	this->previous_location = location;
	this->previous_size = size;
}







OBJECT::OBJECT(OBJECT obj, int number, int deviceCount) {
	this->type = obj.type;
	this->size = size;
	this->initial_size = size;
	this->initial_velocity = initial_velocity;
	this->set_velocity_frequence(velocity_frequence);
	this->location = location;
	this->set_impulseDensity(obj.impulseDensity);
	this->set_impulseTemp(obj.impulseTemp);
	this->name = get_object() + "_COPY";
	this->name += std::to_string(number);
	this->selected = false;
	this->Location[0] = location.x; this->Location[1] = location.y; this->Location[2] = location.z;
	this->force_strength = 0.0f;
	this->vdb_object = GRID3D(deviceCount);
	this->frame_range_max = obj.frame_range_max;
	this->frame_range_min = obj.frame_range_min;

	this->previous_location = location;
	this->previous_size = size;

	this->positions = obj.positions;
	this->velocities = obj.velocities;

	this->particle_filepath = obj.particle_filepath;
}

void OBJECT::reset() {
	this->size = this->initial_size;
}

void OBJECT::update() {
	this->previous_location = this->location;
	this->previous_size = this->size;
}


////////////////////////GETTERS-SETTERS///////////////////////////////
std::string OBJECT::get_type() {
	if (this->type == EMITTER)
		return "emitter";
	else if (this->type == SMOKE)
		return "smoke";
	else if (this->type == VDBOBJECT)
		return "object";
	else if (this->type == VDBSINGLE)
		return "vdbs";
	else if (this->type == FORCE_FIELD_FORCE)
		return "fff";
	else if (this->type == FORCE_FIELD_POWER)
		return "ffp";
	else if (this->type == FORCE_FIELD_TURBULANCE)
		return "fft";
	else if (this->type == FORCE_FIELD_WIND)
		return "ffw";
	else if (this->type == COLLISION_SPHERE)
		return "cols";
	else if (this->type == EXPLOSION)
		return "explosion";
	else if (this->type == PARTICLE)
		return "particle";
}

std::string OBJECT::get_type2() {
	if (this->type == EMITTER)
		return "emitter";
	else if (this->type == SMOKE)
		return "smoke";
	else if (this->type == VDBOBJECT)
		return "object";
	else if (this->type == VDBSINGLE)
		return "vdbs";
	else if (this->type == FORCE_FIELD_FORCE)
		return "force";
	else if (this->type == FORCE_FIELD_POWER)
		return "power";
	else if (this->type == FORCE_FIELD_TURBULANCE)
		return "turbulance";
	else if (this->type == FORCE_FIELD_WIND)
		return "wind";
	else if (this->type == COLLISION_SPHERE)
		return "sphere";
	else if (this->type == EXPLOSION)
		return "explosion";
	else if (this->type == PARTICLE)
		return "particle";
}

void OBJECT::set_type(std::string type) {
	if (type == "EMITTER" || type == "emitter")
		this->type = EMITTER;
	else if (type == "SMOKE" || type == "smoke")
		this->type = SMOKE;
	else if (type == "VDB" || type == "vdb" || type == "object")
		this->type = VDBOBJECT;
	else if (type == "VDBSINGLE" || type == "vdbsingle" || type == "vdbs" || type == "object")
		this->type = VDBSINGLE;
	else if (type == "FFieldFORCE" || type == "force" || type == "fff")
		this->type = FORCE_FIELD_FORCE;
	else if (type == "FFieldPOWER" || type == "power" || type == "ffp")
		this->type = FORCE_FIELD_POWER;
	else if (type == "FFieldTURBULANCE" || type == "turbulance" || type == "fft")
		this->type = FORCE_FIELD_TURBULANCE;
	else if (type == "FFieldWIND" || type == "wind" || type == "ffw")
		this->type = FORCE_FIELD_WIND;
	else if (type == "CollisionSPHERE" || type == "sphere" || type == "cols")
		this->type = COLLISION_SPHERE;
	else if (type == "explosion" || type == "Explosion" || type == "EXPLOSION")
		this->type = EXPLOSION;
	else if (type == "particle" || type == "prt")
		this->type = PARTICLE;
	else {
		std::cout << "Type: " << type << " not known!!!" << std::endl;
		exit(1);
	}
}

std::string OBJECT::get_object() {
	std::string namee = "";
	if (this->type < 5) namee += "OBJ-";
	else if (this->type >= 5 && this->type < 9) namee += "FF-";
	else if (this->type >= 9) namee += "COL-";
	namee += get_type();
	return namee;
}









float OBJECT::get_size() {
	return this->size;
}

float OBJECT::get_initial_velocity() {
	return this->initial_velocity;
}

float OBJECT::get_velocity_frequence() {
	return this->velocity_frequence;
}

float OBJECT::get_impulseTemp() {
	return this->impulseTemp;
}
float OBJECT::get_impulseDensity() {
	return this->impulseDensity;
}

float3 OBJECT::get_location() {
	return this->location;
}

//////////////////




void OBJECT::set_size(float size) {
	this->size = size;
}

void OBJECT::set_initial_velocity(float velocity) {
	this->initial_velocity = velocity;
}

void OBJECT::set_velocity_frequence(float frequence) {
	if (frequence <= 8 && frequence >= 0)
		this->velocity_frequence = frequence;
}

void OBJECT::set_impulseTemp(float temp) {
	this->impulseTemp = temp;
}
void OBJECT::set_impulseDensity(float density) {
	this->impulseDensity = density;
}

void OBJECT::set_location(float x, float y, float z) {
	this->location.x = x;
	this->location.y = y;
	this->location.z = z;
}

void OBJECT::set_location(float3 location) {
	this->location.x = location.x;
	this->location.y = location.y;
	this->location.z = location.z;
}

void OBJECT::load_density_grid(GRID3D obj, float temp, int deviceIndex) {
	this->initial_temperature = temp;
	this->vdb_object = obj;
	this->vdb_object.copyToDevice(deviceIndex);
}

GRID3D OBJECT::get_density_grid() {
	return this->vdb_object;
}

float OBJECT::get_initial_temp() {
	return this->initial_temperature;
}



std::string OBJECT::get_name() {
	return name;
}

void OBJECT::set_name(std::string name) {
	this->name = name;
}