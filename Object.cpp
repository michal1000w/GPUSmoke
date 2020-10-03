#include "Object.h"


////////////////////////CONSTRUCTORS//////////////////////////////////
OBJECT::OBJECT(std::string type, float size, float initial_velocity, float velocity_frequence, float3 location) {
	this->set_type(type);
	this->size = size;
	this->initial_velocity = initial_velocity;
	this->set_velocity_frequence(velocity_frequence);
	//this->velocity_frequence = velocity_frequence;
	this->location = location;
	this->set_impulseDensity();
	this->set_impulseTemp();
}

OBJECT::OBJECT(std::string type, float size, float initial_velocity, float velocity_frequence, float Temp, float Density, float3 location) {
	this->set_type(type);
	this->size = size;
	this->initial_velocity = initial_velocity;
	this->set_velocity_frequence(velocity_frequence);
	this->location = location;
	this->set_impulseDensity(Density);
	this->set_impulseTemp(Temp);
}


////////////////////////GETTERS-SETTERS///////////////////////////////
std::string OBJECT::get_type() {
	if (this->type == EMMITER)
		return "emmiter";
	else if (this->type == SMOKE)
		return "smoke";
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


void OBJECT::set_type(std::string type) {
	if (type == "EMMITER" || type == "emmiter")
		this->type = EMMITER;
	else if (type == "SMOKE" || type == "smoke")
		this->type = SMOKE;
	else {
		std::cout << "Type: " << type << " not known!!!" << std::endl;
		exit(1);
	}
}

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