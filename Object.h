#ifndef __OBJECT__
#define __OBJECT__

#include <string>
#include <iostream>
#include "cutil_math.h"
#include <cuda_runtime.h>
#include "double_buffer.cpp"

#define EMITTER 1
#define SMOKE 2
#define VDBOBJECT 3
#define VDBSINGLE 4
#define MS_TO_SEC 0.001






class Solver;



class OBJECT {
public:
	friend class Solver;
	//Contructors
	//OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f, float3 location = make_float3(0.0,0.0,0.0), int number = -1);
	OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f,float Temp = 5.0f, float Density = 0.9f, float3 location = make_float3(0.0, 0.0, 0.0), int number = -1);
	//SETTERS-GETTERS
	std::string get_type();
	void set_type(std::string type = "emitter");
	float get_size();
	void set_size(float size = 1.0f);
	float get_initial_velocity();
	void set_initial_velocity(float velocity = 1.0f);
	float get_velocity_frequence();
	void set_velocity_frequence(float frequence = 0.9f);
	float get_impulseTemp();
	void set_impulseTemp(float temp = 5.0f);
	float get_impulseDensity();
	void set_impulseDensity(float density = 0.6f);
	float3 get_location();
	void set_location(float x = 0, float y = 0, float z = 0);
	void set_location(float3 location);
    void load_density_grid(GRID3D obj,float temp = 1.0);
    GRID3D get_density_grid();
    float get_initial_temp();
    void cudaFree() { vdb_object.freeCuda(); }
	void free() {
		vdb_object.free();
	}
	std::string get_object();
	std::string get_name();
	void set_name(std::string);

	void UpdateLocation() {
		this->location.x = Location[0];
		this->location.y = Location[1];
		this->location.z = Location[2];
	}

	bool selected;
	float Location[3];
	float size;
private:
	int type;
	float initial_velocity;
	float velocity_frequence;
	float impulseTemp;
	float impulseDensity;
	float3 location;
    GRID3D vdb_object;
    float initial_temperature;
	std::string name;
};


#endif // !OBJECT

