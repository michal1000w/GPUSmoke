#ifndef __OBJECT__
#define __OBJECT__

#include <string>
#include <iostream>
#include "cutil_math.h"
#include <cuda_runtime.h>
#include "double_buffer.cpp"

#define PARTICLE 0
#define EMITTER 1
#define SMOKE -1 //depricated
#define EXPLOSION 2
#define VDBOBJECT 3
#define VDBSINGLE 4
#define FORCE_FIELD_FORCE 5
#define FORCE_FIELD_POWER 6
#define FORCE_FIELD_TURBULANCE 7
#define FORCE_FIELD_WIND 8
#define COLLISION_SPHERE 9
#define MS_TO_SEC 0.001






class Solver;



class OBJECT {
public:
	friend class Solver;
	//Contructors
	//OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f, float3 location = make_float3(0.0,0.0,0.0), int number = -1);
	OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f,float Temp = 5.0f, float Density = 0.9f, float3 location = make_float3(0.0, 0.0, 0.0), int number = -1, int deviceCount = 1);
	OBJECT(std::string type, float size, std::vector<std::vector<float3>> velocities, std::vector<std::vector<float3>> positions, float3 location, float Temp = 5.0f, float Density = 0.9f, int number = -1, int deviceCount = 1);
	OBJECT(std::string type, float size, float3 location, float Temp = 5.0f, float Density = 0.9f, int number = -1, int deviceCount = 1);
	OBJECT(OBJECT &obj, int number, int deviceCount);
	~OBJECT() {
	}
	//SETTERS-GETTERS
	std::string get_type();
	std::string get_type2();
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
    void load_density_grid(GRID3D obj,float temp,int deviceIndex);
    GRID3D get_density_grid();
    float get_initial_temp();
    void cudaFree() { vdb_object.freeCuda(); }
	void free() { vdb_object.free(); }
	std::string get_object();
	std::string get_name();
	void set_name(std::string);
	void set_force_strength(float strength) { force_strength = strength; }
	float get_force_strength() const { return force_strength; }
	void reset();
	void update();
	void UpdateLocation() {this->location.x = Location[0];this->location.y = Location[1];this->location.z = Location[2];}
	void LoadParticles();


	bool selected;
	float Location[3];
	float size;
	float initial_size;
	float force_strength;
	bool square = false;
	int type;
	float initial_velocity;
	float force_direction[3] = { 1.0,0,0 };

	float velocity_frequence;
	bool vel_freq_mov = true;
	float set_vel_freq;
	float max_vel_freq = 20.0;
	float vel_freq_step = 0.4;
	float impulseTemp;
	int frame_range_min = 0;
	int frame_range_max = 30;
	float3 previous_location;
	float previous_size;
	bool edit_frame = false;
	bool edit_frame_translation = false;
	float3 velocity;

	std::vector<std::vector<float3>> velocities;
	std::vector<std::vector<float3>> positions;
	float scale = 1.0f;
	std::string particle_filepath = "";


    GRID3D vdb_object;
private:
	float impulseDensity;
	float3 location;
    float initial_temperature;
	std::string name;

};


#endif // !OBJECT

