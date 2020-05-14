#ifndef __OBJECT
#define __OBJECT

#include <string>
#include <iostream>
#include "cutil_math.h"

#define EMMITER 1
#define SMOKE 2

class OBJECT {
public:
	//Contructors
	OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f, float3 location = make_float3(0.0,0.0,0.0));
	//SETTERS-GETTERS
	std::string get_type();
	void set_type(std::string type = "emmiter");
	float get_size();
	void set_size(float size = 1.0f);
	float get_initial_velocity();
	void set_initial_velocity(float velocity = 1.0f);
	float get_velocity_frequence();
	void set_velocity_frequence(float frequence = 100.0f);
	float3 get_location();
	void set_location(float x = 0, float y = 0, float z = 0);
	void set_location(float3 location);
	

private:
	int type;
	float size;
	float initial_velocity;
	float velocity_frequence;
	float3 location;
};


#endif // !OBJECT

