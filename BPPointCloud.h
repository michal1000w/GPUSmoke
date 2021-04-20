#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "cutil_math.h"


#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

template <class Container>
void split(const std::string& str, Container& cont, char delim = ' ') {
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, delim)) {
		cont.push_back(token);
	}
}



class Particle {
public:
	float3 position;
	float3 velocity;
	int ID;

	Particle() {
		position = make_float3(0);
		velocity = make_float3(0);
		ID = 0;
	}

	void print() {
		std::cout << "ID: " << ID << "   ;   Position: (" << position.x << "," << position.y << ","
			<< position.z << ")   ;   Velocity: (" << velocity.x << "," << velocity.y << "," << velocity.z << ")\n";
	}
};


class BPReader {
public:
	std::string filename;
	std::vector<Particle> particles;

	BPReader(std::string filename) {
		this->filename = filename;
	}

	void ReadData() {
		std::ifstream inFile;

		inFile.open(this->filename);
		if (!inFile) {
			std::cout << "Unable to open file" << std::endl;
			exit(1); // terminate with error
		}

		std::string line = "";

		while (inFile >> line) {
			std::vector<std::string> elements;
			split(line, elements, ';');

			std::vector<std::string> positions;
			split(elements[1], positions, ',');

			std::vector<std::string> velocities;
			split(elements[2], velocities, ',');

			Particle particle;

			try {
				particle.ID = stoi(elements[0]);
				particle.position = make_float3(stof(positions[0]), stof(positions[2]), stof(positions[1]));
				particle.velocity = make_float3(stof(velocities[0]), stof(velocities[2]), stof(velocities[1]));
			}
			catch (std::exception e) {
				std::cout << "Data parsing error!!!!" << std::endl;
				exit(1);
			}
			particles.push_back(particle);
		}

		inFile.close();
	}

	void ReadData1() {
		std::ifstream inFile;


		inFile.open(this->filename, std::ios::binary | std::ios::in);
		if (!inFile) {
			std::cout << "Unable to open file" << std::endl;
			exit(1); // terminate with error
		}

		inFile.seekg(0, std::ios::end);
		int length = inFile.tellg();
		inFile.seekg(0, std::ios::beg);

		//std::cout << length << std::endl;


		char magic[8] = "";
		inFile.read(magic, sizeof(magic));
		std::string Magic = "";
		for (int j = 0; j < 8; j++)
			Magic += magic[j];

		if (Magic != "BPHYSICS") {
			std::cout << "Error loading: " << filename << std::endl;
			std::cout << "   ->   " << Magic << std::endl;
			exit(1);
		}


		int flavor[3];
		inFile.read((char*)flavor, sizeof(flavor));
		//std::cout << "Flavour: " << flavor[0] << "   Count: " << flavor[1] << "   Something: " << flavor[2] << std::endl;

		length -= sizeof(magic) + sizeof(flavor);

		if (flavor[0] == 1) {
			std::cout << "Point Cache" << std::endl;


			while (true) {
				int ID[1];
				float numbers[6];

				length -= 7 * sizeof(float);
				if (length < 0) {
					std::cout << "Too short!!!" << std::endl;
					break;
				}
				//std::cout << length << std::endl;

				inFile.read((char*)ID, sizeof(ID));
				inFile.read((char*)numbers, sizeof(numbers));
				
				Particle particle;
				particle.ID = ID[0];
				particle.position = make_float3(numbers[0], numbers[2], numbers[1]);
				particle.velocity = make_float3(numbers[3], numbers[5], numbers[4]);
				this->particles.push_back(particle);
			}

			inFile.close();

		}
		else {
			std::cout << "Not a pointache!!!!" << std::endl;
			exit(1);
		}



		//std::cout << "No error" << std::endl;
		//PrintParticles();
		//exit(1);
	}

	void PrintParticles() {
		for (int i = 0; i < particles.size(); i++) {
			particles[i].print();
		}
	}
};