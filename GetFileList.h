#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "cutil_math.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <experimental/filesystem>

#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>

#ifndef FILE_LISTT
#define FILE_LISTT

std::vector<std::string> get_file_list(std::string directory) {
	std::vector<std::string> list;
	for (const auto& entry : std::experimental::filesystem::directory_iterator(directory)) {
		//std::cout << entry.path() << std::endl;
		list.push_back(entry.path().string());
	}
	return list;
}
#endif // !FILE_LISTT