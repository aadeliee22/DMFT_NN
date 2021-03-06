# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build

# Include any dependencies generated for this target.
include examples/CMakeFiles/pi_block_mpi.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/pi_block_mpi.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/pi_block_mpi.dir/flags.make

examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o: examples/CMakeFiles/pi_block_mpi.dir/flags.make
examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o: /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/examples/pi_block_mpi.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o"
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o -c /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/examples/pi_block_mpi.cc

examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.i"
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/examples/pi_block_mpi.cc > CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.i

examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.s"
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/examples/pi_block_mpi.cc -o CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.s

# Object files for target pi_block_mpi
pi_block_mpi_OBJECTS = \
"CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o"

# External object files for target pi_block_mpi
pi_block_mpi_EXTERNAL_OBJECTS =

examples/pi_block_mpi: examples/CMakeFiles/pi_block_mpi.dir/pi_block_mpi.cc.o
examples/pi_block_mpi: examples/CMakeFiles/pi_block_mpi.dir/build.make
examples/pi_block_mpi: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
examples/pi_block_mpi: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
examples/pi_block_mpi: examples/CMakeFiles/pi_block_mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pi_block_mpi"
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pi_block_mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/pi_block_mpi.dir/build: examples/pi_block_mpi

.PHONY : examples/CMakeFiles/pi_block_mpi.dir/build

examples/CMakeFiles/pi_block_mpi.dir/clean:
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples && $(CMAKE_COMMAND) -P CMakeFiles/pi_block_mpi.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/pi_block_mpi.dir/clean

examples/CMakeFiles/pi_block_mpi.dir/depend:
	cd /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4 /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/examples /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/examples/CMakeFiles/pi_block_mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/pi_block_mpi.dir/depend

