# Install script for directory: /home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/trng/libtrng4.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so.4.22"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so.22"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/trng/libtrng4.so.4.22"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/trng/libtrng4.so.22"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so.4.22"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so.22"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4.build/trng/libtrng4.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtrng4.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/trng" TYPE FILE FILES
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/bernoulli_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/beta_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/binomial_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/cauchy_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/chi_square_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/constants.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/correlated_normal_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/cuda.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/discrete_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/exponential_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/extreme_value_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/fast_discrete_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/gamma_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/generate_canonical.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/geometric_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/hypergeometric_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/int_math.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/int_types.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lagfib2plus.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lagfib2xor.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lagfib4plus.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lagfib4xor.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lcg64.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lcg64_shift.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/limits.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/logistic_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/lognormal_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/math.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/maxwell_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/minstd.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg2.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg3.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg3s.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg4.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg5.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mrg5s.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mt19937_64.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/mt19937.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/negative_binomial_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/normal_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/pareto_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/poisson_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/powerlaw_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/rayleigh_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/snedecor_f_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/special_functions.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/student_t_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/tent_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/truncated_normal_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/twosided_exponential_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/uniform01_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/uniform_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/uniform_int_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/uniformxx.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/utility.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/weibull_dist.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn2.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn3.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn3s.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn4.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn5.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/yarn5s.hpp"
    "/home/hyejin/Desktop/my_git/DMFT_VAE/1_QMC/Hirsch-Fye-QMC/trng4/trng/zero_truncated_poisson_dist.hpp"
    )
endif()

