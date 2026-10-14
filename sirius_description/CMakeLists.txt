cmake_minimum_required(VERSION 3.8)
project(sirius_description)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(DIRECTORY
  launch
  urdf
  meshes
  robots
  config
  DESTINATION share/${PROJECT_NAME}
)

# Exports the URDF to the JSON + STL model consumed by the Unity simulation (unity_sim).
install(PROGRAMS scripts/urdf_to_unity.py
  DESTINATION lib/${PROJECT_NAME})

ament_auto_package()