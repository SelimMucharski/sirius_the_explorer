cmake_minimum_required(VERSION 3.8)
project(sirius_teleop)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python module (gamepad mapping logic shared by the node and tests)
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS scripts/joy_teleop.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_mapping test/test_mapping.py)
endif()

ament_package()
