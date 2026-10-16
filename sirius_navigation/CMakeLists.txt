cmake_minimum_required(VERSION 3.8)
project(sirius_navigation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(PROGRAMS
  scripts/obstacle_cloud.py
  scripts/gnss_odom_correction.py
  DESTINATION lib/${PROJECT_NAME}
)

# Python module (obstacle_cloud domain logic shared by scripts and tests)
ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_obstacle_cloud test/test_obstacle_cloud.py)
  ament_add_pytest_test(test_gnss_correction test/test_gnss_correction.py)
endif()

ament_package()
