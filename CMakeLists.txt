cmake_minimum_required(VERSION 3.8)
project(image_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install Python scripts as executables.
# Change 'scripts/my_script.py' to the relative path(s) of your script(s)
install(PROGRAMS
  scripts/bag_to_images.py
#   scripts/another_script.py  # if you have more than one script
  DESTINATION lib/${PROJECT_NAME}
)

# Install calibration_data directory to the package share folder.
install(
  DIRECTORY calibration_data/
  DESTINATION share/${PROJECT_NAME}/calibration_data
)

ament_package()
