cmake_minimum_required(VERSION 3.8)
project(image_fusion)

# ROS 2 build system
find_package(ament_cmake REQUIRED)

# Dependencies
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Create extrinsics executable
add_executable(extrinsics src/extrinsics.cpp)
ament_target_dependencies(extrinsics rclcpp)
target_link_libraries(extrinsics
  ${OpenCV_LIBS}
  yaml-cpp
)

# Install Python scripts to the package share folder.
install(PROGRAMS
  scripts/bag_to_images.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install calibration_data directory to the package share folder.
install(
  DIRECTORY calibration_data/
  DESTINATION share/${PROJECT_NAME}/calibration_data
)

# Install C++ executables to the package share folder.
install(TARGETS extrinsics
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
