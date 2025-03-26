#include "image_fusion/bag_time_reader.hpp"

#include <iostream>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_storage/storage_options.hpp> // Include correct header for StorageOptions

BagTimeReader::BagTimeReader(const std::string &bag_path)
    : bag_path_(bag_path)
{
}

void BagTimeReader::readMessages()
{
    // Use the correct namespace for StorageOptions.
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = bag_path_;
    storage_options.storage_id = "sqlite3";

    // Set up converter options.
    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";

    rosbag2_cpp::Reader reader;
    reader.open(storage_options, converter_options);

    while (reader.has_next())
    {
        auto bag_message = reader.read_next();
        // bag_message->time_stamp holds the original bag time (in nanoseconds).
        std::cout << "Topic: " << bag_message->topic_name
                  << " | Bag Timestamp: " << bag_message->time_stamp << std::endl;

        // Read the next bag message.
        auto bag_message = reader.read_next();

        // Create a SerializedMessage and copy the raw buffer from the bag message.
        // Here we assume bag_message->serialized_data is a pointer to an rcutils_uint8_array_t.
        rclcpp::SerializedMessage serialized_msg;
        serialized_msg.reserve(bag_message->serialized_data->buffer_length);
        std::memcpy(
            serialized_msg.get_rcl_serialized_message().buffer,
            bag_message->serialized_data->buffer,
            bag_message->serialized_data->buffer_length);
        serialized_msg.get_rcl_serialized_message().buffer_length = bag_message->serialized_data->buffer_length;

        // Now deserialize the message into a std_msgs::msg::String.
        std_msgs::msg::String str_msg;
        rclcpp::Serialization<std_msgs::msg::String> serializer;
        serializer.deserialize_message(&serialized_msg, &str_msg);

        std::cout << "Received string message: " << str_msg.data << std::endl;
    }
}

// Main function to run the reader.
#include <cstdlib>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <bag_file_path>" << std::endl;
        return 1;
    }
    std::string bag_path = argv[1];

    BagTimeReader reader(bag_path);
    reader.readMessages();
    return 0;
}
