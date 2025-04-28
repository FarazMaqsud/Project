#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/serialization.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mcap_publisher");

    // Set up publishers for /camera/image/raw and /camera/image/edge_raw topics
    auto publisher_raw = node->create_publisher<sensor_msgs::msg::CompressedImage>("/camera/image/raw", 10);
    auto publisher_edge = node->create_publisher<sensor_msgs::msg::CompressedImage>("/camera/image/edge_raw", 10);

    // Configure rosbag2 storage and converter options
    rosbag2_storage::StorageOptions storage_options{};
    storage_options.uri = "/home/farazawan/object/rosbag2_file1/rosbag2_2024_11_05-14_17_54_0.mcap";
    storage_options.storage_id = "mcap";  // Set storage type to "mcap"

    rosbag2_cpp::ConverterOptions converter_options{};
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";

    // Open the rosbag file with SequentialReader
    rosbag2_cpp::readers::SequentialReader reader;
    reader.open(storage_options, converter_options);

    rclcpp::Serialization<sensor_msgs::msg::CompressedImage> serialization;
    sensor_msgs::msg::CompressedImage image_msg;

    // Read and publish each message
    while (rclcpp::ok() && reader.has_next()) {
        auto serialized_message = reader.read_next();
        
        // Deserialize and publish based on topic
        rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
        serialization.deserialize_message(&extracted_serialized_msg, &image_msg);

        if (serialized_message->topic_name == "/camera/image/raw") {
            publisher_raw->publish(image_msg);
            RCLCPP_INFO(node->get_logger(), "Published compressed image on /camera/image/raw");
        } else if (serialized_message->topic_name == "/camera/image/edge_raw") {
            publisher_edge->publish(image_msg);
            RCLCPP_INFO(node->get_logger(), "Published compressed image on /camera/image/edge_raw");
        }

        // Optional: Control playback rate
        rclcpp::sleep_for(std::chrono::milliseconds(100));  // Adjust delay as needed
    }

    rclcpp::shutdown();
    return 0;
}
