#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <fstream>
#include <unordered_map>

std::vector<std::string> get_output_layers(const cv::dnn::Net& net) {
    std::vector<std::string> layer_names = net.getLayerNames();
    std::vector<std::string> output_layers;
    for (int idx : net.getUnconnectedOutLayers()) {
        output_layers.push_back(layer_names[idx - 1]);
    }
    return output_layers;
}

class ObjectDetectionNode : public rclcpp::Node {
public:
    ObjectDetectionNode() : Node("object_detection_node") {
        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/image/raw", 10,
            std::bind(&ObjectDetectionNode::image_callback, this, std::placeholders::_1));

        subscription_edge_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/camera/image/edge_raw", 10,
            std::bind(&ObjectDetectionNode::image_callback_edge, this, std::placeholders::_1));

        // Load YOLO network
        net_ = cv::dnn::readNet("/home/farazawan/demo/yolov4.weights", "/home/farazawan/demo/yolov4.cfg"); // tested with yolov3 & v4 weights and cfg. 
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // Load COCO class labels
        std::ifstream ifs("/home/farazawan/demo/coco.names");
        std::string line;
        while (getline(ifs, line)) {
            class_labels_.push_back(line);
        }

        RCLCPP_INFO(this->get_logger(), "YOLOv4 model loaded successfully.");
    }

private:
    struct TrackedObject {
        cv::Rect box;
        int class_id;
        float confidence;
        int id;
    };

    cv::Mat frame_raw_;
    cv::Mat frame_edge_;
    std::vector<TrackedObject> objects_raw_;
    std::vector<TrackedObject> objects_edge_;

    void image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        frame_raw_ = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (frame_raw_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to decode compressed image from topic 1.");
            return;
        }
        process_frame(frame_raw_, objects_raw_);
        display_combined_frames();
    }

    void image_callback_edge(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        frame_edge_ = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (frame_edge_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to decode compressed image from topic 2.");
            return;
        }
        process_frame(frame_edge_, objects_edge_);
        display_combined_frames();
    }

    void process_frame(cv::Mat& frame, std::vector<TrackedObject>& detected_objects) {
        detected_objects.clear();
        std::vector<cv::Rect> boxes;
        std::vector<int> class_ids;
        std::vector<float> confidences;

        runYOLO(frame, boxes, class_ids, confidences);

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        for (int idx : indices) {
            detected_objects.push_back({boxes[idx], class_ids[idx], confidences[idx], -1});
        }

        assign_ids_based_on_color(frame, detected_objects);

        for (const auto& obj : detected_objects) {
            cv::rectangle(frame, obj.box, cv::Scalar(0, 0, 255), 2); // Green bounding box for Topic 1
            std::string label = class_labels_[obj.class_id] + " ID: " + std::to_string(obj.id) + " Conf: " + cv::format("%.2f", obj.confidence);
            cv::putText(frame, label, obj.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }

    void assign_ids_based_on_color(const cv::Mat& frame, std::vector<TrackedObject>& detected_objects) {
        cv::Rect frame_rect(0, 0, frame.cols, frame.rows);

        for (auto& obj : detected_objects) {
            cv::Rect valid_box = obj.box & frame_rect;
            if (valid_box.width > 0 && valid_box.height > 0) {
                cv::Mat roi = frame(valid_box);
                cv::Scalar mean_color = cv::mean(roi);
                if (is_white(mean_color)) {
                    obj.id = 1;  // White
                } else if (is_black(mean_color)) {
                    obj.id = 2;  // Black
                } else {
                    obj.id = 2; 
                }
            } else {
                obj.id = 3;  
            }
        }
    }

    bool is_white(const cv::Scalar& color) {
        return (color[0] > 90 && color[1] > 90 && color[2] > 90);
    }

    bool is_black(const cv::Scalar& color) {
        return (color[0] < 5 && color[1] < 5 && color[2] < 5);
    }

    void display_combined_frames() {
        if (frame_raw_.empty() || frame_edge_.empty()) {
            return;
        }

        cv::Mat overlayed_frame = frame_edge_.clone();

        // Overlay green and blue bounding boxes (Topic 1 and Topic 2)
        for (auto& obj_raw : objects_raw_) {
            for (auto& obj_edge : objects_edge_) {
                if (obj_raw.id == obj_edge.id) {
                    // Draw green bounding box for Topic 1
                    cv::rectangle(frame_raw_, obj_raw.box, cv::Scalar(0, 0, 255), 2); // red

                    // Draw blue bounding box for Topic 2
                    cv::rectangle(overlayed_frame, obj_edge.box, cv::Scalar(0, 255, 0), 2); // green

                    // Fused bounding box 
                    cv::Rect fused_box = fuse_bounding_boxes(obj_raw.box, obj_edge.box, 0.8, 0.2);
                    cv::rectangle(overlayed_frame, fused_box, cv::Scalar(255, 0, 0), 2); // blue (Resultant)
                }
            }
        }

        // Combine the raw and edge frames side by side
        cv::Mat combined;
        cv::hconcat(frame_raw_, overlayed_frame, combined);

        cv::imshow("Combined View (Green | Blue | Red Resultant)", combined);
        cv::waitKey(1);
    }

    cv::Rect fuse_bounding_boxes(const cv::Rect& box1, const cv::Rect& box2, float weight1, float weight2) {
        // Calculate the weighted center points
        int center_x = static_cast<int>((weight1 * (box1.x + box1.width / 2) + weight2 * (box2.x + box2.width / 2)) / (weight1 + weight2));
        int center_y = static_cast<int>((weight1 * (box1.y + box1.height / 2) + weight2 * (box2.y + box2.height / 2)) / (weight1 + weight2));

        // Calculate the weighted width and height
        int width = static_cast<int>((weight1 * box1.width + weight2 * box2.width) / (weight1 + weight2));
        int height = static_cast<int>((weight1 * box1.height + weight2 * box2.height) / (weight1 + weight2));

        // Create the fused bounding box
        cv::Rect fused_box(center_x - width / 2, center_y - height / 2, width, height);
        return fused_box;
    }

  void runYOLO(cv::Mat& frame, std::vector<cv::Rect>& boxes, std::vector<int>& class_ids, std::vector<float>& confidences) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    std::vector<std::string> output_layer_names = get_output_layers(net_);
    net_.forward(outputs, output_layer_names);

    post_process(frame, outputs, boxes, class_ids, confidences);
}

    void post_process(cv::Mat& frame, const std::vector<cv::Mat>& outputs,
                      std::vector<cv::Rect>& boxes, std::vector<int>& class_ids, std::vector<float>& confidences) {
        for (const auto& output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                float* data = (float*)output.data + i * output.cols;
                float confidence = data[4];

                if (confidence > 0.5) {
                    int center_x = (int)(data[0] * frame.cols);
                    int center_y = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int class_id = -1;
                    float max_class_prob = -1.0;
                    for (int j = 5; j < output.cols; j++) {
                        if (data[j] > max_class_prob) {
                            max_class_prob = data[j];
                            class_id = j - 5;
                        }
                    }

                    boxes.push_back(cv::Rect(center_x - width / 2, center_y - height / 2, width, height));
                    class_ids.push_back(class_id);
                    confidences.push_back(confidence);
                }
            }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_edge_;
    cv::dnn::Net net_;
    std::vector<std::string> class_labels_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetectionNode>());
    rclcpp::shutdown();
    return 0;
}


