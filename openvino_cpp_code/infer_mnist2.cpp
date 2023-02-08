
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include<string>
#include<iostream>

//Configure constant parameters
std::string IMAGE_PATH = "C:\\Users\\19749\\Desktop\\train24.jpg";
std::string MODEL_PATH = "F:\\AI\\ML\\workplace\\my_inference\\FP32\\mnist_FP32.xml";
std::string DEVICE = "CPU";


int main() {
	//1.Create Runtime Core
	ov::Core core;

	//2.Compile the model
	ov::CompiledModel compiled_model = core.compile_model(MODEL_PATH, DEVICE);

	//3.Create inference request
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	//4.Set inputs
	ov::Tensor input_tensor = infer_request.get_input_tensor(0);
	ov::Shape tensor_shape = input_tensor.get_shape();

	//get shape 
	size_t channel = tensor_shape[1];
	size_t height = tensor_shape[2];
	size_t width = tensor_shape[3];

	//Load image
	cv::Mat src = cv::imread(IMAGE_PATH, 1);
	cv::Mat blob_image;
	cv::resize(src, blob_image, cv::Size(width, height));

	//Mat --> IE
	float* image_data = input_tensor.data<float>();

	for (size_t c = 0; c < channel; c++) {
		for (size_t h = 0; h < height; h++) {
			for (size_t w = 0; w < width; w++) {
				size_t index = c * width * height + h * width + w;
				image_data[index] = blob_image.at<cv::Vec3b>(h, w)[c];
			}
		}
	}

	//5.Start inference
	clock_t start_time, end_time;
	start_time = clock();
	infer_request.infer();
	end_time = clock();
	std::cout << "Inference Time:" << (double)(end_time - start_time) << "ms" << std::endl;

	//6.Get output
	auto output = infer_request.get_output_tensor();

	//7.Process output
	const float* output_buffer = output.data<const float>();
	int predict = std::max_element(output_buffer, output_buffer + 10) - output_buffer;
	std::cout << "The prediction digit is:" << predict << std::endl;


	//Visualization
	cv::Mat img, binary;
	cv::threshold(src, binary, 125, 255, cv::THRESH_BINARY_INV);
	cv::resize(binary, img, cv::Size(280, 280), cv::INTER_LINEAR);
	std::string showing1 = "Infer time:" + std::to_string(end_time - start_time) + " ms";
	std::string showing2 = "Prediction:" + std::to_string(predict);
	cv::putText(img, showing1, cv::Point2f(0, 15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1.8, cv::LINE_8);
	cv::putText(img, showing2, cv::Point2f(0, 35), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1.8, cv::LINE_8);
	cv::imshow("Result", img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}