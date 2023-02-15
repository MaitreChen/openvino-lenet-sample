#include<string> //C++ string标准库
#include<inference_engine.hpp> //Inference Engine库
#include<samples/ocv_common.hpp>//OpenCV库及matU8ToBlob函数定义
#include<time.h>
#include<map>

using namespace InferenceEngine;

//从opencv mat对象中复制数据到IE 输入Blob对象
void frameToBlob(const cv::Mat& frame, InferRequest& inferRequest, const std::string& inputName) {
	Blob::Ptr frameBlob = inferRequest.GetBlob(inputName); 
	matU8ToBlob<uint8_t>(frame, frameBlob);
}

//配置常量
std::string DEVICE = "CPU";
std::string IR_File_PATH = "mnist2_FP32";
std::string imageFile = "img.jpg";


int main(void) {
	//1.Load Plugin
	Core ie;

	//2.Reading IR File
	CNNNetwork network = ie.ReadNetwork(IR_File_PATH + ".xml", IR_File_PATH+".bin");
	network.setBatchSize(1);

	//3.Configure input and output
	InputInfo::Ptr input_data = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;
	input_data->setPrecision(Precision::U8);
	input_data->setLayout(Layout::NCHW);

	std::cout << "The input name："<< input_name << std::endl;
	//const SizeVector inputDims = input_data->getTensorDesc().getDims();
	//std::cout << "The shape is：" << inputDims[0] << " " << inputDims[1] << " " << inputDims[2] << " " << inputDims[3] << std::endl;


	//output 
	DataPtr output_data = network.getOutputsInfo().begin()->second;
	std::string output_name = network.getOutputsInfo().begin()->first;
	output_data->setPrecision(Precision::FP32);
	std::cout << "The output name：" << output_name << std::endl;

	//4.Load model into device
	ExecutableNetwork executable_network = ie.LoadNetwork(network, DEVICE);


	//5.Create Infer Request
	InferRequest infer_request = executable_network.CreateInferRequest();


	//6.Prepare Input
	cv::Mat img = cv::imread(imageFile,1);
	std::cout << img.channels() << std::endl;
	frameToBlob(img, infer_request, input_name);//data

	//7.Start inference
	std::clock_t begin, end;
	begin = std::clock();
	infer_request.Infer();
	end = std::clock();
	std::ostringstream infer_time;
	std::cout<< "Infer Time:" << (double)(end*1.0 - begin*1.0) << "ms" << std::endl;

	//8.Process output blobs
	const float* classfication = infer_request.GetBlob(output_name)->buffer().as<float*>();

	float max_pb = classfication[0];
	int idx = 0;//标签编号
	for (int i = 1; i < 10; i++) {
		std::cout << classfication[i] << std::endl;
		if (classfication[i] >= max_pb) {
			max_pb = classfication[i];
			idx = i;
		}
	}
	std::cout << idx << std::endl;
	

	//9.Results visualization
	cv::Mat new_img, binary;
	cv::threshold(img, binary, 125, 255, cv::THRESH_BINARY_INV);
	cv::resize(binary, new_img,cv::Size(280,280), cv::INTER_LINEAR);
	std::string showing1 = "Infer time:" + std::to_string(end - begin) + " ms";
	std::string showing2 = "Result:" + std::to_string(idx);
	cv::putText(new_img, showing1, cv::Point2f(0, 15), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1.8,cv::LINE_8);
	cv::putText(new_img, showing2, cv::Point2f(0, 35), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 0, 255), 1.8,cv::LINE_8);
	cv::imshow("img", new_img);
	cv::waitKey(0);
	cv::destroyAllWindows();

	return 0;
}