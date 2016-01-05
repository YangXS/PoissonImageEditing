//定义Poisson类

class Poisson
{

private:

	cv::Mat s_img;                      //原图像
	cv::Mat gray_img;                   //原图像的灰度形式
	cv::Mat s_mask;                     //待处理区域的mask二值图像
	cv::Mat t_img;                      //colning的目标图像
	//定义存取cv::Mat类型matrix的函数(使用这些函数简便，程序可读性好，不使用此函数，直接存取速度快：
	inline uchar Poisson::get8(cv::Mat src,int i,int j,int c)
	{
		//uchar * d = (uchar*)(src.data + i*src.step);
		return ((uchar*)(src.data + i*src.step))[j*src.channels()+c];
	}
	inline void Poisson::set8(cv::Mat src,int i,int j,int c,uchar value)
	{
		//uchar * d = (uchar*)(src.data + i*src.step);
		((uchar*)(src.data + i*src.step))[j*src.channels()+c] = value;
	}
	inline float Poisson::get32(cv::Mat src,int i,int j,int c)
	{
		//float * d = (float*)(src.data + i*src.step);
		return ((float*)(src.data + i*src.step))[j*src.channels()+c];
	}
	inline void Poisson::set32(cv::Mat src,int i,int j,int c,float value)
	{
		//float * d = (float*)(src.data + i*src.step);
		((float*)(src.data + i*src.step))[j*src.channels()+c] = value;
	}

public:

	//构造函数和析构函数
	Poisson(const std::string & s_name); //初始化til类型
	Poisson(const std::string & s_name,const std::string & mask_name);  //初始化flat，illum，color，decolor类型
	Poisson(const std::string & s_name,const std::string & mask_name,const std::string & t_name);  //初始化import cloning，mix cloning类型
	~Poisson();

	//实现论文中各种editing的功能函数
	cv::Mat PoissonImportCloning(int x = 0,int y = 0,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonMixCloning(int x = 0,int y = 0,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonFlat(float threshold1 = 40,float threshold2 = 50,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonIllum(float alpha = 0.05,float beta = 0.5, float eps = 0.000001f,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonColor(float color_r_ratio = 1.5,float color_g_ratio = 0.5,float color_b_ratio = 0.5,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonDeColor(float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonTil(int Nx = 0,int Ny = 0,float rho = 1.9,int iterN = 1000);

};