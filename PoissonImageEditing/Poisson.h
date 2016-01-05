//����Poisson��

class Poisson
{

private:

	cv::Mat s_img;                      //ԭͼ��
	cv::Mat gray_img;                   //ԭͼ��ĻҶ���ʽ
	cv::Mat s_mask;                     //�����������mask��ֵͼ��
	cv::Mat t_img;                      //colning��Ŀ��ͼ��
	//�����ȡcv::Mat����matrix�ĺ���(ʹ����Щ������㣬����ɶ��Ժã���ʹ�ô˺�����ֱ�Ӵ�ȡ�ٶȿ죺
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

	//���캯������������
	Poisson(const std::string & s_name); //��ʼ��til����
	Poisson(const std::string & s_name,const std::string & mask_name);  //��ʼ��flat��illum��color��decolor����
	Poisson(const std::string & s_name,const std::string & mask_name,const std::string & t_name);  //��ʼ��import cloning��mix cloning����
	~Poisson();

	//ʵ�������и���editing�Ĺ��ܺ���
	cv::Mat PoissonImportCloning(int x = 0,int y = 0,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonMixCloning(int x = 0,int y = 0,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonFlat(float threshold1 = 40,float threshold2 = 50,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonIllum(float alpha = 0.05,float beta = 0.5, float eps = 0.000001f,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonColor(float color_r_ratio = 1.5,float color_g_ratio = 0.5,float color_b_ratio = 0.5,float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonDeColor(float rho = 1.9,int iterN = 1000);
	cv::Mat PoissonTil(int Nx = 0,int Ny = 0,float rho = 1.9,int iterN = 1000);

};