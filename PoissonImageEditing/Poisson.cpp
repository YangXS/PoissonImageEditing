#include "stdafx.h"
#include "time.h"
#include "string.h"
#include "cv.h"
#include "highgui.h"
#include "Poisson.h"


Poisson::Poisson(const std::string & s_name)
{
	s_img = cv::imread(s_name);
	gray_img = cv::imread(s_name,0);
	s_mask = cv::imread(s_name,0);
	t_img = cv::imread(s_name);
}

Poisson::Poisson(const std::string & s_name,const std::string & mask_name)
{
	s_img = cv::imread(s_name);
	gray_img = cv::imread(s_name,0);
	s_mask = cv::imread(mask_name,0);
	t_img = cv::imread(s_name);
}

Poisson::Poisson(const std::string & s_name,const std::string & mask_name,const std::string & t_name)
{
	s_img = cv::imread(s_name);
	gray_img = cv::imread(s_name,0);
	s_mask = cv::imread(mask_name,0);
	t_img = cv::imread(t_name);
}

Poisson::~Poisson(){}

cv::Mat Poisson::PoissonImportCloning(int x,int y,float rho,int iterN)
{   //rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	//x为原图像中所选区域在目标图像中x方向的位移
	//y为原图像中所选区域在目标图像中y方向的位移

	//检测mask指定的区域移动到目标图像中的下标是否越界
	for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			int Erro = 0;
			if(get8(s_mask,i,j,0)){
				if(i+y < 0){
					printf("Erro: index overflow in the top of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(i+y > t_img.rows-1){
					printf("Erro: index overflow in the bottom of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(j+x < 0){
					printf("Erro: index overflow in the left of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(j+x > t_img.cols-1){
					printf("Erro: index overflow in the right of the target image after offset!\n");
					Erro++;
					system("pause");
				}
			}
			if(Erro){
				exit(0);
			}
		}
	}

	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	t_img.convertTo(t_img,CV_32FC3);
	cv::Mat o_img;
	t_img.copyTo(o_img);

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < o_img.channels();c++){
		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0 && (j-1)+x >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i+y,j-1+x,c);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i+y,j-1+x,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
						}
						if((j+1) < s_img.cols && (j+1)+x < o_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i+y,j+1+x,c);
							}
							else{
								sum += get32(t_img,i+y,j+1+x,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
						}
						if((i-1) >= 0 && (i-1)+y >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1+y,j+x,c);
							}
							else{
								sum += get32(t_img,i-1+y,j+x,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
						}
						if((i+1) < s_img.rows && (i+1)+y < o_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1+y,j+x,c);
							}
							else{
								sum += get32(t_img,i+1+y,j+x,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
						}
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(o_img,i+y,j+x,c) + rho/Np * sum;
						set32(o_img,i+y,j+x,c,newValue);
					}
				}
				
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}

cv::Mat Poisson::PoissonMixCloning(int x,int y,float rho,int iterN)
{   //rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	//x为原图像中所选区域在目标图像中x方向的位移
	//y为原图像中所选区域在目标图像中y方向的位移

	//检测mask指定的区域移动到目标图像中的下标是否越界
	for(int i = 0;i < s_mask.rows;i++){
		for(int j = 0;j < s_mask.cols;j++){
			int Erro = 0;
			if(get8(s_mask,i,j,0)){
				if(i+y < 0){
					printf("Erro: index overflow in the top of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(i+y > t_img.rows-1){
					printf("Erro: index overflow in the bottom of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(j+x < 0){
					printf("Erro: index overflow in the left of the target image after offset!\n");
					Erro++;
					system("pause");
				}
				if(j+x > t_img.cols-1){
					printf("Erro: index overflow in the right of the target image after offset!\n");
					Erro++;
					system("pause");
				}
			}
			if(Erro){
				exit(0);
			}
		}
	}

	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	t_img.convertTo(t_img,CV_32FC3);
	cv::Mat o_img;
	t_img.copyTo(o_img);
	printf("start Gaussian Siedal iteration:\n");
	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < o_img.channels();c++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						float Vpq1 = 0;
						float Vpq2 = 0;
						//求四邻域的像素和
						if((j-1) >= 0 && (j-1)+x >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i+y,j-1+x,c);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i+y,j-1+x,c);
							}
							Vpq1 = (get32(t_img,i+y,j+x,c) - get32(t_img,i+y,j-1+x,c));
							Vpq2 = (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((j+1) < s_img.cols && (j+1)+x < o_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i+y,j+1+x,c);
							}
							else{
								sum += get32(t_img,i+y,j+1+x,c);
							}
							Vpq1 = (get32(t_img,i+y,j+x,c) - get32(t_img,i+y,j+1+x,c));
							Vpq2 = (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((i-1) >= 0 && (i-1)+y >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1+y,j+x,c);
							}
							else{
								sum += get32(t_img,i-1+y,j+x,c);
							}
							Vpq1 = (get32(t_img,i+y,j+x,c) - get32(t_img,i-1+y,j+x,c));
							Vpq2 = (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						if((i+1) < s_img.rows && (i+1)+y < o_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1+y,j+x,c);
							}
							else{
								sum += get32(t_img,i+1+y,j+x,c);
							}
							Vpq1 = (get32(t_img,i+y,j+x,c) - get32(t_img,i+1+y,j+x,c));
							Vpq2 = (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
							sum += ( abs(Vpq1) > abs(Vpq2) ? Vpq1 : Vpq2);
						}
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(o_img,i+y,j+x,c) + rho/Np * sum;
						set32(o_img,i+y,j+x,c,newValue);
					}
				}
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}

cv::Mat Poisson::PoissonFlat(float threshold1,float threshold2,float rho,int iterN)
{   //threshold1和threshold2参数为canny边缘检测中的两个阈值
	//rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat o_img;
	s_img.copyTo(o_img);
	cv::Mat edge_img(s_img.rows,s_img.cols,CV_8UC1);
	cv::Canny(gray_img,edge_img,threshold1,threshold2);

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < o_img.channels();c++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							sum += get32(o_img,i,j-1,c);
							if(get8(edge_img,i,j,0) || get8(edge_img,i,j-1,0)){
								sum += (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
							}
						}
						if((j+1) < s_img.cols){ 
							++Np;
							sum += get32(o_img,i,j+1,c);
							if(get8(edge_img,i,j,0) || get8(edge_img,i,j+1,0)){
								sum += (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
							}
						}
						if((i-1) >= 0){
							++Np;
							sum += get32(o_img,i-1,j,c);
							if(get8(edge_img,i,j,0) || get8(edge_img,i-1,j,0)){
								sum += (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
							}
						}
						if((i+1) < s_img.rows){
							++Np;
							sum += get32(o_img,i+1,j,c);
							if(get8(edge_img,i,j,0) || get8(edge_img,i+1,j,0)){
								sum += (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
							}
						}
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(o_img,i,j,c) + rho/Np * sum;
						set32(o_img,i,j,c,newValue);
					}
				}
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}

cv::Mat Poisson::PoissonIllum(float alpha,float beta, float eps,float rho,int iterN)       //自动调整rgb
{   //alpha和beta对应论文(16)式对应的参数，eps防止论文(16)式对应的有负指数的项的基数部分是零
	//rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat log_Simg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Timg(s_img.rows,s_img.cols,s_img.type());
	cv::Mat log_Oimg(s_img.rows,s_img.cols,s_img.type());

	for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			for(int c = 0;c < 3;c++){
				float value = get32(s_img,i,j,c);
				set32(log_Timg,i,j,c,std::log(value+eps));
				set32(log_Oimg,i,j,c,std::log(value+eps));
				set32(log_Simg,i,j,c,std::log(value+eps));
			}
		}
	}

	cv::Mat log_sobel1(s_img.rows,s_img.cols,s_img.type());
	cv::Sobel(log_Simg,log_sobel1,log_sobel1.depth(),1,0,3);
	cv::Mat log_sobel2(s_img.rows,s_img.cols,s_img.type());
	cv::Sobel(log_Simg,log_sobel2,log_sobel2.depth(),0,1,3);
	cv::Mat log_sobel = log_sobel1 + log_sobel2;

	//求解论文中(16)式需要用到的 average gradient norm
	float avGN[3] = {0,0,0};
	int num = 0;
	for(int i = 0;i < log_sobel.rows;i++){
		for(int j = 0;j < log_sobel.cols;j++){
			if(get8(s_mask,i,j,0)){
				for(int c = 0;c < log_sobel.channels();c++){
					avGN[c] += std::abs(get32(log_sobel,i,j,c));
				}
				num++;
			}
		}
	}
	for(int c = 0;c < log_sobel.channels();c++){
		avGN[c] /= num;
	}

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < 3;c++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						float Fp = get32(log_Simg,i,j,c);
						float Fq;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(log_Oimg,i,j-1,c);							
							}
							else{ //位于边界上的邻域点
								sum += get32(log_Timg,i,j-1,c);
							}
							Fq = get32(log_Simg,i,j-1,c);
							sum += std::pow(alpha*avGN[c],beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
						}
						if((j+1) < s_img.cols){ 
							++Np;							
							if(get8(s_mask,i,j+1,0)){		
								sum += get32(log_Oimg,i,j+1,c);								
							}
							else{
								sum += get32(log_Timg,i,j+1,c);
							}
							Fq = get32(log_Simg,i,j+1,c);
							sum += std::pow(alpha*avGN[c],beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
						}
						if((i-1) >= 0){
							++Np;							
							if(get8(s_mask,i-1,j,0)){
								sum += get32(log_Oimg,i-1,j,c);								
							}
							else{
								sum += get32(log_Timg,i-1,j,c);
							}
							Fq = get32(log_Simg,i-1,j,c);
							sum += std::pow(alpha*avGN[c],beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
						}
						if((i+1) < s_img.rows){
							++Np;							
							if(get8(s_mask,i+1,j,0)){
								sum += get32(log_Oimg,i+1,j,c);								
							}
							else{
								sum += get32(log_Timg,i+1,j,c);
							}
							Fq = get32(log_Simg,i+1,j,c);
							sum += std::pow(alpha*avGN[c],beta)*std::pow(std::abs(Fp-Fq)+eps,-beta)*(Fp-Fq);
						}	
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(log_Oimg,i,j,c) + rho/Np * sum;
						set32(log_Oimg,i,j,c,newValue);
					}
				}				
			}
		}
	}
	for(int i = 0;i < log_Oimg.rows;i++){
		for(int j = 0;j < log_Oimg.cols;j++){
			for(int c = 0;c < 3;c++){
				set32(log_Oimg,i,j,c,std::exp(get32(log_Oimg,i,j,c)));
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return log_Oimg;
}

cv::Mat Poisson::PoissonDeColor(float rho,int iterN)     // rho = 1.9; 迭代1000次以上
{	//rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数

	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat t_img(s_img.rows,s_img.cols,s_img.type());
	cv::Mat o_img(t_img.rows,t_img.cols,t_img.type());
	cv::Mat gray_img;
	cv::cvtColor(s_img,gray_img,CV_BGR2GRAY);
	for(int i = 0;i < t_img.rows;i++){
		for(int j = 0;j < t_img.cols;j++){
			float value = get32(gray_img,i,j,0);
			set32(t_img,i,j,0,value);
			set32(t_img,i,j,1,value);
			set32(t_img,i,j,2,value);
		}
	}
	t_img.copyTo(o_img);

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < o_img.channels();c++){
		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i,j-1,c);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i,j-1,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
						}
						if((j+1) < s_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i,j+1,c);
							}
							else{
								sum += get32(t_img,i,j+1,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
						}
						if((i-1) >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1,j,c);
							}
							else{
								sum += get32(t_img,i-1,j,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
						}
						if((i+1) < s_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1,j,c);
							}
							else{
								sum += get32(t_img,i+1,j,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
						}
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(o_img,i,j,c) + rho/Np * sum;
						set32(o_img,i,j,c,newValue);
					}
				}
				
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}

cv::Mat Poisson::PoissonColor(float color_r_ratio,float color_g_ratio,float color_b_ratio,float rho,int iterN)   // rho = 1.9; 迭代1000次以上
{   //rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	//color_r_ratio为color变换中的r通道颜色变换比率
	//color_g_ratio为color变换中的r通道颜色变换比率
	//color_b_ratio为color变换中的r通道颜色变换比率

	clock_t timeBegin = clock();

	s_mask.convertTo(s_mask,CV_8UC1);  //注意：如果imread()第二个参数不设为0，这里转换后的s_mask仍然是3通道；
	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat t_img;
	s_img.copyTo(t_img);
	cv::Mat o_img(s_img.rows,s_img.cols,s_img.type());
	s_img.copyTo(o_img);

	for(int i = 0;i < s_img.rows;i++){
		for(int j = 0;j < s_img.cols;j++){
			set32(s_img,i,j,0,color_b_ratio*get32(s_img,i,j,0));
			set32(s_img,i,j,1,color_g_ratio*get32(s_img,i,j,1));
			set32(s_img,i,j,2,color_r_ratio*get32(s_img,i,j,2));
		}
	}

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int c = 0;c < o_img.channels();c++){		
			for(int i = 0;i < s_mask.rows;i++){
				for(int j = 0;j < s_mask.cols;j++){
					if(get8(s_mask,i,j,0)){
						int Np = 0;
						float sum = 0;
						//求四邻域的像素和
						if((j-1) >= 0){ //判断边界
							++Np;
							if(get8(s_mask,i,j-1,0)){   //位于omega内的邻域点
								sum += get32(o_img,i,j-1,c);
							}
							else{ //位于边界上的邻域点
								sum += get32(t_img,i,j-1,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
						}
						if((j+1) < s_img.cols){ 
							++Np;
							if(get8(s_mask,i,j+1,0)){
								sum += get32(o_img,i,j+1,c);
							}
							else{
								sum += get32(t_img,i,j+1,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
						}
						if((i-1) >= 0){
							++Np;
							if(get8(s_mask,i-1,j,0)){
								sum += get32(o_img,i-1,j,c);
							}
							else{
								sum += get32(t_img,i-1,j,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
						}
						if((i+1) < s_img.rows){
							++Np;
							if(get8(s_mask,i+1,j,0)){
								sum += get32(o_img,i+1,j,c);
							}
							else{
								sum += get32(t_img,i+1,j,c);
							}
							sum += (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
						}
						//更新当前位置的像素值
						float newValue = (1-rho)*get32(o_img,i,j,c) + rho/Np * sum;
						set32(o_img,i,j,c,newValue);

					}
				}
				
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}

cv::Mat Poisson::PoissonTil(int Nx,int Ny,float rho,int iterN)
{   //rho为高斯-赛德尔迭代法的松弛因子，iterN为高斯-赛德尔迭代法的迭代次数
	//Nx为tiling中原图像在目标图像中x方向的重复次数
	//Ny为tiling中原图像在目标图像中y方向的重复次数

	clock_t timeBegin = clock();

	s_img.convertTo(s_img,CV_32FC3);
	cv::Mat t_img(s_img.rows*Ny-(Ny-1),s_img.cols*Nx-(Nx-1),s_img.type());
	cv::Mat o_img(t_img.rows,t_img.cols,t_img.type());
	for(int y = 0;y < Ny;y++){
		for(int x = 0;x < Nx;x++){
			for(int i = 0;i < s_img.rows;i++){
				for(int j = 0;j < s_img.cols;j++){
					for(int c = 0;c < s_img.channels();c++){
						float Value = get32(s_img,i,j,c);	
						if(i == 0 || i == s_img.rows-1){
							Value = 0.5f*(get32(s_img,0,j,c)+get32(s_img,s_img.rows-1,j,c));
						}
						if(j == 0 || j == s_img.cols-1){
							Value = 0.5f*(get32(s_img,i,0,c)+get32(s_img,i,s_img.cols-1,c));
						}
						set32(t_img,y*s_img.rows-y+i,x*s_img.cols-x+j,c,Value);
					}
				}
			}
		}
	}
	t_img.copyTo(o_img);

	printf("start Gaussian Siedal iteration:\n");

	for(int iter = 0;iter < iterN;iter++){
		//printf("iter:%d\n",iter);
		for(int y = 0;y < Ny;y++){
			for(int x = 0;x < Nx;x++){

				for(int c = 0;c < s_img.channels();c++){				
					for(int i = 0;i < s_img.rows;i++){
						for(int j = 0;j < s_img.cols;j++){
							if(i > 0 && i < s_img.rows-1 && j > 0 && j < s_img.cols-1){
								int Np = 0;
								float sum = 0;
								//求四邻域的像素和
								if((j-1) >= 0){ //判断边界
									++Np;
									if(i > 0 && i < s_img.rows-1 && j-1 > 0 && j-1 < s_img.cols-1){   //位于omega内的邻域点
										sum += get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j-1),c);
									}
									else{ //位于边界上的邻域点
										sum += get32(t_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j-1),c);
									}
									sum += (get32(s_img,i,j,c) - get32(s_img,i,j-1,c));
								}
								if((j+1) < s_img.cols){ 
									++Np;
									if(i > 0 && i < s_img.rows-1 && j+1 > 0 && j+1 < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j+1),c);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+(j+1),c);
									}
									sum += (get32(s_img,i,j,c) - get32(s_img,i,j+1,c));
								}
								if((i-1) >= 0){
									++Np;
									if(i-1 > 0 && i-1 < s_img.rows-1 && j > 0 && j < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+(i-1),(x*s_img.cols-x)+j,c);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+(i-1),(x*s_img.cols-x)+j,c);
									}
									sum += (get32(s_img,i,j,c) - get32(s_img,i-1,j,c));
								}
								if((i+1) < s_img.rows){
									++Np;
									if(i+1 > 0 && i+1 < s_img.rows-1 && j > 0 && j < s_img.cols-1){
										sum += get32(o_img,(y*s_img.rows-y)+(i+1),(x*s_img.cols-x)+j,c);
									}
									else{
										sum += get32(t_img,(y*s_img.rows-y)+(i+1),(x*s_img.cols-x)+j,c);
									}
									sum += (get32(s_img,i,j,c) - get32(s_img,i+1,j,c));
								}
								//更新当前位置的像素值
								float newValue = (1-rho)*get32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+j,c) + rho/Np * sum;
								set32(o_img,(y*s_img.rows-y)+i,(x*s_img.cols-x)+j,c,newValue);
							}
						}
						
					}
				}//c
			}
		}
	}
	clock_t timeFinish = clock();
	printf("Computing time: %fs\n",(timeFinish - timeBegin)/1000.f);
	return o_img;
}